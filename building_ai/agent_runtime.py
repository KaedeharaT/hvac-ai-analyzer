"""Bounded structured runtime facade for BuildingAI's read-only agent tools."""
from __future__ import annotations
import uuid
import re
from typing import Literal
from pydantic import BaseModel
from building_ai.evidence import check_reasoning
from building_ai.agent_registry import ToolRegistry, ProjectArgs, KnowledgeArgs
from building_ai.observability import TraceStore
from building_ai.memory import MemoryStore
from building_ai.knowledge import KnowledgeService
from building_ai.security import contains_prompt_injection, find_untrusted_instruction_paths

class Route(BaseModel):
    intent: Literal['general_chat','general_hvac_knowledge','project_summary','energy_analysis','equipment_analysis','diagnosis','recommendation','knowledge_search','drawing_query']
    confidence: float; requires_project: bool; requires_tools: bool; requires_knowledge: bool=False
class PlanStep(BaseModel): tool:str; arguments:dict; purpose:str
class Plan(BaseModel): steps:list[PlanStep]
class RuntimeResult(BaseModel): answer:str; trace_id:str; tools_used:list[str]; grounded:bool; abstained:bool; sources:list[dict]=[]
class AgentRuntime:
    def __init__(self,context): self.context=context
    def route(self,q:str)->Route:
        x=q.casefold()
        if any(t in x for t in ('图纸', 'drawing', 'floor plan', 'where is')):
            return Route(intent='drawing_query',confidence=.95,requires_project=True,requires_tools=True)
        if any(t in x for t in ('你是谁','who are you','你会什么','what can you do','introduce yourself','assistant are you','你能帮我')): return Route(intent='general_chat',confidence=.99,requires_project=False,requires_tools=False)
        if any(t in x for t in ('一般','冷冻水','hvac','暖通','delta t','δt','chilled water','chilled-water','pump','first checks','common hvac','part load')): return Route(intent='general_hvac_knowledge',confidence=.9,requires_project=False,requires_tools=False,requires_knowledge=True)
        if any(t in x for t in ('project overview','project 7','项目目前','项目总览','project results')): intent='project_summary'
        elif any(t in x for t in ('energy rise','higher energy')): intent='diagnosis'
        elif any(t in x for t in ('建议','机会','recommend','opportunit','improvement','improve first','improve it first','what should i improve','改善','应该先做什么','先做什么','先检查什么','quick win','save some energy','省点电','what should i do first','something simple','should look at first')): intent='recommendation'
        elif any(t in x for t in ('能耗','昨天','energy','power','electricity','consumption','load trend','demand','耗电','负荷趋势')): intent='energy_analysis'
        elif any(t in x for t in ('为什么','异常','问题','原因','诊断','why','fault','issue','problem','diagnosis','abnormal','strange','looks odd','怪怪','精确故障')): intent='diagnosis'
        elif any(t in x for t in ('设备','cop','效率','equipment','efficiency','kpi','performance','能效比','machine','heat source','chiller','unit','ahp-','ch-','worst')): intent='equipment_analysis'
        else: intent='project_summary'
        return Route(intent=intent,confidence=.8,requires_project=True,requires_tools=True)
    def plan(self,r:Route,pid:str|None)->Plan:
        names={'energy_analysis':['get_energy_summary','get_energy_timeseries','get_temperature_summary','get_equipment_summary','get_diagnostic_findings'],'equipment_analysis':['get_equipment_kpis','get_diagnostic_findings'],'diagnosis':['get_equipment_kpis','get_diagnostic_findings','get_energy_timeseries'],'recommendation':['get_energy_opportunities','get_diagnostic_findings','get_analysis_results'],'project_summary':['get_project_summary','get_equipment_summary','get_analysis_results']}.get(r.intent,[])
        return Plan(steps=[PlanStep(tool=n,arguments={'project_id':pid},purpose=r.intent) for n in names])
    def run(self,q:str,pid:str|None=None,conversation_id:str='default')->RuntimeResult:
        if pid:self.context.ensure_project_loaded(pid)
        trace_id=str(uuid.uuid4())
        if contains_prompt_injection(q):
            TraceStore(self.context.database).save({'trace_id':trace_id,'project_id':pid,'conversation_id':conversation_id,'query':q,'intent':'security_rejection','plan':[],'tool_calls':[],'evidence_checks':['INSUFFICIENT'],'reflections':[],'memory_used':[],'knowledge_sources':[],'llm_calls':[],'security':{'prompt_injection_detected':True,'untrusted_instruction_paths':['$.query']},'answer':'I can only provide bounded, read-only BuildingAI assistance; I cannot follow instruction-override requests.','grounded':False,'abstained':True,'status':'REJECTED'})
            return RuntimeResult(answer='I can only provide bounded, read-only BuildingAI assistance; I cannot follow instruction-override requests.',trace_id=trace_id,tools_used=[],grounded=False,abstained=True)
        memory=MemoryStore(self.context.database); focus=memory.get(pid,conversation_id,'focus','equipment') if pid else None
        if focus:
            equipment_id=focus.get('equipment_id','')
            q=q.replace('它的',equipment_id).replace('它',equipment_id)
            q=re.sub(r'\b(?:it|its)\b',equipment_id,q,flags=re.IGNORECASE)
            # Short follow-ups such as "what should I do first?" commonly
            # omit a pronoun.  Keep the conversation-scoped focus available
            # to the existing planner without changing general chat requests.
            if equipment_id and equipment_id.casefold() not in q.casefold() and not any(token in q.casefold() for token in ('who are you','你是谁','what can you do','你会什么','哪些设备','设备有哪些','what equipment','equipment list')):
                q=f'{q} {equipment_id}'
        equipment=self.context.agent_controller._equipment_name(q)
        if equipment and pid: memory.put(pid,conversation_id,'focus','equipment',{'equipment_id':equipment})
        r=self.route(q); plan=self.plan(r,pid)
        if r.intent == 'drawing_query':
            plan = Plan(steps=[PlanStep(
                tool='get_equipment_drawing_location' if equipment else 'get_drawing_summary',
                arguments={'project_id': pid, **({'equipment_id': equipment} if equipment else {})},
                purpose='drawing_query',
            )])
        evidence={}; knowledge=KnowledgeService(self.context.database); registry=ToolRegistry(self.context.agent.tools,knowledge); events=[]
        if r.intent=='general_hvac_knowledge' or (equipment and r.intent=='recommendation'):
            chunks=registry.call('search_knowledge',KnowledgeArgs(query=q,top_k=3)); evidence['search_knowledge']={'chunks':chunks}; events.append({'tool':'search_knowledge','success':True})
        for step in plan.steps:
            x=registry.call(step.tool,ProjectArgs(**step.arguments)); evidence[step.tool]=x.data if x.ok else {'error':x.error}; events.append({'tool':step.tool,'success':x.ok})
        # A KPI alone never explains a cause.  Re-plan with diagnostic and time
        # evidence under a fixed one-reflection budget.
        initial=check_reasoning(evidence, r.intent in {'diagnosis','equipment_analysis'})
        reflections=[]
        if initial.value == 'PARTIAL':
            reflections.append({'status':'PARTIAL','reason':'KPI evidence needs diagnostic/time-series cause evidence','action':'REPLAN'})
            for name in ('get_diagnostic_findings','get_energy_timeseries'):
                if name not in evidence:
                    x=registry.call(name,ProjectArgs(project_id=pid)); evidence[name]=x.data if x.ok else {'error':x.error}; events.append({'tool':name,'success':x.ok,'replan':True}); plan.steps.append(PlanStep(tool=name,arguments={'project_id':pid},purpose='reflection evidence'))
        llm_calls=[]
        abstention_reason=self._abstention_reason(evidence,events)
        if abstention_reason:
            answer=f'I cannot provide a reliable project answer: {abstention_reason}.'
        elif not plan.steps:
            answer=self.context.agent_controller.answer(q,llm_event=llm_calls.append)
        elif r.intent == 'drawing_query':
            location = evidence.get('get_equipment_drawing_location', {})
            if location.get('reliable'):
                row = location['locations'][0]
                answer = (f"{location['equipment_id']} 已关联到图纸 {row['file_name']}，第 {row['page_number']} 页，对象 {row['reviewed_class'] or row['class_name']}。"
                          if bool(re.search(r"[\u4e00-\u9fff]", q)) else
                          f"{location['equipment_id']} is associated with {row['file_name']}, page {row['page_number']}, object {row['reviewed_class'] or row['class_name']}.")
            elif equipment:
                answer = (f"{equipment} 当前尚未建立可靠的图纸设备关联。" if bool(re.search(r"[\u4e00-\u9fff]", q)) else f"No reliable drawing association has been confirmed for {equipment}.")
            else:
                answer = ("请先指定设备编号，才能查询其已确认的图纸位置。" if bool(re.search(r"[\u4e00-\u9fff]", q)) else "Specify an equipment ID to query a confirmed drawing location.")
        elif r.intent in {'diagnosis', 'equipment_analysis'} and hasattr(self.context.agent_controller, '_grounded_fallback'):
            # The planner's evidence is intentionally compact.  Rendering a
            # diagnosis/equipment answer from an absent project-summary tool
            # used to produce a misleading "0 rows" fallback.  Reuse the
            # controller's validated analysis view instead; it does not make
            # another inference or change the read-only tool boundary.
            chinese = bool(re.search(r"[\u4e00-\u9fff]", q))
            named_equipment = self.context.agent_controller._equipment_name(q)
            if named_equipment:
                answer = self.context.agent_controller._grounded_fallback(named_equipment, chinese)
            elif any(token in q.casefold() for token in ('最低', '最差', 'lowest', 'worst')):
                answer = self.context.agent_controller._lowest_cop_answer(chinese)
                if chinese:
                    answer += " 当前没有足够的确定性证据把该 KPI 排名归因于单一原因。"
                else:
                    answer += " Current evidence does not support assigning that KPI ranking to a single cause."
            elif 'cop' in q.casefold():
                answer = self.context.agent_controller._lowest_cop_answer(chinese)
            else:
                answer = self.context.agent_controller._grounded_fallback(None, chinese)
        elif hasattr(self.context.agent_controller, 'grounded_answer'):
            # A renderer may use the configured provider, but receives only
            # formal tool evidence after routing/planning is complete.
            answer=self.context.agent_controller.grounded_answer(q,self.context.agent_controller._equipment_name(q),evidence,llm_event=llm_calls.append)
        else:
            answer=self.context.agent_controller._grounded_data_answer(q,self.context.agent_controller._equipment_name(q),True,evidence)
        if pid:
            findings=evidence.get('get_diagnostic_findings',[])
            memory.put_project(pid,'last_analysis',{'project_id':pid,'intent':r.intent,'project_summary':evidence.get('get_project_summary'),'finding_count':len(findings) if isinstance(findings,list) else None})
        final=check_reasoning(evidence,r.intent in {'diagnosis','equipment_analysis'}).value
        sources=evidence.get('search_knowledge',{}).get('chunks',[])
        unsafe_paths=find_untrusted_instruction_paths(evidence)
        # Abstention is a structured evidence decision.  Do not infer it from
        # ordinary explanatory wording returned by an LLM (for example a valid
        # answer mentioning that a secondary detail is insufficient).
        abstained=bool(abstention_reason or (
            r.intent == 'drawing_query' and equipment and
            not evidence.get('get_equipment_drawing_location', {}).get('reliable')
        ))
        TraceStore(self.context.database).save({'trace_id':trace_id,'project_id':pid,'conversation_id':conversation_id,'query':q,'intent':r.intent,'plan':[s.model_dump() for s in plan.steps],'tool_calls':events,'evidence_checks':[initial.value,final],'reflections':reflections,'memory_used':[{'type':'focus','summary':focus}] if focus else [],'knowledge_sources':sources,'llm_calls':llm_calls,'security':{'prompt_injection_detected':bool(unsafe_paths),'untrusted_instruction_paths':unsafe_paths},'answer':answer,'grounded':bool(plan.steps),'abstained':abstained,'status':'ABSTAINED' if abstained else 'SUCCEEDED'})
        return RuntimeResult(answer=answer,trace_id=trace_id,tools_used=[x['tool'] for x in events],grounded=bool(plan.steps),abstained=abstained,sources=sources)

    @staticmethod
    def _abstention_reason(evidence:dict, events:list[dict])->str|None:
        """Turn explicit formal-data absence/failure signals into abstention.

        This is deliberately based only on typed tool outcomes, never on an
        LLM's confidence or an inferred value, so a failed data source cannot
        produce a fabricated project claim.
        """
        if any(not event['success'] for event in events): return 'a required read-only data tool failed'
        for value in evidence.values():
            if not isinstance(value,dict): continue
            if value.get('equipment_found') is False: return 'the requested equipment does not exist in this project'
            if value.get('data_available') is False: return 'the project has insufficient reliable data'
            if value.get('range_available') is False: return 'the requested time range has no reliable data'
        return None
