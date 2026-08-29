"""Bounded structured runtime facade for BuildingAI's read-only agent tools."""
from __future__ import annotations
import uuid
from typing import Literal
from pydantic import BaseModel

class Route(BaseModel):
    intent: Literal['general_chat','general_hvac_knowledge','project_summary','energy_analysis','equipment_analysis','diagnosis','recommendation','knowledge_search']
    confidence: float; requires_project: bool; requires_tools: bool; requires_knowledge: bool=False
class PlanStep(BaseModel): tool:str; arguments:dict; purpose:str
class Plan(BaseModel): steps:list[PlanStep]
class RuntimeResult(BaseModel): answer:str; trace_id:str; tools_used:list[str]; grounded:bool; abstained:bool; sources:list[dict]=[]
class AgentRuntime:
    def __init__(self,context): self.context=context
    def route(self,q:str)->Route:
        x=q.casefold()
        if any(t in x for t in ('你是谁','who are you','你会什么')): return Route(intent='general_chat',confidence=.99,requires_project=False,requires_tools=False)
        if any(t in x for t in ('一般','冷冻水','hvac','delta t','δt')): return Route(intent='general_hvac_knowledge',confidence=.9,requires_project=False,requires_tools=False,requires_knowledge=True)
        intent='energy_analysis' if any(t in x for t in ('能耗','昨天','energy','power')) else 'equipment_analysis' if any(t in x for t in ('设备','cop','效率')) else 'diagnosis' if any(t in x for t in ('为什么','异常','问题')) else 'project_summary'
        return Route(intent=intent,confidence=.8,requires_project=True,requires_tools=True)
    def plan(self,r:Route,pid:str|None)->Plan:
        names={'energy_analysis':['get_energy_summary','get_energy_timeseries','get_temperature_summary','get_equipment_summary','get_diagnostic_findings'],'equipment_analysis':['get_equipment_kpis','get_diagnostic_findings'],'diagnosis':['get_equipment_kpis','get_diagnostic_findings','get_energy_timeseries'],'project_summary':['get_project_summary','get_equipment_summary','get_analysis_results']}.get(r.intent,[])
        return Plan(steps=[PlanStep(tool=n,arguments={'project_id':pid},purpose=r.intent) for n in names])
    def run(self,q:str,pid:str|None=None)->RuntimeResult:
        if pid:self.context.ensure_project_loaded(pid)
        r=self.route(q); plan=self.plan(r,pid); evidence={}
        for step in plan.steps:
            x=self.context.agent.tools.call(step.tool,**step.arguments); evidence[step.tool]=x.data if x.ok else {'error':x.error}
        answer=self.context.agent_controller.answer(q) if not plan.steps else self.context.agent_controller._grounded_data_answer(q,self.context.agent_controller._equipment_name(q),True,evidence)
        return RuntimeResult(answer=answer,trace_id=str(uuid.uuid4()),tools_used=[x.tool for x in plan.steps],grounded=bool(plan.steps),abstained=('不足' in answer or '不存在' in answer))
