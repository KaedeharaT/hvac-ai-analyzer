"""Typed read-only registry for formal BuildingAI AgentService tools."""
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field
class Permission(str,Enum): READ_ONLY='READ_ONLY'; WRITE='WRITE'; DANGEROUS='DANGEROUS'
class ProjectArgs(BaseModel): project_id:str
class KnowledgeArgs(BaseModel): query:str=Field(min_length=1,max_length=2000); top_k:int=Field(default=3,ge=1,le=10); knowledge_type:str|None=None; equipment_type:str|None=None
class ToolSpec(BaseModel): name:str; description:str; permission:Permission=Permission.READ_ONLY; timeout_seconds:int=15; input_schema:str='ProjectArgs'; output_schema:str='ToolResult'
class ToolPermissionError(PermissionError): pass
class ToolRegistry:
    names=('get_project_summary','get_analysis_results','get_energy_summary','get_energy_timeseries','get_temperature_summary','get_equipment_summary','get_equipment_kpis','get_diagnostic_findings','get_energy_opportunities','get_semantic_mapping','get_point_timeseries')
    def __init__(self, tools, knowledge=None, allowed_permissions=frozenset({Permission.READ_ONLY})):
        self.tools=tools; self.knowledge=knowledge; self.allowed_permissions=frozenset(allowed_permissions); self.specs={n:ToolSpec(name=n,description=f'Formal BuildingAI read-only tool: {n}') for n in self.names}; self.specs['search_knowledge']=ToolSpec(name='search_knowledge',description='Search untrusted BuildingAI knowledge chunks')
    def _authorize(self,name):
        spec=self.specs.get(name)
        if spec is None: raise KeyError(name)
        if spec.permission not in self.allowed_permissions: raise ToolPermissionError(f'{name} requires {spec.permission.value}; this runtime permits only READ_ONLY tools')
        return spec
    def call(self,name:str,args:ProjectArgs|KnowledgeArgs):
        self._authorize(name)
        if name=='search_knowledge':
            if self.knowledge is None: raise RuntimeError('Knowledge service is not configured')
            if not isinstance(args,KnowledgeArgs): raise TypeError('search_knowledge requires KnowledgeArgs')
            return self.knowledge.search(args.query,args.top_k)
        if not isinstance(args,ProjectArgs): raise TypeError(f'{name} requires ProjectArgs')
        return self.tools.call(name,project_id=args.project_id)
