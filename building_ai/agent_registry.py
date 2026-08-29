"""Typed read-only registry for formal BuildingAI AgentService tools."""
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel
class Permission(str,Enum): READ_ONLY='READ_ONLY'; WRITE='WRITE'; DANGEROUS='DANGEROUS'
class ProjectArgs(BaseModel): project_id:str
class ToolSpec(BaseModel): name:str; description:str; permission:Permission=Permission.READ_ONLY; timeout_seconds:int=15; input_schema:str='ProjectArgs'; output_schema:str='ToolResult'
class ToolRegistry:
    names=('get_project_summary','get_energy_summary','get_energy_timeseries','get_temperature_summary','get_equipment_summary','get_equipment_kpis','get_diagnostic_findings','get_energy_opportunities','get_semantic_mapping','get_point_timeseries')
    def __init__(self, tools): self.tools=tools; self.specs={n:ToolSpec(name=n,description=f'Formal BuildingAI read-only tool: {n}') for n in self.names}
    def call(self,name:str,args:ProjectArgs):
        if name not in self.specs: raise KeyError(name)
        return self.tools.call(name,project_id=args.project_id)
