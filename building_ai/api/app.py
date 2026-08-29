"""FastAPI application (optional dependency, installed through server extras)."""
from __future__ import annotations
from building_ai.config import Settings
from building_ai.ui.context import ApplicationContext
from building_ai.application.tasks import TaskService
from building_ai.agent_runtime import AgentRuntime

def create_app(context=None):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError('FastAPI is required for the API server. Install requirements-server.txt.') from exc
    ctx=context or ApplicationContext(Settings.load()); tasks=TaskService(ctx.database,ctx); app=FastAPI(title='BuildingAI API',version='1.0')
    def result(data=None, error=None): return {'success':error is None,'data':data,'error':error,'trace_id':None,'request_id':None}
    @app.get('/health')
    def health(): return result({'api':'ok','storage':'ok','queue':'local','worker':'available','llm':ctx.llm_manager.get_provider().display_name})
    @app.get('/projects')
    def projects(): return result([{'project_id':p.project_id,'name':p.name,'time_range':p.time_range} for p in ctx.projects.list()])
    @app.get('/projects/{project_id}')
    def project(project_id:str):
        p=ctx.projects.get(project_id)
        if not p: raise HTTPException(404,detail=result(error={'code':'PROJECT_NOT_FOUND'}))
        return result({'project_id':p.project_id,'name':p.name,'time_range':p.time_range})
    @app.post('/projects/{project_id}/analysis')
    def analysis(project_id:str): return result(tasks.submit_background(project_id).__dict__)
    @app.post('/tasks/{task_id}/run')
    def run(task_id:str): return result(tasks.run(task_id).__dict__)
    @app.get('/tasks/{task_id}')
    def task(task_id:str):
        item=tasks.get(task_id)
        if not item: raise HTTPException(404,detail=result(error={'code':'TASK_NOT_FOUND'}))
        return result(item.__dict__)
    @app.get('/tasks/{task_id}/result')
    def task_result(task_id:str):
        item=tasks.get(task_id)
        if not item: raise HTTPException(404,detail=result(error={'code':'TASK_NOT_FOUND'}))
        return result(item.result if item.status=='SUCCEEDED' else None, None if item.status=='SUCCEEDED' else {'code':item.status})
    @app.get('/projects/{project_id}/analysis/latest')
    def latest(project_id:str):
        ctx.ensure_project_loaded(project_id); analysis=ctx.ensure_analysis_results()
        if not analysis: raise HTTPException(409,detail=result(error={'code':'DATA_NOT_AVAILABLE'}))
        return result({'project_id':project_id,'finding_count':len(analysis.findings),'opportunity_count':len(ctx.opportunities)})
    @app.post('/agent/chat')
    def chat(payload: dict):
        project_id, message = payload.get('project_id'), str(payload.get('message','')).strip()
        if not message: raise HTTPException(422,detail=result(error={'code':'VALIDATION_ERROR'}))
        if project_id: ctx.ensure_project_loaded(project_id)
        response=AgentRuntime(ctx).run(message,project_id)
        return result({**response.model_dump(),'project_id':project_id,'sources':response.sources})
    return app

app = create_app()
