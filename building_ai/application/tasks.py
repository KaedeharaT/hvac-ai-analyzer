"""Persistent local task queue; Redis workers can call the same execute method."""
from __future__ import annotations
import json, time, uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum

class TaskStatus(str, Enum):
    PENDING='PENDING'; RUNNING='RUNNING'; WAITING_REVIEW='WAITING_REVIEW'; SUCCEEDED='SUCCEEDED'; FAILED='FAILED'

@dataclass
class Task:
    task_id: str; project_id: str; idempotency_key: str; status: str; stage: str; progress: int
    created_at: str; started_at: str|None=None; finished_at: str|None=None; error: str|None=None; result: dict|None=None

class TaskService:
    """SQLite-backed service; no GUI/QThread state is used as task storage."""
    def __init__(self, database, context):
        self.database, self.context = database, context
        with database.connect() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS application_tasks (task_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, idempotency_key TEXT UNIQUE NOT NULL, payload TEXT NOT NULL)')
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='buildingai-worker')
    def submit_analysis(self, project_id: str) -> Task:
        project=self.context.projects.get(project_id)
        if not project: raise KeyError(project_id)
        key=f'{project_id}:{project.data_revision}:analysis-v1'
        with self.database.connect() as conn:
            row=conn.execute('SELECT payload FROM application_tasks WHERE idempotency_key=?',(key,)).fetchone()
            if row:
                return Task(**json.loads(row['payload']))
            item=Task(str(uuid.uuid4()),project_id,key,TaskStatus.PENDING.value,'PENDING',0,datetime.now(timezone.utc).isoformat())
            conn.execute('INSERT INTO application_tasks VALUES (?,?,?,?)',(item.task_id,project_id,key,json.dumps(asdict(item))))
        return item
    def submit_background(self, project_id: str) -> Task:
        """Normal local-development queue path; HTTP returns before analysis."""
        item=self.submit_analysis(project_id)
        if item.status == TaskStatus.PENDING.value:
            self._executor.submit(self.run, item.task_id)
        return item
    def get(self, task_id: str) -> Task|None:
        with self.database.connect() as conn: row=conn.execute('SELECT payload FROM application_tasks WHERE task_id=?',(task_id,)).fetchone()
        return Task(**json.loads(row['payload'])) if row else None
    def _save(self, task: Task):
        with self.database.connect() as conn: conn.execute('UPDATE application_tasks SET payload=? WHERE task_id=?',(json.dumps(asdict(task)),task.task_id))
    def run(self, task_id: str) -> Task:
        task=self.get(task_id)
        if not task: raise KeyError(task_id)
        if task.status==TaskStatus.SUCCEEDED.value: return task
        task.status=TaskStatus.RUNNING.value; task.stage='LOAD_DATA'; task.progress=10; task.started_at=datetime.now(timezone.utc).isoformat(); self._save(task)
        try:
            self.context.open_project(task.project_id)
            for stage, progress in [('EQUIPMENT_DISCOVERY',35),('ENERGY_ANALYSIS',60),('DIAGNOSIS',78),('OPPORTUNITY',90),('VALIDATION',96)]:
                task.stage,task.progress=stage,progress; self._save(task)
            result=self.context.ensure_analysis_results()
            task.result={'project_id':task.project_id,'finding_count':len(result.findings) if result else 0,'status':'current'}
            task.status=TaskStatus.SUCCEEDED.value; task.stage='FINALIZE'; task.progress=100
        except Exception as exc:
            task.status=TaskStatus.FAILED.value; task.error='ANALYSIS_FAILED'; task.result={'message':str(exc)[:300]}
        task.finished_at=datetime.now(timezone.utc).isoformat(); self._save(task); return task
