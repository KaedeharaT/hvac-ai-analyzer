"""Persistent local task queue; Redis workers can call the same execute method."""
from __future__ import annotations
import json, time, uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum

class TaskStatus(str, Enum):
    PENDING='PENDING'; RUNNING='RUNNING'; WAITING_REVIEW='WAITING_REVIEW'; SUCCEEDED='SUCCEEDED'; FAILED='FAILED'

ALLOWED_TRANSITIONS={
    'PENDING': {'RUNNING','WAITING_REVIEW','FAILED'}, 'RUNNING': {'WAITING_REVIEW','SUCCEEDED','FAILED'},
    'WAITING_REVIEW': {'RUNNING','FAILED'}, 'SUCCEEDED': set(), 'FAILED': set(),
}

@dataclass
class Task:
    task_id: str; project_id: str; idempotency_key: str; status: str; stage: str; progress: int
    created_at: str; started_at: str|None=None; finished_at: str|None=None; error: str|None=None; result: dict|None=None
    retry_count: int=0; max_retries: int=2; timeout_seconds: int=600; review_items: list|None=None

class LocalTaskQueue:
    def __init__(self, executor): self.executor=executor
    def enqueue(self, callback, task_id): return self.executor.submit(callback, task_id)

class RedisTaskQueue:
    """Optional RQ adapter; local mode remains fully functional without it."""
    def __init__(self, redis_url):
        try:
            import redis
            from rq import Queue
        except ImportError as exc: raise RuntimeError('Install redis and rq for RedisTaskQueue') from exc
        self.queue=Queue(connection=redis.Redis.from_url(redis_url))
    def enqueue(self, callback, task_id): return self.queue.enqueue(callback, task_id)

class TaskService:
    """SQLite-backed service; no GUI/QThread state is used as task storage."""
    def __init__(self, database, context):
        self.database, self.context = database, context
        with database.connect() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS application_tasks (task_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, idempotency_key TEXT UNIQUE NOT NULL, payload TEXT NOT NULL)')
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='buildingai-worker'); self.queue=LocalTaskQueue(self._executor)
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
            self.queue.enqueue(self.run, item.task_id)
        return item
    def get(self, task_id: str) -> Task|None:
        with self.database.connect() as conn: row=conn.execute('SELECT payload FROM application_tasks WHERE task_id=?',(task_id,)).fetchone()
        return Task(**json.loads(row['payload'])) if row else None
    def _save(self, task: Task):
        with self.database.connect() as conn: conn.execute('UPDATE application_tasks SET payload=? WHERE task_id=?',(json.dumps(asdict(task)),task.task_id))
    def transition(self, task: Task, status: TaskStatus, stage: str|None=None):
        if status.value not in ALLOWED_TRANSITIONS.get(task.status,set()): raise ValueError(f'illegal transition {task.status}->{status.value}')
        task.status=status.value
        if stage: task.stage=stage
        self._save(task); return task
    def request_review(self, task_id: str, review_items: list):
        task=self.get(task_id)
        if not task: raise KeyError(task_id)
        task.review_items=review_items; return self.transition(task,TaskStatus.WAITING_REVIEW,'WAITING_REVIEW')
    def resume_review(self, task_id: str):
        task=self.get(task_id)
        if not task: raise KeyError(task_id)
        task.review_items=[]; self.transition(task,TaskStatus.RUNNING,'REVIEW_RESUMED'); self.queue.enqueue(self.run,task_id); return task
    def run(self, task_id: str) -> Task:
        task=self.get(task_id)
        if not task: raise KeyError(task_id)
        if task.status in {TaskStatus.SUCCEEDED.value,TaskStatus.WAITING_REVIEW.value}: return task
        if task.status==TaskStatus.PENDING.value: self.transition(task,TaskStatus.RUNNING,'LOAD_DATA')
        task.progress=10; task.started_at=datetime.now(timezone.utc).isoformat(); self._save(task)
        started=time.monotonic()
        try:
            self.context.open_project(task.project_id)
            for stage, progress in [('EQUIPMENT_DISCOVERY',35),('ENERGY_ANALYSIS',60),('DIAGNOSIS',78),('OPPORTUNITY',90),('VALIDATION',96)]:
                if time.monotonic()-started > task.timeout_seconds: raise TimeoutError('task timeout')
                task.stage,task.progress=stage,progress; self._save(task)
            result=self.context.ensure_analysis_results()
            task.result={'project_id':task.project_id,'finding_count':len(result.findings) if result else 0,'status':'current'}
            self.transition(task,TaskStatus.SUCCEEDED,'FINALIZE'); task.progress=100
        except Exception as exc:
            task.error='TASK_TIMEOUT' if isinstance(exc,TimeoutError) else 'ANALYSIS_FAILED'; task.result={'message':str(exc)[:300]}
            if not isinstance(exc,TimeoutError) and task.retry_count < task.max_retries:
                task.retry_count += 1; task.status=TaskStatus.PENDING.value; task.stage='RETRY'; self._save(task); self.queue.enqueue(self.run,task_id); return task
            self.transition(task,TaskStatus.FAILED,'FAILED')
        task.finished_at=datetime.now(timezone.utc).isoformat(); self._save(task); return task
