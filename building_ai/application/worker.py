"""Local/Redis worker entry point; local queue is used when Redis is absent."""
from __future__ import annotations
import time
from building_ai.config import Settings
from building_ai.ui.context import ApplicationContext
from .tasks import TaskService

def run_task(task_id: str):
    """RQ-importable worker entry point; task state remains in SQLite."""
    context=ApplicationContext(Settings.load())
    return TaskService(context.database, context).run(task_id).__dict__

def main():
    settings=Settings.load()
    if settings.task_queue_backend != 'redis':
        raise RuntimeError('Worker requires BUILDING_AI_TASK_QUEUE=redis')
    try:
        import redis
        from rq import Queue, Worker
    except ImportError as exc:
        raise RuntimeError('Install redis and rq for the worker') from exc
    connection=redis.Redis.from_url(settings.redis_url)
    Worker([Queue('building-ai', connection=connection)], connection=connection).work()
if __name__=='__main__': main()
