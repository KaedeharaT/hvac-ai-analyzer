from types import SimpleNamespace
from building_ai.application.tasks import ALLOWED_TRANSITIONS, RedisTaskQueue, TaskService, TaskStatus
from building_ai.storage import Database
from datetime import datetime, timezone

def test_task_transitions_and_review_resume(tmp_path):
    context=SimpleNamespace(projects=SimpleNamespace(get=lambda _:SimpleNamespace(data_revision=1)))
    service=TaskService(Database(tmp_path/'tasks.sqlite3'), context)
    task=service.submit_analysis('p')
    waiting=service.request_review(task.task_id,[{'point':'ambiguous'}])
    assert waiting.status == TaskStatus.WAITING_REVIEW.value
    service.queue=SimpleNamespace(enqueue=lambda *args: None)
    resumed=service.resume_review(task.task_id)
    assert resumed.status == TaskStatus.RUNNING.value
    assert 'RUNNING' in ALLOWED_TRANSITIONS['PENDING']
    assert 'SUCCEEDED' not in ALLOWED_TRANSITIONS['PENDING']

def test_task_timeout_is_bounded(tmp_path):
    context=SimpleNamespace(projects=SimpleNamespace(get=lambda _:SimpleNamespace(data_revision=1)),open_project=lambda _:None,ensure_analysis_results=lambda:None)
    service=TaskService(Database(tmp_path/'tasks.sqlite3'), context)
    task=service.submit_analysis('p'); task.timeout_seconds=0; service._save(task)
    result=service.run(task.task_id)
    assert result.status == TaskStatus.FAILED.value and result.error == 'TASK_TIMEOUT'

def test_local_queue_is_available(tmp_path):
    class Context: pass
    assert TaskService(Database(tmp_path/'tasks.sqlite3'), Context()).queue.__class__.__name__ == 'LocalTaskQueue'

def test_redis_adapter_dispatches_to_importable_worker():
    captured = {}
    class Queue:
        def __init__(self, name, connection): captured.update(name=name, connection=connection)
        def enqueue(self, target, task_id, **kwargs):
            captured.update(target=target, task_id=task_id, kwargs=kwargs); return 'queued'
    queue=RedisTaskQueue('redis://queue:6379/2', redis_factory=lambda url: {'url': url}, queue_factory=Queue)
    assert queue.enqueue('task-1', lambda _: None) == 'queued'
    assert captured == {
        'name': 'building-ai', 'connection': {'url': 'redis://queue:6379/2'},
        'target': 'building_ai.application.worker.run_task', 'task_id': 'task-1', 'kwargs': {},
    }

def test_task_retries_are_bounded_and_persisted(tmp_path):
    attempts=[]
    def open_project(_):
        attempts.append('open')
        raise RuntimeError('database unavailable')
    context=SimpleNamespace(
        projects=SimpleNamespace(get=lambda _:SimpleNamespace(data_revision=1)),
        open_project=open_project, ensure_analysis_results=lambda:None,
    )
    service=TaskService(Database(tmp_path/'tasks.sqlite3'), context)
    scheduled=[]; service.queue=SimpleNamespace(enqueue=lambda task_id, callback: scheduled.append(task_id))
    task=service.submit_analysis('p'); task.max_retries=1; service._save(task)
    first=service.run(task.task_id)
    assert first.status == TaskStatus.PENDING.value and first.retry_count == 1
    assert scheduled == [task.task_id] and service.get(task.task_id).error == 'ANALYSIS_FAILED'
    final=service.run(task.task_id)
    assert final.status == TaskStatus.FAILED.value and final.retry_count == 1
    assert final.error == 'ANALYSIS_FAILED' and final.result['message'] == 'database unavailable'
