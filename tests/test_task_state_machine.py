from types import SimpleNamespace
from building_ai.application.tasks import ALLOWED_TRANSITIONS, TaskService, TaskStatus
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
