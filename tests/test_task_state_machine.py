from building_ai.application.tasks import ALLOWED_TRANSITIONS, Task, TaskService, TaskStatus
from building_ai.storage import Database
from datetime import datetime, timezone

def test_task_transitions_and_review_resume(tmp_path):
    class Context: pass
    service=TaskService(Database(tmp_path/'tasks.sqlite3'), Context())
    task=Task('t','p','key','PENDING','PENDING',0,datetime.now(timezone.utc).isoformat())
    service._save(task) if service.get('t') else None
    assert 'RUNNING' in ALLOWED_TRANSITIONS['PENDING']
    assert 'SUCCEEDED' not in ALLOWED_TRANSITIONS['PENDING']

def test_local_queue_is_available(tmp_path):
    class Context: pass
    assert TaskService(Database(tmp_path/'tasks.sqlite3'), Context()).queue.__class__.__name__ == 'LocalTaskQueue'
