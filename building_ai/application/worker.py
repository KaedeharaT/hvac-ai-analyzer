"""Local/Redis worker entry point; local queue is used when Redis is absent."""
from __future__ import annotations
import time
from building_ai.config import Settings
from building_ai.ui.context import ApplicationContext
from .tasks import TaskService
def main():
    context=ApplicationContext(Settings.load()); service=TaskService(context.database,context)
    # LocalTaskQueue owns execution in desktop/API development.  A deployed RQ
    # worker imports TaskService.run through RedisTaskQueue.
    while True: time.sleep(60)
if __name__=='__main__': main()
