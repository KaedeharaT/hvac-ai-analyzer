from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


class TaskWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function, self.args, self.kwargs = function, args, kwargs

    @pyqtSlot()
    def run(self):
        try:
            self.finished.emit(self.function(*self.args, **self.kwargs))
        except Exception as exc:
            self.failed.emit(str(exc))
