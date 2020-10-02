from PyQt5.QtCore import QObject, pyqtSignal
from deriva.qt import async_execute, Task


class LauncherTask(QObject):
    status_update_signal = pyqtSignal(bool, str, str, object)
    progress_update_signal = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super(LauncherTask, self).__init__(parent)
        self.task = None

    def start(self):
        async_execute(self.task)

    def cancel(self):
        self.task.cancel()

    def set_status(self, success, status, detail, result):
        self.status_update_signal.emit(success, status, detail, result)

    def result_callback(self, success, result):
        self.set_status(success, str(status), "", result)

    def progress_callback(self, current, maximum):
        if self.task.canceled:
            return False

        self.progress_update_signal.emit(current, maximum)
        return True
