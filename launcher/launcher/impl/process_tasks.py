import os
import subprocess
from PyQt5.QtCore import pyqtSignal
from deriva_common import format_exception
from deriva_qt.common.async_task import async_execute, AsyncTask


class SubprocessTask(AsyncTask):
    def __init__(self, parent=None):
        super(SubprocessTask, self).__init__(parent)


class ViewerTask(SubprocessTask):
    status_update_signal = pyqtSignal(bool, str, str, bool)

    def __init__(self, executable, is_owner, parent=None):
        super(SubprocessTask, self).__init__(parent)
        self.executable = executable
        self.is_owner = is_owner

    def success_callback(self, rid, result):
        if rid != self.rid:
            return
        self.status_update_signal.emit(True, "Viewer subprocess execution success.", "", self.is_owner)

    def error_callback(self, rid, error):
        if rid != self.rid:
            return
        self.status_update_signal.emit(
            False, "Viewer subprocess execution failed", format_exception(error), self.is_owner)

    def run(self, file_path, working_dir=os.getcwd(), env=None):
        self.init_request()
        self.request = async_execute(self._execute,
                                     [self.executable, file_path, working_dir, env],
                                     self.rid,
                                     self.success_callback,
                                     self.error_callback)

    @staticmethod
    def _execute(executable, file_path, working_dir, env=None):
        command = [executable, file_path]
        process = subprocess.Popen(command, cwd=working_dir, env=env)
        ret = process.wait()
        del process
        if ret != 0:
            raise RuntimeError('Non-zero viewer exit status %s!' % ret)

