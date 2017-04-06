import subprocess
import tempfile
import shutil
from PyQt5.QtCore import pyqtSignal
from deriva_common import format_exception, DEFAULT_HEADERS
from deriva_qt.common.async_task import async_execute, AsyncTask


class SubprocessTask(AsyncTask):
    def __init__(self, parent=None):
        super(SubprocessTask, self).__init__(parent)


class ViewerTask(SubprocessTask):
    status_update_signal = pyqtSignal(bool, str, str, str)

    def __init__(self, parent=None):
        super(SubprocessTask, self).__init__(parent)

    def success_callback(self, rid, result):
        if rid != self.rid:
            return
        self.status_update_signal.emit(True, "Viewer subprocess execution success.", "", result)

    def error_callback(self, rid, error):
        if rid != self.rid:
            return
        self.status_update_signal.emit(False, "Viewer subprocess execution failed", format_exception(error), None)

    def run(self, file_path, env=None):
        self.init_request()
        self.request = async_execute(self._execute,
                                     [file_path, env],
                                     self.rid,
                                     self.success_callback,
                                     self.error_callback)

    @staticmethod
    def _execute(file_path, env=None):
        tempdir = tempfile.mkdtemp(prefix="synspy_")
        command = ["synspy-viewer", file_path]
        try:
            process = subprocess.Popen(command, cwd=tempdir, env=env)
            ret = process.wait()
            del process
            if ret != 0:
                shutil.rmtree(tempdir)
                raise RuntimeError('Non-zero viewer exit status %s!' % ret)
            return tempdir
        except:
            shutil.rmtree(tempdir)
            raise

