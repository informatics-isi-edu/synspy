import os
from PyQt5.QtCore import pyqtSignal
from deriva_common import format_exception, DEFAULT_HEADERS
from deriva_qt.common.async_task import async_execute, AsyncTask


class StoreTask(AsyncTask):
    def __init__(self, store, parent=None):
        super(StoreTask, self).__init__(parent)
        self.store = store


class FileRetrieveTask(StoreTask):
    status_update_signal = pyqtSignal(bool, str, str, str)
    progress_update_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super(FileRetrieveTask, self).__init__(parent)
        self.file_path = None

    def success_callback(self, rid, result):
        if rid != self.rid:
            return
        self.status_update_signal.emit(True, "File download success", "", self.file_path)

    def error_callback(self, rid, error):
        if rid != self.rid:
            return
        self.status_update_signal.emit(False, "File download failed", format_exception(error))

    def retrieve(self, path, headers=DEFAULT_HEADERS, destfile=None, progress_callback=None):
        self.init_request()
        self.file_path = os.path.abspath(destfile)
        self.request = async_execute(self.store.get_obj,
                                     [path, headers, destfile, progress_callback],
                                     self.rid,
                                     self.success_callback,
                                     self.error_callback)
