import os
from PyQt5.QtCore import pyqtSignal
from deriva_common import format_exception, DEFAULT_HEADERS, DEFAULT_CHUNK_SIZE
from deriva_qt.common.async_task import async_execute, AsyncTask

HATRAC_UPDATE_URL_TEMPLATE = "/hatrac/Zf/%s/%s"


class StoreTask(AsyncTask):
    def __init__(self, store, parent=None):
        super(StoreTask, self).__init__(parent)
        assert store is not None
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
        self.status_update_signal.emit(False, "File download failed", format_exception(error), None)

    def retrieve(self, path, headers=DEFAULT_HEADERS, destfile=None, progress_callback=None):
        self.init_request()
        self.file_path = os.path.abspath(destfile)
        self.request = async_execute(self.store.get_obj,
                                     [path, headers, destfile, progress_callback],
                                     self.rid,
                                     self.success_callback,
                                     self.error_callback)


class FileUploadTask(StoreTask):
    status_update_signal = pyqtSignal(bool, str, str, object)
    progress_update_signal = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super(FileUploadTask, self).__init__(parent)
        self.update_state = None

    def success_callback(self, rid, result):
        if rid != self.rid:
            return
        ret = (self.update_state, result)
        self.status_update_signal.emit(True, "File upload success", "", ret)

    def error_callback(self, rid, error):
        if rid != self.rid:
            return
        self.status_update_signal.emit(False, "File upload failed", format_exception(error), None)

    def upload(self,
               path,
               file_path,
               update_state,
               headers=DEFAULT_HEADERS,
               md5=None,
               chunked=True,
               chunk_size=DEFAULT_CHUNK_SIZE,
               create_parents=True,
               allow_versioning=True,
               callback=None):
        self.init_request()
        self.update_state = update_state
        self.request = async_execute(self.store.put_loc,
                                     [path,
                                      file_path,
                                      headers,
                                      md5,
                                      chunked,
                                      chunk_size,
                                      create_parents,
                                      allow_versioning,
                                      callback],
                                     self.rid,
                                     self.success_callback,
                                     self.error_callback)
