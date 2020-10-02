import os
from PyQt5.QtCore import pyqtSignal
from deriva.core import format_exception, DEFAULT_HEADERS, DEFAULT_CHUNK_SIZE
from launcher.impl import LauncherTask, Task

HATRAC_UPDATE_URL_TEMPLATE = "/hatrac/Zf/%s/%s"


class StoreTask(LauncherTask):
    def __init__(self, store, parent=None):
        super(StoreTask, self).__init__(parent)
        assert store is not None
        self.store = store


class FileRetrieveTask(StoreTask):

    def __init__(self, parent=None):
        super(FileRetrieveTask, self).__init__(parent)
        self.file_path = None

    def result_callback(self, success, result):
        self.set_status(success,
                        "File download success" if success else "File download failure",
                        "" if success else format_exception(result),
                        self.file_path)

    def retrieve(self, path, headers=DEFAULT_HEADERS, destfile=None, progress_callback=None):
        self.file_path = os.path.abspath(destfile)
        self.task = Task(self.store.get_obj,
                         [path, headers, destfile, progress_callback],
                         self.result_callback)
        self.start()


class FileUploadTask(StoreTask):

    def __init__(self, parent=None):
        super(FileUploadTask, self).__init__(parent)
        self.update_state = None

    def result_callback(self, success, result):
        self.set_status(success,
                        "File upload success" if success else "File upload failed",
                        "" if success else format_exception(result),
                        (self.update_state, result) if success else None)

    def upload(self,
               path,
               file_path,
               update_state,
               headers=DEFAULT_HEADERS,
               md5=None,
               sha256=None,
               content_type=None,
               content_disposition=None,
               chunked=False,
               chunk_size=DEFAULT_CHUNK_SIZE,
               create_parents=True,
               allow_versioning=True,
               callback=None):

        self.update_state = update_state
        self.task = Task(self.store.put_loc,
                         [path,
                          file_path,
                          headers,
                          md5,
                          sha256,
                          content_type,
                          content_disposition,
                          chunked,
                          chunk_size,
                          create_parents,
                          allow_versioning,
                          callback],
                         self.result_callback)
        self.start()
