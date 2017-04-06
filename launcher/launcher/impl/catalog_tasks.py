from PyQt5.QtCore import pyqtSignal
from deriva_common import format_exception
from deriva_qt.common.async_task import async_execute, AsyncTask

WORKLIST_QUERY = "/attributegroup/U:=Synapse:Person/Identities=%s/T:=(Zebrafish:Image Region:Classifier)/" \
                 "!ZYX Slice::null::/Status=\"analysis pending\";Status=\"analysis in progress\"/" \
                 "I:=(Source Image)/!URL::null::/$T/*;source_image:=array(I:*),user:=array(U:*)"


class CatalogTask(AsyncTask):
    def __init__(self, catalog, parent=None):
        super(CatalogTask, self).__init__(parent)
        self.catalog = catalog


class SessionQueryTask(CatalogTask):
    status_update_signal = pyqtSignal(bool, str, str, object)

    def __init__(self, parent=None):
        super(SessionQueryTask, self).__init__(parent)

    def success_callback(self, rid, result):
        if rid != self.rid:
            return
        self.status_update_signal.emit(True, "Session query success", "", result.json())

    def error_callback(self, rid, error):
        if rid != self.rid:
            return
        self.status_update_signal.emit(False, "Session query failure", format_exception(error), None)

    def query(self):
        self.init_request()
        self.request = async_execute(self.catalog.get_authn_session,
                                     [],
                                     self.rid,
                                     self.success_callback,
                                     self.error_callback)


class CatalogQueryTask(CatalogTask):
    status_update_signal = pyqtSignal(bool, str, str, object)

    def __init__(self, parent=None):
        super(CatalogQueryTask, self).__init__(parent)

    def success_callback(self, rid, result):
        if rid != self.rid:
            return
        self.status_update_signal.emit(True, "Catalog query success", "", result.json())

    def error_callback(self, rid, error):
        if rid != self.rid:
            return
        self.status_update_signal.emit(False, "Catalog query failure", format_exception(error), None)

    def query(self, path):
        self.init_request()
        self.request = async_execute(self.catalog.get,
                                     [path],
                                     self.rid,
                                     self.success_callback,
                                     self.error_callback)
