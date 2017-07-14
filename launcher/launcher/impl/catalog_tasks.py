from PyQt5.QtCore import pyqtSignal
from deriva_common import format_exception, DEFAULT_HEADERS
from deriva_qt.common.async_task import async_execute, AsyncTask

WORKLIST_QUERY = "/attributegroup/U:=Synapse:Person/Identities=%s/T:=(Zebrafish:Image Region:Classifier)/" \
                 "!ZYX Slice::null::/I:=(Source Image)/!T:Npz URL::null::/$T/*;source_image:=array(I:*),user:=array(U:*)"

WORKLIST_CURATOR_QUERY = \
    "/attributegroup/U:=Synapse:Person/T:=(Zebrafish:Image Region:Classifier)/" \
    "!ZYX Slice::null::/I:=(Source Image)/!T:Npz URL::null::/$T/*;source_image:=array(I:*),user:=array(U:*)"

WORKLIST_UPDATE = "/attributegroup/Zebrafish:Image Region/ID;Segments URL,Status"

WORKLIST_UPDATE_2D = "/attributegroup/Zebrafish:Image Region/ID;Segments Filtered URL,Status"

WORKLIST_STATUS_UPDATE = "/attributegroup/Zebrafish:Image Region/ID;Status"


class CatalogTask(AsyncTask):
    def __init__(self, catalog, parent=None):
        super(CatalogTask, self).__init__(parent)
        assert catalog is not None
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


class CatalogUpdateTask(CatalogTask):
    status_update_signal = pyqtSignal(bool, str, str, object)

    def __init__(self, parent=None):
        super(CatalogUpdateTask, self).__init__(parent)

    def success_callback(self, rid, result):
        if rid != self.rid:
            return
        self.status_update_signal.emit(True, "Catalog update success", "", result.json())

    def error_callback(self, rid, error):
        if rid != self.rid:
            return
        self.status_update_signal.emit(False, "Catalog update failure", format_exception(error), None)

    def update(self, path, data=None, json=None, headers=DEFAULT_HEADERS, guard_response=None):
        self.init_request()
        self.request = async_execute(self.catalog.put,
                                     [path, data, json, headers, guard_response],
                                     self.rid,
                                     self.success_callback,
                                     self.error_callback)
