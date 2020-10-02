from PyQt5.QtCore import pyqtSignal
from deriva.core import format_exception, DEFAULT_HEADERS
from launcher.impl import LauncherTask, Task

WORKLIST_QUERY = \
    "/attributegroup/U:=public:ERMrest_Client/ID=%s/T:=(Zebrafish:Image Region:Classifier)/" \
    "!ZYX Slice::null::/I:=(Source Image)/!T:Npz URL::null::/$T/*;source_image:=array(I:*),user:=array(U:*)"

WORKLIST_CURATOR_QUERY = \
    "/attributegroup/U:=public:ERMrest_Client/T:=(Zebrafish:Image Region:Classifier)/" \
    "!ZYX Slice::null::/I:=(Source Image)/$T/*;source_image:=array(I:*),user:=array(U:*)"

WORKLIST_UPDATE = "/attributegroup/Zebrafish:Image Region/RID;Segments Filtered URL,Status"

WORKLIST_STATUS_UPDATE = "/attributegroup/Zebrafish:Image Region/RID;Status"


class CatalogTask(LauncherTask):
    def __init__(self, catalog, parent=None):
        super(LauncherTask, self).__init__(parent)
        assert catalog is not None
        self.catalog = catalog


class SessionQueryTask(CatalogTask):
    def __init__(self, parent=None):
        super(SessionQueryTask, self).__init__(parent)

    def result_callback(self, success, result):
        self.set_status(success,
                        "Session query success" if success else "Session query failure",
                        "" if success else format_exception(result),
                        result.json() if success else None)

    def query(self):
        self.task = Task(self.catalog.get_authn_session, [], self.result_callback)
        self.start()


class CatalogQueryTask(CatalogTask):

    def __init__(self, parent=None):
        super(CatalogQueryTask, self).__init__(parent)

    def result_callback(self, success, result):
        self.set_status(success,
                        "Catalog query success" if success else "Catalog query failure",
                        "" if success else format_exception(result),
                        result.json() if success else None)

    def query(self, path):
        self.task = Task(self.catalog.get, [path], self.result_callback)
        self.start()


class CatalogUpdateTask(CatalogTask):
    def __init__(self, parent=None):
        super(CatalogUpdateTask, self).__init__(parent)

    def result_callback(self, success, result):
        self.set_status(success,
                        "Catalog update success" if success else "Catalog update failure",
                        "" if success else format_exception(result),
                        result.json() if success else None)

    def update(self, path, data=None, json=None, headers=DEFAULT_HEADERS, guard_response=None):
        self.task = Task(self.catalog.put, [path, data, json, headers, guard_response], self.result_callback)
        self.start()
