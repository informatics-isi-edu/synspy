import os
import re
import errno
import logging
import shutil
import tempfile
import platform
import datetime
import pytz
from PyQt5.QtCore import Qt, QCoreApplication, QMetaObject, QThreadPool, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import qApp, QMainWindow, QWidget, QAction, QSizePolicy, QMessageBox, QStyle, QSplitter, \
    QToolBar, QStatusBar, QVBoxLayout, QTableWidgetItem, QAbstractItemView, QDialog, QCheckBox, QMenu, QHBoxLayout, \
    QLabel, QLineEdit, QPushButton, QFileDialog
from PyQt5.QtGui import QIcon
from deriva_qt.common import log_widget, table_widget, async_task
from deriva_qt.auth_agent.ui.auth_window import AuthWindow
from deriva_common import ErmrestCatalog, HatracStore, read_config, write_config, format_exception, urlquote, \
    resource_path
from launcher.impl.catalog_tasks import CatalogQueryTask, SessionQueryTask, CatalogUpdateTask, WORKLIST_QUERY, \
    WORKLIST_UPDATE, WORKLIST_CURATOR_QUERY, WORKLIST_STATUS_UPDATE
from launcher.impl.store_tasks import FileRetrieveTask, FileUploadTask, HATRAC_UPDATE_URL_TEMPLATE
from launcher.impl.process_tasks import ViewerTask
from launcher.ui import DEFAULT_CONFIG, CURATORS
from launcher import resources
from synspy import __version__ as synspy_version


# noinspection PyArgumentList
class MainWindow(QMainWindow):
    config = None
    credential = None
    config_path = None
    store = None
    catalog = None
    identity = None
    attributes = None
    server = None
    tempdir = None
    progress_update_signal = pyqtSignal(str)
    use_3D_viewer = False
    curator_mode = False

    def __init__(self, config_path=None):
        super(MainWindow, self).__init__()
        self.ui = MainWindowUI(self)
        self.configure(config_path)
        self.authWindow = AuthWindow(config=self.config,
                                     is_child_window=True,
                                     cookie_persistence=False,
                                     authentication_success_callback=self.onLoginSuccess)
        self.getSession()
        if not self.identity:
            self.ui.actionLaunch.setEnabled(False)
            self.ui.actionRefresh.setEnabled(False)
            self.ui.actionOptions.setEnabled(False)
            self.ui.actionLogout.setEnabled(False)

    def configure(self, config_path):
        # configure logging
        self.ui.logTextBrowser.widget.log_update_signal.connect(self.updateLog)
        self.ui.logTextBrowser.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(self.ui.logTextBrowser)
        logging.getLogger().setLevel(logging.INFO)

        # configure Ermrest/Hatrac
        if not config_path:
            config_path = os.path.join(os.path.expanduser(
                os.path.normpath("~/.deriva/synapse/synspy-launcher")), "config.json")
        self.config_path = config_path
        config = read_config(self.config_path, create_default=True, default=DEFAULT_CONFIG)
        protocol = config["server"]["protocol"]
        self.server = config["server"]["host"]
        catalog_id = config["server"]["catalog_id"]
        session_config = config.get("session")
        self.catalog = ErmrestCatalog(protocol, self.server, catalog_id, self.credential, session_config=session_config)
        self.store = HatracStore(protocol, self.server, self.credential, session_config=session_config)

        # create working dir (tempdir)
        self.tempdir = tempfile.mkdtemp(prefix="synspy_")

        # determine viewer mode
        self.use_3D_viewer = True if config.get("viewer_mode", "2d").lower() == "3d" else False

        # curator mode?
        curator_mode = config.get("curator_mode")
        if not curator_mode:
            config["curator_mode"] = False
        self.curator_mode = config.get("curator_mode")

        # save config
        self.config = config
        write_config(self.config_path, self.config)

    def getSession(self):
        qApp.setOverrideCursor(Qt.WaitCursor)
        self.updateStatus("Validating session.")
        queryTask = SessionQueryTask(self.catalog)
        queryTask.status_update_signal.connect(self.onSessionResult)
        queryTask.query()

    def onLoginSuccess(self, **kwargs):
        self.authWindow.hide()
        self.credential = kwargs["credential"]
        self.catalog.set_credentials(self.credential, self.server)
        self.store.set_credentials(self.credential, self.server)
        self.getSession()

    def enableControls(self):
        self.ui.actionLaunch.setEnabled(True)
        self.ui.actionRefresh.setEnabled(True)
        self.ui.actionOptions.setEnabled(self.authWindow.authenticated())
        self.ui.actionLogin.setEnabled(not self.authWindow.authenticated())
        self.ui.actionLogout.setEnabled(self.authWindow.authenticated())
        self.ui.actionExit.setEnabled(True)
        self.ui.workList.setEnabled(True)

    def disableControls(self):
        self.ui.actionLaunch.setEnabled(False)
        self.ui.actionRefresh.setEnabled(False)
        self.ui.actionOptions.setEnabled(False)
        self.ui.actionLogin.setEnabled(False)
        self.ui.actionLogout.setEnabled(False)
        self.ui.actionExit.setEnabled(False)
        self.ui.workList.setEnabled(False)

    def closeEvent(self, event=None):
        self.disableControls()
        self.cancelTasks()
        shutil.rmtree(self.tempdir)
        if event:
            event.accept()

    def cancelTasks(self):
        async_task.Request.shutdown()
        self.statusBar().showMessage("Waiting for background tasks to terminate...")

        while True:
            qApp.processEvents()
            if QThreadPool.globalInstance().waitForDone(10):
                break

        self.statusBar().showMessage("All background tasks terminated successfully")

    def is_curator(self):
        for attr in self.attributes:
            if attr.get('id') == CURATORS:
                return True
        return False

    def displayWorklist(self, worklist):
        keys = ["ID",
                "Classifier",
                "Subject Issue Date",
                "Status",
                "Identities",
                "URL",
                "Npz URL",
                "ZYX Slice",
                "Segmentation Mode",
                "Segments URL",
                "Segments Filtered URL",
                "Subject",
                "History"]
        self.ui.workList.clear()
        self.ui.workList.setRowCount(0)
        self.ui.workList.setColumnCount(0)
        displayed = ["ID", "Classifier", "Subject Issue Date", "Status"]
        self.ui.workList.setRowCount(len(worklist))
        self.ui.workList.setColumnCount(len(keys))

        self.ui.workList.removeAction(self.ui.markIncompleteAction)
        if self.is_curator() and self.curator_mode:
            self.ui.workList.addAction(self.ui.markIncompleteAction)

        rows = 0
        for row in worklist:
            value = row.get("Status")
            if not (value == "analysis pending" or value == "analysis in progress") \
                    and not (self.is_curator() and self.curator_mode):
                self.ui.workList.hideRow(rows)
            cols = 0
            for key in keys:
                item = QTableWidgetItem()
                if key == "Classifier":
                    value = row['user'][0]['Full Name']
                elif key == "Identities":
                    value = row['user'][0]['Identities']
                    item.setData(Qt.UserRole, value)
                elif key == "URL" or key == "Subject":
                    value = row["source_image"][0].get(key)
                elif key == "History":
                    value = row.get(key)
                    if not value:
                        value = []
                    item.setData(Qt.UserRole, value)
                else:
                    value = row.get(key)
                if isinstance(value, str):
                    item.setText(value)
                    item.setToolTip(value)
                self.ui.workList.setItem(rows, cols, item)
                cols += 1
            rows += 1

        cols = 0
        for key in keys:
            if key not in displayed:
                self.ui.workList.hideColumn(cols)
            cols += 1

        self.ui.workList.setHorizontalHeaderLabels(keys)  # add header names
        self.ui.workList.horizontalHeader().setDefaultAlignment(Qt.AlignLeft)  # set alignment
        self.ui.workList.resizeColumnToContents(0)
        self.ui.workList.resizeColumnToContents(1)
        self.ui.workList.sortByColumn(2, Qt.DescendingOrder)
        if (self.ui.workList.rowCount() > 0) and self.identity:
            self.ui.actionLaunch.setEnabled(True)
        else:
            self.ui.actionLaunch.setEnabled(False)

    def getCacheDir(self):
        cwd = os.getcwd()
        cache_dir = os.path.expanduser(self.config.get("cache_dir", cwd))
        if not os.path.isdir(cache_dir):
            try:
                os.makedirs(cache_dir)
            except OSError as error:
                if error.errno != errno.EEXIST:
                    logging.error(format_exception(error))
                    cache_dir = cwd
        return cache_dir

    def downloadCallback(self, **kwargs):
        status = kwargs.get("progress")
        if status:
            self.progress_update_signal.emit(status)
        return True

    def uploadCallback(self, **kwargs):
        completed = kwargs.get("completed")
        total = kwargs.get("total")
        file_path = kwargs.get("file_path")
        if completed and total:
            file_path = " [%s]" % os.path.basename(file_path) if file_path else ""
            status = "Uploading file%s: %d%% complete" % (file_path, round(((completed / total) % 100) * 100))
        else:
            summary = kwargs.get("summary", "")
            file_path = "Uploaded file: [%s] " % os.path.basename(file_path) if file_path else ""
            status = file_path  # + summary
        if status:
            self.progress_update_signal.emit(status)
        return True

    def serverProblemMessageBox(self, text, detail):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Confirm Action")
        msg.setText(text)
        msg.setInformativeText(detail + "\n\nWould you like to remove this item from the current worklist?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        ret = msg.exec_()
        if ret == QMessageBox.No:
            return
        else:
            row = self.ui.workList.getCurrentTableRow()
            self.ui.workList.removeRow(row)
            return

    def retrieveFiles(self):
        # if there is an existing segments file, download it first, otherwise just initiate the input file download
        segments_url = self.ui.workList.getCurrentTableItemTextByName("Segments Filtered URL")
        if segments_url:
            segments_filename = os.path.basename(segments_url).split(":")[0]
            segments_destfile = os.path.abspath(os.path.join(self.tempdir, segments_filename))
            self.updateStatus("Downloading file: [%s]" % segments_destfile)
            downloadTask = FileRetrieveTask(self.store)
            downloadTask.status_update_signal.connect(self.onRetrieveAnalysisFileResult)
            self.progress_update_signal.connect(self.updateProgress)
            downloadTask.retrieve(
                segments_url,
                destfile=segments_destfile,
                progress_callback=self.downloadCallback)
        else:
            self.retrieveInputFile()

    def retrieveInputFile(self):
        # get the main TIFF file for analysis if not already cached
        input_url = "URL" if self.use_3D_viewer else "Npz URL"
        url = self.ui.workList.getCurrentTableItemTextByName(input_url)
        if not url and not self.use_3D_viewer:
            self.resetUI("Unable to launch 2D viewer due to missing NPZ file for %s." %
                         self.ui.workList.getCurrentTableItemTextByName("ID"))
            self.serverProblemMessageBox(
                "2D viewer requires NPZ data to be present!",
                "The launcher is currently configured to execute the 2D viewer, which requires NPZ files for input. " +
                "No NPZ file could be found on the server for this task.")
            return
        filename = os.path.basename(url).split(":")[0]
        destfile = os.path.abspath(os.path.join(self.getCacheDir(), filename))
        if not os.path.isfile(destfile):
            self.updateStatus("Downloading file: [%s]" % destfile)
            downloadTask = FileRetrieveTask(self.store)
            downloadTask.status_update_signal.connect(self.onRetrieveInputFileResult)
            self.progress_update_signal.connect(self.updateProgress)
            downloadTask.retrieve(
                url,
                destfile=destfile,
                progress_callback=self.downloadCallback)
        else:
            self.onRetrieveInputFileResult(True, "The file [%s] already exists" % destfile, None, destfile)

    def getSubprocessPath(self):
        executable = "synspy-viewer" if self.use_3D_viewer else "synspy-viewer2d"
        if platform.system() == "Windows":
            base_path = None
        else:
            base_path = "/bin"
        return os.path.normpath(resource_path(executable, base_path))

    def executeViewer(self, file_path):
        self.updateStatus("Executing viewer...")
        env = os.environ
        env["SYNSPY_AUTO_DUMP_LOAD"] = "true"
        env["DUMP_PREFIX"] = "./%s." % self.ui.workList.getCurrentTableItemTextByName("ID")
        env["ZYX_SLICE"] = self.ui.workList.getCurrentTableItemTextByName("ZYX Slice")
        env["ZYX_IMAGE_GRID"] = "0.4, 0.26, 0.26"
        env["SYNSPY_DETECT_NUCLEI"] = str(
            "nucleic" == self.ui.workList.getCurrentTableItemTextByName("Segmentation Mode")).lower()
        output_path = os.path.join(os.path.dirname(self.config_path), "viewer.log")
        identities = self.ui.workList.getTableItemByName(
            self.ui.workList.getCurrentTableRow(), "Identities").data(Qt.UserRole)
        viewerTask = ViewerTask(self.getSubprocessPath(), self.identity in identities, proc_output_path=output_path)
        viewerTask.status_update_signal.connect(self.onSubprocessExecuteResult)
        viewerTask.run(file_path, self.tempdir, env)

    def uploadAnalysisResult(self, update_state):
        qApp.setOverrideCursor(Qt.WaitCursor)
        # generate hatrac upload params
        basename = self.ui.workList.getCurrentTableItemTextByName("ID")
        match = "%s\..*\.csv" % basename
        output_files = [f for f in os.listdir(self.tempdir)
                        if os.path.isfile(os.path.join(self.tempdir, f)) and re.match(match, f)]
        if not output_files:
            self.resetUI("Could not locate output file from viewer subprocess -- aborting.")
            return
        seg_mode = self.ui.workList.getCurrentTableItemTextByName("Segmentation Mode")
        if seg_mode == "synaptic":
            extension = ".synapses-only.csv"
        elif seg_mode == "nucleic":
            extension = ".nuclei-only.csv"
        else:
            self.updateStatus("Unknown segmentation mode \"%s\" -- aborting." % seg_mode)
            return
        file_name = basename + extension
        hatrac_path = HATRAC_UPDATE_URL_TEMPLATE % \
            (self.ui.workList.getCurrentTableItemTextByName("Subject"), file_name)
        file_path = os.path.abspath(os.path.join(self.tempdir, file_name))

        # upload to object store
        self.updateStatus("Uploading file %s to server..." % file_name)
        self.progress_update_signal.connect(self.updateProgress)
        uploadTask = FileUploadTask(self.store)
        uploadTask.status_update_signal.connect(self.onUploadFileResult)
        uploadTask.upload(hatrac_path, file_path, update_state, callback=self.uploadCallback)

    def markIncomplete(self):
        ID = self.ui.workList.getCurrentTableItemTextByName("ID")
        body = [{"ID": ID, "Status":  "analysis in progress"}]
        self.updateStatus("Updating task status for %s..." % ID)
        updateTask = CatalogUpdateTask(self.catalog)
        updateTask.status_update_signal.connect(self.onCatalogUpdateResult)
        updateTask.update(WORKLIST_STATUS_UPDATE, json=body)

    @pyqtSlot()
    def taskTriggered(self):
        self.ui.logTextBrowser.widget.clear()
        self.disableControls()

    @pyqtSlot(str)
    def updateProgress(self, status):
        self.statusBar().showMessage(status)

    @pyqtSlot(str, str)
    def updateStatus(self, status, detail=None):
        logging.info(status + ((": %s" % detail) if detail else ""))
        self.statusBar().showMessage(status)

    @pyqtSlot(str, str)
    def resetUI(self, status, detail=None):
        qApp.restoreOverrideCursor()
        self.updateStatus(status, detail)
        self.enableControls()

    @pyqtSlot(str)
    def updateLog(self, text):
        self.ui.logTextBrowser.widget.appendPlainText(text)

    @pyqtSlot(bool, str, str, object)
    def onSessionResult(self, success, status, detail, result):
        qApp.restoreOverrideCursor()
        if success:
            self.identity = result["client"]["id"]
            self.attributes = result["attributes"]
            display_name = result["client"]["full_name"]
            self.setWindowTitle("%s (%s - %s)" % (self.windowTitle(), self.server, display_name))
            self.ui.actionLaunch.setEnabled(True)
            self.ui.actionLogout.setEnabled(True)
            self.ui.actionLogin.setEnabled(False)
            if not self.is_curator():
                self.curator_mode = self.config["curator_mode"] = False
            self.on_actionRefresh_triggered()
        else:
            self.updateStatus(status, detail)

    @pyqtSlot()
    def on_actionLaunch_triggered(self):
        self.disableControls()
        qApp.setOverrideCursor(Qt.WaitCursor)
        # create working dir (tempdir)
        if self.tempdir:
            shutil.rmtree(self.tempdir)
        self.tempdir = tempfile.mkdtemp(prefix="synspy_")
        self.retrieveFiles()

    @pyqtSlot(bool, str, str, str)
    def onRetrieveAnalysisFileResult(self, success, status, detail, file_path):
        if not success:
            try:
                os.remove(file_path)
            except Exception as e:
                logging.warning("Unable to remove file [%s]: %s" % (file_path, format_exception(e)))
            self.resetUI(status, detail)
            self.serverProblemMessageBox(
                "Unable to download required input file",
                "The in-progress analysis file was not downloaded successfully.")
            return

        self.retrieveInputFile()

    @pyqtSlot(bool, str, str, str)
    def onRetrieveInputFileResult(self, success, status, detail, file_path):
        if not success:
            try:
                os.remove(file_path)
            except Exception as e:
                logging.warning("Unable to remove file [%s]: %s" % (file_path, format_exception(e)))
            self.resetUI(status, detail)
            self.serverProblemMessageBox(
                "Unable to download required input file",
                "The image input file was not downloaded successfully.")
            return

        self.executeViewer(file_path)

    @pyqtSlot(bool, str, str, bool)
    def onSubprocessExecuteResult(self, success, status, detail, is_owner):
        qApp.restoreOverrideCursor()
        if not success:
            self.resetUI(status, detail)
            return

        if not is_owner:
            self.resetUI(status, detail)
            return

        # prompt for save/complete/discard
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Confirm Action")
        msg.setText("How would you like to proceed?")
        msg.setInformativeText(
            "Select \"Save Progress\" to save your progress and upload the output to the server.\n\n"
            "Select \"Complete\" to upload the output to the server and mark this task as completed.\n\n"
            "Select \"Discard\" to abort the process and leave the task state unchanged.")
        saveButton = msg.addButton("Save Progress", QMessageBox.ActionRole)
        completeButton = msg.addButton("Complete", QMessageBox.ActionRole)
        discardButton = msg.addButton("Discard", QMessageBox.RejectRole)
        msg.exec_()
        if msg.clickedButton() == discardButton:
            self.resetUI("Aborted.")
            return
        update_state = None
        if msg.clickedButton() == saveButton:
            update_state = ("incomplete", "analysis in progress")
        elif msg.clickedButton() == completeButton:
            update_state = ("complete", "analysis complete")

        self.uploadAnalysisResult(update_state)

    @pyqtSlot(bool, str, str, object)
    def onUploadFileResult(self, success, status, detail, result):
        if not success:
            self.resetUI(status, detail)
            self.serverProblemMessageBox(
                "Unable to upload required file(s)",
                "One or more required files were not uploaded successfully.")
            return
        state = result[0]
        ID = self.ui.workList.getCurrentTableItemTextByName("ID")
        history = self.ui.workList.getTableItemByName(
            self.ui.workList.getCurrentTableRow(), "History").data(Qt.UserRole)
        history.append({
            "program": "synspy-viewer" if self.use_3D_viewer else "synspy-viewer2d",
            "version": synspy_version,
            "ts": datetime.datetime.now(pytz.utc).isoformat()
        })
        body = [{"ID": ID, "Segments Filtered URL": result[1], "Status":  state[1], "History":history}]
        self.updateStatus("Updating task status for %s..." % ID)
        updateTask = CatalogUpdateTask(self.catalog)
        updateTask.status_update_signal.connect(self.onCatalogUpdateResult)
        updateTask.update(WORKLIST_UPDATE, json=body)

    @pyqtSlot(bool, str, str, object)
    def onCatalogUpdateResult(self, success, status, detail, result):
        if not success:
            self.resetUI(status, detail)
            self.serverProblemMessageBox(
                "Unable to update catalog data",
                "The catalog state was not updated successfully.")
            return
        qApp.restoreOverrideCursor()
        self.on_actionRefresh_triggered()

    @pyqtSlot()
    def on_actionRefresh_triggered(self):
        if not self.identity:
            self.updateStatus("Unable to get worklist -- not logged in.")
            return
        qApp.setOverrideCursor(Qt.WaitCursor)
        self.disableControls()
        self.updateStatus("Refreshing worklist...")
        queryTask = CatalogQueryTask(self.catalog)
        queryTask.status_update_signal.connect(self.onRefreshResult)
        if self.is_curator() and self.curator_mode:
            queryTask.query(WORKLIST_CURATOR_QUERY)
        else:
            queryTask.query(WORKLIST_QUERY % urlquote(self.identity, ""))

    @pyqtSlot(bool, str, str, object)
    def onRefreshResult(self, success, status, detail, result):
        if success:
            self.displayWorklist(result)
            self.resetUI("Ready.")
        else:
            self.resetUI(status, detail)

    @pyqtSlot()
    def on_actionLogin_triggered(self):
        self.authWindow.show()
        self.authWindow.login()

    @pyqtSlot()
    def on_actionLogout_triggered(self):
        self.authWindow.logout()
        self.setWindowTitle(self.ui.title)
        self.ui.workList.clearContents()
        self.ui.workList.setRowCount(0)
        self.identity = None
        self.ui.actionLaunch.setEnabled(False)
        self.ui.actionLogout.setEnabled(False)
        self.ui.actionLogin.setEnabled(True)

    @pyqtSlot()
    def on_actionHelp_triggered(self):
        pass

    @pyqtSlot()
    def on_actionOptions_triggered(self):
        OptionsDialog.getOptions(self)

    @pyqtSlot()
    def on_actionExit_triggered(self):
        self.closeEvent()
        QCoreApplication.quit()


class OptionsDialog(QDialog):
    def __init__(self, parent):
        super(OptionsDialog, self).__init__(parent)
        self.refreshWorklist = False
        self.setWindowTitle("Options")
        self.setWindowIcon(QIcon(":/images/synapse.png"))
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setMinimumWidth(400)
        self.resize(400, 100)
        layout = QVBoxLayout(self)

        self.horizontalLayout = QHBoxLayout()
        self.pathLabel = QLabel("Downloads:")
        self.horizontalLayout.addWidget(self.pathLabel)
        self.pathTextBox = QLineEdit()
        self.pathTextBox.setReadOnly(True)
        self.cache_path = os.path.expanduser(os.path.normpath(parent.config.get("cache_dir", ".")))
        self.pathTextBox.setText(os.path.normpath(self.cache_path))
        self.horizontalLayout.addWidget(self.pathTextBox)
        self.browseButton = QPushButton("Browse", parent)
        self.browseButton.clicked.connect(self.on_actionBrowse_triggered)
        self.horizontalLayout.addWidget(self.browseButton)
        layout.addLayout(self.horizontalLayout)

        curator_mode = QCheckBox("&Curator Mode", parent)
        curator_mode.setChecked(parent.curator_mode)
        layout.addWidget(curator_mode)
        if not parent.is_curator():
            curator_mode.setEnabled(False)
        curator_mode.toggled.connect(self.onCuratorModeToggled)
        use_3D_viewer = QCheckBox("Use &3D Viewer", parent)
        use_3D_viewer.setChecked(parent.use_3D_viewer)
        layout.addWidget(use_3D_viewer)
        if platform.system() != "Linux":
            use_3D_viewer.setEnabled(False)
        use_3D_viewer.toggled.connect(self.onUse3DViewerToggled)


    @pyqtSlot(bool)
    def onCuratorModeToggled(self, toggled):
        parent = self.parent()
        parent.curator_mode = toggled
        parent.config["curator_mode"] = toggled
        self.refreshWorklist = True

    @pyqtSlot(bool)
    def onUse3DViewerToggled(self, toggled):
        parent = self.parent()
        parent.use_3D_viewer = toggled
        parent.config["viewer_mode"] = "3D" if toggled else "2D"

    @pyqtSlot()
    def on_actionBrowse_triggered(self):
        parent = self.parent()
        dialog = QFileDialog()
        path = dialog.getExistingDirectory(self,
                                           "Select Directory",
                                           self.cache_path,
                                           QFileDialog.ShowDirsOnly)
        if path:
            self.pathTextBox.setText(os.path.normpath(path))
            parent.config["cache_dir"] = os.path.normpath(path)

    @staticmethod
    def getOptions(parent):
        dialog = OptionsDialog(parent)
        dialog.exec_()
        write_config(parent.config_path, parent.config)
        if dialog.refreshWorklist:
            parent.on_actionRefresh_triggered()


# noinspection PyArgumentList
class MainWindowUI(object):

    title = "Synspy Launcher"

    def __init__(self, MainWin):
        super(MainWindow).__init__()

        # Main Window
        MainWin.setObjectName("MainWindow")
        MainWin.setWindowTitle(MainWin.tr(self.title))
        MainWin.setWindowIcon(QIcon(":/images/synapse.png"))
        MainWin.resize(640, 600)
        self.centralWidget = QWidget(MainWin)
        self.centralWidget.setObjectName("centralWidget")
        MainWin.setCentralWidget(self.centralWidget)
        self.verticalLayout = QVBoxLayout(self.centralWidget)
        self.verticalLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")

        # Splitter for Worklist/Log
        self.splitter = QSplitter(Qt.Vertical)

        # Table View (Work list)
        self.workList = table_widget.TableWidget(self.centralWidget)
        self.workList.setObjectName("tableWidget")
        self.workList.setStyleSheet(
            """
            QTableWidget {
                    border: 2px solid grey;
                    border-radius: 5px;
            }
            """)
        self.workList.setEditTriggers(QAbstractItemView.NoEditTriggers)  # use NoEditTriggers to disable editing
        self.workList.setAlternatingRowColors(True)
        self.workList.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.workList.setSelectionMode(QAbstractItemView.SingleSelection)
        self.workList.verticalHeader().setDefaultSectionSize(18)  # tighten up the row size
        self.workList.horizontalHeader().setStretchLastSection(True)
        # self.workList.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.workList.setSortingEnabled(True)  # allow sorting
        self.workList.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.workList.doubleClicked.connect(MainWin.on_actionLaunch_triggered)
        self.splitter.addWidget(self.workList)

        # Log Widget
        self.logTextBrowser = log_widget.QPlainTextEditLogger(self.centralWidget)
        self.logTextBrowser.widget.setObjectName("logTextBrowser")
        self.logTextBrowser.widget.setStyleSheet(
            """
            QPlainTextEdit {
                    border: 2px solid grey;
                    border-radius: 5px;
                    background-color: lightgray;
            }
            """)
        self.splitter.addWidget(self.logTextBrowser.widget)

        # add splitter
        self.splitter.setSizes([600, 100])
        self.verticalLayout.addWidget(self.splitter)

    # Actions

        # Launch
        self.actionLaunch = QAction(MainWin)
        self.actionLaunch.setObjectName("actionLaunch")
        self.actionLaunch.setText(MainWin.tr("Launch Analysis"))
        self.actionLaunch.setToolTip(MainWin.tr("Launch the synspy-viewer process"))
        self.actionLaunch.setShortcut(MainWin.tr("Ctrl+L"))

        # Refresh
        self.actionRefresh = QAction(MainWin)
        self.actionRefresh.setObjectName("actionRefresh")
        self.actionRefresh.setText(MainWin.tr("Refresh Work List"))
        self.actionRefresh.setToolTip(MainWin.tr("Refresh the work list"))
        self.actionRefresh.setShortcut(MainWin.tr("Ctrl+R"))

        # Options
        self.actionOptions = QAction(MainWin)
        self.actionOptions.setObjectName("actionOptions")
        self.actionOptions.setText(MainWin.tr("Options"))
        self.actionOptions.setToolTip(MainWin.tr("Configuration Options"))
        self.actionOptions.setShortcut(MainWin.tr("Ctrl+P"))

        # Login
        self.actionLogin = QAction(MainWin)
        self.actionLogin.setObjectName("actionLogin")
        self.actionLogin.setText(MainWin.tr("Login"))
        self.actionLogin.setToolTip(MainWin.tr("Login to the server"))
        self.actionLogin.setShortcut(MainWin.tr("Ctrl+G"))

        # Logout
        self.actionLogout = QAction(MainWin)
        self.actionLogout.setObjectName("actionLogout")
        self.actionLogout.setText(MainWin.tr("Logout"))
        self.actionLogout.setToolTip(MainWin.tr("Logout of the server"))
        self.actionLogout.setShortcut(MainWin.tr("Ctrl+O"))

        # Exit
        self.actionExit = QAction(MainWin)
        self.actionExit.setObjectName("actionExit")
        self.actionExit.setText(MainWin.tr("Exit"))
        self.actionExit.setToolTip(MainWin.tr("Exit the application"))
        self.actionExit.setShortcut(MainWin.tr("Ctrl+Z"))

        # Help
        self.actionHelp = QAction(MainWin)
        self.actionHelp.setObjectName("actionHelp")
        self.actionHelp.setText(MainWin.tr("Help"))
        self.actionHelp.setToolTip(MainWin.tr("Help"))
        self.actionHelp.setShortcut(MainWin.tr("Ctrl+H"))

        # Mark Incomplete
        self.markIncompleteAction = QAction('Mark Incomplete', self.workList)
        self.markIncompleteAction.triggered.connect(MainWin.markIncomplete)

    # Tool Bar

        self.mainToolBar = QToolBar(MainWin)
        self.mainToolBar.setObjectName("mainToolBar")
        self.mainToolBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        MainWin.addToolBar(Qt.TopToolBarArea, self.mainToolBar)

        # Launch
        self.mainToolBar.addAction(self.actionLaunch)
        self.actionLaunch.setIcon(qApp.style().standardIcon(QStyle.SP_MediaPlay))

        # Reload
        self.mainToolBar.addAction(self.actionRefresh)
        self.actionRefresh.setIcon(qApp.style().standardIcon(QStyle.SP_BrowserReload))

        # Options
        self.mainToolBar.addAction(self.actionOptions)
        self.actionOptions.setIcon(qApp.style().standardIcon(QStyle.SP_FileDialogDetailedView))

        # this spacer right justifies everything that comes after it
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mainToolBar.addWidget(spacer)

        # Login
        self.mainToolBar.addAction(self.actionLogin)
        self.actionLogin.setIcon(qApp.style().standardIcon(QStyle.SP_DialogApplyButton))

        # Logout
        self.mainToolBar.addAction(self.actionLogout)
        self.actionLogout.setIcon(qApp.style().standardIcon(QStyle.SP_DialogOkButton))

        # Help
        #self.mainToolBar.addAction(self.actionHelp)
        self.actionHelp.setIcon(qApp.style().standardIcon(QStyle.SP_MessageBoxQuestion))

        # Exit
        self.mainToolBar.addAction(self.actionExit)
        self.actionExit.setIcon(qApp.style().standardIcon(QStyle.SP_DialogCancelButton))

    # Status Bar

        self.statusBar = QStatusBar(MainWin)
        self.statusBar.setToolTip("")
        self.statusBar.setStatusTip("")
        self.statusBar.setObjectName("statusBar")
        MainWin.setStatusBar(self.statusBar)

    # finalize UI setup
        QMetaObject.connectSlotsByName(MainWin)
