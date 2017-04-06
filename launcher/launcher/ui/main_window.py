import os
import errno
import logging
import shutil
from PyQt5.QtCore import Qt, QCoreApplication,QMetaObject, QThreadPool, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QAction, QSizePolicy, QMessageBox, QStyle, \
     QToolBar, QStatusBar, QVBoxLayout, QTableWidget, QTableWidgetItem,QAbstractItemView, qApp
from PyQt5.QtGui import QIcon
from deriva_qt.common import log_widget
from deriva_qt.common import async_task
from deriva_common import ErmrestCatalog, HatracStore, read_config, read_credentials, resource_path, format_exception, \
    urlquote
from ..impl.catalog_tasks import CatalogQueryTask, SessionQueryTask, WORKLIST_QUERY
from ..impl.store_tasks import FileRetrieveTask
from ..impl.process_tasks import ViewerTask


# noinspection PyArgumentList
class MainWindow(QMainWindow):
    config = None
    store = None
    catalog = None
    identity = None
    progress_update_signal = pyqtSignal(str)

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = MainWindowUI(self)
        self.configure()
        self.getSession()

    def configure(self):
        # configure logging
        self.ui.logTextBrowser.widget.log_update_signal.connect(self.updateLog)
        self.ui.logTextBrowser.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(self.ui.logTextBrowser)
        logging.getLogger().setLevel(logging.INFO)

        # configure Ermrest/Hatrac
        config = read_config(resource_path(os.path.join('conf', 'config.json')))
        credentials = read_credentials(resource_path(os.path.join('conf', 'credentials.json')))
        protocol = config['server']['protocol']
        server = config['server']['host']
        catalog_id = config['server']['catalog_id']
        session_config = config.get('session')
        self.catalog = ErmrestCatalog(protocol, server, catalog_id, credentials, session_config=session_config)
        self.store = HatracStore(protocol, server, credentials, session_config=session_config)

        self.config = config

    def getSession(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.updateStatus("Validating session.")
        queryTask = SessionQueryTask(self.catalog)
        queryTask.status_update_signal.connect(self.onSessionResult)
        queryTask.query()

    def enableControls(self):
        self.ui.actionLaunch.setEnabled(True)
        self.ui.actionRefresh.setEnabled(True)
        self.ui.workList.setEnabled(True)

    def disableControls(self):
        self.ui.actionLaunch.setEnabled(False)
        self.ui.actionRefresh.setEnabled(False)
        self.ui.workList.setEnabled(False)

    def closeEvent(self, event):
        self.cancelTasks()
        event.accept()

    def cancelTasks(self):
        self.disableControls()
        async_task.Request.shutdown()
        self.statusBar().showMessage("Waiting for background tasks to terminate...")

        while True:
            qApp.processEvents()
            if QThreadPool.globalInstance().waitForDone(10):
                break

        self.enableControls()
        self.statusBar().showMessage("All background tasks terminated successfully")

    def displayWorklist(self, worklist):
        keys = ["ID",
                "Classifier",
                "Subject Issue Date",
                "Sub-Sequence",
                "Status",
                "URL",
                "ZYX Slice",
                "Segmentation Mode"]
        hidden = ["URL", "ZYX Slice", "Segmentation Mode"]
        self.ui.workList.setRowCount(len(worklist))
        self.ui.workList.setColumnCount(len(keys))

        rows = 0
        for row in worklist:
            cols = 0
            for key in keys:
                item = QTableWidgetItem()
                if key == "Classifier":
                    value = row['user'][0]['Full Name']
                elif key == "URL":
                    value = row['source_image'][0][key]
                else:
                    value = row[key]
                if isinstance(value, str):
                    item.setText(value or '')  # or '' for any None values
                self.ui.workList.setItem(rows, cols, item)
                if key in hidden:
                    self.ui.workList.hideColumn(cols)
                cols += 1
            rows += 1

        self.ui.workList.setHorizontalHeaderLabels(keys)  # add header names
        self.ui.workList.horizontalHeader().setDefaultAlignment(Qt.AlignLeft)  # set alignment
        self.ui.workList.resizeColumnToContents(0)
        self.ui.workList.resizeColumnToContents(1)
        self.ui.workList.resizeColumnToContents(2)

    def getCacheDir(self):
        cwd = os.getcwd()
        cache_dir = self.config.get('cache_dir', cwd)
        if not os.path.isdir(cache_dir):
            try:
                os.makedirs(cache_dir)
            except OSError as error:
                if error.errno != errno.EEXIST:
                    logging.error(format_exception(error))
                    cache_dir = cwd
        return cache_dir

    def downloadCallback(self, status):
        self.progress_update_signal.emit(status)
        return True

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
    def updateUI(self, status, detail=None):
        self.updateStatus(status, detail)
        self.enableControls()

    @pyqtSlot(str)
    def updateLog(self, text):
        self.ui.logTextBrowser.widget.appendPlainText(text)

    @pyqtSlot(bool, str, str, object)
    def onSessionResult(self, success, status, detail, result):
        QApplication.restoreOverrideCursor()
        if success:
            self.identity = result['client']['id']
            self.on_actionRefresh_triggered()
        else:
            self.updateStatus(status, detail)

    @pyqtSlot()
    def on_actionLaunch_triggered(self):
        self.disableControls()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        row = self.ui.getWorkListSelectedRow()
        url = self.ui.getWorkListItemByName(row, "URL")

        filename = os.path.basename(url).split(":")[0]
        destfile = os.path.abspath(os.path.join(self.getCacheDir(), filename))
        if not os.path.isfile(destfile):
            self.updateStatus("Downloading file: [%s]" % destfile)
            downloadTask = FileRetrieveTask(self.store)
            downloadTask.status_update_signal.connect(self.onRetrieveFileResult)
            self.progress_update_signal.connect(self.updateProgress)
            downloadTask.retrieve(
                url,
                destfile=destfile,
                progress_callback=self.downloadCallback)
        else:
            self.onRetrieveFileResult(True, "The file [%s] already exists" % destfile, None, destfile)

    @pyqtSlot(bool, str, str, str)
    def onRetrieveFileResult(self, success, status, detail, file_path):
        QApplication.restoreOverrideCursor()
        if not success:
            self.updateUI(status, detail)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Confirm Action")
            msg.setText("Unable to download required file(s)")
            msg.setInformativeText(
                "One or more required files were not downloaded successfully.\n\n"
                "Would you like to remove this item from the current worklist?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            ret = msg.exec_()
            if ret == QMessageBox.No:
                return
            else:
                row = self.ui.getWorkListSelectedRow()
                self.ui.workList.removeRow(row)
                return

        self.updateStatus("%s. Executing viewer..." % status)
        env = dict()
        env["ZYX_SLICE"] = self.ui.getWorkListItemByName(self.ui.getWorkListSelectedRow(), "ZYX Slice")
        env["SYNSPY_DETECT_NUCLEI"] = \
            str("nucleic" == self.ui.getWorkListItemByName(
                self.ui.getWorkListSelectedRow(), "Segmentation Mode")).lower()
        viewerTask = ViewerTask()
        viewerTask.status_update_signal.connect(self.onSubprocessExecuteResult)
        viewerTask.run(file_path, env)

    @pyqtSlot(bool, str, str, str)
    def onSubprocessExecuteResult(self, success, status, detail, output_dir):
        if not success:
            self.updateUI(status, detail)

        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)

    @pyqtSlot()
    def on_actionRefresh_triggered(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.disableControls()
        self.updateStatus("Refreshing worklist...")
        queryTask = CatalogQueryTask(self.catalog)
        queryTask.status_update_signal.connect(self.onRefreshResult)
        queryTask.query(WORKLIST_QUERY % urlquote(self.identity, ''))

    @pyqtSlot(bool, str, str, object)
    def onRefreshResult(self, success, status, detail, result):
        QApplication.restoreOverrideCursor()
        if success:
            self.displayWorklist(result)
            self.updateUI("Ready.")
        else:
            self.updateUI(status, detail)

    @pyqtSlot()
    def on_actionHelp_triggered(self):
        pass

    @pyqtSlot()
    def on_actionExit_triggered(self):
        self.cancelTasks()
        QCoreApplication.quit()


# noinspection PyArgumentList
class MainWindowUI(object):

    def __init__(self, MainWin):
        super(MainWindow).__init__()

        # Main Window
        MainWin.setObjectName("MainWindow")
        MainWin.setWindowTitle(MainWin.tr("Synspy Launcher"))
        # MainWin.setWindowIcon(QIcon(":/images/bag.png"))
        MainWin.resize(640, 600)
        self.centralWidget = QWidget(MainWin)
        self.centralWidget.setObjectName("centralWidget")
        MainWin.setCentralWidget(self.centralWidget)
        self.verticalLayout = QVBoxLayout(self.centralWidget)
        self.verticalLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")

        # Table View (Work list)
        self.workList = QTableWidget(self.centralWidget)
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
        self.verticalLayout.addWidget(self.workList)

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
        self.verticalLayout.addWidget(self.logTextBrowser.widget)

    # Actions

        # Launch
        self.actionLaunch = QAction(MainWin)
        self.actionLaunch.setObjectName("actionLaunch")
        self.actionLaunch.setText(MainWin.tr("Launch Viewer"))
        self.actionLaunch.setToolTip(
            MainWin.tr("Launch the synspy-viewer process"))
        self.actionLaunch.setShortcut(MainWin.tr("Ctrl+L"))

        # Refresh
        self.actionRefresh = QAction(MainWin)
        self.actionRefresh.setObjectName("actionRefresh")
        self.actionRefresh.setText(MainWin.tr("Refresh Work List"))
        self.actionRefresh.setToolTip(
            MainWin.tr("Refresh the work list"))
        self.actionLaunch.setShortcut(MainWin.tr("Ctrl+R"))

        # Exit
        self.actionExit = QAction(MainWin)
        self.actionExit.setObjectName("actionExit")
        self.actionExit.setText(MainWin.tr("Exit"))
        self.actionExit.setToolTip(MainWin.tr("Exit the application"))
        self.actionExit.setShortcut(MainWin.tr("Ctrl+X"))

        # Help
        self.actionHelp = QAction(MainWin)
        self.actionHelp.setObjectName("actionHelp")
        self.actionHelp.setText(MainWin.tr("Help"))
        self.actionHelp.setToolTip(MainWin.tr("Help"))
        self.actionHelp.setShortcut(MainWin.tr("Ctrl+H"))

    # Tool Bar

        self.mainToolBar = QToolBar(MainWin)
        self.mainToolBar.setObjectName("mainToolBar")
        self.mainToolBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        MainWin.addToolBar(Qt.TopToolBarArea, self.mainToolBar)

        # Launch
        self.mainToolBar.addAction(self.actionLaunch)
        self.actionLaunch.setIcon(
            self.actionLaunch.parentWidget().style().standardIcon(getattr(QStyle, "SP_MediaPlay")))

        # Reload
        self.mainToolBar.addAction(self.actionRefresh)
        self.actionRefresh.setIcon(
            self.actionRefresh.parentWidget().style().standardIcon(getattr(QStyle, "SP_BrowserReload")))

        # this spacer right justifies everything that comes after it
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mainToolBar.addWidget(spacer)

        # Help
        self.mainToolBar.addAction(self.actionHelp)
        self.actionHelp.setIcon(
            self.actionHelp.parentWidget().style().standardIcon(getattr(QStyle, "SP_MessageBoxQuestion")))

        # Exit
        self.mainToolBar.addAction(self.actionExit)
        self.actionExit.setIcon(
            self.actionExit.parentWidget().style().standardIcon(getattr(QStyle, "SP_DialogCancelButton")))

    # Status Bar

        self.statusBar = QStatusBar(MainWin)
        self.statusBar.setToolTip("")
        self.statusBar.setStatusTip("")
        self.statusBar.setObjectName("statusBar")
        MainWin.setStatusBar(self.statusBar)

    # finalize UI setup
        QMetaObject.connectSlotsByName(MainWin)

    def getWorkListItemByName(self, row, column_name):
        column = None
        header_count = self.workList.columnCount()
        # noinspection PyTypeChecker
        for column in range(header_count):
            header_text = self.workList.horizontalHeaderItem(column).text()
            if column_name == header_text:
                break
        item = self.workList.item(row, column)
        return item.text()

    def getWorkListSelectedRow(self):
        row = self.workList.currentRow()
        if row == -1 and self.workList.rowCount() > 0:
            row = 0

        return row
