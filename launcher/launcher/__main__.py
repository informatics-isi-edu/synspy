import sys
import traceback
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QStyleFactory, QMessageBox
from deriva_common import format_exception
from deriva_common.base_cli import BaseCLI
from launcher.ui import main_window as mw


def excepthook(etype, value, tb):
    traceback.print_tb(tb)
    sys.stderr.write(format_exception(value))
    msg = QMessageBox()
    msg.setText(str(value))
    msg.setStandardButtons(QMessageBox.Close)
    msg.setWindowTitle("Unhandled Exception: %s" % etype.__name__)
    msg.setIcon(QMessageBox.Critical)
    msg.setDetailedText('\n'.join(traceback.format_exception(etype, value, tb)))
    msg.exec_()


def main():
    sys.excepthook = excepthook
    try:
        QApplication.setDesktopSettingsAware(False)
        QApplication.setStyle(QStyleFactory.create("Fusion"))
        app = QApplication(sys.argv)
        app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
        app.setWindowIcon(QIcon(":/images/synapse.png"))
        cli = BaseCLI("Synapse Viewer Launcher",
                      "For more information see: https://github.com/informatics-isi-edu/synspy")
        cli.remove_options(["--credential-file"])
        args = cli.parse_cli()
        mainWindow = mw.MainWindow(args.config_file)
        mainWindow.show()
        ret = app.exec_()
        return ret
    except Exception as e:
        print(e)

if __name__ == '__main__':
    sys.exit(main())
