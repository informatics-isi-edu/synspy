import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QStyleFactory
from deriva_common.base_cli import BaseCLI
from launcher.ui import main_window as mw


def main():
    try:
        QApplication.setDesktopSettingsAware(False)
        QApplication.setStyle(QStyleFactory.create("Fusion"))
        app = QApplication(sys.argv)
        app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

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
