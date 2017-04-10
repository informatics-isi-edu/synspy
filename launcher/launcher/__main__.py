import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QStyleFactory
from deriva_common.base_cli import BaseCLI
from .ui import main_window as mw


def main():
    try:
        app = QApplication(sys.argv)
        app.setStyle(QStyleFactory.create('Fusion'))
        app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

        cli = BaseCLI("Synapse Viewer Launcher",
                      "For more information see: https://github.com/informatics-isi-edu/synspy")
        args = cli.parse_cli()
        mainWindow = mw.MainWindow(args.config_file, args.credential_file)
        mainWindow.show()
        ret = app.exec_()
        return ret
    except Exception as e:
        print(e)

if __name__ == '__main__':
    sys.exit(main())
