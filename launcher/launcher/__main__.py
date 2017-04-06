import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QStyleFactory
from ui import main_window as mw


def main():
    try:
        app = QApplication(sys.argv)
        app.setStyle(QStyleFactory.create('Fusion'))
        app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
        mainWindow = mw.MainWindow()
        mainWindow.show()
        ret = app.exec()
        return ret
    except Exception as e:
        print(e)

if __name__ == '__main__':
    sys.exit(main())
