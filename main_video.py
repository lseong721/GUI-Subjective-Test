# coding: utf-8
import sys

from PyQt5.QtWidgets import QApplication
from ui_video import UiMainWindow

if __name__ == '__main__':
    print("=============== Start ===============")
    app = QApplication(sys.argv)
    window = UiMainWindow()
    sys.exit(app.exec_())
