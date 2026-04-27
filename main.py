from gui_controller import HVACAnalyzerGUI
import sys
from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置全局样式
    app.setStyle('Fusion')

    # 创建并显示主窗口
    window = HVACAnalyzerGUI()
    window.show()

    sys.exit(app.exec_())