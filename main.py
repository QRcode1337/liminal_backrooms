import sys
import os
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow

def main():
    # Ensure src is in python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    app = QApplication(sys.argv)
    
    # Apply dark theme
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
