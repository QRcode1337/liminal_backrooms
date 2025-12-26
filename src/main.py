import sys
import os
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import LiminalBackroomsApp
from src.ui.widgets.custom_widgets import COLORS # Ensure fonts or styles are loaded if needed
from src.utils.font_loader import load_fonts

def main():
    """Main entry point for the application"""
    # Create the application
    app = QApplication(sys.argv)
    app.setApplicationName("Liminal Backrooms")

    # Load custom fonts
    load_fonts()

    # Create the main window
    main_window = LiminalBackroomsApp()

    # Show the window
    main_window.show()

    # Run the event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
