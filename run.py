#!/usr/bin/env python
"""
Wrapper script to properly set Qt plugin paths before launching the Liminal Backrooms application.
This resolves the "Could not find the Qt platform plugin" error on macOS.
"""

import os
import sys
import site

# Get the virtual environment's site-packages directory
venv_path = site.getsitepackages()[0]
qt_plugin_path = os.path.join(venv_path, 'PyQt6', 'Qt6', 'plugins')

# Set the Qt plugin path environment variable
os.environ['QT_PLUGIN_PATH'] = qt_plugin_path

# Also set the specific platform plugin path for extra safety
platform_plugin_path = os.path.join(qt_plugin_path, 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = platform_plugin_path

# Print for debugging (can be commented out once working)
print(f"Qt plugin path set to: {qt_plugin_path}")
print(f"Platform plugin path set to: {platform_plugin_path}")

# Now import and run the main application
from main import create_gui, run_gui

if __name__ == "__main__":
    main_window, app = create_gui()
    run_gui(main_window, app)
