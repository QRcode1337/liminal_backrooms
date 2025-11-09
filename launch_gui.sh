#!/bin/bash

# Launch Liminal Backrooms GUI
cd "$(dirname "$0")"

echo "========================================="
echo "Launching Liminal Backrooms GUI..."
echo "========================================="

# Check for required dependencies
if ! python -c "import PyQt6" 2>/dev/null; then
    echo "ERROR: PyQt6 not installed!"
    echo "Please run: pip install PyQt6"
    exit 1
fi

# Launch the GUI
python main.py

echo "========================================="
echo "GUI closed"
echo "========================================="
