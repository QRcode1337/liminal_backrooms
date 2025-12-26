from PyQt6.QtGui import QFontDatabase
from pathlib import Path
import os

def load_fonts():
    """Load custom fonts for the application"""
    # Look for fonts in the root 'fonts' directory
    # If running from run.py, root is current directory
    # If running from src/main.py, root is one level up

    font_dir = Path("fonts")
    if not font_dir.exists():
        # Try going up one level if we are inside src
        font_dir = Path("../fonts")

    if not font_dir.exists():
        # Try absolute path from file location
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        font_dir = project_root / "fonts"

    # List of fonts to load
    fonts = [
        ("IosevkaTerm-Regular.ttf", "Iosevka Term"),
        ("IosevkaTerm-Bold.ttf", "Iosevka Term"),
        ("IosevkaTerm-Italic.ttf", "Iosevka Term"),
    ]

    loaded_fonts = []

    if font_dir.exists():
        for font_file, font_name in fonts:
            font_path = font_dir / font_file
            if font_path.exists():
                font_id = QFontDatabase.addApplicationFont(str(font_path))
                if font_id >= 0:
                    if font_name not in loaded_fonts:
                        loaded_fonts.append(font_name)
                    print(f"Loaded font: {font_name} from {font_file}")
                else:
                    print(f"Failed to load font: {font_file}")
            else:
                pass
                # Silent fail for missing font files, fallback will be used

    return loaded_fonts
