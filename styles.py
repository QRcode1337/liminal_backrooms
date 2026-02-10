# styles.py
"""
Centralized styling for the Liminal Backrooms application.

This module is the SINGLE SOURCE OF TRUTH for all colors, fonts, and widget styles.
Import from here - never hardcode colors or duplicate style definitions.

Usage:
    from styles import COLORS, FONTS, get_combobox_style, get_button_style
"""

# =============================================================================
# COLOR PALETTE - Phosphor Green CRT Terminal Theme
# =============================================================================

COLORS = {
    # Backgrounds - true black to very dark green-black
    'bg_dark': '#000000',           # Pure black (CRT off-pixel)
    'bg_medium': '#0A1A0A',         # Very dark green-black (message blocks)
    'bg_light': '#142814',          # Slightly lighter green-black (hover/selection)

    # Primary accents - phosphor green spectrum
    'accent_cyan': '#00FF41',       # Bright phosphor green (primary accent)
    'accent_cyan_hover': '#00CC33', # Medium phosphor green (hover)
    'accent_cyan_active': '#009926',# Darker phosphor green (pressed)

    # Secondary accents - functional differentiation
    'accent_pink': '#FF3333',       # Red (danger, errors, human-related)
    'accent_purple': '#00AA22',     # Deep green (tertiary, export buttons)
    'accent_yellow': '#CCFF00',     # Yellow-green (warnings)
    'accent_green': '#00FF41',      # Same as primary (rabbithole)

    # AI-specific colors (for chat message headers/borders - green spectrum)
    'ai_1': '#00FF41',              # Bright phosphor green - AI-1
    'ai_2': '#33FF66',              # Lighter green - AI-2
    'ai_3': '#66FFAA',              # Mint green - AI-3
    'ai_4': '#99FFCC',              # Pale green - AI-4
    'ai_5': '#CCFFEE',              # Near-white green - AI-5
    'human': '#FF6600',             # Amber/orange - Human User (stands out from green)

    # Notification colors
    'notify_error': '#FF3333',      # Red - Error/Failure
    'notify_success': '#00FF41',    # Phosphor green - Success
    'notify_info': '#CCFF00',       # Yellow-green - Informational

    # Text colors - WHITE for message body, green for UI chrome
    'text_normal': '#E0E0E0',       # White/light gray - message body text (readability)
    'text_dim': '#1A6B1A',          # Dim green - timestamps, secondary labels
    'text_bright': '#F0F0F0',       # Near-white - emphasis text
    'text_glow': '#00FF41',         # Phosphor green - headers, glowing elements
    'text_timestamp': '#1A6B1A',    # Dim green - same as text_dim
    'text_error': '#FF3333',        # Red - Error text

    # Borders and effects
    'border': '#0D3B0D',            # Very dark green
    'border_glow': '#00FF41',       # Bright phosphor green borders
    'border_highlight': '#1A6B1A',  # Medium green borders
    'shadow': 'rgba(0, 255, 65, 0.2)',  # Green glow shadows

    # Legacy color mappings for compatibility
    'accent_blue': '#00FF41',       # Map old blue to phosphor green
    'accent_blue_hover': '#00CC33',
    'accent_blue_active': '#009926',
    'accent_orange': '#CCFF00',     # Yellow-green
    'chain_of_thought': '#00AA22',  # Deep green
    'user_header': '#00FF41',       # Phosphor green
    'ai_header': '#33FF66',         # Lighter green
    'system_message': '#CCFF00',    # Yellow-green
}


# =============================================================================
# FONT CONFIGURATION
# =============================================================================

FONTS = {
    # All monospace - full terminal feel
    'family_mono': "'Iosevka Term', 'Consolas', 'Courier New', monospace",
    'family_display': "'Iosevka Term', 'Consolas', 'Courier New', monospace",
    'family_ui': "'Iosevka Term', 'Consolas', 'Courier New', monospace",
    
    # Font sizes
    'size_xs': '8px',
    'size_sm': '10px',
    'size_md': '12px',
    'size_lg': '14px',
    'size_xl': '16px',
    
    # Common combinations
    'default': '10px',              # Default UI font size
    'code': '10pt',                 # Code/monospace size
}


# =============================================================================
# WIDGET STYLE GENERATORS
# =============================================================================

def get_combobox_style():
    """Get the style for comboboxes - phosphor green CRT themed."""
    return f"""
        QComboBox {{
            background-color: {COLORS['bg_medium']};
            color: {COLORS['text_normal']};
            border: 1px solid {COLORS['border_glow']};
            border-radius: 0px;
            padding: 4px 8px;
            min-height: 20px;
            font-size: {FONTS['size_sm']};
        }}
        QComboBox:hover {{
            border: 1px solid {COLORS['accent_cyan']};
            color: {COLORS['text_bright']};
        }}
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left: 1px solid {COLORS['border_glow']};
            border-radius: 0px;
        }}
        QComboBox::down-arrow {{
            width: 12px;
            height: 12px;
            image: none;
        }}
        QComboBox QAbstractItemView {{
            background-color: {COLORS['bg_dark']};
            color: {COLORS['text_normal']};
            border: 1px solid {COLORS['border_glow']};
            border-radius: 0px;
            padding: 2px;
            outline: none;
        }}
        QComboBox QAbstractItemView::item {{
            min-height: 22px;
            padding: 2px 4px;
            padding-left: 8px;
        }}
        QComboBox QAbstractItemView::item:selected {{
            background-color: {COLORS['bg_light']};
            color: {COLORS['text_bright']};
        }}
        QComboBox QAbstractItemView::item:hover {{
            background-color: {COLORS['bg_light']};
            color: {COLORS['text_bright']};
        }}
    """


def get_button_style(accent_color=None):
    """
    Get cyberpunk-themed button style.
    
    Args:
        accent_color: Override accent color (defaults to accent_cyan)
    """
    accent = accent_color or COLORS['accent_cyan']
    return f"""
        QPushButton {{
            background-color: {COLORS['bg_medium']};
            color: {accent};
            border: 1px solid {accent};
            border-radius: 0px;
            padding: 10px 14px;
            font-size: {FONTS['size_sm']};
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {accent};
            color: {COLORS['bg_dark']};
        }}
        QPushButton:pressed {{
            background-color: {COLORS['bg_light']};
        }}
        QPushButton:disabled {{
            background-color: {COLORS['bg_dark']};
            color: {COLORS['text_dim']};
            border-color: {COLORS['text_dim']};
        }}
    """


def get_input_style():
    """Get style for text inputs - phosphor green CRT themed."""
    return f"""
        QLineEdit, QTextEdit {{
            background-color: {COLORS['bg_medium']};
            color: {COLORS['text_normal']};
            border: 1px solid {COLORS['border_glow']};
            border-radius: 0px;
            padding: 8px;
            font-size: {FONTS['size_sm']};
        }}
        QLineEdit:focus, QTextEdit:focus {{
            border: 1px solid {COLORS['accent_cyan']};
            color: {COLORS['text_bright']};
        }}
    """


def get_label_style(style_type='normal'):
    """
    Get style for labels.
    
    Args:
        style_type: One of 'normal', 'header', 'glow', 'dim'
    """
    styles = {
        'normal': f"""
            QLabel {{
                color: {COLORS['text_normal']};
                font-size: {FONTS['size_sm']};
            }}
        """,
        'header': f"""
            QLabel {{
                color: {COLORS['text_glow']};
                font-size: {FONTS['size_sm']};
                font-weight: bold;
                letter-spacing: 1px;
            }}
        """,
        'glow': f"""
            QLabel {{
                color: {COLORS['text_glow']};
                font-size: {FONTS['size_sm']};
            }}
        """,
        'dim': f"""
            QLabel {{
                color: {COLORS['text_dim']};
                font-size: {FONTS['size_xs']};
            }}
        """,
    }
    return styles.get(style_type, styles['normal'])


def get_checkbox_style():
    """Get style for checkboxes - phosphor green CRT themed."""
    return f"""
        QCheckBox {{
            color: {COLORS['text_dim']};
            font-size: 10px;
            spacing: 6px;
            padding: 4px 0px;
        }}
        QCheckBox::indicator {{
            width: 14px;
            height: 14px;
            border: 1px solid {COLORS['border_glow']};
            border-radius: 0px;
            background-color: {COLORS['bg_dark']};
        }}
        QCheckBox::indicator:checked {{
            background-color: {COLORS['accent_cyan']};
            border-color: {COLORS['accent_cyan']};
        }}
        QCheckBox::indicator:hover {{
            border-color: {COLORS['accent_cyan']};
        }}
    """


def get_scrollbar_style():
    """
    Get style for scrollbars - retro CRT/phosphor green CRT theme.
    
    Features:
    - No rounded corners (sharp edges for retro look)
    - Cyan glow on hover
    - Minimal design
    """
    return f"""
        QScrollBar:vertical {{
            background-color: {COLORS['bg_dark']};
            width: 12px;
            border: 1px solid {COLORS['border']};
            border-radius: 0px;
            margin: 0px;
        }}
        QScrollBar::handle:vertical {{
            background-color: {COLORS['border_glow']};
            border: none;
            border-radius: 0px;
            min-height: 30px;
            margin: 2px;
        }}
        QScrollBar::handle:vertical:hover {{
            background-color: {COLORS['accent_cyan']};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
            border: none;
        }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: none;
        }}
        QScrollBar:horizontal {{
            background-color: {COLORS['bg_dark']};
            height: 12px;
            border: 1px solid {COLORS['border']};
            border-radius: 0px;
            margin: 0px;
        }}
        QScrollBar::handle:horizontal {{
            background-color: {COLORS['border_glow']};
            border: none;
            border-radius: 0px;
            min-width: 30px;
            margin: 2px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background-color: {COLORS['accent_cyan']};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
            border: none;
        }}
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
            background: none;
        }}
    """


def get_frame_style(style_type='default'):
    """
    Get style for frames/containers.
    
    Args:
        style_type: One of 'default', 'bordered', 'glow'
    """
    styles = {
        'default': f"""
            QFrame {{
                background-color: {COLORS['bg_dark']};
                border: none;
            }}
        """,
        'bordered': f"""
            QFrame {{
                background-color: {COLORS['bg_dark']};
                border: 1px solid {COLORS['border']};
                border-radius: 0px;
            }}
        """,
        'glow': f"""
            QFrame {{
                background-color: {COLORS['bg_dark']};
                border: 1px solid {COLORS['border_glow']};
                border-radius: 0px;
            }}
        """,
    }
    return styles.get(style_type, styles['default'])


def get_tooltip_style():
    """Get style for tooltips."""
    return f"""
        QToolTip {{
            background-color: {COLORS['bg_medium']};
            color: {COLORS['text_bright']};
            border: 1px solid {COLORS['accent_cyan']};
            padding: 6px;
            font-size: {FONTS['size_sm']};
        }}
    """


def get_menu_style():
    """Get style for context menus."""
    return f"""
        QMenu {{
            background-color: {COLORS['bg_medium']};
            color: {COLORS['text_normal']};
            border: 1px solid {COLORS['border_glow']};
            padding: 4px;
        }}
        QMenu::item {{
            padding: 6px 20px;
        }}
        QMenu::item:selected {{
            background-color: {COLORS['accent_cyan']};
            color: {COLORS['bg_dark']};
        }}
        QMenu::separator {{
            height: 1px;
            background-color: {COLORS['border']};
            margin: 4px 8px;
        }}
    """


# =============================================================================
# COMPLETE APPLICATION STYLESHEET
# =============================================================================

def get_app_stylesheet():
    """
    Get a complete application stylesheet combining all widget styles.
    Apply this to QApplication for global styling.
    """
    return f"""
        {get_tooltip_style()}
        {get_menu_style()}
        {get_scrollbar_style()}
    """