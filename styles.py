# styles.py
"""
Centralized styling for the Liminal Backrooms application.

This module is the SINGLE SOURCE OF TRUTH for all colors, fonts, and widget styles.
Import from here - never hardcode colors or duplicate style definitions.

Usage:
    from styles import COLORS, FONTS, get_combobox_style, get_button_style
"""

# =============================================================================
# COLOR PALETTE - Cypher OS (ink-black shell, acid signal, hairline chrome)
# =============================================================================

COLORS = {
    # Backgrounds - ink black, panel, titlebar
    'bg_dark': '#06080B',           # Ground / desktop
    'bg_medium': '#0C1015',         # Window panels, message blocks
    'bg_light': '#121821',          # Hover / selection
    'bg_titlebar': '#10151C',       # Window title strips, section headers

    # Primary accent - acid signal (key name kept for compatibility)
    'accent_cyan': '#C6FF3D',       # Signal (primary accent)
    'accent_cyan_hover': '#A8E01F', # Signal hover
    'accent_cyan_active': '#8BBF0F',# Signal pressed

    # Secondary accents - functional differentiation
    'accent_pink': '#FF5C7A',       # Alert (danger, errors)
    'accent_purple': '#5CE1E6',     # Ice (tertiary, export buttons)
    'accent_yellow': '#FFB547',     # Amber (warnings)
    'accent_green': '#C6FF3D',      # Same as primary (rabbithole)
    'accent_violet': '#B69CFF',     # System events

    # AI-specific colors (agent rails/tags) - distinct hue AND lightness
    'ai_1': '#C6FF3D',              # Signal
    'ai_2': '#5CE1E6',              # Ice
    'ai_3': '#B69CFF',              # Violet
    'ai_4': '#FF8FCB',              # Rose
    'ai_5': '#EDF2F6',              # White
    'human': '#FFB547',             # Operator amber

    # Notification colors
    'notify_error': '#FF5C7A',      # Alert
    'notify_success': '#C6FF3D',    # Signal
    'notify_info': '#5CE1E6',       # Ice

    # Notification / code tints (dark backgrounds behind colored text)
    'tint_error': '#1E0E13',
    'tint_success': '#131A08',
    'tint_info': '#0B181B',
    'tint_warn': '#1E170A',
    'code_bg': '#0A0D12',
    'code_header': '#10151C',

    # Text colors
    'text_normal': '#D7E0E8',       # Message body
    'text_dim': '#7D8A99',          # Meta, labels, timestamps (4.5:1 on panels)
    'text_bright': '#F2F5F8',       # Emphasis
    'text_glow': '#C6FF3D',         # Headers, live elements
    'text_timestamp': '#7D8A99',
    'text_error': '#FF5C7A',

    # Borders and effects - hairlines, accent only on focus
    'border': '#1F2A36',            # Hairline
    'border_glow': '#2A3746',       # Control outline
    'border_highlight': '#3A4A5C',  # Hover outline
    'shadow': 'rgba(198, 255, 61, 0.18)',

    # Legacy color mappings for compatibility
    'accent_blue': '#C6FF3D',
    'accent_blue_hover': '#A8E01F',
    'accent_blue_active': '#8BBF0F',
    'accent_orange': '#FFB547',
    'chain_of_thought': '#8E9BAA',
    'user_header': '#FFB547',
    'ai_header': '#5CE1E6',
    'system_message': '#B69CFF',
}


# =============================================================================
# FONT CONFIGURATION
# =============================================================================

FONTS = {
    # Mono body + angular display. Drop JetBrainsMono / ChakraPetch TTFs into
    # fonts/ to bundle them; otherwise the stack falls back to Iosevka Term.
    'family_mono': "'JetBrains Mono', 'Iosevka Term', 'SF Mono', 'Menlo', 'Consolas', monospace",
    'family_display': "'Chakra Petch', 'JetBrains Mono', 'Iosevka Term', 'Menlo', monospace",
    'family_ui': "'JetBrains Mono', 'Iosevka Term', 'SF Mono', 'Menlo', 'Consolas', monospace",
    
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
    """Get the style for comboboxes - Cypher OS themed."""
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
            font-family: {FONTS['family_display']};
            font-size: {FONTS['size_sm']};
            font-weight: bold;
            letter-spacing: 1px;
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
    """Get style for text inputs - Cypher OS themed."""
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
    """Get style for checkboxes - Cypher OS themed."""
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
    Get style for scrollbars - Cypher OS theme.
    
    Features:
    - No rounded corners (sharp edges for retro look)
    - Signal accent on hover
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


def get_window_title_style(accent_color=None):
    """Title strip for a Cypher OS window/pane (hairline under a titlebar fill)."""
    accent = accent_color or COLORS['text_bright']
    return f"""
        color: {accent};
        font-family: {FONTS['family_display']};
        font-size: 12px;
        font-weight: bold;
        letter-spacing: 2px;
        padding: 9px 12px;
        background-color: {COLORS['bg_titlebar']};
        border: none;
        border-bottom: 1px solid {COLORS['border']};
    """


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