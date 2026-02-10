# grouped_model_selector.py
"""
Grouped Model Selector - A QComboBox with hierarchical grouping for AI models.

Provides a 3-level hierarchy:
  - Tier (Paid / Free)
    - Provider (Anthropic, Google, Meta, etc.)
      - Model (Display Name (model-id), e.g., "Claude Sonnet 4.5 (anthropic/claude-sonnet-4.5)")

Only the leaf model items are selectable; tier and provider headers are visual groupings.

All model data is imported from config.AI_MODELS - this file contains NO duplicate model data.
All styling is imported from styles.py - the single source of truth for colors/fonts.
"""

from PyQt6.QtWidgets import QComboBox, QStyledItemDelegate
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QFont, QColor
from PyQt6.QtCore import Qt

from config import AI_MODELS, get_model_id
from styles import COLORS


class GroupedItemDelegate(QStyledItemDelegate):
    """
    Custom delegate to render grouped items with different styling.
    
    - Tier headers (Paid/Free): Custom painted - bold, accent color
    - Provider headers: Custom painted - semi-bold, subtle background
    - Model items: Stylesheet-matching rendering with indentation
    
    Args:
        colors: Dict of color values (defaults to styles.COLORS). Expected keys:
                - text_bright, text_normal, accent_cyan, bg_dark, bg_light, bg_medium
        parent: Parent QWidget (should be the GroupedModelComboBox)
    """
    
    def __init__(self, colors=None, parent=None):
        super().__init__(parent)
        self._combo = parent  # Reference to combo box for checking current selection
        
        # Use provided colors or fall back to styles.COLORS
        colors = colors or COLORS
        
        # Helper to get color as QColor
        def get_color(key):
            return QColor(colors.get(key, COLORS.get(key, '#FFFFFF')))
        
        # Map colors
        self.bg_dark = get_color('bg_dark')
        self.bg_medium = get_color('bg_medium')
        self.bg_light = get_color('bg_light')
        self.text_bright = get_color('text_bright')
        self.text_normal = get_color('text_normal')
        self.accent_cyan = get_color('accent_cyan')
        
        # Current selection (the actual selected item in combobox)
        self.current_bg = QColor(self.bg_light)
        
        # Hover color - subtle highlight  
        self.hover_bg = get_color('bg_light')
        
        # Derived colors for headers
        self.tier_bg = QColor(self.bg_medium).lighter(110)
        self.provider_bg = QColor(self.bg_medium)
        
        # Cache for current index (updated by combobox)
        self._current_row = -1
    
    def setCurrentRow(self, row):
        """Called by combobox to update cached current row"""
        self._current_row = row
        
    def paint(self, painter, option, index):
        item_type = index.data(Qt.ItemDataRole.UserRole + 1)
        
        # Get the font from the option (inherits from widget/stylesheet)
        base_font = option.font
        
        if item_type == "tier":
            # Tier header - custom painted
            painter.save()
            painter.fillRect(option.rect, self.tier_bg)
            painter.setPen(self.accent_cyan)
            font = QFont(base_font)
            font.setBold(True)
            painter.setFont(font)
            text_rect = option.rect.adjusted(8, 0, 0, 0)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, 
                           f"▾ {index.data()}")
            painter.restore()
                           
        elif item_type == "provider":
            # Provider header - custom painted
            painter.save()
            painter.fillRect(option.rect, self.provider_bg)
            painter.setPen(self.text_bright)
            font = QFont(base_font)
            font.setBold(True)
            painter.setFont(font)
            text_rect = option.rect.adjusted(20, 0, 0, 0)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, 
                           index.data())
            painter.restore()
                           
        else:
            # Model item
            from PyQt6.QtWidgets import QStyle
            
            painter.save()
            painter.setFont(base_font)
            
            # Check if this is the currently selected item (cached)
            is_current = (index.row() == self._current_row)
            
            # Check if hovered/focused
            is_hover = option.state & QStyle.StateFlag.State_Selected
            
            # Draw background based on state
            if is_current:
                # Currently selected item - always show with darker cyan
                painter.fillRect(option.rect, self.current_bg)
                painter.setPen(self.text_bright)
            elif is_hover:
                # Hovered item - subtle bg_light
                painter.fillRect(option.rect, self.hover_bg)
                painter.setPen(self.text_bright)
            else:
                # Normal state
                painter.setPen(self.text_normal)
            
            # Draw text with indentation
            text_rect = option.rect.adjusted(28, 0, -4, 0)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, 
                           index.data())
            
            painter.restore()
    
    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        item_type = index.data(Qt.ItemDataRole.UserRole + 1)
        
        # All items use same height to match gui.py QComboBox styling
        if item_type == "tier":
            size.setHeight(32)  # Slightly taller tier headers
        else:
            size.setHeight(28)  # Match gui.py QComboBox QAbstractItemView::item min-height
        
        return size


class GroupedModelComboBox(QComboBox):
    """
    A QComboBox that displays AI models in a hierarchical grouped structure.

    Structure (from config.AI_MODELS):
    ▾ Paid
        Anthropic Claude
            Claude Sonnet 4.5 (anthropic/claude-sonnet-4.5)
            Claude Opus 4.5 (anthropic/claude-opus-4.5)
            ...
        OpenAI
            GPT-5 (openai/gpt-5)
            ...
    ▾ Free
        Google
            Gemini 2.0 Flash Exp (google/gemini-2.0-flash-exp:free)
            ...

    Each model displays as: "Display Name (model-id)"

    All model data comes from config.AI_MODELS - no duplication.

    Args:
        colors: Dict of color values (typically gui.COLORS). Passed to delegate for styling.
                See GroupedItemDelegate for expected keys.
        parent: Parent QWidget

    Usage in gui.py:
        from grouped_model_selector import GroupedModelComboBox

        model_dropdown = GroupedModelComboBox(colors=COLORS, parent=self)
        model_dropdown.currentIndexChanged.connect(self.on_model_changed)

        # Get selected model
        model_id = model_dropdown.get_selected_model_id()
    """
    
    def __init__(self, colors=None, parent=None):
        super().__init__(parent)
        
        # Use a QStandardItemModel for hierarchical data
        self.item_model = QStandardItemModel(self)
        self.setModel(self.item_model)
        
        # Set custom delegate for rendering
        self._delegate = GroupedItemDelegate(colors, self)
        self.setItemDelegate(self._delegate)
        
        # Track model_id -> index mapping for programmatic selection
        self._model_id_to_index = {}
        self._display_name_to_index = {}
        
        # Populate the dropdown from config.AI_MODELS
        self._populate_models()
        
        # Set placeholder text and no initial selection
        self.setPlaceholderText("Select a free or paid model...")
        self.setCurrentIndex(-1)  # No selection initially
        
        # Update delegate when selection changes
        self.currentIndexChanged.connect(self._update_delegate_selection)
        
        # Make dropdown wider to fit content
        self.setMinimumWidth(280)
        self.view().setMinimumWidth(350)
        
        # Disable scroll wheel changing selection - let parent scroll instead
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    def wheelEvent(self, event):
        """Ignore wheel events so parent scroll area can scroll instead."""
        # Only handle wheel events when dropdown is expanded (popup visible)
        if self.view().isVisible():
            super().wheelEvent(event)
        else:
            # Pass to parent for scrolling
            event.ignore()
    
    def _update_delegate_selection(self, index):
        """Update the delegate's cached current row"""
        self._delegate.setCurrentRow(index)
    
    def _populate_models(self):
        """Build the hierarchical model list from config.AI_MODELS."""
        self.item_model.clear()
        self._model_id_to_index = {}
        self._display_name_to_index = {}
        
        row = 0
        for tier_name, providers in AI_MODELS.items():
            # Add tier header (Paid / Free)
            tier_item = QStandardItem(tier_name)
            tier_item.setData("tier", Qt.ItemDataRole.UserRole + 1)
            tier_item.setEnabled(False)
            tier_item.setSelectable(False)
            self.item_model.appendRow(tier_item)
            row += 1
            
            for provider_name, models in providers.items():
                # Add provider header (Anthropic Claude, Google, etc.)
                provider_item = QStandardItem(provider_name)
                provider_item.setData("provider", Qt.ItemDataRole.UserRole + 1)
                provider_item.setEnabled(False)
                provider_item.setSelectable(False)
                self.item_model.appendRow(provider_item)
                row += 1
                
                for display_name, model_id in models.items():
                    # Add model item (selectable) - show both display name and model ID
                    full_display = f"{display_name} ({model_id})"
                    model_item = QStandardItem(full_display)
                    model_item.setData("model", Qt.ItemDataRole.UserRole + 1)
                    model_item.setData(model_id, Qt.ItemDataRole.UserRole + 2)
                    self.item_model.appendRow(model_item)

                    # Track indices for lookup
                    self._model_id_to_index[model_id] = row
                    self._display_name_to_index[display_name] = row
                    row += 1
    
    def get_selected_model_id(self):
        """
        Get the model_id of the currently selected model.
        
        Returns:
            The model_id string (e.g., "claude-sonnet-4"), or None if nothing selected.
        """
        index = self.currentIndex()
        if index >= 0:
            item = self.item_model.item(index)
            if item and item.data(Qt.ItemDataRole.UserRole + 1) == "model":
                return item.data(Qt.ItemDataRole.UserRole + 2)
        return None
    
    def get_model_id_at_index(self, index):
        """
        Get the model_id at a specific index.
        
        Returns:
            The model_id string, or None if index is invalid or not a model item.
        """
        if index >= 0:
            item = self.item_model.item(index)
            if item and item.data(Qt.ItemDataRole.UserRole + 1) == "model":
                return item.data(Qt.ItemDataRole.UserRole + 2)
        return None
    
    def get_selected_display_name(self):
        """
        Get the display name of the currently selected model.
        
        Returns:
            The display name (e.g., "Claude Sonnet 4 (claude-sonnet-4)"), or empty string.
        """
        index = self.currentIndex()
        if index >= 0:
            item = self.item_model.item(index)
            if item and item.data(Qt.ItemDataRole.UserRole + 1) == "model":
                return item.text()
        return ""
    
    def set_model_by_id(self, model_id):
        """
        Set the selection by model_id.
        
        Args:
            model_id: The model ID string (e.g., "claude-sonnet-4")
        """
        if model_id in self._model_id_to_index:
            self.setCurrentIndex(self._model_id_to_index[model_id])
    
    def set_model_by_display_name(self, display_name):
        """
        Set the selection by display name.
        
        Args:
            display_name: The full display name (e.g., "Claude Sonnet 4 (claude-sonnet-4)")
        """
        if display_name in self._display_name_to_index:
            self.setCurrentIndex(self._display_name_to_index[display_name])
    
    def refresh_models(self):
        """
        Refresh the model list from config.AI_MODELS.
        
        Call this if AI_MODELS has been modified at runtime.
        """
        current_model_id = self.get_selected_model_id()
        self._populate_models()
        if current_model_id:
            self.set_model_by_id(current_model_id)


# =============================================================================
# Standalone testing - run with: poetry run python grouped_model_selector.py
# =============================================================================

if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget, QLabel
    
    # Import styles (single source of truth)
    from styles import COLORS, get_combobox_style
    
    app = QApplication(sys.argv)
    
    # Apply dark theme for testing (matches gui.py styling via styles.py)
    app.setStyleSheet(f"""
        QWidget {{
            background-color: {COLORS['bg_dark']};
            color: {COLORS['text_normal']};
            font-family: "Segoe UI", sans-serif;
            font-size: 10px;
        }}
    """)
    
    window = QWidget()
    window.setWindowTitle("Grouped Model Selector Test")
    window.setMinimumSize(500, 300)
    
    layout = QVBoxLayout(window)
    
    label = QLabel("Select AI Model:")
    layout.addWidget(label)
    
    # Pass COLORS to the widget and apply combobox style
    combo = GroupedModelComboBox(colors=COLORS)
    combo.setStyleSheet(get_combobox_style())
    layout.addWidget(combo)
    
    # Show selected model info
    info_label = QLabel("Selected: None")
    info_label.setWordWrap(True)
    layout.addWidget(info_label)
    
    def on_selection_changed():
        model_id = combo.get_selected_model_id()
        display = combo.get_selected_display_name()
        if model_id:
            info_label.setText(f"Display: {display}\nModel ID: {model_id}")
        else:
            info_label.setText("Selected: (header - not selectable)")
    
    combo.currentIndexChanged.connect(on_selection_changed)
    
    layout.addStretch()
    window.show()
    
    sys.exit(app.exec())