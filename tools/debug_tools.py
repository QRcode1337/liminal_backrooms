# debug_tools.py
"""
Debug Tools for PyQt6 Applications

Provides Chrome DevTools-like inspection capabilities:
- F12: Toggle debug panel
- Ctrl+Shift+C: Enable element picker (click any widget to inspect)
- Live stylesheet editing
- Widget hierarchy viewer
- Property inspector

Usage:
    from debug_tools import DebugManager
    
    # In your main window __init__ or after app creation:
    self.debug_manager = DebugManager(self)
    
    # Or attach to QApplication for global access:
    debug_manager = DebugManager(main_window)
"""

from PyQt6.QtCore import Qt, QObject, QEvent, pyqtSignal, QTimer, QSize, QRect, QPoint
from PyQt6.QtGui import QFont, QColor, QCursor, QPainter, QPen, QKeySequence, QShortcut, QBrush, QPainterPath, QPolygon
from PyQt6.QtWidgets import (
    QWidget, QApplication, QMainWindow, QDockWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QTreeWidget, QTreeWidgetItem, QLabel, QPushButton, QFrame,
    QSplitter, QTabWidget, QLineEdit, QScrollArea, QSizePolicy, QStyle,
    QStyleOption, QComboBox, QCheckBox, QGroupBox, QPlainTextEdit, QMenu,
    QStyledItemDelegate
)


class CyanArrowTreeWidget(QTreeWidget):
    """
    Custom QTreeWidget that draws cyan branch indicators.
    Overrides drawBranches to render custom colored expand/collapse arrows.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._arrow_color = QColor("#06B6D4")  # Cyan
        self._arrow_hover_color = QColor("#38BDF8")  # Brighter cyan
        self._hovered_index = None
        self.setMouseTracking(True)
    
    def drawBranches(self, painter, rect, index):
        """Override to draw custom colored branch indicators"""
        # Let Qt draw the branch lines (but not the arrows - we'll override those via stylesheet)
        # We need to draw our own arrows on top
        
        # Get the item
        item = self.itemFromIndex(index)
        if item is None:
            return
        
        # Check if item has children OR is marked as expandable
        has_children = item.childCount() > 0
        
        if not has_children:
            return
        
        # Calculate arrow position (left side of the row)
        indent = self.indentation()
        level = 0
        parent = item.parent()
        while parent:
            level += 1
            parent = parent.parent()
        
        arrow_size = 8
        x = rect.left() + (level * indent) + (indent - arrow_size) // 2
        y = rect.top() + (rect.height() - arrow_size) // 2
        
        # Determine if this is the hovered item
        is_hovered = (self._hovered_index is not None and 
                      self._hovered_index == index)
        
        color = self._arrow_hover_color if is_hovered else self._arrow_color
        
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        
        if item.isExpanded():
            # Down arrow (expanded)
            points = [
                QPoint(x, y + 2),
                QPoint(x + arrow_size, y + 2),
                QPoint(x + arrow_size // 2, y + arrow_size - 1)
            ]
        else:
            # Right arrow (collapsed)
            points = [
                QPoint(x + 2, y),
                QPoint(x + arrow_size - 1, y + arrow_size // 2),
                QPoint(x + 2, y + arrow_size)
            ]
        
        painter.drawPolygon(QPolygon(points))
        painter.restore()
    
    def mouseMoveEvent(self, event):
        """Track hover for branch indicators"""
        index = self.indexAt(event.pos())
        if index != self._hovered_index:
            self._hovered_index = index
            self.viewport().update()
        super().mouseMoveEvent(event)
    
    def leaveEvent(self, event):
        """Clear hover state"""
        self._hovered_index = None
        self.viewport().update()
        super().leaveEvent(event)


class WidgetHighlighter(QWidget):
    """Transparent overlay that highlights the currently inspected widget"""
    
    def __init__(self, main_window=None):
        super().__init__(None)  # No parent - top-level window
        self._main_window = main_window
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        # Tool window stays on top but doesn't steal focus
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.Tool |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self._color = QColor(6, 182, 212, 80)  # Cyan with transparency
        self._border_color = QColor(6, 182, 212, 255)
        self._target_widget = None
    
    def set_main_window(self, main_window):
        """Set the main window reference"""
        self._main_window = main_window
        
    def highlight_widget(self, widget):
        """Position overlay over the target widget"""
        if widget is None:
            self.hide()
            self._target_widget = None
            return
        
        self._target_widget = widget
            
        try:
            # Use global coordinates - simpler and more reliable
            global_pos = widget.mapToGlobal(QPoint(0, 0))
            self.setGeometry(global_pos.x(), global_pos.y(), widget.width(), widget.height())
            
            # Only show if main window is active/visible
            if self._main_window and self._main_window.isActiveWindow():
                self.show()
                self.raise_()
            elif not self._main_window:
                self.show()
                self.raise_()
        except RuntimeError:
            self.hide()
    
    def hideIfNotActive(self):
        """Hide if the main window is not active"""
        if self._main_window and not self._main_window.isActiveWindow():
            self.hide()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._color)
        painter.setPen(QPen(self._border_color, 2))
        painter.drawRect(self.rect().adjusted(1, 1, -1, -1))


class ElementPicker(QObject):
    """Allows clicking on any widget to select it for inspection"""
    
    widget_picked = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._active = False
        self._highlighter = WidgetHighlighter()
    
    def _is_debug_widget(self, widget):
        """Check if widget is part of the debug panel"""
        if widget is None:
            return True
        if isinstance(widget, (WidgetHighlighter,)):
            return True
        
        # Walk up the parent chain to check if any parent is the debug panel
        current = widget
        while current is not None:
            obj_name = current.objectName()
            class_name = current.__class__.__name__
            
            # Check by object name or class name
            if (obj_name and 'Debug' in obj_name) or class_name in ('DebugPanel', 'TitleBarButton'):
                return True
            if isinstance(current, QDockWidget) and 'Debug' in (current.windowTitle() or ''):
                return True
                
            current = current.parent()
        
        return False
        
    def start(self):
        """Start element picking mode"""
        self._active = True
        QApplication.instance().installEventFilter(self)
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.CrossCursor))
        
    def stop(self):
        """Stop element picking mode"""
        self._active = False
        QApplication.instance().removeEventFilter(self)
        QApplication.restoreOverrideCursor()
        self._highlighter.hide()
        
    def eventFilter(self, obj, event):
        if not self._active:
            return False
            
        if event.type() == QEvent.Type.MouseMove:
            widget = QApplication.widgetAt(QCursor.pos())
            if widget and not self._is_debug_widget(widget):
                self._highlighter.highlight_widget(widget)
            else:
                self._highlighter.hide()
            return False
            
        elif event.type() == QEvent.Type.MouseButtonPress:
            widget = QApplication.widgetAt(QCursor.pos())
            if widget and not self._is_debug_widget(widget):
                self.widget_picked.emit(widget)
            self.stop()
            return True
            
        elif event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Escape:
                self.stop()
                return True
                
        return False


class StylesheetEditor(QWidget):
    """Live stylesheet editor with apply button"""
    
    stylesheet_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._target_widget = None
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Header
        header = QLabel("Stylesheet Editor")
        header.setStyleSheet("font-weight: bold; color: #06B6D4;")
        layout.addWidget(header)
        
        # Editor
        self.editor = QPlainTextEdit()
        self.editor.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1E293B;
                color: #E2E8F0;
                border: 1px solid #334155;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10px;
            }
        """)
        self.editor.setPlaceholderText("Enter stylesheet here...")
        layout.addWidget(self.editor)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._apply_stylesheet)
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #06B6D4;
                color: #0A0E1A;
                border: none;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0891B2;
            }
        """)
        btn_layout.addWidget(self.apply_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self._reset_stylesheet)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #E2E8F0;
                border: none;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #475569;
            }
        """)
        btn_layout.addWidget(self.reset_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
    def set_target(self, widget):
        """Set the widget to edit"""
        self._target_widget = widget
        if widget:
            self._original_stylesheet = widget.styleSheet()
            self.editor.setPlainText(self._original_stylesheet)
        else:
            self._original_stylesheet = ""
            self.editor.clear()
            
    def _apply_stylesheet(self):
        if self._target_widget:
            try:
                self._target_widget.setStyleSheet(self.editor.toPlainText())
                self.stylesheet_changed.emit(self.editor.toPlainText())
            except Exception as e:
                print(f"Error applying stylesheet: {e}")
                
    def _reset_stylesheet(self):
        if self._target_widget:
            self.editor.setPlainText(self._original_stylesheet)
            self._target_widget.setStyleSheet(self._original_stylesheet)
            # Emit signal so any listeners know about the change
            self.stylesheet_changed.emit(self._original_stylesheet)
            # Force widget to update its appearance
            self._target_widget.style().unpolish(self._target_widget)
            self._target_widget.style().polish(self._target_widget)
            self._target_widget.update()


class PropertyInspector(QWidget):
    """Shows properties of the selected widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Property tree with custom cyan arrows
        self.tree = CyanArrowTreeWidget()
        self.tree.setHeaderLabels(["Property", "Value"])
        self.tree.setColumnWidth(0, 150)
        self.tree.setStyleSheet(self._get_tree_stylesheet())
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        
        layout.addWidget(self.tree)
    
    def _show_context_menu(self, position):
        """Show context menu for copying values"""
        item = self.tree.itemAt(position)
        if not item:
            return
        
        menu = QMenu(self.tree)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1E293B;
                color: #E2E8F0;
                border: 1px solid #334155;
            }
            QMenu::item:selected {
                background-color: #334155;
            }
        """)
        
        copy_value = menu.addAction("Copy Value")
        copy_prop = menu.addAction("Copy Property Name")
        copy_both = menu.addAction("Copy Both")
        
        action = menu.exec(self.tree.viewport().mapToGlobal(position))
        
        clipboard = QApplication.clipboard()
        if action == copy_value:
            clipboard.setText(item.text(1))
        elif action == copy_prop:
            clipboard.setText(item.text(0))
        elif action == copy_both:
            clipboard.setText(f"{item.text(0)}: {item.text(1)}")
    
    def _get_tree_stylesheet(self):
        """Get stylesheet for tree widget - uses cyan accent for selection"""
        return """
            QTreeWidget {
                background-color: #0A0E1A;
                color: #CBD5E1;
                border: 1px solid #334155;
                font-size: 10px;
            }
            QTreeWidget::item {
                padding: 2px;
            }
            QTreeWidget::item:selected {
                background-color: #164E63;
                border-left: 2px solid #06B6D4;
            }
            QTreeWidget::item:hover:!selected {
                background-color: #1E293B;
            }
            QHeaderView::section {
                background-color: #111827;
                color: #94A3B8;
                border: none;
                padding: 4px;
                font-weight: bold;
            }
        """
        
    def inspect_widget(self, widget):
        """Populate tree with widget properties"""
        self.tree.clear()
        
        if widget is None:
            return
        
        # Helper to create expandable sections
        def create_section(name, expanded=True):
            item = QTreeWidgetItem([name, ""])
            item.setExpanded(expanded)
            item.setForeground(0, QColor("#06B6D4"))  # Cyan for section headers
            return item
            
        # Basic info
        basic = create_section("Basic Info")
        self.tree.addTopLevelItem(basic)
        
        QTreeWidgetItem(basic, ["Class", widget.__class__.__name__])
        QTreeWidgetItem(basic, ["Object Name", widget.objectName() or "(none)"])
        QTreeWidgetItem(basic, ["Visible", str(widget.isVisible())])
        QTreeWidgetItem(basic, ["Enabled", str(widget.isEnabled())])
        
        # Geometry
        geo = create_section("Geometry")
        self.tree.addTopLevelItem(geo)
        
        rect = widget.geometry()
        QTreeWidgetItem(geo, ["Position", f"({rect.x()}, {rect.y()})"])
        QTreeWidgetItem(geo, ["Size", f"{rect.width()} × {rect.height()}"])
        QTreeWidgetItem(geo, ["Min Size", f"{widget.minimumWidth()} × {widget.minimumHeight()}"])
        QTreeWidgetItem(geo, ["Max Size", f"{widget.maximumWidth()} × {widget.maximumHeight()}"])
        
        # Font
        font_item = create_section("Font")
        self.tree.addTopLevelItem(font_item)
        
        font = widget.font()
        QTreeWidgetItem(font_item, ["Family", font.family()])
        QTreeWidgetItem(font_item, ["Size", f"{font.pointSize()}pt / {font.pixelSize()}px"])
        QTreeWidgetItem(font_item, ["Bold", str(font.bold())])
        QTreeWidgetItem(font_item, ["Italic", str(font.italic())])
        
        # Colors (from palette)
        colors = create_section("Palette Colors")
        self.tree.addTopLevelItem(colors)
        
        palette = widget.palette()
        for role_name in ['Window', 'WindowText', 'Base', 'Text', 'Button', 'ButtonText', 'Highlight', 'HighlightedText']:
            try:
                role = getattr(palette.ColorRole, role_name)
                color = palette.color(role)
                item = QTreeWidgetItem(colors, [role_name, color.name()])
                item.setBackground(1, color)
                # Set text color for readability
                if color.lightness() > 128:
                    item.setForeground(1, QColor('black'))
                else:
                    item.setForeground(1, QColor('white'))
            except:
                pass
        
        # Widget-specific properties
        if isinstance(widget, QComboBox):
            combo = create_section("QComboBox")
            self.tree.addTopLevelItem(combo)
            QTreeWidgetItem(combo, ["Current Index", str(widget.currentIndex())])
            QTreeWidgetItem(combo, ["Current Text", widget.currentText()])
            QTreeWidgetItem(combo, ["Item Count", str(widget.count())])
            QTreeWidgetItem(combo, ["Editable", str(widget.isEditable())])
            
            # Check for custom delegate
            delegate = widget.itemDelegate()
            QTreeWidgetItem(combo, ["Delegate Class", delegate.__class__.__name__])
            
        # Stylesheet
        ss = create_section("Stylesheet", expanded=False)  # Collapsed by default
        self.tree.addTopLevelItem(ss)
        
        stylesheet = widget.styleSheet()
        if stylesheet:
            # Truncate long stylesheets
            preview = stylesheet[:200] + "..." if len(stylesheet) > 200 else stylesheet
            QTreeWidgetItem(ss, ["Value", preview])
            QTreeWidgetItem(ss, ["Length", f"{len(stylesheet)} chars"])
        else:
            QTreeWidgetItem(ss, ["Value", "(none - inherited)"])
            
        # Parent chain
        parents = create_section("Parent Chain")
        self.tree.addTopLevelItem(parents)
        
        parent = widget.parent()
        depth = 0
        while parent and depth < 10:
            QTreeWidgetItem(parents, [f"Level {depth}", f"{parent.__class__.__name__} ({parent.objectName() or 'unnamed'})"])
            parent = parent.parent()
            depth += 1
    
    def inspect_model_item(self, item_data):
        """Populate tree with model item properties"""
        self.tree.clear()
        
        if item_data is None:
            return
        
        # Helper to create expandable sections
        def create_section(name, expanded=True):
            item = QTreeWidgetItem([name, ""])
            item.setExpanded(expanded)
            item.setForeground(0, QColor("#A855F7"))  # Purple for model items
            return item
        
        # Basic info
        basic = create_section("Model Item Info")
        self.tree.addTopLevelItem(basic)
        
        QTreeWidgetItem(basic, ["Text", str(item_data.get('text', ''))])
        QTreeWidgetItem(basic, ["Type", str(item_data.get('type', ''))])
        QTreeWidgetItem(basic, ["Row", str(item_data.get('row', ''))])
        QTreeWidgetItem(basic, ["Selectable", str(item_data.get('selectable', True))])
        
        # Flags
        flags = item_data.get('flags')
        if flags:
            flags_section = create_section("Item Flags")
            self.tree.addTopLevelItem(flags_section)
            QTreeWidgetItem(flags_section, ["IsSelectable", str(bool(flags & Qt.ItemFlag.ItemIsSelectable))])
            QTreeWidgetItem(flags_section, ["IsEnabled", str(bool(flags & Qt.ItemFlag.ItemIsEnabled))])
            QTreeWidgetItem(flags_section, ["IsEditable", str(bool(flags & Qt.ItemFlag.ItemIsEditable))])
        
        # All data roles
        data = item_data.get('data', {})
        if data:
            data_section = create_section("Data Roles")
            self.tree.addTopLevelItem(data_section)
            
            for role_name, value in data.items():
                # Format the value nicely
                if hasattr(value, 'name'):  # QColor, etc.
                    value_str = f"{value.name()} ({type(value).__name__})"
                elif isinstance(value, QColor):
                    value_str = f"{value.name()} (QColor)"
                else:
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:100] + "..."
                
                QTreeWidgetItem(data_section, [str(role_name), value_str])
        
        # Parent list view info
        list_view = item_data.get('list_view')
        if list_view:
            view_section = create_section("Parent View")
            self.tree.addTopLevelItem(view_section)
            
            QTreeWidgetItem(view_section, ["Class", list_view.__class__.__name__])
            QTreeWidgetItem(view_section, ["Object Name", list_view.objectName() or "(none)"])
            
            # Delegate info
            delegate = list_view.itemDelegate()
            if delegate:
                QTreeWidgetItem(view_section, ["Delegate", delegate.__class__.__name__])


class WidgetTree(QWidget):
    """Hierarchical view of all widgets"""
    
    widget_selected = pyqtSignal(object)
    model_item_selected = pyqtSignal(dict)  # Emits model item data dict
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._root_widget = None
        self._widget_map = {}
        self._model_item_map = {}  # For model items (not widgets)
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Refresh button
        self.refresh_btn = QPushButton("↻ Refresh Tree")
        self.refresh_btn.clicked.connect(self.refresh)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #E2E8F0;
                border: none;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #475569;
            }
        """)
        layout.addWidget(self.refresh_btn)
        
        # Tree with custom cyan arrows
        self.tree = CyanArrowTreeWidget()
        self.tree.setHeaderLabels(["Widget", "Class"])
        self.tree.setColumnWidth(0, 200)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.setStyleSheet(self._get_tree_stylesheet())
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        
        layout.addWidget(self.tree)
    
    def _show_context_menu(self, position):
        """Show context menu for copying values"""
        item = self.tree.itemAt(position)
        if not item:
            return
        
        menu = QMenu(self.tree)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1E293B;
                color: #E2E8F0;
                border: 1px solid #334155;
            }
            QMenu::item:selected {
                background-color: #334155;
            }
        """)
        
        copy_name = menu.addAction("Copy Widget Name")
        copy_class = menu.addAction("Copy Class Name")
        copy_both = menu.addAction("Copy Both")
        
        action = menu.exec(self.tree.viewport().mapToGlobal(position))
        
        clipboard = QApplication.clipboard()
        if action == copy_name:
            clipboard.setText(item.text(0))
        elif action == copy_class:
            clipboard.setText(item.text(1))
        elif action == copy_both:
            clipboard.setText(f"{item.text(0)}: {item.text(1)}")
        
    def _get_tree_stylesheet(self):
        """Get stylesheet for tree widget - uses cyan accent for selection"""
        return """
            QTreeWidget {
                background-color: #0A0E1A;
                color: #CBD5E1;
                border: 1px solid #334155;
                font-size: 10px;
            }
            QTreeWidget::item {
                padding: 2px;
            }
            QTreeWidget::item:selected {
                background-color: #164E63;
                border-left: 2px solid #06B6D4;
            }
            QTreeWidget::item:hover:!selected {
                background-color: #1E293B;
            }
            QHeaderView::section {
                background-color: #111827;
                color: #94A3B8;
                border: none;
                padding: 4px;
                font-weight: bold;
            }
        """
        
    def set_root(self, widget):
        """Set the root widget to display"""
        self._root_widget = widget
        self.refresh()
        
    def refresh(self):
        """Rebuild the tree, including popup widgets"""
        self.tree.clear()
        self._widget_map.clear()
        self._reverse_widget_map = {}  # Map widget id to tree item
        self._model_item_map = {}  # Map tree item id to model item data
        
        if self._root_widget:
            self._add_widget_to_tree(self._root_widget, None)
        
        # Also add any visible popup widgets (menus, combo dropdowns, etc.)
        self._add_popup_widgets()
    
    def _add_popup_widgets(self):
        """Find and add popup widgets to the tree"""
        app = QApplication.instance()
        if not app:
            return
            
        # Create a "Popups" section if we find any
        popups_found = []
        
        for widget in app.topLevelWidgets():
            # Skip the main window (already in tree) and debug panel
            if widget == self._root_widget:
                continue
            if 'Debug' in (widget.objectName() or '') or 'Debug' in widget.__class__.__name__:
                continue
            if isinstance(widget, WidgetHighlighter):
                continue
                
            # Check if it's a popup-like widget
            flags = widget.windowFlags()
            if (flags & Qt.WindowType.Popup or 
                flags & Qt.WindowType.ToolTip or
                widget.__class__.__name__ in ['QMenu', 'QComboBoxPrivateContainer']):
                if widget.isVisible():
                    popups_found.append(widget)
        
        if popups_found:
            # Add a "Popups" parent item
            popups_item = QTreeWidgetItem(["[Popups]", ""])
            popups_item.setForeground(0, QColor("#06B6D4"))
            self.tree.addTopLevelItem(popups_item)
            popups_item.setExpanded(True)
            
            for popup in popups_found:
                popup_tree_item = self._add_widget_to_tree_with_model_items(popup, popups_item)
    
    def _add_widget_to_tree_with_model_items(self, widget, parent_item):
        """Add widget to tree, and if it's a list view, add model items as children"""
        from PyQt6.QtWidgets import QAbstractItemView
        
        name = widget.objectName() or "(unnamed)"
        class_name = widget.__class__.__name__
        
        # Get children
        children = widget.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly)
        
        display_name = name
        if len(children) > 0:
            display_name = f"{name} ({len(children)})"
        
        item = QTreeWidgetItem([display_name, class_name])
        self._widget_map[id(item)] = widget
        self._reverse_widget_map[id(widget)] = item
        
        if parent_item:
            parent_item.addChild(item)
        else:
            self.tree.addTopLevelItem(item)
        
        item.setExpanded(True)
        
        # If this is a list view, add model items as children
        if isinstance(widget, QAbstractItemView):
            model = widget.model()
            if model and model.rowCount() > 0:
                self._add_model_items_to_tree(widget, model, item)
        
        # Recurse for child widgets
        for child in children:
            self._add_widget_to_tree_with_model_items(child, item)
        
        return item
    
    def _add_model_items_to_tree(self, list_view, model, parent_item):
        """Add model items as purple children under the list view"""
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            item_text = model.data(index, Qt.ItemDataRole.DisplayRole) or "(no text)"
            item_type = model.data(index, Qt.ItemDataRole.UserRole + 1) or "item"
            
            # Get all available data roles for inspection
            item_data = {}
            for role in [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.DecorationRole,
                        Qt.ItemDataRole.EditRole, Qt.ItemDataRole.ToolTipRole,
                        Qt.ItemDataRole.StatusTipRole, Qt.ItemDataRole.WhatsThisRole,
                        Qt.ItemDataRole.FontRole, Qt.ItemDataRole.TextAlignmentRole,
                        Qt.ItemDataRole.BackgroundRole, Qt.ItemDataRole.ForegroundRole,
                        Qt.ItemDataRole.CheckStateRole, Qt.ItemDataRole.SizeHintRole]:
                data = model.data(index, role)
                if data is not None:
                    item_data[role.name if hasattr(role, 'name') else str(role)] = data
            
            # Also get custom user roles
            for user_role_offset in range(10):
                role = Qt.ItemDataRole.UserRole + user_role_offset
                data = model.data(index, role)
                if data is not None:
                    item_data[f"UserRole+{user_role_offset}"] = data
            
            # Check if item is selectable
            flags = model.flags(index)
            selectable = bool(flags & Qt.ItemFlag.ItemIsSelectable) if flags else True
            
            item_node = QTreeWidgetItem([f"[{row}] {item_text}", str(item_type)])
            item_node.setForeground(0, QColor("#A855F7"))  # Purple for model items
            item_node.setForeground(1, QColor("#A855F7"))
            
            if not selectable:
                item_node.setForeground(0, QColor("#64748B"))  # Dim non-selectable
                item_node.setForeground(1, QColor("#64748B"))
            
            parent_item.addChild(item_node)
            
            # Store model item info for inspection
            self._model_item_map[id(item_node)] = {
                'list_view': list_view,
                'model': model,
                'row': row,
                'index': index,
                'text': str(item_text),
                'type': str(item_type),
                'selectable': selectable,
                'flags': flags,
                'data': item_data
            }
            
    def _add_widget_to_tree(self, widget, parent_item):
        """Recursively add widgets to tree"""
        name = widget.objectName() or "(unnamed)"
        class_name = widget.__class__.__name__
        
        # Get children first so we can show count
        children = widget.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly)
        
        # Add child count to name if has children (helps debug tree structure)
        display_name = name
        if len(children) > 0:
            display_name = f"{name} ({len(children)})"
        
        item = QTreeWidgetItem([display_name, class_name])
        self._widget_map[id(item)] = widget
        self._reverse_widget_map[id(widget)] = item  # Reverse lookup
        
        if parent_item:
            parent_item.addChild(item)
        else:
            self.tree.addTopLevelItem(item)
            item.setExpanded(True)
            
        # Add children
        for child in children:
            self._add_widget_to_tree(child, item)
            
    def _on_item_clicked(self, item, column):
        # Check if it's a model item first
        model_item_data = self._model_item_map.get(id(item))
        if model_item_data:
            self.model_item_selected.emit(model_item_data)
            return
        
        # Otherwise it's a widget
        widget = self._widget_map.get(id(item))
        if widget:
            self.widget_selected.emit(widget)
    
    def _on_item_double_clicked(self, item, column):
        """Handle double-click to select widget or model item"""
        # Check if it's a model item first
        model_item_data = self._model_item_map.get(id(item))
        if model_item_data:
            self.model_item_selected.emit(model_item_data)
            return
        
        widget = self._widget_map.get(id(item))
        if widget:
            self.widget_selected.emit(widget)
    
    def select_widget(self, widget):
        """Select and reveal a widget in the tree"""
        if widget is None:
            return
            
        # Look up the tree item for this widget
        item = self._reverse_widget_map.get(id(widget))
        
        if item is None:
            # Widget might not be in tree yet, refresh and try again
            self.refresh()
            item = self._reverse_widget_map.get(id(widget))
        
        if item:
            # Expand all parents to make item visible
            parent = item.parent()
            while parent:
                parent.setExpanded(True)
                parent = parent.parent()
            
            # Select and scroll to the item
            self.tree.setCurrentItem(item)
            self.tree.scrollToItem(item)


class TitleBarButton(QPushButton):
    """Custom painted button for title bar icons"""
    
    def __init__(self, icon_type, parent=None):
        super().__init__(parent)
        self._icon_type = icon_type  # 'float' or 'close'
        self._hovered = False
        self._color = QColor("#94A3B8")  # Default gray
        self._hover_color = QColor("#E2E8F0")  # Bright on hover
        self._hover_bg = QColor("#475569")  # Hover background
        self._close_hover_bg = QColor("#EF4444")  # Red for close
        self.setFixedSize(24, 24)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
    
    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        if self._hovered:
            if self._icon_type == 'close':
                painter.fillRect(self.rect(), self._close_hover_bg)
            else:
                painter.fillRect(self.rect(), self._hover_bg)
        
        # Icon color
        color = self._hover_color if self._hovered else self._color
        painter.setPen(QPen(color, 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        # Center the icon
        cx, cy = self.width() // 2, self.height() // 2
        
        if self._icon_type == 'float':
            # Draw window/float icon (overlapping squares)
            painter.drawRect(cx - 5, cy - 3, 8, 6)
            painter.drawRect(cx - 3, cy - 5, 8, 6)
        elif self._icon_type == 'close':
            # Draw X
            painter.drawLine(cx - 4, cy - 4, cx + 4, cy + 4)
            painter.drawLine(cx + 4, cy - 4, cx - 4, cy + 4)


class DebugPanel(QDockWidget):
    """Main debug panel - dockable window with all debug tools"""
    
    def __init__(self, parent=None):
        super().__init__("", parent)  # Empty title, we'll use custom title bar
        self.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.BottomDockWidgetArea)
        self.setMinimumWidth(400)
        self.setObjectName("DebugPanel")
        
        # Store main window reference
        self._main_window = parent
        
        # Use custom title bar for better icon visibility
        self._setup_title_bar()
        
        self._current_widget = None
        self._highlighter = WidgetHighlighter(parent)
        self._element_picker = ElementPicker()
        self._element_picker.widget_picked.connect(self._on_widget_picked)
        
        self._setup_ui()
        self._apply_styling()
        
        # ESC key to deselect
        self._esc_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self._esc_shortcut.activated.connect(self._deselect_all)
        
        # Hide highlighter when app loses focus
        if QApplication.instance():
            QApplication.instance().applicationStateChanged.connect(self._on_app_state_changed)
    
    def _on_app_state_changed(self, state):
        """Hide highlighter when app is not active"""
        from PyQt6.QtCore import Qt as QtCore_Qt
        if state != Qt.ApplicationState.ApplicationActive:
            self._highlighter.hide()
        elif self._current_widget:
            # Re-show when app becomes active again
            self._highlighter.highlight_widget(self._current_widget)
    
    def _setup_title_bar(self):
        """Create custom title bar with painted icons"""
        title_bar = QWidget()
        title_bar.setObjectName("DebugPanelTitleBar")
        title_bar.setStyleSheet("background-color: #111827;")
        layout = QHBoxLayout(title_bar)
        layout.setContentsMargins(8, 4, 4, 4)
        layout.setSpacing(6)
        
        # Title
        title_label = QLabel("Debug Tools")
        title_label.setStyleSheet("color: #06B6D4; font-weight: bold; font-size: 11px;")
        layout.addWidget(title_label)
        
        layout.addStretch()
        
        # Float/dock button - custom painted
        self.float_btn = TitleBarButton('float')
        self.float_btn.setToolTip("Undock panel (pop out to separate window)")
        self.float_btn.clicked.connect(self._toggle_float)
        layout.addWidget(self.float_btn)
        
        # Close button - custom painted
        close_btn = TitleBarButton('close')
        close_btn.setToolTip("Close debug panel (F12 to reopen)")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        self.setTitleBarWidget(title_bar)
    
    def _toggle_float(self):
        """Toggle between floating and docked state"""
        self.setFloating(not self.isFloating())
        # Update tooltip based on new state
        if self.isFloating():
            self.float_btn.setToolTip("Dock panel (attach to main window)")
        else:
            self.float_btn.setToolTip("Undock panel (pop out to separate window)")
        
    def _setup_ui(self):
        # Main container
        container = QWidget()
        container.setObjectName("DebugPanelContainer")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Toolbar
        toolbar = QHBoxLayout()
        
        self.pick_btn = QPushButton("Pick Element")
        self.pick_btn.clicked.connect(self._start_picking)
        toolbar.addWidget(self.pick_btn)
        
        self.selected_label = QLabel("No widget selected")
        self.selected_label.setStyleSheet("color: #94A3B8;")
        toolbar.addWidget(self.selected_label, 1)
        
        layout.addLayout(toolbar)
        
        # Tab widget
        tabs = QTabWidget()
        tabs.setObjectName("DebugPanelTabs")
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #334155;
                background-color: #0A0E1A;
            }
            QTabBar::tab {
                background-color: #111827;
                color: #94A3B8;
                padding: 8px 16px;
                border: none;
            }
            QTabBar::tab:selected {
                background-color: #1E293B;
                color: #06B6D4;
            }
        """)
        
        # Properties tab
        self.property_inspector = PropertyInspector()
        tabs.addTab(self.property_inspector, "Properties")
        
        # Stylesheet tab
        self.stylesheet_editor = StylesheetEditor()
        tabs.addTab(self.stylesheet_editor, "Stylesheet")
        
        # Widget tree tab
        self.widget_tree = WidgetTree()
        self.widget_tree.widget_selected.connect(self._on_widget_picked)
        self.widget_tree.model_item_selected.connect(self._on_model_item_picked)
        tabs.addTab(self.widget_tree, "Widget Tree")
        
        layout.addWidget(tabs)
        
        self.setWidget(container)
        
    def _apply_styling(self):
        self.setStyleSheet("""
            QDockWidget {
                background-color: #0A0E1A;
                color: #E2E8F0;
                font-size: 11px;
            }
            QPushButton {
                background-color: #1E293B;
                color: #E2E8F0;
                border: 1px solid #334155;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #334155;
                border-color: #06B6D4;
            }
            /* Standardized scrollbar style - retro CRT theme */
            QScrollBar:vertical {
                background-color: #0A0E1A;
                width: 12px;
                border: 1px solid #1E293B;
                border-radius: 0px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #06B6D4;
                border: none;
                border-radius: 0px;
                min-height: 30px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #38BDF8;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
                border: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            QScrollBar:horizontal {
                background-color: #0A0E1A;
                height: 12px;
                border: 1px solid #1E293B;
                border-radius: 0px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background-color: #06B6D4;
                border: none;
                border-radius: 0px;
                min-width: 30px;
                margin: 2px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #38BDF8;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
                border: none;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
        """)
        
    def _start_picking(self):
        """Enter element picker mode"""
        self.pick_btn.setText("Picking... (ESC to cancel)")
        self._element_picker.start()
    
    def _delayed_capture(self):
        """Capture widget tree after a delay - allows time to open dropdowns"""
    def _on_widget_picked(self, widget):
        """Handle widget selection"""
        self.pick_btn.setText("Pick Element")
        self._current_widget = widget
        
        # Update label
        class_name = widget.__class__.__name__
        obj_name = widget.objectName() or "(unnamed)"
        self.selected_label.setText(f"{class_name}: {obj_name}")
        self.selected_label.setStyleSheet("color: #06B6D4; font-weight: bold;")
        
        # Update inspectors
        self.property_inspector.inspect_widget(widget)
        self.stylesheet_editor.set_target(widget)
        
        # Highlight the widget (stays until another widget is selected or panel closed)
        self._highlighter.highlight_widget(widget)
        
        # Auto-expand and select in widget tree
        self.widget_tree.select_widget(widget)
    
    def _on_model_item_picked(self, item_data):
        """Handle model item selection (dropdown items, etc.)"""
        self.pick_btn.setText("Pick Element")
        self._current_widget = None
        
        # Update label with model item info
        text = item_data.get('text', '(unknown)')
        item_type = item_data.get('type', 'item')
        row = item_data.get('row', '?')
        self.selected_label.setText(f"[Model Item] Row {row}: {text}")
        self.selected_label.setStyleSheet("color: #A855F7; font-weight: bold;")
        
        # Update property inspector with model item data
        self.property_inspector.inspect_model_item(item_data)
        
        # Clear stylesheet editor (model items don't have stylesheets)
        self.stylesheet_editor.set_target(None)
        
        # Hide widget highlight
        self._highlighter.hide()
    
    def _deselect_all(self):
        """Deselect all elements and clear highlight (ESC key)"""
        self._current_widget = None
        self._highlighter.hide()
        
        # Reset label
        self.selected_label.setText("No widget selected")
        self.selected_label.setStyleSheet("color: #94A3B8;")
        
        # Clear property inspector
        self.property_inspector.inspect_widget(None)
        self.stylesheet_editor.set_target(None)
        
        # Clear tree selection and refresh
        self.widget_tree.tree.clearSelection()
        self.widget_tree.refresh()
    
    def _clear_highlight(self):
        """Clear the current highlight"""
        self._highlighter.hide()
        
    def closeEvent(self, event):
        """Clean up when panel is closed"""
        self._highlighter.hide()
        super().closeEvent(event)
        
    def set_root_widget(self, widget):
        """Set the root widget for the tree view"""
        self.widget_tree.set_root(widget)


class DebugManager:
    """
    Main entry point for debug tools.
    
    Usage:
        debug_manager = DebugManager(main_window)
        
    Keyboard shortcuts:
        F12: Toggle debug panel
        Ctrl+Shift+C: Pick element
    """
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.debug_panel = None
        
        # Setup keyboard shortcuts
        self._setup_shortcuts()
        
        print("🔧 Debug tools initialized. Press F12 to toggle debug panel.")
        
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # F12 - Toggle debug panel
        self.toggle_shortcut = QShortcut(QKeySequence("F12"), self.main_window)
        self.toggle_shortcut.activated.connect(self.toggle_panel)
        self.toggle_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        
        # Ctrl+Shift+C - Pick element
        self.pick_shortcut = QShortcut(QKeySequence("Ctrl+Shift+C"), self.main_window)
        self.pick_shortcut.activated.connect(self.pick_element)
        self.pick_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        
    def toggle_panel(self):
        """Show/hide the debug panel"""
        if self.debug_panel is None:
            self.debug_panel = DebugPanel(self.main_window)
            self.debug_panel.set_root_widget(self.main_window)
            self.main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.debug_panel)
        else:
            if self.debug_panel.isVisible():
                self.debug_panel.hide()
            else:
                self.debug_panel.show()
                
    def pick_element(self):
        """Start element picking mode"""
        if self.debug_panel is None:
            self.toggle_panel()
        self.debug_panel._start_picking()
        
    def inspect(self, widget):
        """Programmatically inspect a widget"""
        if self.debug_panel is None:
            self.toggle_panel()
        self.debug_panel._on_widget_picked(widget)


# =============================================================================
# Standalone test - run with: poetry run python debug_tools.py
# =============================================================================

if __name__ == "__main__":
    import sys
    
    app = QApplication(sys.argv)
    
    # Create a test window
    window = QMainWindow()
    window.setWindowTitle("Debug Tools Test")
    window.setMinimumSize(800, 600)
    
    # Apply dark theme
    window.setStyleSheet("""
        QMainWindow {
            background-color: #0A0E1A;
        }
    """)
    
    # Central widget with some test controls
    central = QWidget()
    layout = QVBoxLayout(central)
    
    label = QLabel("Test Label")
    label.setStyleSheet("color: #06B6D4; font-size: 14px;")
    layout.addWidget(label)
    
    combo = QComboBox()
    combo.addItems(["Option 1", "Option 2", "Option 3"])
    combo.setStyleSheet("""
        QComboBox {
            background-color: #111827;
            color: #CBD5E1;
            border: 1px solid #06B6D4;
            padding: 8px;
        }
    """)
    layout.addWidget(combo)
    
    button = QPushButton("Test Button")
    button.setStyleSheet("""
        QPushButton {
            background-color: #06B6D4;
            color: #0A0E1A;
            border: none;
            padding: 10px 20px;
            font-weight: bold;
        }
    """)
    layout.addWidget(button)
    
    layout.addStretch()
    window.setCentralWidget(central)
    
    # Initialize debug manager
    debug_manager = DebugManager(window)
    
    # Add a button to show debug panel
    show_debug_btn = QPushButton("Show Debug Panel (or press F12)")
    show_debug_btn.clicked.connect(debug_manager.toggle_panel)
    layout.addWidget(show_debug_btn)
    
    window.show()
    
    print("\n" + "="*50)
    print("DEBUG TOOLS TEST")
    print("="*50)
    print("F12: Toggle debug panel")
    print("Ctrl+Shift+C: Pick element")
    print("="*50 + "\n")
    
    sys.exit(app.exec())