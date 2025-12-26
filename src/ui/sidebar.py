from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget
from PyQt6.QtCore import pyqtSignal
from src.ui.colors import COLORS
from src.ui.control_panel import ControlPanel
from src.ui.widgets.network_graph import NetworkGraphWidget
from src.ui.widgets.media_panes import ImagePreviewPane, VideoPreviewPane

class NetworkPane(QWidget):
    nodeSelected = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Network view
        from src.ui.widgets.network_graph import NetworkGraphWidget
        from PyQt6.QtWidgets import QSizePolicy, QLabel
        from PyQt6.QtCore import Qt

        # Title
        title = QLabel("Propagation Network")
        title.setStyleSheet("color: #D4D4D4; font-size: 14px; font-weight: bold; font-family: 'Orbitron', sans-serif;")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)

        self.network_view = NetworkGraphWidget()
        self.network_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.network_view, 1)

        # Connect signals
        self.network_view.nodeSelected.connect(self.nodeSelected)

        # Initialize graph logic - moved from original NetworkPane to avoid circular imports if needed
        # But for now, we'll implement wrappers

    def add_node(self, node_id, label, node_type='branch'):
        # Simplified: Pass to widget if it had full logic, or implement here
        # Since logic was mixed in original, I'll reimplement minimal logic here or in widget
        # The widget handles drawing, we handle graph structure
        # Wait, the original NetworkGraphWidget was just drawing. NetworkPane had the graph logic.
        # I need to port the graph logic here.
        import networkx as nx
        import math
        import random

        if not hasattr(self, 'graph'):
            self.graph = nx.DiGraph()
            self.node_positions = {}
            self.node_colors = {}
            self.node_labels = {}
            self.node_sizes = {}

        try:
            self.graph.add_node(node_id)

            if node_type == 'main':
                color = '#569CD6'
                size = 800
            elif node_type == 'rabbithole':
                color = '#B5CEA8'
                size = 600
            elif node_type == 'fork':
                color = '#DCDCAA'
                size = 600
            else:
                color = '#CE9178'
                size = 400

            self.node_colors[node_id] = color
            self.node_labels[node_id] = label
            self.node_sizes[node_id] = size

            # Position calculation
            num_nodes = len(self.graph.nodes) - 1
            if node_type == 'main':
                self.node_positions[node_id] = (0, 0)
            else:
                golden_ratio = 1.618033988749895
                angle = 2 * math.pi * golden_ratio * num_nodes
                base_distance = 200
                count_factor = min(1.0, num_nodes / 20)

                if node_type == 'rabbithole':
                    distance = base_distance * (1.0 + count_factor * 0.5)
                elif node_type == 'fork':
                    distance = base_distance * (1.2 + count_factor * 0.5)
                else:
                    distance = base_distance * (1.4 + count_factor * 0.5)

                x = distance * math.cos(angle)
                y = distance * math.sin(angle)
                x += random.uniform(-30, 30)
                y += random.uniform(-30, 30)

                self.node_positions[node_id] = (x, y)

            self.update_graph()

        except Exception as e:
            print(f"Error adding node: {e}")

    def add_edge(self, source_id, target_id):
        if not hasattr(self, 'graph'):
            return
        try:
            self.graph.add_edge(source_id, target_id)
            self.update_graph()
        except Exception as e:
            print(f"Error adding edge: {e}")

    def update_graph(self):
        if hasattr(self, 'network_view'):
            self.network_view.nodes = list(self.graph.nodes())
            self.network_view.edges = list(self.graph.edges())
            self.network_view.node_positions = self.node_positions
            self.network_view.node_colors = self.node_colors
            self.network_view.node_labels = self.node_labels
            self.network_view.node_sizes = self.node_sizes
            self.network_view.update()

class RightSidebar(QWidget):
    """Right sidebar with tabbed interface for Setup and Network Graph"""
    nodeSelected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(300)
        self.setup_ui()

    def setup_ui(self):
        """Set up the tabbed sidebar interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(0)

        # Create tab bar at the top (custom styled)
        tab_container = QWidget()
        tab_container.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_medium']};
                border-bottom: 1px solid {COLORS['border_glow']};
            }}
        """)
        tab_layout = QHBoxLayout(tab_container)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(0)

        # Tab buttons
        self.setup_button = QPushButton("⚙ SETUP")
        self.graph_button = QPushButton("🌐 GRAPH")
        self.image_button = QPushButton("🖼 IMAGE")
        self.video_button = QPushButton("🎬 VIDEO")

        # Cyberpunk tab button styling
        tab_style = f"""
            QPushButton {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_dim']};
                border: none;
                border-bottom: 2px solid transparent;
                padding: 12px 12px;
                font-weight: bold;
                font-size: 10px;
                letter-spacing: 1px;
                text-transform: uppercase;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_normal']};
            }}
            QPushButton:checked {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['accent_cyan']};
                border-bottom: 2px solid {COLORS['accent_cyan']};
            }}
        """

        self.setup_button.setStyleSheet(tab_style)
        self.graph_button.setStyleSheet(tab_style)
        self.image_button.setStyleSheet(tab_style)
        self.video_button.setStyleSheet(tab_style)

        # Make buttons checkable for tab behavior
        self.setup_button.setCheckable(True)
        self.graph_button.setCheckable(True)
        self.image_button.setCheckable(True)
        self.video_button.setCheckable(True)
        self.setup_button.setChecked(True)  # Start with setup tab active

        # Connect tab buttons
        self.setup_button.clicked.connect(lambda: self.switch_tab(0))
        self.graph_button.clicked.connect(lambda: self.switch_tab(1))
        self.image_button.clicked.connect(lambda: self.switch_tab(2))
        self.video_button.clicked.connect(lambda: self.switch_tab(3))

        tab_layout.addWidget(self.setup_button)
        tab_layout.addWidget(self.graph_button)
        tab_layout.addWidget(self.image_button)
        tab_layout.addWidget(self.video_button)

        layout.addWidget(tab_container)

        # Create stacked widget for tab content
        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"""
            QStackedWidget {{
                background-color: {COLORS['bg_dark']};
                border: none;
            }}
        """)

        # Create tab pages
        self.control_panel = ControlPanel()
        self.network_pane = NetworkPane()
        self.image_preview_pane = ImagePreviewPane()
        self.video_preview_pane = VideoPreviewPane()

        # Add pages to stack
        self.stack.addWidget(self.control_panel)
        self.stack.addWidget(self.network_pane)
        self.stack.addWidget(self.image_preview_pane)
        self.stack.addWidget(self.video_preview_pane)

        layout.addWidget(self.stack, 1)  # Stretch to fill

        # Connect network pane signal to forward it
        self.network_pane.nodeSelected.connect(self.nodeSelected)

    def switch_tab(self, index):
        """Switch between tabs"""
        self.stack.setCurrentIndex(index)

        # Update button states
        self.setup_button.setChecked(index == 0)
        self.graph_button.setChecked(index == 1)
        self.image_button.setChecked(index == 2)
        self.video_button.setChecked(index == 3)

    def update_image_preview(self, image_path):
        """Update the image preview pane with a new image"""
        if hasattr(self, 'image_preview_pane'):
            self.image_preview_pane.set_image(image_path)

    def update_video_preview(self, video_path):
        """Update the video preview pane with a new video"""
        if hasattr(self, 'video_preview_pane'):
            self.video_preview_pane.set_video(video_path)

    def add_node(self, node_id, label, node_type):
        """Forward to network pane"""
        self.network_pane.add_node(node_id, label, node_type)

    def add_edge(self, source_id, target_id):
        """Forward to network pane"""
        self.network_pane.add_edge(source_id, target_id)

    def update_graph(self):
        """Forward to network pane"""
        self.network_pane.update_graph()
