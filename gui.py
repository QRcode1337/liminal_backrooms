# new_gui.py

import os
import json
import requests
import threading
import math
import random
from datetime import datetime
from io import BytesIO
from PIL import Image
import time
from pathlib import Path
import uuid
import shutil
import networkx as nx
import re
import sys
import webbrowser
from PyQt6.QtCore import Qt, QRect, QTimer, QRectF, QPointF, QSize, pyqtSignal, QEvent, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QBrush, QFontDatabase, QTextCursor, QAction, QKeySequence, QTextCharFormat, QLinearGradient, QRadialGradient, QPainterPath, QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QApplication, QMainWindow, QSplitter, QVBoxLayout, QHBoxLayout, QTextEdit, QFrame, QLineEdit, QPushButton, QLabel, QComboBox, QMenu, QFileDialog, QMessageBox, QScrollArea, QToolTip, QSizePolicy, QCheckBox

from config import (
    AI_MODELS,
    SYSTEM_PROMPT_PAIRS,
    SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT
)

# Add import for the HTML viewing functionality 
from shared_utils import open_html_in_browser, generate_image_from_text

# Define global color palette for consistent styling
COLORS = {
    'bg_dark': '#1E1E1E',           # Main background
    'bg_medium': '#252526',         # Widget backgrounds
    'bg_light': '#2D2D30',          # Lighter elements
    'accent_blue': '#569CD6',       # Primary accent
    'accent_blue_hover': '#4E8CC2', # Hover state
    'accent_blue_active': '#007ACC',# Active state
    'accent_green': '#B5CEA8',      # Secondary accent (rabbithole)
    'accent_yellow': '#DCDCAA',     # Tertiary accent (fork)
    'accent_orange': '#CE9178',     # Quaternary accent
    'text_normal': '#D4D4D4',       # Normal text
    'text_dim': '#9CDCFE',          # Dimmed text
    'text_bright': '#FFFFFF',       # Bright text
    'text_error': '#F44747',        # Error text
    'border': '#3E3E42',            # Borders
    'border_highlight': '#555555',  # Highlighted borders
    'chain_of_thought': '#608B4E',  # Chain of thought text
    'user_header': '#4EC9B0',       # User message headers
    'ai_header': '#569CD6',         # AI message headers
    'system_message': '#CE9178',    # System messages
}

# Load custom fonts
def load_fonts():
    """Load custom fonts for the application"""
    font_dir = Path("fonts")
    font_dir.mkdir(exist_ok=True)
    
    # List of fonts to load - these would need to be included with the application
    fonts = [
        # ("JetBrainsMono-Regular.ttf", "JetBrains Mono"),
        # ("Orbitron-Regular.ttf", "Orbitron"),
    ]
    
    loaded_fonts = []
    for font_file, font_name in fonts:
        font_path = font_dir / font_file
        if font_path.exists():
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            if font_id >= 0:
                loaded_fonts.append(font_name)
    
    return loaded_fonts

class NetworkGraphWidget(QWidget):
    nodeSelected = pyqtSignal(str)
    nodeHovered = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        # Graph data
        self.nodes = []
        self.edges = []
        self.node_positions = {}
        self.node_colors = {}
        self.node_labels = {}
        self.node_sizes = {}
        
        # Edge animation data
        self.growing_edges = {}  # Dictionary to track growing edges: {(source, target): growth_progress}
        self.edge_growth_speed = 0.05  # Increased speed of edge growth animation (was 0.02)
        
        # Visual settings
        self.margin = 50
        self.selected_node = None
        self.hovered_node = None
        self.animation_progress = 0
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(50)  # 20 FPS animation
        
        # Mycelial node settings
        self.hyphae_count = 5  # Number of hyphae per node
        self.hyphae_length_factor = 0.4  # Length of hyphae relative to node radius
        self.hyphae_variation = 0.3  # Random variation in hyphae
        
        # Node colors - use global color palette with mycelial theme
        self.node_colors_by_type = {
            'main': '#8E9DCC',  # Soft blue-purple
            'rabbithole': '#7FB069',  # Soft green
            'fork': '#F2C14E',  # Soft yellow
            'branch': '#F78154'   # Soft orange
        }
        
        # Collision dynamics
        self.node_velocities = {}  # Store velocities for each node
        self.repulsion_strength = 0.5  # Strength of repulsion between nodes
        self.attraction_strength = 0.1  # Strength of attraction along edges
        self.damping = 0.8  # Damping factor to prevent oscillation
        self.apply_physics = True  # Toggle for physics simulation
        
        # Set up the widget
        self.setMinimumSize(300, 300)
        self.setMouseTracking(True)
        
    def add_edge(self, source, target):
        """Add an edge with growth animation"""
        if (source, target) not in self.edges:
            self.edges.append((source, target))
            # Initialize edge growth at 0
            self.growing_edges[(source, target)] = 0.0
            # Force update to start animation immediately
            self.update()
        
    def update_animation(self):
        """Update animation state"""
        self.animation_progress = (self.animation_progress + 0.05) % 1.0
        
        # Update growing edges
        edges_to_remove = []
        has_growing_edges = False
        
        for edge, progress in self.growing_edges.items():
            if progress < 1.0:
                self.growing_edges[edge] = min(progress + self.edge_growth_speed, 1.0)
                has_growing_edges = True
            else:
                # Mark fully grown edges for removal from animation tracking
                edges_to_remove.append(edge)
        
        # Remove fully grown edges from tracking
        for edge in edges_to_remove:
            if edge in self.growing_edges:
                self.growing_edges.pop(edge)
        
        # Apply collision dynamics if enabled
        if self.apply_physics and len(self.nodes) > 1:
            self.apply_collision_dynamics()
        
        # Update the widget
        self.update()
    
    def apply_collision_dynamics(self):
        """Apply collision dynamics to prevent node overlap"""
        # Skip physics if too few nodes or physics has settled
        if len(self.nodes) < 3:
            return

        # Check if physics has settled (all velocities near zero)
        if hasattr(self, '_physics_settled') and self._physics_settled:
            max_velocity = max(
                (abs(vx) + abs(vy) for vx, vy in self.node_velocities.values()),
                default=0
            )
            if max_velocity < 0.01:  # Physics threshold
                return

        # Initialize velocities if needed
        for node_id in self.nodes:
            if node_id not in self.node_velocities:
                self.node_velocities[node_id] = (0, 0)

        # Calculate repulsive forces between nodes
        new_velocities = {}
        node_list = list(self.nodes)  # Convert to list once

        for i, node_id in enumerate(node_list):
            if node_id not in self.node_positions:
                continue

            # Start with any velocity already accumulated for this node
            vx, vy = new_velocities.get(
                node_id,
                self.node_velocities.get(node_id, (0, 0))
            )
            x1, y1 = self.node_positions[node_id]

            # Apply repulsion between nodes - only check nodes after current (avoid duplicate pairs)
            for other_id in node_list[i+1:]:
                if other_id not in self.node_positions:
                    continue

                x2, y2 = self.node_positions[other_id]

                # Calculate distance
                dx = x1 - x2
                dy = y1 - y2
                distance = max(0.1, math.sqrt(dx*dx + dy*dy))  # Avoid division by zero

                # Get node sizes
                size1 = math.sqrt(self.node_sizes.get(node_id, 400))
                size2 = math.sqrt(self.node_sizes.get(other_id, 400))
                min_distance = (size1 + size2) / 2

                # Apply repulsive force if nodes are too close
                if distance < min_distance * 2:
                    # Normalize direction vector
                    nx = dx / distance
                    ny = dy / distance

                    # Calculate repulsion strength (stronger when closer)
                    strength = self.repulsion_strength * (1.0 - distance / (min_distance * 2))

                    # Apply force to both nodes (Newton's third law)
                    vx += nx * strength
                    vy += ny * strength

                    # Apply opposite force to other node
                    other_vx, other_vy = new_velocities.get(other_id, self.node_velocities.get(other_id, (0, 0)))
                    new_velocities[other_id] = (other_vx - nx * strength, other_vy - ny * strength)
            
            # Apply attraction along edges
            for edge in self.edges:
                source, target = edge
                
                # Skip edges that are still growing
                if (source, target) in self.growing_edges and self.growing_edges[(source, target)] < 1.0:
                    continue
                
                if source == node_id and target in self.node_positions:
                    # This node is the source, attract towards target
                    x2, y2 = self.node_positions[target]
                    dx = x2 - x1
                    dy = y2 - y1
                    distance = max(0.1, math.sqrt(dx*dx + dy*dy))
                    
                    # Normalize and apply attraction
                    vx += (dx / distance) * self.attraction_strength
                    vy += (dy / distance) * self.attraction_strength
                    
                elif target == node_id and source in self.node_positions:
                    # This node is the target, attract towards source
                    x2, y2 = self.node_positions[source]
                    dx = x2 - x1
                    dy = y2 - y1
                    distance = max(0.1, math.sqrt(dx*dx + dy*dy))
                    
                    # Normalize and apply attraction
                    vx += (dx / distance) * self.attraction_strength
                    vy += (dy / distance) * self.attraction_strength
            
            # Apply damping to prevent oscillation
            vx *= self.damping
            vy *= self.damping

            # Store new velocity
            new_velocities[node_id] = (vx, vy)

        # Update positions based on velocities
        max_velocity = 0.0
        for node_id, (vx, vy) in new_velocities.items():
            if node_id in self.node_positions:
                # Skip the main node to keep it centered
                if node_id == 'main':
                    continue

                x, y = self.node_positions[node_id]
                self.node_positions[node_id] = (x + vx, y + vy)

                # Track maximum velocity for settling detection
                velocity_magnitude = abs(vx) + abs(vy)
                max_velocity = max(max_velocity, velocity_magnitude)

        # Update velocities for next frame
        self.node_velocities = new_velocities

        # Check if physics has settled
        self._physics_settled = max_velocity < 0.01
        
    def paintEvent(self, event):
        """Paint the network graph"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        
        # Set background with subtle gradient
        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0, QColor('#1A1A1E'))  # Dark blue-gray
        gradient.setColorAt(1, QColor('#0F0F12'))  # Darker at bottom
        painter.fillRect(0, 0, width, height, gradient)
        
        # Draw subtle grid lines
        painter.setPen(QPen(QColor(COLORS['border']).darker(150), 0.5, Qt.PenStyle.DotLine))
        grid_size = 40
        for x in range(0, width, grid_size):
            painter.drawLine(x, 0, x, height)
        for y in range(0, height, grid_size):
            painter.drawLine(0, y, width, y)
        
        # Calculate center point and scale factor
        center_x = width / 2
        center_y = height / 2
        scale = min(width, height) / 500
        
        # Draw edges first so they appear behind nodes
        for edge in self.edges:
            source, target = edge
            if source in self.node_positions and target in self.node_positions:
                src_x, src_y = self.node_positions[source]
                dst_x, dst_y = self.node_positions[target]
                
                # Transform coordinates to screen space
                screen_src_x = center_x + src_x * scale
                screen_src_y = center_y + src_y * scale
                screen_dst_x = center_x + dst_x * scale
                screen_dst_y = center_y + dst_y * scale
                
                # Get growth progress for this edge (default to 1.0 if not growing)
                growth_progress = self.growing_edges.get((source, target), 1.0)
                
                # Calculate the actual destination based on growth progress
                if growth_progress < 1.0:
                    # Interpolate between source and destination
                    actual_dst_x = screen_src_x + (screen_dst_x - screen_src_x) * growth_progress
                    actual_dst_y = screen_src_y + (screen_dst_y - screen_src_y) * growth_progress
                else:
                    actual_dst_x = screen_dst_x
                    actual_dst_y = screen_dst_y
                
                # Draw mycelial connection (multiple thin lines with variations)
                source_color = QColor(self.node_colors.get(source, self.node_colors_by_type['main']))
                target_color = QColor(self.node_colors.get(target, self.node_colors_by_type['main']))
                
                # Number of filaments per connection
                num_filaments = 3
                
                for i in range(num_filaments):
                    # Create a path with multiple segments for organic look
                    path = QPainterPath()
                    path.moveTo(screen_src_x, screen_src_y)
                    
                    # Calculate distance between points
                    distance = math.sqrt((actual_dst_x - screen_src_x)**2 + (actual_dst_y - screen_src_y)**2)
                    
                    # Number of segments increases with distance
                    num_segments = max(3, int(distance / 40))
                    
                    # Create intermediate points with slight random variations
                    prev_x, prev_y = screen_src_x, screen_src_y
                    
                    for j in range(1, num_segments):
                        # Calculate position along the line
                        ratio = j / num_segments
                        
                        # Base position
                        base_x = screen_src_x + (actual_dst_x - screen_src_x) * ratio
                        base_y = screen_src_y + (actual_dst_y - screen_src_y) * ratio
                        
                        # Add random variation perpendicular to the line
                        angle = math.atan2(actual_dst_y - screen_src_y, actual_dst_x - screen_src_x) + math.pi/2
                        variation = (random.random() - 0.5) * 10 * scale
                        
                        # Variation decreases near endpoints
                        endpoint_factor = min(ratio, 1 - ratio) * 4  # Maximum at middle
                        variation *= endpoint_factor
                        
                        # Apply variation
                        point_x = base_x + variation * math.cos(angle)
                        point_y = base_y + variation * math.sin(angle)
                        
                        # Add point to path
                        path.lineTo(point_x, point_y)
                        prev_x, prev_y = point_x, point_y
                    
                    # Complete the path to destination
                    path.lineTo(actual_dst_x, actual_dst_y)
                    
                    # Create gradient along the path
                    gradient = QLinearGradient(screen_src_x, screen_src_y, actual_dst_x, actual_dst_y)
                    
                    # Make colors more transparent for mycelial effect
                    source_color_trans = QColor(source_color)
                    target_color_trans = QColor(target_color)
                    
                    # Vary transparency by filament
                    alpha = 70 + i * 20
                    source_color_trans.setAlpha(alpha)
                    target_color_trans.setAlpha(alpha)
                    
                    gradient.setColorAt(0, source_color_trans)
                    gradient.setColorAt(1, target_color_trans)
                    
                    # Animate flow along edge
                    flow_pos = (self.animation_progress + i * 0.3) % 1.0
                    flow_color = QColor(255, 255, 255, 100)
                    gradient.setColorAt(flow_pos, flow_color)
                    
                    # Draw the edge with varying thickness
                    thickness = 1.0 + (i * 0.5)
                    pen = QPen(QBrush(gradient), thickness)
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    painter.setPen(pen)
                    painter.drawPath(path)
                
                # Draw small nodes along the path for mycelial effect
                if growth_progress == 1.0:  # Only for fully grown edges
                    num_nodes = int(distance / 50)
                    for j in range(1, num_nodes):
                        ratio = j / num_nodes
                        node_x = screen_src_x + (screen_dst_x - screen_src_x) * ratio
                        node_y = screen_src_y + (screen_dst_y - screen_src_y) * ratio
                        
                        # Add small random offset
                        offset_angle = random.random() * math.pi * 2
                        offset_dist = random.random() * 5
                        node_x += math.cos(offset_angle) * offset_dist
                        node_y += math.sin(offset_angle) * offset_dist
                        
                        # Draw small node
                        node_color = QColor(source_color)
                        node_color.setAlpha(100)
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.setBrush(QBrush(node_color))
                        node_size = 1 + random.random() * 2
                        painter.drawEllipse(QPointF(node_x, node_y), node_size, node_size)
        
        # Draw nodes
        for node_id in self.nodes:
            if node_id in self.node_positions:
                x, y = self.node_positions[node_id]
                
                # Transform coordinates to screen space
                screen_x = center_x + x * scale
                screen_y = center_y + y * scale
                
                # Get node properties
                node_color = self.node_colors.get(node_id, self.node_colors_by_type['branch'])
                node_label = self.node_labels.get(node_id, 'Node')
                node_size = self.node_sizes.get(node_id, 400)
                
                # Scale the node size
                radius = math.sqrt(node_size) * scale / 2
                
                # Adjust radius for hover/selection
                if node_id == self.selected_node:
                    radius *= 1.1  # Larger when selected
                elif node_id == self.hovered_node:
                    radius *= 1.05  # Slightly larger when hovered
                
                # Draw node glow for selected/hovered nodes
                if node_id == self.selected_node or node_id == self.hovered_node:
                    glow_radius = radius * 1.5
                    glow_color = QColor(node_color)
                    
                    for i in range(5):
                        r = glow_radius - (i * radius * 0.1)
                        alpha = 40 - (i * 8)
                        glow_color.setAlpha(alpha)
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.setBrush(glow_color)
                        painter.drawEllipse(QPointF(screen_x, screen_y), r, r)
                
                # Draw mycelial node (irregular shape with hyphae)
                painter.setPen(Qt.PenStyle.NoPen)
                
                # Create gradient fill for node
                gradient = QRadialGradient(screen_x, screen_y, radius)
                base_color = QColor(node_color)
                lighter_color = QColor(node_color).lighter(130)
                darker_color = QColor(node_color).darker(130)
                
                gradient.setColorAt(0, lighter_color)
                gradient.setColorAt(0.7, base_color)
                gradient.setColorAt(1, darker_color)
                
                # Fill main node body
                painter.setBrush(QBrush(gradient))
                
                # Draw irregular node shape
                path = QPainterPath()
                
                # Create irregular circle with random variations
                num_points = 20
                start_angle = random.random() * math.pi * 2
                
                for i in range(num_points + 1):
                    angle = start_angle + (i * 2 * math.pi / num_points)
                    # Vary radius slightly for organic look
                    variation = 1.0 + (random.random() - 0.5) * 0.2
                    point_radius = radius * variation
                    
                    x_point = screen_x + math.cos(angle) * point_radius
                    y_point = screen_y + math.sin(angle) * point_radius
                    
                    if i == 0:
                        path.moveTo(x_point, y_point)
                    else:
                        # Use quadratic curves for smoother shape
                        control_angle = start_angle + ((i - 0.5) * 2 * math.pi / num_points)
                        control_radius = radius * (1.0 + (random.random() - 0.5) * 0.1)
                        control_x = screen_x + math.cos(control_angle) * control_radius
                        control_y = screen_y + math.sin(control_angle) * control_radius
                        
                        path.quadTo(control_x, control_y, x_point, y_point)
                
                # Draw the main node body
                painter.drawPath(path)
                
                # Draw hyphae (mycelial extensions)
                hyphae_count = self.hyphae_count
                if node_id == 'main':
                    hyphae_count += 3  # More hyphae for main node
                
                for i in range(hyphae_count):
                    # Random angle for hyphae
                    angle = random.random() * math.pi * 2
                    
                    # Base length varies by node type
                    base_length = radius * self.hyphae_length_factor
                    if node_id == 'main':
                        base_length *= 1.5
                    
                    # Random variation in length
                    length = base_length * (1.0 + (random.random() - 0.5) * self.hyphae_variation)
                    
                    # Calculate end point
                    end_x = screen_x + math.cos(angle) * (radius + length)
                    end_y = screen_y + math.sin(angle) * (radius + length)
                    
                    # Start point is on the node perimeter
                    start_x = screen_x + math.cos(angle) * radius * 0.9
                    start_y = screen_y + math.sin(angle) * radius * 0.9
                    
                    # Create hyphae path with slight curve
                    hypha_path = QPainterPath()
                    hypha_path.moveTo(start_x, start_y)
                    
                    # Control point for curve
                    ctrl_angle = angle + (random.random() - 0.5) * 0.5  # Slight angle variation
                    ctrl_dist = radius + length * 0.5
                    ctrl_x = screen_x + math.cos(ctrl_angle) * ctrl_dist
                    ctrl_y = screen_y + math.sin(ctrl_angle) * ctrl_dist
                    
                    hypha_path.quadTo(ctrl_x, ctrl_y, end_x, end_y)
                    
                    # Draw hypha with gradient
                    hypha_gradient = QLinearGradient(start_x, start_y, end_x, end_y)
                    
                    # Hypha color starts as node color and fades out
                    hypha_start_color = QColor(node_color)
                    hypha_end_color = QColor(node_color)
                    hypha_start_color.setAlpha(150)
                    hypha_end_color.setAlpha(30)
                    
                    hypha_gradient.setColorAt(0, hypha_start_color)
                    hypha_gradient.setColorAt(1, hypha_end_color)
                    
                    # Draw hypha with varying thickness
                    thickness = 1.0 + random.random() * 1.5
                    hypha_pen = QPen(QBrush(hypha_gradient), thickness)
                    hypha_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    painter.setPen(hypha_pen)
                    painter.drawPath(hypha_path)
                    
                    # Add small nodes at the end of some hyphae
                    if random.random() > 0.5:
                        small_node_color = QColor(node_color)
                        small_node_color.setAlpha(100)
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.setBrush(QBrush(small_node_color))
                        small_node_size = 1 + random.random() * 2
                        painter.drawEllipse(QPointF(end_x, end_y), small_node_size, small_node_size)
    
    def draw_arrow_head(self, painter, x1, y1, x2, y2):
        """Draw an arrow head at the end of a line"""
        # For mycelial style, we don't need arrow heads
        pass
    
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Get click position
            pos = event.position()
            
            # Check if a node was clicked
            clicked_node = self.get_node_at_position(pos)
            if clicked_node:
                self.selected_node = clicked_node
                self.update()
                self.nodeSelected.emit(clicked_node)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for hover effects"""
        pos = event.position()
        hovered_node = self.get_node_at_position(pos)
        
        if hovered_node != self.hovered_node:
            self.hovered_node = hovered_node
            self.update()
            if hovered_node:
                self.nodeHovered.emit(hovered_node)
                
                # Show tooltip with node info
                if hovered_node in self.node_labels:
                    # Get node type from the ID
                    node_type = "main"
                    if "rabbithole_" in hovered_node:
                        node_type = "rabbithole"
                    elif "fork_" in hovered_node:
                        node_type = "fork"
                    
                    # Set emoji based on node type
                    emoji = "🌱"  # Default/main
                    if node_type == "rabbithole":
                        emoji = "🕳️"  # Rabbithole emoji
                    elif node_type == "fork":
                        emoji = "🔱"  # Fork emoji
                    
                    # Show tooltip with emoji and label
                    QToolTip.showText(
                        event.globalPosition().toPoint(),
                        f"{emoji} {self.node_labels[hovered_node]}",
                        self
                    )
    
    def get_node_at_position(self, pos):
        """Get the node at the given position"""
        # Calculate center point and scale factor
        width = self.width()
        height = self.height()
        center_x = width / 2
        center_y = height / 2
        scale = min(width, height) / 500
                    
        # Check each node
        for node_id in self.nodes:
            if node_id in self.node_positions:
                    x, y = self.node_positions[node_id]
                    screen_x = center_x + x * scale
                    screen_y = center_y + y * scale
                    
                    # Get node size
                    node_size = self.node_sizes.get(node_id, 400)
                    radius = math.sqrt(node_size) * scale / 2
                    
            # Check if click is inside the node
                    distance = math.sqrt((pos.x() - screen_x)**2 + (pos.y() - screen_y)**2)
                    if distance <= radius:
                        return node_id
        
        return None
    
    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        self.update()

class NetworkPane(QWidget):
    nodeSelected = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("Propagation Network")
        title.setStyleSheet("color: #D4D4D4; font-size: 14px; font-weight: bold; font-family: 'Orbitron', sans-serif;")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Network view - set to expand to fill available space
        self.network_view = NetworkGraphWidget()
        self.network_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.network_view, 1)  # Add stretch factor of 1 to make it expand
        
        # Connect signals
        self.network_view.nodeSelected.connect(self.nodeSelected)
    
        # Initialize graph
        self.graph = nx.DiGraph()
        self.node_positions = {}
        self.node_colors = {}
        self.node_labels = {}
        self.node_sizes = {}
        
        # Add main node
        self.add_node('main', 'Seed', 'main')
    
    def add_node(self, node_id, label, node_type='branch'):
        """Add a node to the graph"""
        try:
            # Add the node to the graph
            self.graph.add_node(node_id)
            
            # Set node properties based on type
            if node_type == 'main':
                color = '#569CD6'  # Blue
                size = 800
            elif node_type == 'rabbithole':
                color = '#B5CEA8'  # Green
                size = 600
            elif node_type == 'fork':
                color = '#DCDCAA'  # Yellow
                size = 600
            else:
                color = '#CE9178'  # Orange
                size = 400
            
            # Store node properties
            self.node_colors[node_id] = color
            self.node_labels[node_id] = label
            self.node_sizes[node_id] = size
            
            # Calculate position based on existing nodes
            self.calculate_node_position(node_id, node_type)
            
            # Redraw the graph
            self.update_graph()
            
        except Exception as e:
            print(f"Error adding node: {e}")
    
    def add_edge(self, source_id, target_id):
        """Add an edge between two nodes"""
        try:
            # Add the edge to the graph
            self.graph.add_edge(source_id, target_id)
            
            # Redraw the graph
            self.update_graph()
            
        except Exception as e:
            print(f"Error adding edge: {e}")
    
    def calculate_node_position(self, node_id, node_type):
        """Calculate position for a new node"""
        # Get number of existing nodes
        num_nodes = len(self.graph.nodes) - 1  # Exclude the main node
        
        if node_type == 'main':
            # Main node is at center
            self.node_positions[node_id] = (0, 0)
        else:
            # Calculate angle based on node count with better distribution
            # Use golden ratio to distribute nodes more evenly
            golden_ratio = 1.618033988749895
            angle = 2 * math.pi * golden_ratio * num_nodes
            
            # Calculate distance from center based on node type and node count
            # Increase distance as more nodes are added
            base_distance = 200
            count_factor = min(1.0, num_nodes / 20)  # Scale up to 20 nodes
            
            if node_type == 'rabbithole':
                distance = base_distance * (1.0 + count_factor * 0.5)
            elif node_type == 'fork':
                distance = base_distance * (1.2 + count_factor * 0.5)
            else:
                distance = base_distance * (1.4 + count_factor * 0.5)
            
            # Calculate position using polar coordinates
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            
            # Add some random offset for natural appearance
            x += random.uniform(-30, 30)
            y += random.uniform(-30, 30)
            
            # Check for potential overlaps with existing nodes and adjust if needed
            overlap = True
            max_attempts = 5
            attempt = 0
            
            while overlap and attempt < max_attempts:
                overlap = False
                for existing_id, (ex, ey) in self.node_positions.items():
                    # Skip comparing with self
                    if existing_id == node_id:
                        continue
                        
                    # Calculate distance between nodes
                    dx = x - ex
                    dy = y - ey
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # Get node sizes
                    new_size = math.sqrt(self.node_sizes.get(node_id, 400))
                    existing_size = math.sqrt(self.node_sizes.get(existing_id, 400))
                    min_distance = (new_size + existing_size) / 2
                    
                    # If too close, adjust position
                    if distance < min_distance * 1.5:
                        overlap = True
                        # Move away from the overlapping node
                        angle = math.atan2(dy, dx)
                        adjustment = min_distance * 1.5 - distance
                        x += math.cos(angle) * adjustment * 1.2
                        y += math.sin(angle) * adjustment * 1.2
                        break
                
                attempt += 1
            
            # Store the position
            self.node_positions[node_id] = (x, y)
    
    def update_graph(self):
        """Update the network graph visualization"""
        if hasattr(self, 'network_view'):
            # Update the network view with current graph data
            self.network_view.nodes = list(self.graph.nodes())
            self.network_view.edges = list(self.graph.edges())
            self.network_view.node_positions = self.node_positions
            self.network_view.node_colors = self.node_colors
            self.network_view.node_labels = self.node_labels
            self.network_view.node_sizes = self.node_sizes
            
            # Redraw
            self.network_view.update()

class ControlPanel(QWidget):
    """Control panel with mode, model selections, etc."""
    def __init__(self):
        super().__init__()
        
        # Set up the UI
        self.setup_ui()
        
        # Initialize with models and prompt pairs
        self.initialize_selectors()
    
    def setup_ui(self):
        """Set up the user interface for the control panel"""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 10, 0, 0)
        main_layout.setSpacing(15)
        
        # Add a title
        title = QLabel("Control Panel")
        title.setStyleSheet(f"""
            color: {COLORS['text_bright']};
            font-size: 14px;
            font-weight: bold;
            padding-bottom: 5px;
            border-bottom: 1px solid {COLORS['border']};
        """)
        main_layout.addWidget(title)
        
        # Create a grid layout for the controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)
        
        # Left column - Mode and iterations
        left_column = QVBoxLayout()
        left_column.setSpacing(10)
        
        # Mode selection with icon
        mode_container = QWidget()
        mode_layout = QVBoxLayout(mode_container)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(5)
        
        mode_label = QLabel("Conversation Mode")
        mode_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px;")
        mode_layout.addWidget(mode_label)
        
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["AI-AI", "Human-AI"])
        self.mode_selector.setStyleSheet(self.get_combobox_style())
        mode_layout.addWidget(self.mode_selector)
        
        left_column.addWidget(mode_container)
        
        # Iterations with slider
        iterations_container = QWidget()
        iterations_layout = QVBoxLayout(iterations_container)
        iterations_layout.setContentsMargins(0, 0, 0, 0)
        iterations_layout.setSpacing(5)
        
        iterations_label = QLabel("Iterations")
        iterations_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px;")
        iterations_layout.addWidget(iterations_label)
        
        self.iterations_selector = QComboBox()
        self.iterations_selector.addItems(["1", "2", "4", "6", "12", "100"])
        self.iterations_selector.setStyleSheet(self.get_combobox_style())
        iterations_layout.addWidget(self.iterations_selector)
        
        left_column.addWidget(iterations_container)
        
        # Middle column - AI models
        middle_column = QVBoxLayout()
        middle_column.setSpacing(10)
        
        # AI-1 Model selection
        ai1_container = QWidget()
        ai1_layout = QVBoxLayout(ai1_container)
        ai1_layout.setContentsMargins(0, 0, 0, 0)
        ai1_layout.setSpacing(5)
        
        ai1_label = QLabel("AI-1 Model")
        ai1_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px;")
        ai1_layout.addWidget(ai1_label)
        
        self.ai1_model_selector = QComboBox()
        self.ai1_model_selector.setStyleSheet(self.get_combobox_style())
        ai1_layout.addWidget(self.ai1_model_selector)
        
        middle_column.addWidget(ai1_container)
        
        # AI-2 Model selection
        ai2_container = QWidget()
        ai2_layout = QVBoxLayout(ai2_container)
        ai2_layout.setContentsMargins(0, 0, 0, 0)
        ai2_layout.setSpacing(5)
        
        ai2_label = QLabel("AI-2 Model")
        ai2_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px;")
        ai2_layout.addWidget(ai2_label)
        
        self.ai2_model_selector = QComboBox()
        self.ai2_model_selector.setStyleSheet(self.get_combobox_style())
        ai2_layout.addWidget(self.ai2_model_selector)
        
        middle_column.addWidget(ai2_container)
        
        # Right column - Prompt pair and export
        right_column = QVBoxLayout()
        right_column.setSpacing(10)
        
        # Prompt pair selection
        prompt_container = QWidget()
        prompt_layout = QVBoxLayout(prompt_container)
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_layout.setSpacing(5)
        
        prompt_label = QLabel("Conversation Scenario")
        prompt_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px;")
        prompt_layout.addWidget(prompt_label)
        
        self.prompt_pair_selector = QComboBox()
        self.prompt_pair_selector.setStyleSheet(self.get_combobox_style())
        prompt_layout.addWidget(self.prompt_pair_selector)
        
        right_column.addWidget(prompt_container)
        
        # Action buttons container
        action_container = QWidget()
        action_layout = QVBoxLayout(action_container)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(5)
        
        action_label = QLabel("Actions")
        action_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px;")
        action_layout.addWidget(action_label)
        
        # Auto-generate images checkbox
        self.auto_image_checkbox = QCheckBox("Auto-generate images")
        self.auto_image_checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {COLORS['text_normal']};
                spacing: 5px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {COLORS['border']};
                border-radius: 3px;
                background-color: {COLORS['bg_light']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {COLORS['accent_blue']};
                border: 1px solid {COLORS['accent_blue']};
            }}
        """)
        self.auto_image_checkbox.setToolTip("Automatically generate images from AI responses using OpenAI's GPT-image-1 model")
        action_layout.addWidget(self.auto_image_checkbox)
        
        # Removed: HTML contributions checkbox
        
        # Buttons layout (horizontal)
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(5)
        
        # Export button
        self.export_button = QPushButton("Export")
        self.export_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: {COLORS['border']};
                border: 1px solid {COLORS['border_highlight']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['border_highlight']};
            }}
        """)
        
        # View HTML button (removed)
        self.view_html_button = QPushButton("View HTML")
        self.view_html_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_green']};
                color: {COLORS['bg_dark']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_blue']};
                color: {COLORS['text_bright']};
                border: 1px solid {COLORS['border_highlight']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['accent_blue_active']};
            }}
        """)
        self.view_html_button.clicked.connect(lambda: open_html_in_browser("shared_document.html"))
        
        # View Full HTML button
        self.view_full_html_button = QPushButton("View Dark HTML")
        self.view_full_html_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #212121;
                color: #789922;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: #2d2d2d;
                color: #98c379;
                border: 1px solid #555;
            }}
            QPushButton:pressed {{
                background-color: #1a1a1a;
            }}
        """)
        self.view_full_html_button.setToolTip("View the full conversation in dark mode with greentext styling")
        self.view_full_html_button.clicked.connect(lambda: open_html_in_browser("conversation_full.html"))
        
        # Removed: View Living Document button
        
        # Add buttons to layout
        buttons_layout.addWidget(self.export_button)
        buttons_layout.addWidget(self.view_html_button)
        buttons_layout.addWidget(self.view_full_html_button)
        # Removed: living doc button from layout
        
        action_layout.addLayout(buttons_layout)
        
        right_column.addWidget(action_container)
        
        # Add columns to the controls layout
        controls_layout.addLayout(left_column, 1)
        controls_layout.addLayout(middle_column, 1)
        controls_layout.addLayout(right_column, 1)
        
        # Add controls layout to main layout
        main_layout.addLayout(controls_layout)
    
    def get_combobox_style(self):
        """Get the style for comboboxes"""
        return f"""
            QComboBox {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 5px 10px;
                min-width: 150px;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid {COLORS['border']};
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
            }}
            QComboBox::down-arrow {{
                width: 12px;
                height: 12px;
                image: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_normal']};
                selection-background-color: {COLORS['accent_blue']};
                selection-color: {COLORS['text_bright']};
                border: 1px solid {COLORS['border']};
                border-radius: 0px;
            }}
        """
    
    def initialize_selectors(self):
        """Initialize the selector dropdowns with values from config"""
        # Add AI models
        self.ai1_model_selector.clear()
        self.ai2_model_selector.clear()
        self.ai1_model_selector.addItems(list(AI_MODELS.keys()))
        self.ai2_model_selector.addItems(list(AI_MODELS.keys()))
        
        # Add prompt pairs
        self.prompt_pair_selector.clear()
        self.prompt_pair_selector.addItems(list(SYSTEM_PROMPT_PAIRS.keys()))

class ConversationContextMenu(QMenu):
    """Context menu for the conversation display"""
    rabbitholeSelected = pyqtSignal()
    forkSelected = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create actions
        self.rabbithole_action = QAction("🕳️ Rabbithole", self)
        self.fork_action = QAction("🔱 Fork", self)
        
        # Add actions to menu
        self.addAction(self.rabbithole_action)
        self.addAction(self.fork_action)
        
        # Connect actions to signals
        self.rabbithole_action.triggered.connect(self.on_rabbithole_selected)
        self.fork_action.triggered.connect(self.on_fork_selected)
        
        # Apply styling
        self.setStyleSheet("""
            QMenu {
                background-color: #2D2D30;
                color: #D4D4D4;
                border: 1px solid #3E3E42;
            }
            QMenu::item {
                padding: 5px 20px 5px 20px;
            }
            QMenu::item:selected {
                background-color: #3E3E42;
            }
        """)
    
    def on_rabbithole_selected(self):
        """Signal that rabbithole action was selected"""
        if self.parent() and hasattr(self.parent(), 'rabbithole_from_selection'):
            cursor = self.parent().conversation_display.textCursor()
            selected_text = cursor.selectedText()
            if selected_text and hasattr(self.parent(), 'rabbithole_callback'):
                self.parent().rabbithole_callback(selected_text)
    
    def on_fork_selected(self):
        """Signal that fork action was selected"""
        if self.parent() and hasattr(self.parent(), 'fork_from_selection'):
            cursor = self.parent().conversation_display.textCursor()
            selected_text = cursor.selectedText()
            if selected_text and hasattr(self.parent(), 'fork_callback'):
                self.parent().fork_callback(selected_text)

class ConversationPane(QWidget):
    """Left pane containing the conversation and input area"""
    def __init__(self):
        super().__init__()
        
        # Set up the UI
        self.setup_ui()
        
        # Connect signals and slots
        self.connect_signals()
        
        # Initialize state
        self.conversation = []
        self.input_callback = None
        self.rabbithole_callback = None
        self.fork_callback = None
        self.loading = False
        self.loading_dots = 0
        self.loading_timer = QTimer()
        self.loading_timer.timeout.connect(self.update_loading_animation)
        self.loading_timer.setInterval(300)  # Update every 300ms for smoother animation
        
        # Context menu
        self.context_menu = ConversationContextMenu(self)
        
        # Initialize with empty conversation
        self.update_conversation([])
        
        # Images list - to prevent garbage collection
        self.images = []
        self.image_paths = []

        # Create text formats with different colors
        self.text_formats = {
            "user": QTextCharFormat(),
            "ai": QTextCharFormat(),
            "system": QTextCharFormat(),
            "ai_label": QTextCharFormat(),
            "normal": QTextCharFormat(),
            "error": QTextCharFormat()
        }

        # Configure text formats using global color palette
        self.text_formats["user"].setForeground(QColor(COLORS['text_normal']))
        self.text_formats["ai"].setForeground(QColor(COLORS['text_normal']))
        self.text_formats["system"].setForeground(QColor(COLORS['text_normal']))
        self.text_formats["ai_label"].setForeground(QColor(COLORS['accent_blue']))
        self.text_formats["normal"].setForeground(QColor(COLORS['text_normal']))
        self.text_formats["error"].setForeground(QColor(COLORS['text_error']))
        
        # Make AI labels bold
        self.text_formats["ai_label"].setFontWeight(QFont.Weight.Bold)
    
    def setup_ui(self):
        """Set up the user interface for the conversation pane"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)  # Reduced spacing
        
        # Title and info area
        title_layout = QHBoxLayout()
        self.title_label = QLabel("Liminal Backrooms")
        self.title_label.setStyleSheet(f"""
            color: {COLORS['text_bright']};
            font-size: 14px;
            font-weight: bold;
            padding: 2px;
        """)
        
        self.info_label = QLabel("AI-to-AI conversation")
        self.info_label.setStyleSheet(f"""
            color: {COLORS['text_dim']};
            font-size: 11px;
            padding: 2px;
        """)
        
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.info_label)
        
        layout.addLayout(title_layout)
        
        # Conversation display (read-only text edit in a scroll area)
        self.conversation_display = QTextEdit()
        self.conversation_display.setReadOnly(True)
        self.conversation_display.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.conversation_display.customContextMenuRequested.connect(self.show_context_menu)
        
        # Set font for conversation display
        font = QFont("Segoe UI", 10)
        self.conversation_display.setFont(font)
        
        # Apply modern styling
        self.conversation_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 10px;
                selection-background-color: {COLORS['accent_blue']};
                selection-color: {COLORS['text_bright']};
            }}
            QScrollBar:vertical {{
                background: {COLORS['bg_medium']};
                width: 12px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS['border']};
                min-height: 20px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {COLORS['border_highlight']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
        """)
        
        # Input area with label
        input_container = QWidget()
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(2)  # Reduced spacing
        
        input_label = QLabel("Your message:")
        input_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        input_layout.addWidget(input_label)
        
        # Input field with modern styling
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Seed the conversation or just click propagate...")
        self.input_field.setMaximumHeight(60)  # Reduced height
        self.input_field.setFont(font)
        self.input_field.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px;
                selection-background-color: {COLORS['accent_blue']};
                selection-color: {COLORS['text_bright']};
            }}
        """)
        input_layout.addWidget(self.input_field)
        
        # Button container for better layout
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(5)  # Reduced spacing
        
        # Clear button
        self.clear_button = QPushButton("Clear")
        self.clear_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 4px 10px;
                font-weight: bold;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['border']};
                border: 1px solid {COLORS['border_highlight']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['border_highlight']};
            }}
        """)
        
        # Submit button with modern styling
        self.submit_button = QPushButton("Propagate")
        self.submit_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_blue']};
                color: {COLORS['text_bright']};
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-weight: bold;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_blue_hover']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['accent_blue_active']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['border']};
                color: {COLORS['text_dim']};
            }}
        """)
        
        # Add buttons to layout
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        button_layout.addWidget(self.submit_button)
        
        # Add input container to main layout
        input_layout.addWidget(button_container)
        
        # Control panel (buttons for various actions)
        self.control_panel = ControlPanel()
        
        # Add widgets to layout with adjusted stretch factors
        layout.addWidget(self.conversation_display, 1)  # Main conversation area gets most space
        layout.addWidget(input_container, 0)  # Input area gets minimal space
        layout.addWidget(self.control_panel, 0)  # Control panel gets minimal space
    
    def connect_signals(self):
        """Connect signals and slots"""
        # Submit button
        self.submit_button.clicked.connect(self.handle_propagate_click)
        
        # Clear button
        self.clear_button.clicked.connect(self.clear_input)
        
        # Enter key in input field
        self.input_field.installEventFilter(self)
    
    def clear_input(self):
        """Clear the input field"""
        self.input_field.clear()
        self.input_field.setFocus()
    
    def eventFilter(self, obj, event):
        """Filter events to handle Enter key in input field"""
        if obj is self.input_field and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.handle_propagate_click()
                return True
        return super().eventFilter(obj, event)
    
    def handle_propagate_click(self):
        """Handle click on the propagate button"""
        # Get the input text (might be empty)
        input_text = self.input_field.toPlainText().strip()
        
        # Clear the input box
        self.input_field.clear()
        
        # Always call the input callback, even with empty input
        if hasattr(self, 'input_callback') and self.input_callback:
            self.input_callback(input_text)
        
        # Start loading animation
        self.start_loading()
    
    def set_input_callback(self, callback):
        """Set callback function for input submission"""
        self.input_callback = callback
    
    def set_rabbithole_callback(self, callback):
        """Set callback function for rabbithole creation"""
        self.rabbithole_callback = callback
    
    def set_fork_callback(self, callback):
        """Set callback function for fork creation"""
        self.fork_callback = callback
    
    def update_conversation(self, conversation):
        """Update conversation display"""
        self.conversation = conversation
        self.render_conversation()
    
    def render_conversation(self):
        """Render conversation in the display"""
        # Clear display
        self.conversation_display.clear()
        
        # Create HTML for conversation with modern styling
        html = "<style>"
        html += f"body {{ font-family: 'Segoe UI', sans-serif; font-size: 10pt; line-height: 1.4; }}"
        html += f".message {{ margin-bottom: 10px; padding: 8px; border-radius: 4px; }}"
        html += f".user {{ background-color: {COLORS['bg_medium']}; }}"
        html += f".assistant {{ background-color: {COLORS['bg_medium']}; }}"
        html += f".system {{ background-color: {COLORS['bg_medium']}; font-style: italic; }}"
        html += f".header {{ font-weight: bold; margin: 10px 0; color: {COLORS['accent_blue']}; }}"
        html += f".content {{ white-space: pre-wrap; color: {COLORS['text_normal']}; }}"
        html += f".branch-indicator {{ color: {COLORS['text_dim']}; font-style: italic; text-align: center; margin: 8px 0; }}"
        html += f".rabbithole {{ color: {COLORS['accent_green']}; }}"
        html += f".fork {{ color: {COLORS['accent_yellow']}; }}"
        # Removed HTML contribution styling
        html += f"pre {{ background-color: {COLORS['bg_dark']}; border: 1px solid {COLORS['border']}; border-radius: 3px; padding: 8px; overflow-x: auto; margin: 8px 0; }}"
        html += f"code {{ font-family: 'Consolas', 'Courier New', monospace; color: {COLORS['text_bright']}; }}"
        html += "</style>"
        
        for i, message in enumerate(self.conversation):
            role = message.get("role", "")
            content = message.get("content", "")
            ai_name = message.get("ai_name", "")
            model = message.get("model", "")
            
            # Skip empty messages
            if not content:
                continue
                
            # Handle branch indicators with special styling
            if role == 'system' and message.get('_type') == 'branch_indicator':
                if "Rabbitholing down:" in content:
                    html += f'<div class="branch-indicator rabbithole">{content}</div>'
                elif "Forking off:" in content:
                    html += f'<div class="branch-indicator fork">{content}</div>'
                continue
            
            # Removed HTML contribution indicator logic
            
            # Process content to handle code blocks
            processed_content = self.process_content_with_code_blocks(content)
            
            # Format based on role
            if role == 'user':
                # User message
                html += f'<div class="message user">'
                html += f'<div class="content">{processed_content}</div>'
                html += f'</div>'
            elif role == 'assistant':
                # AI message
                display_name = ai_name
                if model:
                    display_name += f" ({model})"
                html += f'<div class="message assistant">'
                html += f'<div class="header">\n{display_name}\n</div>'
                html += f'<div class="content">{processed_content}</div>'
                
                # Removed HTML contribution indicator
                
                html += f'</div>'
            elif role == 'system':
                # System message
                html += f'<div class="message system">'
                html += f'<div class="content">{processed_content}</div>'
                html += f'</div>'
        
        # Set HTML in display
        self.conversation_display.setHtml(html)
        
        # Scroll to bottom
        self.conversation_display.verticalScrollBar().setValue(
            self.conversation_display.verticalScrollBar().maximum()
        )
    
    def process_content_with_code_blocks(self, content):
        """Process content to properly format code blocks"""
        import re
        from html import escape
        
        # First, escape HTML in the content
        escaped_content = escape(content)
        
        # Check if there are any code blocks in the content
        if "```" not in escaped_content:
            return escaped_content
        
        # Split the content by code block markers
        parts = re.split(r'(```(?:[a-zA-Z0-9_]*)\n.*?```)', escaped_content, flags=re.DOTALL)
        
        result = []
        for part in parts:
            if part.startswith("```") and part.endswith("```"):
                # This is a code block
                try:
                    # Extract language if specified
                    language_match = re.match(r'```([a-zA-Z0-9_]*)\n', part)
                    language = language_match.group(1) if language_match else ""
                    
                    # Extract code content
                    code_content = part[part.find('\n')+1:part.rfind('```')]
                    
                    # Format as HTML
                    formatted_code = f'<pre><code class="language-{language}">{code_content}</code></pre>'
                    result.append(formatted_code)
                except Exception as e:
                    # If there's an error, just add the original escaped content
                    print(f"Error processing code block: {e}")
                    result.append(part)
            else:
                # Process inline code in non-code-block parts
                inline_parts = re.split(r'(`[^`]+`)', part)
                processed_part = []
                
                for inline_part in inline_parts:
                    if inline_part.startswith("`") and inline_part.endswith("`") and len(inline_part) > 2:
                        # This is inline code
                        code = inline_part[1:-1]
                        processed_part.append(f'<code>{code}</code>')
                    else:
                        processed_part.append(inline_part)
                
                result.append(''.join(processed_part))
        
        return ''.join(result)
    
    def start_loading(self):
        """Start loading animation"""
        self.loading = True
        self.loading_dots = 0
        self.input_field.setEnabled(False)
        self.submit_button.setEnabled(False)
        self.submit_button.setText("Processing")
        self.loading_timer.start()
        
        # Add subtle pulsing animation to the button
        self.pulse_animation = QPropertyAnimation(self.submit_button, b"styleSheet")
        self.pulse_animation.setDuration(1000)
        self.pulse_animation.setLoopCount(-1)  # Infinite loop
        
        # Define keyframes for the animation
        normal_style = f"""
            QPushButton {{
                background-color: {COLORS['border']};
                color: {COLORS['text_dim']};
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-weight: bold;
                font-size: 11px;
            }}
        """
        
        pulse_style = f"""
            QPushButton {{
                background-color: {COLORS['border_highlight']};
                color: {COLORS['text_dim']};
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-weight: bold;
                font-size: 11px;
            }}
        """
        
        self.pulse_animation.setStartValue(normal_style)
        self.pulse_animation.setEndValue(pulse_style)
        self.pulse_animation.start()
    
    def stop_loading(self):
        """Stop loading animation"""
        self.loading = False
        self.loading_timer.stop()
        self.input_field.setEnabled(True)
        self.submit_button.setEnabled(True)
        self.submit_button.setText("Propagate")
        
        # Stop the pulsing animation
        if hasattr(self, 'pulse_animation'):
            self.pulse_animation.stop()
            
        # Reset button style
        self.submit_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_blue']};
                color: {COLORS['text_bright']};
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-weight: bold;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_blue_hover']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['accent_blue_active']};
            }}
        """)
    
    def update_loading_animation(self):
        """Update loading animation dots"""
        self.loading_dots = (self.loading_dots + 1) % 4
        dots = "." * self.loading_dots
        self.submit_button.setText(f"Processing{dots}")
    
    def show_context_menu(self, position):
        """Show context menu at the given position"""
        # Get selected text
        cursor = self.conversation_display.textCursor()
        selected_text = cursor.selectedText()
        
        # Only show context menu if text is selected
        if selected_text:
            # Show the context menu at cursor position
            self.context_menu.exec(self.conversation_display.mapToGlobal(position))
    
    def rabbithole_from_selection(self):
        """Create a rabbithole branch from selected text"""
        cursor = self.conversation_display.textCursor()
        selected_text = cursor.selectedText()
        
        if selected_text and hasattr(self, 'rabbithole_callback'):
            self.rabbithole_callback(selected_text)
    
    def fork_from_selection(self):
        """Create a fork branch from selected text"""
        cursor = self.conversation_display.textCursor()
        selected_text = cursor.selectedText()
        
        if selected_text and hasattr(self, 'fork_callback'):
            self.fork_callback(selected_text)
    
    def append_text(self, text, format_type="normal"):
        """Append text to the conversation display with the specified format"""
        cursor = self.conversation_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Apply the format if specified
        if format_type in self.text_formats:
            self.conversation_display.setCurrentCharFormat(self.text_formats[format_type])
        
        # Insert the text
        cursor.insertText(text)
        
        # Reset to normal format after insertion
        if format_type != "normal":
            self.conversation_display.setCurrentCharFormat(self.text_formats["normal"])
        
        # Scroll to bottom
        self.conversation_display.setTextCursor(cursor)
        self.conversation_display.ensureCursorVisible()
    
    def clear_conversation(self):
        """Clear the conversation display"""
        self.conversation_display.clear()
        self.images = []
        
    def display_conversation(self, conversation, branch_data=None):
        """Display the conversation in the text edit widget"""
        # Clear the current text
        self.conversation_display.clear()
        
        # Store conversation data
        self.conversation = conversation
        
        # Check if we're in a branch
        is_branch = branch_data is not None
        branch_type = branch_data.get('type', '') if is_branch else ''
        selected_text = branch_data.get('selected_text', '') if is_branch else ''
        
        # Update title if in a branch
        if is_branch:
            branch_emoji = "🐇" if branch_type == "rabbithole" else "🍴"
            self.title_label.setText(f"{branch_emoji} {branch_type.capitalize()}: {selected_text[:30]}...")
            self.info_label.setText(f"Branch conversation")
        else:
            self.title_label.setText("Liminal Backrooms")
            self.info_label.setText("AI-to-AI conversation")
        
        # Debug: Print conversation to console
        print("\n--- DEBUG: Conversation Content ---")
        for msg in conversation:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if "```" in content:
                print(f"Found code block in {role} message")
                print(f"Content snippet: {content[:100]}...")
        print("--- End Debug ---\n")
        
        # Render conversation
        self.render_conversation()
        
    def display_image(self, image_path):
        """Display an image in the conversation"""
        try:
            # Check if the image path is valid
            if not image_path or not os.path.exists(image_path):
                self.append_text("[Image not found]\n", "error")
                return
            
            # Load the image
            image = QImage(image_path)
            if image.isNull():
                self.append_text("[Invalid image format]\n", "error")
                return
            
            # Create a pixmap from the image
            pixmap = QPixmap.fromImage(image)
            
            # Scale the image to fit the conversation display
            max_width = self.conversation_display.width() - 50
            if pixmap.width() > max_width:
                pixmap = pixmap.scaledToWidth(max_width, Qt.TransformationMode.SmoothTransformation)
            
            # Insert the image into the conversation display
            cursor = self.conversation_display.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertImage(pixmap.toImage())
            cursor.insertText("\n\n")
            
            # Store the image to prevent garbage collection
            self.images.append(pixmap)
            self.image_paths.append(image_path)
            
        except Exception as e:
            self.append_text(f"[Error displaying image: {str(e)}]\n", "error")
    
    def export_conversation(self):
        """Export the conversation to a file"""
        # Set default directory to user's documents folder or a custom exports folder
        default_dir = ""
        
        # Try to use user's Documents folder
        documents_path = os.path.join(os.path.expanduser("~"), "Documents")
        if os.path.exists(documents_path):
            default_dir = os.path.join(documents_path, "LiminalBackrooms")
        else:
            # Fallback to a local exports directory
            default_dir = os.path.join(os.getcwd(), "exports")
        
        # Create the directory if it doesn't exist
        os.makedirs(default_dir, exist_ok=True)
        
        # Generate a default filename based on date/time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = os.path.join(default_dir, f"conversation_{timestamp}.txt")
        
        # Get the file name from a save dialog
        file_name, selected_filter = QFileDialog.getSaveFileName(
            self, 
            "Export Conversation",
            default_filename,
            "Text Files (*.txt);;Markdown Files (*.md);;HTML Files (*.html);;Full HTML Document (*.html);;All Files (*)"
        )
        
        if not file_name:
            return  # User cancelled the dialog
        
        try:
            # Determine export format based on file extension
            _, ext = os.path.splitext(file_name)
        
            # Export as Full HTML Document
            if selected_filter == "Full HTML Document (*.html)":
                # Copy the existing conversation_full.html file if it exists
                full_html_path = os.path.join(os.getcwd(), "conversation_full.html")
                if os.path.exists(full_html_path):
                    shutil.copy2(full_html_path, file_name)
                    print(f"Full HTML document exported to {file_name}")
                    
                    # Get main window
                    main_window = self.window()
                    main_window.statusBar().showMessage(f"Full HTML document exported to {file_name}")
                    return
                else:
                    # Fallback to regular HTML if full document doesn't exist
                    content = self.conversation_display.toHtml()
            elif ext.lower() == '.html':
                # Export as HTML - the QTextEdit already contains HTML formatting
                content = self.conversation_display.toHtml()
            elif ext.lower() == '.md':
                # Export as Markdown - convert HTML to markdown
                html_content = self.conversation_display.toHtml()
                # Simple conversion for now (could be improved with a proper HTML->MD converter)
                content = html_content.replace('<b>', '**').replace('</b>', '**')
                content = re.sub(r'<[^>]*>', '', content)  # Remove other HTML tags
            else:
                # Export as plain text
                content = self.conversation_display.toPlainText()
            
            # Write content to file
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # For status message - get main window
            main_window = self.window()
            main_window.statusBar().showMessage(f"Conversation exported to {file_name}")
            print(f"Conversation exported to {file_name}")
            
        except Exception as e:
            error_msg = f"Error exporting conversation: {str(e)}"
            QMessageBox.critical(self, "Export Error", error_msg)
            print(error_msg)

class LiminalBackroomsApp(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        
        # Main app state
        self.conversation = []
        self.turn_count = 0
        self.images = []
        self.image_paths = []
        self.branch_conversations = {}  # Store branch conversations by ID
        self.active_branch = None      # Currently displayed branch
        
        # Set up the UI
        self.setup_ui()
        
        # Connect signals and slots
        self.connect_signals()
        
        # Dark theme
        self.apply_dark_theme()
        
        # Restore splitter state if available
        self.restore_splitter_state()
        
        # Start maximized
        self.showMaximized()
    
    def setup_ui(self):
        """Set up the user interface"""
        self.setWindowTitle("Liminal Backrooms v0.7")
        self.setGeometry(100, 100, 1600, 900)  # Initial size before maximizing
        self.setMinimumSize(1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(0)
        
        # Create horizontal splitter for left and right panes
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(8)  # Make the handle wider for easier grabbing
        self.splitter.setChildrenCollapsible(False)  # Prevent panes from being collapsed
        self.splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {COLORS['border']};
                border: 1px solid {COLORS['border_highlight']};
                border-radius: 2px;
            }}
            QSplitter::handle:hover {{
                background-color: {COLORS['accent_blue']};
            }}
        """)
        main_layout.addWidget(self.splitter)
        
        # Create left pane (conversation) and right pane (network view)
        self.left_pane = ConversationPane()
        self.right_pane = NetworkPane()
        
        self.splitter.addWidget(self.left_pane)
        self.splitter.addWidget(self.right_pane)
        
        # Set initial splitter sizes (65:35 ratio for more space for the network)
        total_width = 1600  # Based on default window width
        self.splitter.setSizes([int(total_width * 0.65), int(total_width * 0.35)])
        
        # Initialize main conversation as root node
        self.right_pane.add_node('main', 'Seed', 'main')
        
        # Status bar with modern styling
        self.statusBar().setStyleSheet(f"""
            QStatusBar {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_dim']};
                border-top: 1px solid {COLORS['border']};
                padding: 3px;
                font-size: 11px;
            }}
        """)
        self.statusBar().showMessage("Ready")
        
        # Set up input callback
        self.left_pane.set_input_callback(self.handle_user_input)
    
    def connect_signals(self):
        """Connect all signals and slots"""
        # Node selection in network view
        self.right_pane.nodeSelected.connect(self.on_branch_select)
        
        # Node hover in network view
        if hasattr(self.right_pane.network_view, 'nodeHovered'):
            self.right_pane.network_view.nodeHovered.connect(self.on_node_hover)
        
        # Export button
        self.left_pane.control_panel.export_button.clicked.connect(self.export_conversation)
        
        # Connect context menu actions to the main app methods
        self.left_pane.set_rabbithole_callback(self.branch_from_selection)
        self.left_pane.set_fork_callback(self.fork_from_selection)
        
        # Save splitter state when it moves
        self.splitter.splitterMoved.connect(self.save_splitter_state)
    
    def handle_user_input(self, text):
        """Handle user input from the conversation pane"""
        # Add user message to conversation
        if text:
            user_message = {
                "role": "user",
                "content": text
            }
            self.conversation.append(user_message)
            
            # Update conversation display
            self.left_pane.update_conversation(self.conversation)
        
        # Process the conversation (this will be implemented in main.py)
        if hasattr(self, 'process_conversation'):
            self.process_conversation()
    
    def append_text(self, text, format_type="normal"):
        """Append text to the conversation display with the specified format"""
        self.left_pane.append_text(text, format_type)
    
    def clear_conversation(self):
        """Clear the conversation display and reset images"""
        self.left_pane.clear_conversation()
        self.conversation = []
        self.images = []
        self.image_paths = []
    
    def display_conversation(self, conversation, branch_data=None):
        """Display the conversation in the text edit widget"""
        self.left_pane.display_conversation(conversation, branch_data)
    
    def display_image(self, image_path):
        """Display an image in the conversation"""
        self.left_pane.display_image(image_path)
    
    def export_conversation(self):
        """Export the current conversation"""
        self.left_pane.export_conversation()
    
    def on_node_hover(self, node_id):
        """Handle node hover in the network view"""
        if node_id == 'main':
            self.statusBar().showMessage("Main conversation")
        elif node_id in self.branch_conversations:
            branch_data = self.branch_conversations[node_id]
            branch_type = branch_data.get('type', 'branch')
            selected_text = branch_data.get('selected_text', '')
            self.statusBar().showMessage(f"{branch_type.capitalize()}: {selected_text[:50]}...")
    
    def apply_dark_theme(self):
        """Apply dark theme to the application"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_normal']};
            }}
            QWidget {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_normal']};
            }}
            QToolTip {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border']};
                padding: 5px;
            }}
        """)
        
        # Add specific styling for branch messages
        branch_header_format = QTextCharFormat()
        branch_header_format.setForeground(QColor(COLORS['ai_header']))
        branch_header_format.setFontWeight(QFont.Weight.Bold)
        branch_header_format.setFontPointSize(11)
        
        branch_inline_format = QTextCharFormat()
        branch_inline_format.setForeground(QColor(COLORS['ai_header']))
        branch_inline_format.setFontItalic(True)
        branch_inline_format.setFontPointSize(10)
        
        # Add formats to the left pane
        self.left_pane.text_formats["branch_header"] = branch_header_format
        self.left_pane.text_formats["branch_inline"] = branch_inline_format
    
    def on_branch_select(self, branch_id):
        """Handle branch selection in the network view"""
        try:
            # Check if branch exists
            if branch_id == 'main':
                # Switch to main conversation
                self.active_branch = None
                # Make sure we have a main_conversation attribute
                if not hasattr(self, 'main_conversation'):
                    self.main_conversation = []
                self.conversation = self.main_conversation
                self.left_pane.update_conversation(self.conversation)
                self.statusBar().showMessage("Switched to main conversation")
                return
            
            if branch_id not in self.branch_conversations:
                self.statusBar().showMessage(f"Branch {branch_id} not found")
                return
            
            # Get branch data
            branch_data = self.branch_conversations[branch_id]
            
            # Set active branch
            self.active_branch = branch_id
            
            # Update conversation
            self.conversation = branch_data['conversation']
            
            # Display the conversation with branch metadata
            self.left_pane.display_conversation(self.conversation, branch_data)
            
            # Update status bar
            self.statusBar().showMessage(f"Switched to {branch_data['type']} branch: {branch_id}")
            
        except Exception as e:
            print(f"Error selecting branch: {e}")
            self.statusBar().showMessage(f"Error selecting branch: {e}")
    
    def branch_from_selection(self, selected_text):
        """Create a rabbithole branch from selected text"""
        if not selected_text:
            return
        
        # Create branch
        branch_id = self.create_branch(selected_text, 'rabbithole')
        
        # Switch to branch
        self.on_branch_select(branch_id)
    
    def fork_from_selection(self, selected_text):
        """Create a fork branch from selected text"""
        if not selected_text:
            return
        
        # Create branch
        branch_id = self.create_branch(selected_text, 'fork')
        
        # Switch to branch
        self.on_branch_select(branch_id)
    
    def create_branch(self, selected_text, branch_type="rabbithole", parent_branch=None):
        """Create a new branch in the conversation"""
        try:
            # Generate a unique ID for the branch
            branch_id = str(uuid.uuid4())
            
            # Get parent branch ID
            parent_id = parent_branch if parent_branch else (self.active_branch if self.active_branch else 'main')
            
            # Get current conversation
            if parent_id == 'main':
                # If parent is main, use main conversation
                if not hasattr(self, 'main_conversation'):
                    self.main_conversation = []
                current_conversation = self.main_conversation.copy()
            else:
                # Otherwise, use parent branch conversation
                parent_data = self.branch_conversations.get(parent_id)
                if parent_data:
                    current_conversation = parent_data['conversation'].copy()
                else:
                    current_conversation = []
            
            # Create initial message based on branch type
            if branch_type == 'fork':
                initial_message = {
                    "role": "user",
                    "content": f"Complete this thought or sentence naturally, continuing forward from exactly this point: '{selected_text}'"
                }
            else:  # rabbithole
                initial_message = {
                    "role": "user",
                    "content": f"Let's explore and expand upon the concept of '{selected_text}' from our previous discussion."
                }
            
            # Create branch conversation with initial message
            branch_conversation = current_conversation.copy()
            branch_conversation.append(initial_message)
            
            # Create branch data
            branch_data = {
                'id': branch_id,
                'parent': parent_id,
                'type': branch_type,
                'selected_text': selected_text,
                'conversation': branch_conversation,
                'turn_count': 0,
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'history': current_conversation
            }
            
            # Store branch data
            self.branch_conversations[branch_id] = branch_data
            
            # Add node to network graph - make sure parameters are in the correct order
            node_label = f"{branch_type.capitalize()}: {selected_text[:20]}{'...' if len(selected_text) > 20 else ''}"
            self.right_pane.add_node(branch_id, node_label, branch_type)
            self.right_pane.add_edge(parent_id, branch_id)
            
            # Set active branch to this new branch
            self.active_branch = branch_id
            self.conversation = branch_conversation
            
            # Display the conversation
            self.left_pane.display_conversation(branch_conversation, branch_data)
            
            # Trigger AI response processing for this branch
            if hasattr(self, 'process_branch_conversation'):
                # Add a small delay to ensure UI updates first
                QTimer.singleShot(100, lambda: self.process_branch_conversation(branch_id))
            
            # Return branch ID
            return branch_id
            
        except Exception as e:
            print(f"Error creating branch: {e}")
            self.statusBar().showMessage(f"Error creating branch: {e}")
            return None
    
    def get_branch_path(self, branch_id):
        """Get the full path of branch names from root to the given branch"""
        try:
            path = []
            current_id = branch_id
            
            # Prevent potential infinite loops by tracking visited branches
            visited = set()
            
            while current_id != 'main' and current_id not in visited:
                visited.add(current_id)
                branch_data = self.branch_conversations.get(current_id)
                if not branch_data:
                    break
                    
                # Get a readable version of the selected text (truncated if needed)
                selected_text = branch_data.get('selected_text', '')
                if selected_text:
                    display_text = f"{selected_text[:20]}{'...' if len(selected_text) > 20 else ''}"
                    path.append(display_text)
                else:
                    path.append(f"{branch_data.get('type', 'Branch').capitalize()}")
                
                # Check for valid parent attribute
                current_id = branch_data.get('parent')
                if not current_id:
                    break
            
            path.append('Seed')
            return ' → '.join(reversed(path))
        except Exception as e:
            print(f"Error building branch path: {e}")
            return f"Branch {branch_id}"
    
    def save_splitter_state(self):
        """Save the current splitter state to a file"""
        try:
            # Create settings directory if it doesn't exist
            if not os.path.exists('settings'):
                os.makedirs('settings')
                
            # Save splitter state to file
            with open('settings/splitter_state.json', 'w') as f:
                json.dump({
                    'sizes': self.splitter.sizes()
                }, f)
        except Exception as e:
            print(f"Error saving splitter state: {e}")
    
    def restore_splitter_state(self):
        """Restore the splitter state from a file if available"""
        try:
            if os.path.exists('settings/splitter_state.json'):
                with open('settings/splitter_state.json', 'r') as f:
                    state = json.load(f)
                    if 'sizes' in state:
                        self.splitter.setSizes(state['sizes'])
        except Exception as e:
            print(f"Error restoring splitter state: {e}")
            # Fall back to default sizes
            total_width = self.width()
            self.splitter.setSizes([int(total_width * 0.7), int(total_width * 0.3)])

    def process_branch_conversation(self, branch_id):
        """Process the branch conversation using the selected models"""
        # This method will be implemented in main.py to avoid circular imports
        pass

    def node_clicked(self, node_id):
        """Handle node click in the network view"""
        print(f"Node clicked: {node_id}")
        
        # Check if this is the main conversation or a branch
        if node_id == 'main':
            # Switch to main conversation
            self.active_branch = None
            self.left_pane.display_conversation(self.main_conversation)
        elif node_id in self.branch_conversations:
            # Switch to branch conversation
            self.active_branch = node_id
            branch_data = self.branch_conversations[node_id]
            conversation = branch_data['conversation']
            
            # Filter hidden messages for display
            visible_conversation = [msg for msg in conversation if not msg.get('hidden', False)]
            self.left_pane.display_conversation(visible_conversation, branch_data)

    def initialize_selectors(self):
        """Initialize the AI model selectors and prompt pair selector"""
        pass

    # Removed: create_new_living_document