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
import base64
from PyQt6.QtCore import Qt, QRect, QTimer, QRectF, QPointF, QSize, pyqtSignal, QEvent, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QBrush, QFontDatabase, QTextCursor, QAction, QKeySequence, QTextCharFormat, QLinearGradient, QRadialGradient, QPainterPath, QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QApplication, QMainWindow, QSplitter, QVBoxLayout, QHBoxLayout, QTextEdit, QFrame, QLineEdit, QPushButton, QLabel, QComboBox, QMenu, QFileDialog, QMessageBox, QScrollArea, QToolTip, QSizePolicy, QCheckBox, QGraphicsDropShadowEffect

from config import (
    AI_MODELS,
    SYSTEM_PROMPT_PAIRS,
    SHOW_CHAIN_OF_THOUGHT_IN_CONTEXT
)

# Add import for the HTML viewing functionality 
from shared_utils import open_html_in_browser, generate_image_from_text

# Define global color palette for consistent styling - Cyberpunk theme
COLORS = {
    # Backgrounds - darker, moodier
    'bg_dark': '#0A0E1A',           # Deep blue-black
    'bg_medium': '#111827',         # Slate dark
    'bg_light': '#1E293B',          # Lighter slate
    
    # Primary accents - neon but muted
    'accent_cyan': '#06B6D4',       # Cyan (primary)
    'accent_cyan_hover': '#0891B2',
    'accent_cyan_active': '#0E7490',
    
    # Secondary accents
    'accent_pink': '#EC4899',       # Hot pink (secondary)
    'accent_purple': '#A855F7',     # Purple (tertiary)
    'accent_yellow': '#FBBF24',     # Amber for warnings
    'accent_green': '#10B981',      # Emerald (rabbithole)
    
    # Text colors
    'text_normal': '#CBD5E1',       # Slate-200
    'text_dim': '#64748B',          # Slate-500
    'text_bright': '#F1F5F9',       # Slate-50
    'text_glow': '#38BDF8',         # Sky-400 (glowing text)
    'text_error': '#EF4444',        # Red-500
    
    # Borders and effects
    'border': '#1E293B',            # Slate-800
    'border_glow': '#06B6D4',       # Glowing cyan borders
    'border_highlight': '#334155',  # Slate-700
    'shadow': 'rgba(6, 182, 212, 0.2)',  # Cyan glow shadows
    
    # Legacy color mappings for compatibility
    'accent_blue': '#06B6D4',       # Map old blue to cyan
    'accent_blue_hover': '#0891B2',
    'accent_blue_active': '#0E7490',
    'accent_orange': '#F59E0B',     # Amber-500
    'chain_of_thought': '#10B981',  # Emerald
    'user_header': '#06B6D4',       # Cyan
    'ai_header': '#A855F7',         # Purple
    'system_message': '#F59E0B',    # Amber
}


def apply_glow_effect(widget, color, blur_radius=15, offset=(0, 2)):
    """Apply a glowing drop shadow effect to a widget"""
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(blur_radius)
    shadow.setColor(QColor(color))
    shadow.setOffset(offset[0], offset[1])
    widget.setGraphicsEffect(shadow)
    return shadow


class GlowButton(QPushButton):
    """Enhanced button with glow effect on hover"""
    
    def __init__(self, text, glow_color=COLORS['accent_cyan'], parent=None):
        super().__init__(text, parent)
        self.glow_color = glow_color
        self.base_blur = 8
        self.hover_blur = 20
        
        # Create shadow effect
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(self.base_blur)
        self.shadow.setColor(QColor(glow_color))
        self.shadow.setOffset(0, 2)
        self.setGraphicsEffect(self.shadow)
        
        # Track hover state for animation
        self.setMouseTracking(True)
    
    def enterEvent(self, event):
        """Increase glow on hover"""
        self.shadow.setBlurRadius(self.hover_blur)
        self.shadow.setColor(QColor(self.glow_color))
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Decrease glow when not hovering"""
        self.shadow.setBlurRadius(self.base_blur)
        super().leaveEvent(event)

# Load custom fonts
def load_fonts():
    """Load custom fonts for the application"""
    font_dir = Path("fonts")
    font_dir.mkdir(exist_ok=True)
    
    # List of fonts to load - these would need to be included with the application
    fonts = [
        ("IosevkaTerm-Regular.ttf", "Iosevka Term"),
        ("IosevkaTerm-Bold.ttf", "Iosevka Term"),
        ("IosevkaTerm-Italic.ttf", "Iosevka Term"),
    ]
    
    loaded_fonts = []
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
            print(f"Font file not found: {font_path}")
    
    return loaded_fonts


# ═══════════════════════════════════════════════════════════════════════════════
# ATMOSPHERIC EFFECT WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

class DepthGauge(QWidget):
    """Vertical gauge showing conversation depth/turn progress"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_turn = 0
        self.max_turns = 10
        self.setFixedWidth(24)
        self.setMinimumHeight(100)
        
        # Animation
        self.pulse_offset = 0
        self.pulse_timer = QTimer(self)
        self.pulse_timer.timeout.connect(self._animate_pulse)
        self.pulse_timer.start(50)
        
    def _animate_pulse(self):
        self.pulse_offset = (self.pulse_offset + 2) % 360
        self.update()
    
    def set_progress(self, current, maximum):
        """Update the gauge progress"""
        self.current_turn = current
        self.max_turns = max(maximum, 1)
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        margin = 4
        gauge_width = w - margin * 2
        gauge_height = h - margin * 2
        
        # Background track
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS['bg_dark']))
        painter.drawRoundedRect(margin, margin, gauge_width, gauge_height, 4, 4)
        
        # Border
        painter.setPen(QPen(QColor(COLORS['border_glow']), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(margin, margin, gauge_width, gauge_height, 4, 4)
        
        # Calculate fill height (fills from bottom to top)
        progress = min(self.current_turn / self.max_turns, 1.0)
        fill_height = int(gauge_height * progress)
        fill_y = margin + gauge_height - fill_height
        
        if fill_height > 0:
            # Gradient fill
            gradient = QLinearGradient(0, fill_y, 0, margin + gauge_height)
            
            # Color shifts based on depth - deeper = more purple/pink
            if progress < 0.33:
                gradient.setColorAt(0, QColor(COLORS['accent_cyan']))
                gradient.setColorAt(1, QColor(COLORS['accent_cyan']).darker(130))
            elif progress < 0.66:
                gradient.setColorAt(0, QColor(COLORS['accent_purple']))
                gradient.setColorAt(1, QColor(COLORS['accent_cyan']))
            else:
                gradient.setColorAt(0, QColor(COLORS['accent_pink']))
                gradient.setColorAt(1, QColor(COLORS['accent_purple']))
            
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(gradient)
            painter.drawRoundedRect(margin + 2, fill_y, gauge_width - 4, fill_height, 2, 2)
            
            # Pulsing glow line at top of fill
            pulse_alpha = int(100 + 80 * math.sin(math.radians(self.pulse_offset)))
            glow_color = QColor(COLORS['accent_cyan'])
            glow_color.setAlpha(pulse_alpha)
            painter.setPen(QPen(glow_color, 2))
            painter.drawLine(margin + 2, fill_y, margin + gauge_width - 2, fill_y)
        
        # Turn counter text
        painter.setPen(QColor(COLORS['text_dim']))
        font = painter.font()
        font.setPixelSize(9)
        painter.setFont(font)
        text = f"{self.current_turn}"
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, text)


class SignalIndicator(QWidget):
    """Signal strength/latency indicator"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(80, 20)
        self.signal_strength = 1.0  # 0.0 to 1.0
        self.latency_ms = 0
        self.is_active = False
        
        # Animation for activity
        self.bar_offset = 0
        self.activity_timer = QTimer(self)
        self.activity_timer.timeout.connect(self._animate)
        
    def _animate(self):
        self.bar_offset = (self.bar_offset + 1) % 5
        self.update()
    
    def set_active(self, active):
        """Set whether we're actively waiting for a response"""
        self.is_active = active
        if active:
            self.activity_timer.start(100)
        else:
            self.activity_timer.stop()
        self.update()
    
    def set_latency(self, latency_ms):
        """Update the latency display"""
        self.latency_ms = latency_ms
        # Calculate signal strength based on latency
        if latency_ms < 500:
            self.signal_strength = 1.0
        elif latency_ms < 1500:
            self.signal_strength = 0.75
        elif latency_ms < 3000:
            self.signal_strength = 0.5
        else:
            self.signal_strength = 0.25
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw signal bars
        bar_heights = [4, 7, 10, 13, 16]
        bar_width = 4
        spacing = 2
        start_x = 5
        base_y = 18
        
        for i, bar_h in enumerate(bar_heights):
            x = start_x + i * (bar_width + spacing)
            y = base_y - bar_h
            
            # Determine if this bar should be lit
            threshold = (i + 1) / len(bar_heights)
            is_lit = self.signal_strength >= threshold
            
            if self.is_active:
                # Animated pattern when active
                is_lit = ((i + self.bar_offset) % 5) < 3
                color = QColor(COLORS['accent_cyan']) if is_lit else QColor(COLORS['bg_light'])
            else:
                if is_lit:
                    # Color based on signal strength
                    if self.signal_strength > 0.7:
                        color = QColor(COLORS['accent_green'])
                    elif self.signal_strength > 0.4:
                        color = QColor(COLORS['accent_yellow'])
                    else:
                        color = QColor(COLORS['accent_pink'])
                else:
                    color = QColor(COLORS['bg_light'])
            
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(x, y, bar_width, bar_h, 1, 1)
        
        # Draw latency text
        painter.setPen(QColor(COLORS['text_dim']))
        font = painter.font()
        font.setPixelSize(9)
        painter.setFont(font)
        
        if self.is_active:
            text = "···"
        elif self.latency_ms > 0:
            text = f"{self.latency_ms}ms"
        else:
            text = "IDLE"
        
        painter.drawText(40, 3, 40, 16, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)


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
        # Initialize velocities if needed
        for node_id in self.nodes:
            if node_id not in self.node_velocities:
                self.node_velocities[node_id] = (0, 0)
        
        # Calculate repulsive forces between nodes
        new_velocities = {}
        for node_id in self.nodes:
            if node_id not in self.node_positions:
                continue
                
            vx, vy = self.node_velocities.get(node_id, (0, 0))
            x1, y1 = self.node_positions[node_id]
            
            # Apply repulsion between nodes
            for other_id in self.nodes:
                if other_id == node_id or other_id not in self.node_positions:
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
                    
                    # Apply force
                    vx += nx * strength
                    vy += ny * strength
            
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
        for node_id, (vx, vy) in new_velocities.items():
            if node_id in self.node_positions:
                # Skip the main node to keep it centered
                if node_id == 'main':
                    continue
                    
                x, y = self.node_positions[node_id]
                self.node_positions[node_id] = (x + vx, y + vy)
        
        # Update velocities for next frame
        self.node_velocities = new_velocities
        
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
        
        # Cyberpunk tab button styling
        tab_style = f"""
            QPushButton {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_dim']};
                border: none;
                border-bottom: 2px solid transparent;
                padding: 12px 20px;
                font-weight: bold;
                font-size: 11px;
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
        
        # Make buttons checkable for tab behavior
        self.setup_button.setCheckable(True)
        self.graph_button.setCheckable(True)
        self.setup_button.setChecked(True)  # Start with setup tab active
        
        # Connect tab buttons
        self.setup_button.clicked.connect(lambda: self.switch_tab(0))
        self.graph_button.clicked.connect(lambda: self.switch_tab(1))
        
        tab_layout.addWidget(self.setup_button)
        tab_layout.addWidget(self.graph_button)
        
        layout.addWidget(tab_container)
        
        # Create stacked widget for tab content
        from PyQt6.QtWidgets import QStackedWidget
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
        
        # Add pages to stack
        self.stack.addWidget(self.control_panel)
        self.stack.addWidget(self.network_pane)
        
        layout.addWidget(self.stack, 1)  # Stretch to fill
        
        # Connect network pane signal to forward it
        self.network_pane.nodeSelected.connect(self.nodeSelected)
    
    def switch_tab(self, index):
        """Switch between tabs"""
        self.stack.setCurrentIndex(index)
        
        # Update button states
        self.setup_button.setChecked(index == 0)
        self.graph_button.setChecked(index == 1)
    
    def add_node(self, node_id, label, node_type):
        """Forward to network pane"""
        self.network_pane.add_node(node_id, label, node_type)
    
    def add_edge(self, source_id, target_id):
        """Forward to network pane"""
        self.network_pane.add_edge(source_id, target_id)
    
    def update_graph(self):
        """Forward to network pane"""
        self.network_pane.update_graph()

class ControlPanel(QWidget):
    """Control panel with mode, model selections, etc."""
    def __init__(self):
        super().__init__()
        
        # Set up the UI
        self.setup_ui()
        
        # Initialize with models and prompt pairs
        self.initialize_selectors()
    
    def setup_ui(self):
        """Set up the user interface for the control panel - vertical sidebar layout"""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(8)
        
        # Add a title with cyberpunk styling
        title = QLabel("═ CONTROL PANEL ═")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"""
            color: {COLORS['accent_cyan']};
            font-size: 12px;
            font-weight: bold;
            padding: 10px;
            background-color: {COLORS['bg_medium']};
            border: 1px solid {COLORS['border_glow']};
            border-radius: 0px;
            letter-spacing: 2px;
        """)
        main_layout.addWidget(title)
        
        # Create scrollable area for controls
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                background: {COLORS['bg_medium']};
                width: 10px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS['border_glow']};
                min-height: 20px;
                border-radius: 0px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {COLORS['accent_cyan']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
        """)
        
        # Container widget for scrollable content
        scroll_content = QWidget()
        scroll_content.setStyleSheet(f"background-color: transparent;")
        
        # All controls in vertical layout
        controls_layout = QVBoxLayout(scroll_content)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        controls_layout.setSpacing(10)
        
        # Mode selection with icon
        mode_container = QWidget()
        mode_layout = QVBoxLayout(mode_container)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(5)
        
        mode_label = QLabel("▸ MODE")
        mode_label.setStyleSheet(f"color: {COLORS['text_glow']}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        mode_layout.addWidget(mode_label)
        
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["AI-AI", "Human-AI"])
        self.mode_selector.setStyleSheet(self.get_combobox_style())
        mode_layout.addWidget(self.mode_selector)
        controls_layout.addWidget(mode_container)
        
        # Iterations with slider
        iterations_container = QWidget()
        iterations_layout = QVBoxLayout(iterations_container)
        iterations_layout.setContentsMargins(0, 0, 0, 0)
        iterations_layout.setSpacing(5)
        
        iterations_label = QLabel("▸ ITERATIONS")
        iterations_label.setStyleSheet(f"color: {COLORS['text_glow']}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        iterations_layout.addWidget(iterations_label)
        
        self.iterations_selector = QComboBox()
        self.iterations_selector.addItems(["1", "2", "4", "6", "12", "100"])
        self.iterations_selector.setStyleSheet(self.get_combobox_style())
        iterations_layout.addWidget(self.iterations_selector)
        controls_layout.addWidget(iterations_container)
        
        # Number of AIs selection
        num_ais_container = QWidget()
        num_ais_layout = QVBoxLayout(num_ais_container)
        num_ais_layout.setContentsMargins(0, 0, 0, 0)
        num_ais_layout.setSpacing(5)
        
        num_ais_label = QLabel("▸ NUMBER OF AIs")
        num_ais_label.setStyleSheet(f"color: {COLORS['text_glow']}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        num_ais_layout.addWidget(num_ais_label)
        
        self.num_ais_selector = QComboBox()
        self.num_ais_selector.addItems(["1", "2", "3", "4", "5"])
        self.num_ais_selector.setCurrentText("3")  # Default to 3 AIs
        self.num_ais_selector.setStyleSheet(self.get_combobox_style())
        num_ais_layout.addWidget(self.num_ais_selector)
        controls_layout.addWidget(num_ais_container)
        
        # AI-1 Model selection
        self.ai1_container = QWidget()
        ai1_layout = QVBoxLayout(self.ai1_container)
        ai1_layout.setContentsMargins(0, 0, 0, 0)
        ai1_layout.setSpacing(5)
        
        ai1_label = QLabel("AI-1")
        ai1_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        ai1_layout.addWidget(ai1_label)
        
        self.ai1_model_selector = QComboBox()
        self.ai1_model_selector.setStyleSheet(self.get_combobox_style())
        ai1_layout.addWidget(self.ai1_model_selector)
        controls_layout.addWidget(self.ai1_container)
        
        # AI-2 Model selection
        self.ai2_container = QWidget()
        ai2_layout = QVBoxLayout(self.ai2_container)
        ai2_layout.setContentsMargins(0, 0, 0, 0)
        ai2_layout.setSpacing(5)
        
        ai2_label = QLabel("AI-2")
        ai2_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        ai2_layout.addWidget(ai2_label)
        
        self.ai2_model_selector = QComboBox()
        self.ai2_model_selector.setStyleSheet(self.get_combobox_style())
        ai2_layout.addWidget(self.ai2_model_selector)
        controls_layout.addWidget(self.ai2_container)
        
        # AI-3 Model selection
        self.ai3_container = QWidget()
        ai3_layout = QVBoxLayout(self.ai3_container)
        ai3_layout.setContentsMargins(0, 0, 0, 0)
        ai3_layout.setSpacing(5)
        
        ai3_label = QLabel("AI-3")
        ai3_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        ai3_layout.addWidget(ai3_label)
        
        self.ai3_model_selector = QComboBox()
        self.ai3_model_selector.setStyleSheet(self.get_combobox_style())
        ai3_layout.addWidget(self.ai3_model_selector)
        controls_layout.addWidget(self.ai3_container)
        
        # AI-4 Model selection
        self.ai4_container = QWidget()
        ai4_layout = QVBoxLayout(self.ai4_container)
        ai4_layout.setContentsMargins(0, 0, 0, 0)
        ai4_layout.setSpacing(5)
        
        ai4_label = QLabel("AI-4")
        ai4_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        ai4_layout.addWidget(ai4_label)
        
        self.ai4_model_selector = QComboBox()
        self.ai4_model_selector.setStyleSheet(self.get_combobox_style())
        ai4_layout.addWidget(self.ai4_model_selector)
        controls_layout.addWidget(self.ai4_container)
        
        # AI-5 Model selection
        self.ai5_container = QWidget()
        ai5_layout = QVBoxLayout(self.ai5_container)
        ai5_layout.setContentsMargins(0, 0, 0, 0)
        ai5_layout.setSpacing(5)
        
        ai5_label = QLabel("AI-5")
        ai5_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        ai5_layout.addWidget(ai5_label)
        
        self.ai5_model_selector = QComboBox()
        self.ai5_model_selector.setStyleSheet(self.get_combobox_style())
        ai5_layout.addWidget(self.ai5_model_selector)
        controls_layout.addWidget(self.ai5_container)
        
        # Prompt pair selection
        prompt_container = QWidget()
        prompt_layout = QVBoxLayout(prompt_container)
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_layout.setSpacing(5)
        
        prompt_label = QLabel("Conversation Scenario")
        prompt_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        prompt_layout.addWidget(prompt_label)
        
        self.prompt_pair_selector = QComboBox()
        self.prompt_pair_selector.setStyleSheet(self.get_combobox_style())
        prompt_layout.addWidget(self.prompt_pair_selector)
        controls_layout.addWidget(prompt_container)
        
        # Action buttons container
        action_container = QWidget()
        action_layout = QVBoxLayout(action_container)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(5)
        
        action_label = QLabel("▸ OPTIONS")
        action_label.setStyleSheet(f"color: {COLORS['text_glow']}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        action_layout.addWidget(action_label)
        
        # Auto-generate images checkbox
        self.auto_image_checkbox = QCheckBox("Auto-generate images")
        self.auto_image_checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {COLORS['text_normal']};
                spacing: 5px;
                font-size: 10px;
                padding: 4px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 1px solid {COLORS['border_glow']};
                border-radius: 0px;
                background-color: {COLORS['bg_medium']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {COLORS['accent_cyan']};
                border: 1px solid {COLORS['accent_cyan']};
            }}
            QCheckBox::indicator:hover {{
                border: 1px solid {COLORS['accent_cyan']};
            }}
        """)
        self.auto_image_checkbox.setToolTip("Automatically generate images from AI responses using Google Gemini 3 Pro Image Preview via OpenRouter")
        action_layout.addWidget(self.auto_image_checkbox)
        
        # Removed: HTML contributions checkbox
        
        # Actions - buttons in vertical layout
        actions_label = QLabel("▸ ACTIONS")
        actions_label.setStyleSheet(f"color: {COLORS['text_glow']}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        action_layout.addWidget(actions_label)
        
        # Export button with glow
        self.export_button = self.create_glow_button("📡 EXPORT", COLORS['accent_purple'])
        action_layout.addWidget(self.export_button)
        
        # View HTML button with glow
        self.view_html_button = self.create_glow_button("🌐 VIEW HTML", COLORS['accent_green'])
        self.view_html_button.clicked.connect(lambda: open_html_in_browser("shared_document.html"))
        action_layout.addWidget(self.view_html_button)
        
        # View Full HTML button
        self.view_full_html_button = QPushButton("🌐 FULL HTML")
        self.view_full_html_button.setStyleSheet(self.get_cyberpunk_button_style(COLORS['accent_yellow']))
        self.view_full_html_button.setToolTip("View the full conversation in dark mode")
        self.view_full_html_button.clicked.connect(lambda: open_html_in_browser("conversation_full.html"))
        action_layout.addWidget(self.view_full_html_button)
        
        controls_layout.addWidget(action_container)
        
        # Add all controls directly to controls_layout (now vertical)
        controls_layout.addWidget(mode_container)
        controls_layout.addWidget(iterations_container)
        controls_layout.addWidget(num_ais_container)
        
        # Divider
        divider1 = QLabel("─" * 20)
        divider1.setStyleSheet(f"color: {COLORS['border_glow']}; font-size: 8px;")
        controls_layout.addWidget(divider1)
        
        models_label = QLabel("▸ AI MODELS")
        models_label.setStyleSheet(f"color: {COLORS['text_glow']}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        controls_layout.addWidget(models_label)
        
        controls_layout.addWidget(self.ai1_container)
        controls_layout.addWidget(self.ai2_container)
        controls_layout.addWidget(self.ai3_container)
        controls_layout.addWidget(self.ai4_container)
        controls_layout.addWidget(self.ai5_container)
        
        # Divider
        divider2 = QLabel("─" * 20)
        divider2.setStyleSheet(f"color: {COLORS['border_glow']}; font-size: 8px;")
        controls_layout.addWidget(divider2)
        
        scenario_label = QLabel("▸ SCENARIO")
        scenario_label.setStyleSheet(f"color: {COLORS['text_glow']}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        controls_layout.addWidget(scenario_label)
        
        controls_layout.addWidget(prompt_container)
        
        # Divider
        divider3 = QLabel("─" * 20)
        divider3.setStyleSheet(f"color: {COLORS['border_glow']}; font-size: 8px;")
        controls_layout.addWidget(divider3)
        
        controls_layout.addWidget(action_container)
        
        # Add spacer
        controls_layout.addStretch()
        
        # Set the scroll area widget and add to main layout
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, 1)  # Stretch to fill
    
    def get_combobox_style(self):
        """Get the style for comboboxes - cyberpunk themed"""
        return f"""
            QComboBox {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border_glow']};
                border-radius: 0px;
                padding: 8px 10px;
                min-height: 30px;
                font-size: 10px;
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
                selection-background-color: {COLORS['accent_cyan']};
                selection-color: {COLORS['bg_dark']};
                border: 1px solid {COLORS['border_glow']};
                border-radius: 0px;
                padding: 4px;
            }}
            QComboBox QAbstractItemView::item {{
                min-height: 28px;
                padding: 4px;
            }}
        """
    
    def get_cyberpunk_button_style(self, accent_color):
        """Get cyberpunk-themed button style with given accent color"""
        return f"""
            QPushButton {{
                background-color: {COLORS['bg_medium']};
                color: {accent_color};
                border: 2px solid {accent_color};
                border-radius: 3px;
                padding: 10px 14px;
                font-weight: bold;
                font-size: 10px;
                letter-spacing: 1px;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: {accent_color};
                color: {COLORS['bg_dark']};
                border: 2px solid {accent_color};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['bg_light']};
                color: {accent_color};
            }}
        """
    
    def create_glow_button(self, text, accent_color):
        """Create a button with glow effect"""
        button = GlowButton(text, accent_color)
        button.setStyleSheet(self.get_cyberpunk_button_style(accent_color))
        return button
    
    def initialize_selectors(self):
        """Initialize the selector dropdowns with values from config"""
        # Add AI models
        self.ai1_model_selector.clear()
        self.ai2_model_selector.clear()
        self.ai3_model_selector.clear()
        self.ai4_model_selector.clear()
        self.ai5_model_selector.clear()
        self.ai1_model_selector.addItems(list(AI_MODELS.keys()))
        self.ai2_model_selector.addItems(list(AI_MODELS.keys()))
        self.ai3_model_selector.addItems(list(AI_MODELS.keys()))
        self.ai4_model_selector.addItems(list(AI_MODELS.keys()))
        self.ai5_model_selector.addItems(list(AI_MODELS.keys()))
        
        # Add prompt pairs
        self.prompt_pair_selector.clear()
        self.prompt_pair_selector.addItems(list(SYSTEM_PROMPT_PAIRS.keys()))
        
        # Connect number of AIs selector to update visibility
        self.num_ais_selector.currentTextChanged.connect(self.update_ai_selector_visibility)
        
        # Set initial visibility based on default number of AIs (3)
        self.update_ai_selector_visibility("3")
    
    def update_ai_selector_visibility(self, num_ais_text):
        """Show/hide AI model selectors based on number of AIs selected"""
        num_ais = int(num_ais_text)
        
        # AI-1 is always visible
        # AI-2 visible if num_ais >= 2
        # AI-3 visible if num_ais >= 3
        # AI-4 visible if num_ais >= 4
        # AI-5 visible if num_ais >= 5
        
        self.ai1_container.setVisible(num_ais >= 1)
        self.ai2_container.setVisible(num_ais >= 2)
        self.ai3_container.setVisible(num_ais >= 3)
        self.ai4_container.setVisible(num_ais >= 4)
        self.ai5_container.setVisible(num_ais >= 5)

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
        # NOTE: Fork/Rabbithole temporarily disabled - needs rebuild
        # self.addAction(self.rabbithole_action)
        # self.addAction(self.fork_action)
        
        # Connect actions to signals
        # self.rabbithole_action.triggered.connect(self.on_rabbithole_selected)
        # self.fork_action.triggered.connect(self.on_fork_selected)
        
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
        
        # Uploaded image for current message
        self.uploaded_image_path = None
        self.uploaded_image_base64 = None

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
        self.title_label = QLabel("╔═ LIMINAL BACKROOMS ═╗")
        self.title_label.setStyleSheet(f"""
            color: {COLORS['accent_cyan']};
            font-size: 14px;
            font-weight: bold;
            padding: 4px;
            letter-spacing: 2px;
        """)
        
        self.info_label = QLabel("[ AI-TO-AI PROPAGATION ]")
        self.info_label.setStyleSheet(f"""
            color: {COLORS['text_glow']};
            font-size: 10px;
            padding: 2px;
            letter-spacing: 1px;
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
        
        # Set font for conversation display - use Iosevka Term for better ASCII art rendering
        font = QFont("Iosevka Term", 10)
        # Set fallbacks in case Iosevka Term isn't loaded
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.conversation_display.setFont(font)
        
        # Apply cyberpunk styling
        self.conversation_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border_glow']};
                border-radius: 0px;
                padding: 15px;
                selection-background-color: {COLORS['accent_cyan']};
                selection-color: {COLORS['bg_dark']};
            }}
            QScrollBar:vertical {{
                background: {COLORS['bg_medium']};
                width: 10px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS['border_glow']};
                min-height: 20px;
                border-radius: 0px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {COLORS['accent_cyan']};
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
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border_glow']};
                border-radius: 0px;
                padding: 8px;
                selection-background-color: {COLORS['accent_cyan']};
                selection-color: {COLORS['bg_dark']};
            }}
        """)
        input_layout.addWidget(self.input_field)
        
        # Button container for better layout
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(5)  # Reduced spacing
        
        # Upload image button
        self.upload_image_button = QPushButton("📎 IMAGE")
        self.upload_image_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border_glow']};
                border-radius: 0px;
                padding: 6px 10px;
                font-weight: bold;
                font-size: 10px;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
                border: 1px solid {COLORS['accent_cyan']};
                color: {COLORS['accent_cyan']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['border_glow']};
            }}
        """)
        self.upload_image_button.setToolTip("Upload an image to include in your message")
        
        # Clear button with subtle glow
        self.clear_button = GlowButton("CLEAR", COLORS['accent_pink'])
        self.clear_button.shadow.setBlurRadius(5)  # Subtler glow
        self.clear_button.base_blur = 5
        self.clear_button.hover_blur = 12
        self.clear_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_normal']};
                border: 1px solid {COLORS['border_glow']};
                border-radius: 3px;
                padding: 8px 12px;
                font-weight: bold;
                font-size: 10px;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
                border: 2px solid {COLORS['accent_pink']};
                color: {COLORS['accent_pink']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['border_glow']};
            }}
        """)
        
        # Submit button with cyberpunk styling and glow effect
        self.submit_button = GlowButton("⚡ PROPAGATE", COLORS['accent_cyan'])
        self.submit_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_cyan']};
                color: {COLORS['bg_dark']};
                border: 2px solid {COLORS['accent_cyan']};
                border-radius: 3px;
                padding: 8px 20px;
                font-weight: bold;
                font-size: 11px;
                letter-spacing: 2px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['accent_cyan']};
                border: 2px solid {COLORS['accent_cyan']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['accent_cyan_active']};
                color: {COLORS['text_bright']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['border']};
                color: {COLORS['text_dim']};
                border: 2px solid {COLORS['border']};
            }}
        """)
        
        # Add buttons to layout
        button_layout.addWidget(self.upload_image_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        button_layout.addWidget(self.submit_button)
        
        # Add input container to main layout
        input_layout.addWidget(button_container)
        
        # Add widgets to layout with adjusted stretch factors
        layout.addWidget(self.conversation_display, 1)  # Main conversation area gets most space
        layout.addWidget(input_container, 0)  # Input area gets minimal space
    
    def connect_signals(self):
        """Connect signals and slots"""
        # Submit button
        self.submit_button.clicked.connect(self.handle_propagate_click)
        
        # Upload image button
        self.upload_image_button.clicked.connect(self.handle_upload_image)
        
        # Clear button
        self.clear_button.clicked.connect(self.clear_input)
        
        # Enter key in input field
        self.input_field.installEventFilter(self)
    
    def clear_input(self):
        """Clear the input field"""
        self.input_field.clear()
        self.uploaded_image_path = None
        self.uploaded_image_base64 = None
        self.upload_image_button.setText("📎 Image")
        self.input_field.setFocus()
    
    def handle_upload_image(self):
        """Handle image upload button click"""
        # Open file dialog
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.gif *.webp);;All Files (*)"
        )
        
        if file_path:
            try:
                # Read and encode the image to base64
                with open(file_path, 'rb') as image_file:
                    image_data = image_file.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                
                # Determine media type
                file_extension = os.path.splitext(file_path)[1].lower()
                media_type_map = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp'
                }
                media_type = media_type_map.get(file_extension, 'image/jpeg')
                
                # Store the image data
                self.uploaded_image_path = file_path
                self.uploaded_image_base64 = {
                    'data': image_base64,
                    'media_type': media_type
                }
                
                # Update button text to show an image is attached
                file_name = os.path.basename(file_path)
                self.upload_image_button.setText(f"📎 {file_name[:15]}...")
                
                # Update placeholder text
                self.input_field.setPlaceholderText("Add a message about your image (optional)...")
                
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Upload Error",
                    f"Failed to load image: {str(e)}"
                )
    
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
        
        # Prepare message data (text + optional image)
        message_data = {
            'text': input_text,
            'image': None
        }
        
        # Include image if one was uploaded
        if self.uploaded_image_base64:
            message_data['image'] = {
                'path': self.uploaded_image_path,
                'base64': self.uploaded_image_base64['data'],
                'media_type': self.uploaded_image_base64['media_type']
            }
        
        # Clear the input box and image
        self.input_field.clear()
        self.uploaded_image_path = None
        self.uploaded_image_base64 = None
        self.upload_image_button.setText("📎 Image")
        self.input_field.setPlaceholderText("Seed the conversation or just click propagate...")
        
        # Always call the input callback, even with empty input
        if hasattr(self, 'input_callback') and self.input_callback:
            self.input_callback(message_data)
        
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
        # Check if user is at the bottom before re-rendering
        scrollbar = self.conversation_display.verticalScrollBar()
        was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 20
        
        # Clear display
        self.conversation_display.clear()
        
        # Create HTML for conversation with modern styling
        html = "<style>"
        html += f"body {{ font-family: 'Iosevka Term', 'Consolas', 'Monaco', monospace; font-size: 10pt; line-height: 1.4; }}"
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
        html += f"code {{ font-family: 'Iosevka Term', 'Consolas', 'Monaco', monospace; color: {COLORS['text_bright']}; }}"
        html += "</style>"
        
        for i, message in enumerate(self.conversation):
            role = message.get("role", "")
            content = message.get("content", "")
            ai_name = message.get("ai_name", "")
            model = message.get("model", "")
            
            # Handle structured content (with images)
            has_image = False
            image_base64 = None
            text_content = ""
            
            if isinstance(content, list):
                # Structured content with potential images
                for part in content:
                    if part.get('type') == 'text':
                        text_content += part.get('text', '')
                    elif part.get('type') == 'image':
                        has_image = True
                        source = part.get('source', {})
                        if source.get('type') == 'base64':
                            image_base64 = source.get('data', '')
            else:
                # Plain text content
                text_content = content
            
            # Skip empty messages (no text and no image)
            if not text_content and not has_image:
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
            processed_content = self.process_content_with_code_blocks(text_content) if text_content else ""
            
            # Add image display if present
            image_html = ""
            if has_image and image_base64:
                image_html = f'<div style="margin: 10px 0;"><img src="data:image/jpeg;base64,{image_base64}" style="max-width: 100%; border-radius: 8px;" /></div>'
            
            # Format based on role
            if role == 'user':
                # User message
                html += f'<div class="message user">'
                if image_html:
                    html += image_html
                if processed_content:
                    html += f'<div class="content">{processed_content}</div>'
                html += f'</div>'
            elif role == 'assistant':
                # AI message
                display_name = ai_name
                if model:
                    display_name += f" ({model})"
                html += f'<div class="message assistant">'
                html += f'<div class="header">\n{display_name}\n</div>'
                if image_html:
                    html += image_html
                if processed_content:
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
        
        # Only auto-scroll if user was already at the bottom
        if was_at_bottom:
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
                background-color: {COLORS['accent_cyan']};
                color: {COLORS['bg_dark']};
                border: 1px solid {COLORS['accent_cyan']};
                border-radius: 0px;
                padding: 6px 16px;
                font-weight: bold;
                font-size: 11px;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['accent_cyan']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['accent_cyan_active']};
                color: {COLORS['text_bright']};
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
        # Check if user is at the bottom before appending (within 20 pixels is considered "at bottom")
        scrollbar = self.conversation_display.verticalScrollBar()
        was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 20
        
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
        
        # Only auto-scroll if user was already at the bottom
        if was_at_bottom:
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


class CentralContainer(QWidget):
    """Central container widget with animated background and overlay support"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Background animation state
        self.bg_offset = 0
        self.noise_offset = 0
        
        # Animation timer for background
        self.bg_timer = QTimer(self)
        self.bg_timer.timeout.connect(self._animate_bg)
        self.bg_timer.start(80)  # ~12 FPS for subtle movement
        
        # Create scanline overlay as child widget
        self.scanline_overlay = ScanlineOverlayWidget(self)
        self.scanline_overlay.hide()
    
    def _animate_bg(self):
        self.bg_offset = (self.bg_offset + 1) % 360
        self.noise_offset = (self.noise_offset + 0.5) % 100
        self.update()
    
    def set_scanlines_enabled(self, enabled):
        """Toggle scanline effect"""
        if enabled:
            # Ensure overlay has correct geometry before showing
            self.scanline_overlay.setGeometry(self.rect())
            self.scanline_overlay.show()
            self.scanline_overlay.raise_()
            self.scanline_overlay.start_animation()
        else:
            self.scanline_overlay.stop_animation()
            self.scanline_overlay.hide()
    
    def resizeEvent(self, event):
        """Update scanline overlay size when container resizes"""
        super().resizeEvent(event)
        self.scanline_overlay.setGeometry(self.rect())
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # ═══ ANIMATED BACKGROUND ═══
        # Create shifting gradient with more visible movement
        center_x = self.width() / 2 + math.sin(math.radians(self.bg_offset)) * 100
        center_y = self.height() / 2 + math.cos(math.radians(self.bg_offset * 0.7)) * 60
        
        gradient = QRadialGradient(center_x, center_y, max(self.width(), self.height()) * 0.9)
        
        # More visible atmospheric colors with cyan tint
        pulse = 0.5 + 0.5 * math.sin(math.radians(self.bg_offset * 2))
        center_r = int(10 + 8 * pulse)
        center_g = int(15 + 10 * pulse)
        center_b = int(30 + 15 * pulse)
        
        gradient.setColorAt(0, QColor(center_r, center_g, center_b))
        gradient.setColorAt(0.4, QColor(10, 14, 26))
        gradient.setColorAt(1, QColor(6, 8, 14))
        
        painter.fillRect(self.rect(), gradient)
        
        # Add subtle glow lines at edges
        glow_alpha = int(15 + 10 * pulse)
        glow_color = QColor(6, 182, 212, glow_alpha)  # Cyan glow
        painter.setPen(QPen(glow_color, 2))
        
        # Top edge glow
        painter.drawLine(0, 0, self.width(), 0)
        # Bottom edge glow  
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        # Left edge glow
        painter.drawLine(0, 0, 0, self.height())
        # Right edge glow
        painter.drawLine(self.width() - 1, 0, self.width() - 1, self.height())
        
        # Add subtle noise/grain pattern
        noise_color = QColor(COLORS['accent_cyan'])
        noise_color.setAlpha(8)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(noise_color)
        
        # Sparse random dots for grain effect
        random.seed(int(self.noise_offset))
        for _ in range(50):
            x = random.randint(0, self.width())
            y = random.randint(0, self.height())
            painter.drawEllipse(x, y, 1, 1)


class ScanlineOverlayWidget(QWidget):
    """Transparent overlay widget for CRT scanline effect"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.scanline_offset = 0
        self.intensity = 0.25  # More visible scanlines
        
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate)
    
    def start_animation(self):
        self.anim_timer.start(100)
    
    def stop_animation(self):
        self.anim_timer.stop()
    
    def _animate(self):
        self.scanline_offset = (self.scanline_offset + 1) % 4
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        
        # Draw horizontal scanlines - more visible
        line_alpha = int(255 * self.intensity)
        line_color = QColor(0, 0, 0, line_alpha)
        painter.setPen(QPen(line_color, 1))
        
        # Draw every 2nd line for more visible effect
        for y in range(self.scanline_offset, self.height(), 2):
            painter.drawLine(0, y, self.width(), y)
        
        # Subtle vignette effect at edges
        gradient = QRadialGradient(self.width() / 2, self.height() / 2, 
                                   max(self.width(), self.height()) * 0.7)
        gradient.setColorAt(0, QColor(0, 0, 0, 0))
        gradient.setColorAt(0.7, QColor(0, 0, 0, 0))
        gradient.setColorAt(1, QColor(0, 0, 0, int(255 * self.intensity * 1.5)))
        
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRect(self.rect())


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
        self.setWindowTitle("╔═ LIMINAL BACKROOMS v0.7 ═╗")
        self.setGeometry(100, 100, 1600, 900)  # Initial size before maximizing
        self.setMinimumSize(1200, 800)
        
        # Create central widget - this will be a custom widget that paints the background
        self.central_container = CentralContainer()
        self.setCentralWidget(self.central_container)
        
        # Main layout for content
        main_layout = QHBoxLayout(self.central_container)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        
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
        
        # Create left pane (conversation) and right sidebar (tabbed: setup + network)
        self.left_pane = ConversationPane()
        self.right_sidebar = RightSidebar()
        
        self.splitter.addWidget(self.left_pane)
        self.splitter.addWidget(self.right_sidebar)
        
        # Set initial splitter sizes (70:30 ratio - more space for conversation)
        total_width = 1600  # Based on default window width
        self.splitter.setSizes([int(total_width * 0.70), int(total_width * 0.30)])
        
        # Initialize main conversation as root node
        self.right_sidebar.add_node('main', 'Seed', 'main')
        
        # ═══ SIGNAL INDICATOR ═══
        self.signal_indicator = SignalIndicator()
        
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
        
        # Add signal indicator to status bar
        self.statusBar().addPermanentWidget(self.signal_indicator)
        
        # ═══ CRT TOGGLE CHECKBOX ═══
        self.crt_checkbox = QCheckBox("CRT")
        self.crt_checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {COLORS['text_dim']};
                font-size: 10px;
                spacing: 4px;
            }}
            QCheckBox::indicator {{
                width: 12px;
                height: 12px;
                border: 1px solid {COLORS['border_glow']};
                border-radius: 2px;
                background: {COLORS['bg_dark']};
            }}
            QCheckBox::indicator:checked {{
                background: {COLORS['accent_cyan']};
            }}
        """)
        self.crt_checkbox.setToolTip("Toggle CRT scanline effect")
        self.crt_checkbox.toggled.connect(self.toggle_crt_effect)
        self.statusBar().addPermanentWidget(self.crt_checkbox)
        
        # Set up input callback
        self.left_pane.set_input_callback(self.handle_user_input)
    
    def toggle_crt_effect(self, enabled):
        """Toggle the CRT scanline effect"""
        if hasattr(self, 'central_container'):
            self.central_container.set_scanlines_enabled(enabled)
    
    def set_signal_active(self, active):
        """Set signal indicator to active (waiting for response)"""
        self.signal_indicator.set_active(active)
    
    def update_signal_latency(self, latency_ms):
        """Update signal indicator with response latency"""
        self.signal_indicator.set_latency(latency_ms)
    
    def connect_signals(self):
        """Connect all signals and slots"""
        # Node selection in network view
        self.right_sidebar.nodeSelected.connect(self.on_branch_select)
        
        # Node hover in network view
        if hasattr(self.right_sidebar.network_pane.network_view, 'nodeHovered'):
            self.right_sidebar.network_pane.network_view.nodeHovered.connect(self.on_node_hover)
        
        # Export button
        self.right_sidebar.control_panel.export_button.clicked.connect(self.export_conversation)
        
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
            self.right_sidebar.add_node(branch_id, node_label, branch_type)
            self.right_sidebar.add_edge(parent_id, branch_id)
            
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