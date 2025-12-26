from PyQt6.QtWidgets import QWidget, QToolTip
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QLinearGradient, QRadialGradient, QPainterPath
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPointF
import math
import random
from src.ui.colors import COLORS

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
        self.edge_growth_speed = 0.05  # Increased speed of edge growth animation

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

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
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
        width = self.width()
        height = self.height()
        center_x = width / 2
        center_y = height / 2
        scale = min(width, height) / 500

        for node_id in self.nodes:
            if node_id in self.node_positions:
                    x, y = self.node_positions[node_id]
                    screen_x = center_x + x * scale
                    screen_y = center_y + y * scale

                    node_size = self.node_sizes.get(node_id, 400)
                    radius = math.sqrt(node_size) * scale / 2

                    distance = math.sqrt((pos.x() - screen_x)**2 + (pos.y() - screen_y)**2)
                    if distance <= radius:
                        return node_id

        return None
