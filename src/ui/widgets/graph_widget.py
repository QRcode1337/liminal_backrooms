from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QLinearGradient
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
import math

class GraphWidget(QWidget):
    node_clicked = pyqtSignal(str) # branch_id

    def __init__(self):
        super().__init__()
        self.nodes = {} # id -> data
        self.edges = [] # (source, target)
        self.positions = {}
        self.setMouseTracking(True)
        self.selected_node = None

    def update_graph(self, branches, active_id):
        self.nodes = branches
        self.active_id = active_id
        self.calculate_layout()
        self.update()

    def calculate_layout(self):
        # Very simple radial layout
        self.positions = {}
        self.edges = []

        # Main node at center
        self.positions["main"] = (self.width()/2, self.height()/2)

        # Traverse branches
        # This is a simplification. A real implementation needs a proper tree traversal.
        # Here we just place branches in a circle around main for demo purposes
        # or recursively if we had the structure handy.

        # Let's assume nodes are stored with parent info
        levels = {"main": 0}

        # Identify levels
        to_process = [("main", 0)]
        processed = set(["main"])

        # We need a proper way to get children.
        # Since self.nodes is a dict of Branch objects, we can scan it.
        children_map = {}
        for bid, branch in self.nodes.items():
            pid = branch.parent or "main"
            if pid not in children_map: children_map[pid] = []
            children_map[pid].append(bid)

        while to_process:
            pid, level = to_process.pop(0)
            if pid in children_map:
                for child_id in children_map[pid]:
                    levels[child_id] = level + 1
                    self.edges.append((pid, child_id))
                    to_process.append((child_id, level + 1))

        # Position nodes
        # Center
        cx, cy = self.width() / 2, self.height() / 2
        self.positions["main"] = (cx, cy)

        # For each level > 0, place in circle
        level_nodes = {}
        for nid, lvl in levels.items():
            if lvl == 0: continue
            if lvl not in level_nodes: level_nodes[lvl] = []
            level_nodes[lvl].append(nid)

        for lvl, nodes in level_nodes.items():
            radius = 100 * lvl
            angle_step = 2 * math.pi / len(nodes)
            for i, nid in enumerate(nodes):
                angle = i * angle_step
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                self.positions[nid] = (x, y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor("#1E1E1E"))

        # Edges
        painter.setPen(QPen(QColor("#555555"), 2))
        for u, v in self.edges:
            if u in self.positions and v in self.positions:
                p1 = QPointF(*self.positions[u])
                p2 = QPointF(*self.positions[v])
                painter.drawLine(p1, p2)

        # Nodes
        for nid, pos in self.positions.items():
            x, y = pos

            # Determine color
            color = QColor("#569CD6") # Blue main
            if nid != "main":
                branch_type = self.nodes[nid].type
                if branch_type == "rabbithole": color = QColor("#B5CEA8")
                elif branch_type == "fork": color = QColor("#DCDCAA")

            if nid == self.active_id:
                painter.setBrush(QBrush(color.lighter(150)))
                size = 15
            else:
                painter.setBrush(QBrush(color))
                size = 10

            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(x, y), size, size)

            # Label
            if nid != "main":
                text = self.nodes[nid].selected_text[:15] + "..."
                painter.setPen(QColor("#D4D4D4"))
                painter.drawText(int(x+15), int(y+5), text)

    def mousePressEvent(self, event):
        pos = event.position()
        # Find clicked node
        for nid, (nx, ny) in self.positions.items():
            dist = math.sqrt((pos.x() - nx)**2 + (pos.y() - ny)**2)
            if dist < 20: # Hit radius
                self.node_clicked.emit(nid)
                return
