from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QPushButton
from src.core.config import config

class ControlPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # AI 1
        layout.addWidget(QLabel("AI 1:"))
        self.ai1_combo = QComboBox()
        self.ai1_combo.addItems(list(config.ai_models.keys()))
        layout.addWidget(self.ai1_combo)

        # AI 2
        layout.addWidget(QLabel("AI 2:"))
        self.ai2_combo = QComboBox()
        self.ai2_combo.addItems(list(config.ai_models.keys()))
        layout.addWidget(self.ai2_combo)

        # Iterations
        layout.addWidget(QLabel("Turns:"))
        self.iter_combo = QComboBox()
        self.iter_combo.addItems(["1", "2", "4", "10", "100"])
        layout.addWidget(self.iter_combo)

        # Stats
        self.stats_label = QLabel("Turns: 0")
        layout.addWidget(self.stats_label)

        layout.addStretch()

    def get_config(self):
        return {
            "ai1_model": self.ai1_combo.currentText(),
            "ai1_model_id": config.ai_models[self.ai1_combo.currentText()],
            "ai2_model": self.ai2_combo.currentText(),
            "ai2_model_id": config.ai_models[self.ai2_combo.currentText()],
            "iterations": int(self.iter_combo.currentText())
        }
