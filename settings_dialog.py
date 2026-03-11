"""settings_dialog.py — API Key & Routing Settings for Liminal Backrooms"""

import os
from pathlib import Path
from dotenv import load_dotenv, set_key, dotenv_values
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGroupBox, QFormLayout, QScrollArea, QWidget,
    QFrame, QMessageBox, QTabWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

ENV_PATH = Path(__file__).parent / ".env"

# Provider definitions: display name, env var, base URL, which model prefixes it handles
PROVIDERS = [
    {
        "name": "Anthropic",
        "env": "ANTHROPIC_API_KEY",
        "url": "https://api.anthropic.com",
        "routes": ["anthropic/*"],
        "color": "#D97706",
        "placeholder": "sk-ant-api03-...",
    },
    {
        "name": "OpenAI",
        "env": "OPENAI_API_KEY",
        "url": "https://api.openai.com/v1",
        "routes": ["openai/*", "gpt-*", "o1", "o3"],
        "color": "#10B981",
        "placeholder": "sk-proj-...",
    },
    {
        "name": "OpenRouter (FREE models only)",
        "env": "OPENROUTER_API_KEY",
        "url": "https://openrouter.ai/api/v1",
        "routes": ["*:free", "deepseek/*", "meta-llama/*", "qwen/*", "mistralai/*"],
        "color": "#6366F1",
        "placeholder": "sk-or-v1-...",
    },
    {
        "name": "Groq",
        "env": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1",
        "routes": ["groq::*"],
        "color": "#EC4899",
        "placeholder": "gsk_...",
    },
    {
        "name": "Google (Gemini)",
        "env": "GOOGLE_API_KEY",
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "routes": ["google/gemini-*"],
        "color": "#3B82F6",
        "placeholder": "AIzaSy...",
    },
    {
        "name": "xAI (Grok)",
        "env": "XAI_API_KEY",
        "url": "https://api.x.ai/v1",
        "routes": ["x-ai/*"],
        "color": "#8B5CF6",
        "placeholder": "xai-...",
    },
    {
        "name": "Kimi / Moonshot",
        "env": "KIMIK2_API_KEY",
        "url": "https://api.moonshot.cn/v1",
        "routes": ["moonshotai/*", "kimi-direct::*"],
        "color": "#F59E0B",
        "placeholder": "sk-...",
    },
    {
        "name": "Ollama (Local/Remote)",
        "env": "OLLAMA_API_KEY",
        "url_env": "OLLAMA_BASE_URL",
        "url": "http://localhost:11434/v1",
        "routes": ["ollama::*"],
        "color": "#14B8A6",
        "placeholder": "ollama (or auth key)",
    },
]


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️  API Keys & Provider Routing")
        self.setMinimumSize(700, 600)
        self.setModal(True)
        self._apply_style()

        load_dotenv(ENV_PATH, override=True)
        self._env_vals = dict(dotenv_values(ENV_PATH))
        self._fields = {}  # env_var → QLineEdit

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header
        title = QLabel("API Keys & Provider Routing")
        title.setFont(QFont("monospace", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #00FF41; margin-bottom: 4px;")
        layout.addWidget(title)

        subtitle = QLabel(
            "OpenRouter is used ONLY for :free models. "
            "All paid models route directly to their provider."
        )
        subtitle.setStyleSheet("color: #888; font-size: 11px;")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #333;")
        layout.addWidget(sep)

        # Tabs
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #333; background: #0a0a0a; }
            QTabBar::tab { background: #111; color: #888; padding: 6px 16px; border: 1px solid #333; }
            QTabBar::tab:selected { background: #1a1a1a; color: #00FF41; border-bottom: 2px solid #00FF41; }
        """)

        # Tab 1: API Keys
        keys_tab = self._build_keys_tab()
        tabs.addTab(keys_tab, "🔑  API Keys")

        # Tab 2: Routing map
        routing_tab = self._build_routing_tab()
        tabs.addTab(routing_tab, "🔀  Routing Map")

        layout.addWidget(tabs, 1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        save_btn = QPushButton("💾  Save & Restart")
        save_btn.setStyleSheet(
            "QPushButton { background: #00FF41; color: #000; font-weight: bold; "
            "padding: 8px 20px; border-radius: 4px; }"
            "QPushButton:hover { background: #00cc33; }"
        )
        save_btn.clicked.connect(self._save)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(
            "QPushButton { background: #222; color: #aaa; padding: 8px 20px; border-radius: 4px; }"
            "QPushButton:hover { background: #333; }"
        )
        cancel_btn.clicked.connect(self.reject)

        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

    def _apply_style(self):
        self.setStyleSheet("""
            QDialog { background: #0a0a0a; color: #ccc; }
            QGroupBox { border: 1px solid #333; border-radius: 4px; margin-top: 8px;
                        padding: 8px; color: #aaa; font-size: 11px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #666; }
            QLabel { color: #ccc; }
            QLineEdit { background: #111; color: #00FF41; border: 1px solid #333;
                        border-radius: 3px; padding: 4px 8px; font-family: monospace; }
            QLineEdit:focus { border-color: #00FF41; }
            QScrollArea { border: none; background: transparent; }
        """)

    def _build_keys_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setSpacing(10)

        for prov in PROVIDERS:
            env = prov["env"]
            current_val = self._env_vals.get(env, "")
            masked = self._mask(current_val)

            box = QGroupBox()
            box.setStyleSheet(
                f"QGroupBox {{ border-left: 3px solid {prov['color']}; padding: 8px 12px; }}"
            )
            form = QFormLayout(box)
            form.setSpacing(6)

            # Provider name + status dot
            configured = bool(current_val and current_val not in ("", "your_key_here"))
            dot = "🟢" if configured else "🔴"
            name_label = QLabel(f"{dot}  <b>{prov['name']}</b>")
            name_label.setTextFormat(Qt.TextFormat.RichText)

            # URL label
            url = prov.get("url", "")
            url_label = QLabel(f"<span style='color:#555; font-size:10px;'>{url}</span>")
            url_label.setTextFormat(Qt.TextFormat.RichText)

            # Routes label
            routes_str = "  ".join(f"<code style='color:#666'>{r}</code>" for r in prov["routes"])
            routes_label = QLabel(routes_str)
            routes_label.setTextFormat(Qt.TextFormat.RichText)

            # Input field
            field = QLineEdit()
            field.setPlaceholderText(prov.get("placeholder", "Enter API key..."))
            field.setEchoMode(QLineEdit.EchoMode.Password)
            field.setText(current_val)
            field.setToolTip(f"Current (masked): {masked}")
            self._fields[env] = field

            # Show/hide toggle
            toggle = QPushButton("👁")
            toggle.setFixedWidth(32)
            toggle.setStyleSheet("background: #222; border: 1px solid #333; border-radius: 3px;")
            toggle.setCheckable(True)
            toggle.toggled.connect(lambda checked, f=field: f.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            ))

            row = QHBoxLayout()
            row.addWidget(field)
            row.addWidget(toggle)

            form.addRow(name_label, url_label)
            form.addRow("Routes:", routes_label)
            form.addRow(f"{env}:", row)

            # Extra URL field for Ollama
            if prov.get("url_env"):
                url_field = QLineEdit()
                url_field.setPlaceholderText("http://localhost:11434/v1")
                url_field.setText(self._env_vals.get(prov["url_env"], ""))
                self._fields[prov["url_env"]] = url_field
                form.addRow("Base URL:", url_field)

            vbox.addWidget(box)

        vbox.addStretch()
        scroll.setWidget(container)
        return scroll

    def _build_routing_tab(self) -> QWidget:
        widget = QWidget()
        vbox = QVBoxLayout(widget)

        info = QLabel(
            "<b>How routing works:</b><br><br>"
            "When you pick a model and start a conversation, the app looks at the model ID<br>"
            "and routes to the cheapest/most direct provider:<br><br>"
            "<code style='color:#00FF41'>anthropic/claude-*</code> → Anthropic API (direct)<br>"
            "<code style='color:#00FF41'>openai/gpt-*, o1, o3</code> → OpenAI API (direct)<br>"
            "<code style='color:#00FF41'>google/gemini-*</code> → Google API (direct)<br>"
            "<code style='color:#00FF41'>x-ai/grok-*</code> → xAI API (direct)<br>"
            "<code style='color:#00FF41'>moonshotai/*</code> → Moonshot/Kimi API (direct)<br>"
            "<code style='color:#00FF41'>groq::*</code> → Groq API (fast + cheap)<br>"
            "<code style='color:#00FF41'>ollama::*</code> → Ollama (local/remote)<br>"
            "<code style='color:#F59E0B'>*:free</code> → OpenRouter (your $2 account)<br>"
            "<code style='color:#F59E0B'>deepseek/*, qwen/*, meta-llama/*</code> → OpenRouter<br><br>"
            "<span style='color:#888; font-size:10px;'>"
            "OpenRouter is only charged for :free tier models and providers without direct keys.<br>"
            "Add direct API keys above to avoid OpenRouter charges for paid models."
            "</span>"
        )
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setWordWrap(True)
        info.setStyleSheet("padding: 16px; background: #0d0d0d; border: 1px solid #222; border-radius: 4px; line-height: 1.6;")
        vbox.addWidget(info)
        vbox.addStretch()
        return widget

    def _mask(self, val: str) -> str:
        if not val or len(val) < 8:
            return "not set"
        return val[:6] + "..." + val[-4:]

    def _save(self):
        """Write updated keys to .env file."""
        changed = []
        for env_var, field in self._fields.items():
            new_val = field.text().strip()
            old_val = self._env_vals.get(env_var, "")
            if new_val != old_val:
                set_key(str(ENV_PATH), env_var, new_val)
                changed.append(env_var)

        if changed:
            QMessageBox.information(
                self, "Saved",
                f"Updated {len(changed)} key(s):\n" + "\n".join(changed) +
                "\n\nRestart the app to apply changes."
            )
        else:
            QMessageBox.information(self, "No Changes", "Nothing was changed.")
        self.accept()
