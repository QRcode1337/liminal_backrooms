from PyQt6.QtWidgets import QPushButton, QWidget, QGraphicsDropShadowEffect
from PyQt6.QtGui import QColor, QPainter, QPen, QLinearGradient
from PyQt6.QtCore import Qt, QTimer, QRectF
import math
import random
from src.ui.colors import COLORS

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

            # Color shifts based on depth
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

            # Pulsing glow line
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
        from PyQt6.QtGui import QRadialGradient
        gradient = QRadialGradient(self.width() / 2, self.height() / 2,
                                   max(self.width(), self.height()) * 0.7)
        gradient.setColorAt(0, QColor(0, 0, 0, 0))
        gradient.setColorAt(0.7, QColor(0, 0, 0, 0))
        gradient.setColorAt(1, QColor(0, 0, 0, int(255 * self.intensity * 1.5)))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRect(self.rect())
