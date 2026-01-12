"""Timeline widget for visualizing sync results."""

from typing import Optional
import logging

from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QFrame,
    QSlider,
)
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath

try:
    from qfluentwidgets import SubtitleLabel, Slider
    HAS_FLUENT = True
except ImportError:
    HAS_FLUENT = False
    SubtitleLabel = QLabel
    Slider = QSlider

logger = logging.getLogger(__name__)


class TimelineCanvas(QFrame):
    """Canvas for drawing timeline."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._sync_results: list[dict] = []
        self._scale = 10.0  # Pixels per second
        self._min_start = 0.0
        self._max_end = 0.0

        self.setMinimumHeight(150)
        self.setFrameStyle(QFrame.Shape.StyledPanel)

        # Colors
        self._colors = {
            "synced": QColor("#4CAF50"),
            "unsynced": QColor("#FF9800"),
            "failed": QColor("#F44336"),
            "reference": QColor("#2196F3"),
            "background": QColor("#1E1E1E"),
            "grid": QColor("#333333"),
            "text": QColor("#E0E0E0"),
        }

    def set_sync_results(self, results: list[dict]):
        """Set sync results to display."""
        self._sync_results = results

        if results:
            starts = [r["start"] for r in results]
            ends = [r["start"] + r["duration"] for r in results]
            self._min_start = min(starts)
            self._max_end = max(ends)
        else:
            self._min_start = 0.0
            self._max_end = 0.0

        self._update_size()
        self.update()

    def set_scale(self, pixels_per_second: float):
        """Set zoom scale."""
        self._scale = max(1.0, min(100.0, pixels_per_second))
        self._update_size()
        self.update()

    def _update_size(self):
        """Update widget size based on content."""
        if self._sync_results:
            duration = self._max_end - self._min_start
            width = int(duration * self._scale) + 100
            height = max(150, len(self._get_tracks()) * 40 + 60)
            self.setMinimumWidth(width)
            self.setMinimumHeight(height)

    def _get_tracks(self) -> list[str]:
        """Get unique track names."""
        tracks = []
        for r in self._sync_results:
            track = r.get("track", "Unknown")
            if track not in tracks:
                tracks.append(track)
        return tracks

    def paintEvent(self, event):
        """Paint the timeline."""
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), self._colors["background"])

        if not self._sync_results:
            # Draw placeholder text
            painter.setPen(self._colors["text"])
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No sync results to display"
            )
            return

        # Draw timecode ruler
        self._draw_ruler(painter)

        # Draw tracks
        tracks = self._get_tracks()
        track_height = 30
        y_offset = 30

        for i, track_name in enumerate(tracks):
            y = y_offset + i * (track_height + 5)
            self._draw_track(painter, track_name, y, track_height)

    def _draw_ruler(self, painter: QPainter):
        """Draw timecode ruler at top."""
        painter.setPen(self._colors["grid"])

        duration = self._max_end - self._min_start
        interval = self._get_ruler_interval(duration)

        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(self._colors["text"])

        # Draw ruler marks
        t = 0.0
        while t <= duration:
            x = int(t * self._scale) + 50

            # Tick mark
            painter.drawLine(x, 15, x, 25)

            # Time label
            minutes = int(t // 60)
            seconds = int(t % 60)
            label = f"{minutes:02d}:{seconds:02d}"
            painter.drawText(x - 15, 12, label)

            t += interval

    def _get_ruler_interval(self, duration: float) -> float:
        """Get appropriate ruler interval based on duration."""
        if duration < 60:
            return 5.0
        elif duration < 300:
            return 30.0
        elif duration < 600:
            return 60.0
        else:
            return 300.0

    def _draw_track(self, painter: QPainter, track_name: str, y: int, height: int):
        """Draw a single track."""
        # Track label
        painter.setPen(self._colors["text"])
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(5, y + height // 2 + 4, track_name[:10])

        # Draw clips on this track
        for result in self._sync_results:
            if result.get("track") != track_name:
                continue

            self._draw_clip(painter, result, y, height)

    def _draw_clip(self, painter: QPainter, result: dict, y: int, height: int):
        """Draw a clip rectangle."""
        # Calculate position
        start = result["start"] - self._min_start
        duration = result["duration"]

        x = int(start * self._scale) + 50
        width = max(int(duration * self._scale), 5)

        # Determine color based on status
        status = result.get("status", "unknown")
        if result.get("is_reference"):
            color = self._colors["reference"]
        elif status == "synced":
            color = self._colors["synced"]
        elif status == "unsynced":
            color = self._colors["unsynced"]
        else:
            color = self._colors["failed"]

        # Draw clip rectangle
        rect = QRectF(x, y, width, height)

        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color.darker(120), 1))
        painter.drawRoundedRect(rect, 3, 3)

        # Draw clip name if wide enough
        if width > 40:
            painter.setPen(Qt.GlobalColor.white)
            font = QFont()
            font.setPointSize(8)
            painter.setFont(font)

            name = result.get("path", "")
            if hasattr(name, "stem"):
                name = name.stem
            else:
                name = str(name).split("/")[-1].split("\\")[-1]
                if "." in name:
                    name = name.rsplit(".", 1)[0]

            # Clip text to fit
            metrics = painter.fontMetrics()
            text = metrics.elidedText(name, Qt.TextElideMode.ElideRight, width - 6)
            painter.drawText(rect.adjusted(3, 0, -3, 0), Qt.AlignmentFlag.AlignVCenter, text)


class TimelineWidget(QWidget):
    """Timeline widget with zoom controls."""

    # Signals
    clip_clicked = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header with title and zoom controls
        header = QHBoxLayout()

        title = SubtitleLabel("TIMELINE") if HAS_FLUENT else QLabel("TIMELINE")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        header.addWidget(title)

        header.addStretch()

        # Zoom controls
        zoom_label = QLabel("Zoom:")
        header.addWidget(zoom_label)

        if HAS_FLUENT:
            self.zoom_slider = Slider(Qt.Orientation.Horizontal)
        else:
            self.zoom_slider = QSlider(Qt.Orientation.Horizontal)

        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(1000)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(100)
        header.addWidget(self.zoom_slider)

        layout.addLayout(header)

        # Scroll area for canvas
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.canvas = TimelineCanvas()
        scroll.setWidget(self.canvas)

        layout.addWidget(scroll)

        # Connect signals
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)

    def set_sync_results(self, results: list[dict]):
        """Set sync results to display."""
        self.canvas.set_sync_results(results)

    def _on_zoom_changed(self, value: int):
        """Handle zoom slider change."""
        scale = value / 10.0  # 1-100 pixels per second
        self.canvas.set_scale(scale)

    def clear(self):
        """Clear timeline."""
        self.canvas.set_sync_results([])
