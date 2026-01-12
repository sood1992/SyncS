"""Waveform display widget."""

from typing import Optional
import logging

import numpy as np
from PySide6.QtCore import Qt, QRectF
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
)
from PySide6.QtGui import QPainter, QColor, QPen, QPainterPath

# Use standard PySide6 widgets (qfluentwidgets requires PyQt5)
HAS_FLUENT = False
SubtitleLabel = QLabel

logger = logging.getLogger(__name__)


class WaveformCanvas(QFrame):
    """Canvas for drawing waveform."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._audio: Optional[np.ndarray] = None
        self._sample_rate: int = 8000
        self._downsampled: Optional[np.ndarray] = None

        self.setMinimumHeight(80)
        self.setFrameStyle(QFrame.Shape.StyledPanel)

        # Colors
        self._bg_color = QColor("#1E1E1E")
        self._wave_color = QColor("#4CAF50")
        self._center_color = QColor("#333333")
        self._text_color = QColor("#E0E0E0")

    def set_audio(self, audio: np.ndarray, sample_rate: int):
        """Set audio data to display.

        Args:
            audio: Audio samples as numpy array.
            sample_rate: Sample rate in Hz.
        """
        self._audio = audio
        self._sample_rate = sample_rate
        self._downsample()
        self.update()

    def clear(self):
        """Clear waveform."""
        self._audio = None
        self._downsampled = None
        self.update()

    def _downsample(self):
        """Downsample audio for display."""
        if self._audio is None or len(self._audio) == 0:
            self._downsampled = None
            return

        # Target ~2000 points for smooth display
        target_points = min(2000, len(self._audio))
        chunk_size = max(1, len(self._audio) // target_points)

        # Compute min/max for each chunk (envelope)
        n_chunks = len(self._audio) // chunk_size
        if n_chunks == 0:
            self._downsampled = self._audio.copy()
            return

        audio_trimmed = self._audio[: n_chunks * chunk_size]
        chunks = audio_trimmed.reshape(n_chunks, chunk_size)

        mins = chunks.min(axis=1)
        maxs = chunks.max(axis=1)

        # Interleave min/max for envelope display
        self._downsampled = np.empty(n_chunks * 2)
        self._downsampled[0::2] = mins
        self._downsampled[1::2] = maxs

    def paintEvent(self, event):
        """Paint the waveform."""
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), self._bg_color)

        rect = self.rect()
        width = rect.width()
        height = rect.height()
        center_y = height / 2

        # Center line
        painter.setPen(QPen(self._center_color, 1))
        painter.drawLine(0, int(center_y), width, int(center_y))

        if self._downsampled is None or len(self._downsampled) == 0:
            # Draw placeholder text
            painter.setPen(self._text_color)
            painter.drawText(
                rect,
                Qt.AlignmentFlag.AlignCenter,
                "Select a clip to view waveform"
            )
            return

        # Draw waveform
        self._draw_waveform(painter, width, height, center_y)

        # Draw duration label
        if self._audio is not None:
            duration = len(self._audio) / self._sample_rate
            label = f"{duration:.1f}s"
            painter.setPen(self._text_color)
            painter.drawText(width - 50, 15, label)

    def _draw_waveform(
        self,
        painter: QPainter,
        width: int,
        height: int,
        center_y: float,
    ):
        """Draw the waveform envelope."""
        data = self._downsampled
        n_points = len(data)

        if n_points < 2:
            return

        # Scale factor for amplitude
        amplitude = height * 0.45

        # Create path for filled waveform
        path = QPainterPath()

        # Draw top half
        x_scale = width / (n_points / 2)

        # Start at first max point
        first_max = data[1] * amplitude
        path.moveTo(0, center_y - first_max)

        for i in range(1, n_points // 2):
            x = i * x_scale
            # Max values are at odd indices
            y = center_y - data[i * 2 + 1] * amplitude
            path.lineTo(x, y)

        # Connect to bottom half (min values)
        for i in range(n_points // 2 - 1, -1, -1):
            x = i * x_scale
            # Min values are at even indices
            y = center_y - data[i * 2] * amplitude
            path.lineTo(x, y)

        path.closeSubpath()

        # Fill waveform
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._wave_color)
        painter.drawPath(path)

        # Draw outline
        painter.setPen(QPen(self._wave_color.lighter(120), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)


class WaveformWidget(QWidget):
    """Waveform display widget with label."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QHBoxLayout()

        title = SubtitleLabel("WAVEFORM") if HAS_FLUENT else QLabel("WAVEFORM")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        header.addWidget(title)

        header.addStretch()

        self.info_label = QLabel("")
        header.addWidget(self.info_label)

        layout.addLayout(header)

        # Waveform canvas
        self.canvas = WaveformCanvas()
        layout.addWidget(self.canvas)

    def set_audio(self, audio: np.ndarray, sample_rate: int):
        """Set audio data to display."""
        self.canvas.set_audio(audio, sample_rate)

        duration = len(audio) / sample_rate
        self.info_label.setText(f"{duration:.1f}s @ {sample_rate}Hz")

    def clear(self):
        """Clear waveform."""
        self.canvas.clear()
        self.info_label.setText("")
