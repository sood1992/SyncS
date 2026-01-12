"""Settings dialog for SyncFox configuration."""

from typing import Optional
from pathlib import Path
import logging

from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QCheckBox,
    QSlider,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QGroupBox,
    QFormLayout,
    QDialogButtonBox,
    QDoubleSpinBox,
)

try:
    from qfluentwidgets import (
        ComboBox,
        CheckBox,
        Slider,
        LineEdit,
        PushButton,
        SpinBox,
        DoubleSpinBox,
    )
    HAS_FLUENT = True
except ImportError:
    HAS_FLUENT = False
    ComboBox = QComboBox
    CheckBox = QCheckBox
    Slider = QSlider
    LineEdit = QLineEdit
    PushButton = QPushButton
    DoubleSpinBox = QDoubleSpinBox

from ..core.cache import AudioCache

logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """Settings dialog."""

    def __init__(self, settings: QSettings, parent: Optional[QDialog] = None):
        super().__init__(parent)

        self._settings = settings
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)

        # Sync Settings Group
        sync_group = QGroupBox("Sync Settings")
        sync_layout = QFormLayout()

        # Accuracy
        self.accuracy_combo = ComboBox() if HAS_FLUENT else QComboBox()
        self.accuracy_combo.addItems(["Quick", "Standard", "Thorough", "Maximum"])
        sync_layout.addRow("Accuracy:", self.accuracy_combo)

        # Drift Correction
        self.drift_check = CheckBox("Enable drift correction") if HAS_FLUENT else QCheckBox("Enable drift correction")
        sync_layout.addRow("", self.drift_check)

        # Confidence Threshold
        threshold_widget = QHBoxLayout()
        if HAS_FLUENT:
            self.confidence_slider = Slider(Qt.Orientation.Horizontal)
        else:
            self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setMinimum(30)
        self.confidence_slider.setMaximum(90)
        self.confidence_slider.setTickInterval(10)
        threshold_widget.addWidget(self.confidence_slider)

        self.confidence_label = QLabel("0.6")
        self.confidence_label.setFixedWidth(30)
        threshold_widget.addWidget(self.confidence_label)

        sync_layout.addRow("Confidence Threshold:", threshold_widget)

        self.confidence_slider.valueChanged.connect(
            lambda v: self.confidence_label.setText(f"{v/100:.2f}")
        )

        sync_group.setLayout(sync_layout)
        layout.addWidget(sync_group)

        # Timeline Settings Group
        timeline_group = QGroupBox("Timeline Settings")
        timeline_layout = QFormLayout()

        # Chronology
        self.chronology_combo = ComboBox() if HAS_FLUENT else QComboBox()
        self.chronology_combo.addItems(["Auto", "Timestamp", "Filename", "Original Order"])
        timeline_layout.addRow("Clip Ordering:", self.chronology_combo)

        # Export Format
        self.format_combo = ComboBox() if HAS_FLUENT else QComboBox()
        self.format_combo.addItems(["Premiere Pro (XML)", "Final Cut Pro X", "DaVinci Resolve", "OpenTimelineIO"])
        timeline_layout.addRow("Default Export:", self.format_combo)

        # Frame Rate
        if HAS_FLUENT:
            self.fps_spin = DoubleSpinBox()
        else:
            self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 120.0)
        self.fps_spin.setDecimals(3)
        self.fps_spin.setValue(24.0)
        timeline_layout.addRow("Frame Rate:", self.fps_spin)

        timeline_group.setLayout(timeline_layout)
        layout.addWidget(timeline_group)

        # Cache Settings Group
        cache_group = QGroupBox("Cache Settings")
        cache_layout = QFormLayout()

        # Cache Location
        cache_path_layout = QHBoxLayout()
        self.cache_path_edit = LineEdit() if HAS_FLUENT else QLineEdit()
        self.cache_path_edit.setReadOnly(True)
        cache_path_layout.addWidget(self.cache_path_edit)

        browse_btn = PushButton("Browse...") if HAS_FLUENT else QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_cache)
        cache_path_layout.addWidget(browse_btn)

        cache_layout.addRow("Cache Location:", cache_path_layout)

        # Cache stats
        cache = AudioCache()
        stats = cache.get_stats()

        stats_text = f"Size: {stats['total_size_mb']:.1f} MB | "
        stats_text += f"Entries: {stats['valid_entries']} | "
        stats_text += f"Duration: {stats['total_duration_hours']:.1f} hours"

        stats_label = QLabel(stats_text)
        stats_label.setStyleSheet("color: gray;")
        cache_layout.addRow("", stats_label)

        # Clear cache button
        clear_btn = PushButton("Clear Cache") if HAS_FLUENT else QPushButton("Clear Cache")
        clear_btn.clicked.connect(self._on_clear_cache)
        cache_layout.addRow("", clear_btn)

        cache_group.setLayout(cache_layout)
        layout.addWidget(cache_group)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)

        layout.addWidget(button_box)

    def _load_settings(self):
        """Load current settings into UI."""
        # Accuracy
        accuracy = self._settings.value("accuracy", "standard")
        accuracy_map = {"quick": 0, "standard": 1, "thorough": 2, "maximum": 3}
        self.accuracy_combo.setCurrentIndex(accuracy_map.get(accuracy, 1))

        # Drift correction
        drift = self._settings.value("drift_correction", True, type=bool)
        self.drift_check.setChecked(drift)

        # Confidence threshold
        confidence = self._settings.value("confidence", 0.6, type=float)
        self.confidence_slider.setValue(int(confidence * 100))

        # Chronology
        chronology = self._settings.value("chronology", "auto")
        chronology_map = {"auto": 0, "timestamp": 1, "filename": 2, "original": 3}
        self.chronology_combo.setCurrentIndex(chronology_map.get(chronology, 0))

        # Export format
        format_val = self._settings.value("export_format", "premiere")
        format_map = {"premiere": 0, "fcpx": 1, "resolve": 2, "otio": 3}
        self.format_combo.setCurrentIndex(format_map.get(format_val, 0))

        # FPS
        fps = self._settings.value("fps", 24.0, type=float)
        self.fps_spin.setValue(fps)

        # Cache path
        cache = AudioCache()
        self.cache_path_edit.setText(str(cache.cache_dir))

    def _on_accept(self):
        """Save settings and close."""
        # Accuracy
        accuracy_map = {0: "quick", 1: "standard", 2: "thorough", 3: "maximum"}
        self._settings.setValue(
            "accuracy",
            accuracy_map[self.accuracy_combo.currentIndex()]
        )

        # Drift correction
        self._settings.setValue("drift_correction", self.drift_check.isChecked())

        # Confidence threshold
        self._settings.setValue("confidence", self.confidence_slider.value() / 100.0)

        # Chronology
        chronology_map = {0: "auto", 1: "timestamp", 2: "filename", 3: "original"}
        self._settings.setValue(
            "chronology",
            chronology_map[self.chronology_combo.currentIndex()]
        )

        # Export format
        format_map = {0: "premiere", 1: "fcpx", 2: "resolve", 3: "otio"}
        self._settings.setValue(
            "export_format",
            format_map[self.format_combo.currentIndex()]
        )

        # FPS
        self._settings.setValue("fps", self.fps_spin.value())

        self.accept()

    def _on_browse_cache(self):
        """Browse for cache directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Cache Directory",
            self.cache_path_edit.text(),
        )

        if dir_path:
            self.cache_path_edit.setText(dir_path)
            self._settings.setValue("cache_dir", dir_path)

    def _on_clear_cache(self):
        """Clear the audio cache."""
        from PySide6.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self,
            "Clear Cache",
            "Are you sure you want to clear the audio cache?\n\n"
            "This will remove all cached audio extractions.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            cache = AudioCache()
            cache.clear()

            QMessageBox.information(
                self,
                "Cache Cleared",
                "Audio cache has been cleared.",
            )
