"""Main application window for SyncFox.

Uses PySide6 for the GUI. qfluentwidgets is disabled as it requires PyQt5.
"""

from pathlib import Path
from typing import Optional
import logging

from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QFileDialog,
    QMessageBox,
    QApplication,
    QPushButton,
    QToolButton,
    QProgressBar,
)

# qfluentwidgets requires PyQt5 and is incompatible with PySide6
# Use standard PySide6 widgets instead
HAS_FLUENT = False
PrimaryPushButton = QPushButton
PushButton = QPushButton
ToolButton = QToolButton
ProgressBar = QProgressBar

from ..core.engine import (
    SyncEngine,
    SyncEngineSettings,
    SyncProject,
    SyncAccuracy,
    ClipStatus,
)
from ..core.cache import AudioCache
from ..timeline.builder import TimelineBuilder, TimelineSettings
from ..timeline.exporter import TimelineExporter, ExportFormat
from .project_panel import ProjectPanel
from .timeline_widget import TimelineWidget
from .waveform_widget import WaveformWidget
from .settings_dialog import SettingsDialog

logger = logging.getLogger(__name__)

# Supported media extensions
MEDIA_EXTENSIONS = "*.mp4 *.mov *.avi *.mkv *.mxf *.m4v *.wav *.mp3 *.aiff *.aif *.flac *.m4a"


class SyncWorker(QThread):
    """Background worker for sync operations."""

    progress = Signal(str, float, str)  # stage, progress, message
    finished = Signal(bool, str)  # success, message
    error = Signal(str)

    def __init__(
        self,
        engine: SyncEngine,
        project: SyncProject,
        output_dir: Optional[Path] = None,
    ):
        super().__init__()
        self.engine = engine
        self.project = project
        self.output_dir = output_dir

    def run(self):
        """Run sync in background thread."""
        try:
            self.engine.set_progress_callback(
                lambda stage, progress, msg: self.progress.emit(stage, progress, msg or "")
            )

            self.engine.run_full_sync(self.project, self.output_dir)

            synced = sum(1 for c in self.project.clips if c.status == ClipStatus.SYNCED)
            total = len(self.project.clips)

            self.finished.emit(True, f"Synced {synced}/{total} clips")

        except Exception as e:
            logger.exception("Sync failed")
            self.error.emit(str(e))
            self.finished.emit(False, str(e))


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("SyncFox")
        self.setMinimumSize(1200, 800)

        # State
        self.project = SyncProject()
        self.engine: Optional[SyncEngine] = None
        self.sync_worker: Optional[SyncWorker] = None
        self._settings = QSettings("SyncFox", "SyncFox")

        # Setup UI
        self._setup_ui()
        self._setup_connections()
        self._load_settings()

        # Initialize engine
        self._init_engine()

    def _setup_ui(self):
        """Setup UI components."""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Project
        self.project_panel = ProjectPanel()
        splitter.addWidget(self.project_panel)

        # Right side - Waveform and Timeline
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)

        # Waveform
        self.waveform_widget = WaveformWidget()
        right_layout.addWidget(self.waveform_widget, 1)

        # Timeline
        self.timeline_widget = TimelineWidget()
        right_layout.addWidget(self.timeline_widget, 2)

        splitter.addWidget(right_widget)

        # Set splitter sizes
        splitter.setSizes([300, 900])

        layout.addWidget(splitter, 1)

        # Status bar with progress
        self._setup_statusbar()

    def _create_toolbar(self) -> QWidget:
        """Create toolbar widget."""
        toolbar = QWidget()
        toolbar.setFixedHeight(48)

        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(8, 4, 8, 4)

        # Add Media button
        if HAS_FLUENT:
            self.add_media_btn = PushButton("Add Media", self)
            self.add_media_btn.setIcon(FluentIcon.FOLDER_ADD)
        else:
            self.add_media_btn = PushButton("Add Media")
        layout.addWidget(self.add_media_btn)

        # Import XML button
        if HAS_FLUENT:
            self.import_btn = PushButton("Import XML", self)
            self.import_btn.setIcon(FluentIcon.DOWNLOAD)
        else:
            self.import_btn = PushButton("Import XML")
        layout.addWidget(self.import_btn)

        layout.addStretch()

        # Settings button
        if HAS_FLUENT:
            self.settings_btn = ToolButton()
            self.settings_btn.setIcon(FluentIcon.SETTING)
        else:
            self.settings_btn = ToolButton()
            self.settings_btn.setText("Settings")
        layout.addWidget(self.settings_btn)

        layout.addStretch()

        # Sync button
        if HAS_FLUENT:
            self.sync_btn = PrimaryPushButton("SYNC", self)
            self.sync_btn.setIcon(FluentIcon.PLAY)
        else:
            self.sync_btn = PrimaryPushButton("SYNC")
        self.sync_btn.setFixedWidth(120)
        layout.addWidget(self.sync_btn)

        # Export button
        if HAS_FLUENT:
            self.export_btn = PushButton("Export", self)
            self.export_btn.setIcon(FluentIcon.SAVE)
        else:
            self.export_btn = PushButton("Export")
        self.export_btn.setEnabled(False)
        layout.addWidget(self.export_btn)

        return toolbar

    def _setup_statusbar(self):
        """Setup status bar with progress."""
        self.progress_bar = ProgressBar()
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()

        self.statusBar().addPermanentWidget(self.progress_bar, 1)
        self.statusBar().showMessage("Ready")

    def _setup_connections(self):
        """Connect signals and slots."""
        self.add_media_btn.clicked.connect(self._on_add_media)
        self.import_btn.clicked.connect(self._on_import_xml)
        self.settings_btn.clicked.connect(self._on_settings)
        self.sync_btn.clicked.connect(self._on_sync)
        self.export_btn.clicked.connect(self._on_export)

        self.project_panel.clip_selected.connect(self._on_clip_selected)
        self.project_panel.set_reference_requested.connect(self._on_set_reference)

    def _init_engine(self):
        """Initialize sync engine with current settings."""
        settings = SyncEngineSettings(
            accuracy=SyncAccuracy(self._settings.value("accuracy", "standard")),
            enable_drift_correction=self._settings.value("drift_correction", True, type=bool),
            confidence_threshold=self._settings.value("confidence", 0.6, type=float),
        )

        cache = AudioCache()
        self.engine = SyncEngine(settings, cache)

    def _load_settings(self):
        """Load application settings."""
        geometry = self._settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        state = self._settings.value("windowState")
        if state:
            self.restoreState(state)

    def _save_settings(self):
        """Save application settings."""
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("windowState", self.saveState())

    def closeEvent(self, event):
        """Handle window close."""
        self._save_settings()

        if self.sync_worker and self.sync_worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Sync is in progress. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

            self.sync_worker.terminate()
            self.sync_worker.wait()

        event.accept()

    def _on_add_media(self):
        """Handle add media button click."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Media Files",
            "",
            f"Media Files ({MEDIA_EXTENSIONS});;All Files (*.*)",
        )

        if files:
            for file_path in files:
                self.engine.add_media(self.project, Path(file_path))

            self._update_project_view()
            self._show_info(f"Added {len(files)} files")

    def _on_import_xml(self):
        """Handle import XML button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Timeline",
            "",
            "Timeline Files (*.xml *.fcpxml *.otio);;All Files (*.*)",
        )

        if file_path:
            try:
                from ..timeline.exporter import TimelineImporter

                importer = TimelineImporter()
                clips = importer.import_clips(Path(file_path))

                for clip in clips:
                    if clip.get("path"):
                        self.engine.add_media(self.project, Path(clip["path"]))

                self._update_project_view()
                self._show_info(f"Imported {len(clips)} clips from timeline")

            except Exception as e:
                self._show_error(f"Import failed: {e}")

    def _on_settings(self):
        """Handle settings button click."""
        dialog = SettingsDialog(self._settings, self)

        if dialog.exec():
            # Reload settings and reinitialize engine
            self._init_engine()
            self._show_info("Settings updated")

    def _on_sync(self):
        """Handle sync button click."""
        if not self.project.clips:
            self._show_warning("No media files added")
            return

        if self.sync_worker and self.sync_worker.isRunning():
            self._show_warning("Sync already in progress")
            return

        # Disable UI during sync
        self.sync_btn.setEnabled(False)
        self.add_media_btn.setEnabled(False)

        self.progress_bar.setValue(0)
        self.progress_bar.show()

        # Create worker
        self.sync_worker = SyncWorker(self.engine, self.project)
        self.sync_worker.progress.connect(self._on_sync_progress)
        self.sync_worker.finished.connect(self._on_sync_finished)
        self.sync_worker.error.connect(self._on_sync_error)

        self.sync_worker.start()

    def _on_sync_progress(self, stage: str, progress: float, message: str):
        """Handle sync progress update."""
        self.progress_bar.setValue(int(progress * 100))
        self.statusBar().showMessage(f"{stage}: {message}")

    def _on_sync_finished(self, success: bool, message: str):
        """Handle sync completion."""
        self.progress_bar.hide()
        self.sync_btn.setEnabled(True)
        self.add_media_btn.setEnabled(True)

        if success:
            self.export_btn.setEnabled(True)
            self._update_project_view()
            self._update_timeline_view()
            self._show_success(message)
        else:
            self._show_error(message)

    def _on_sync_error(self, error: str):
        """Handle sync error."""
        self._show_error(f"Sync error: {error}")

    def _on_export(self):
        """Handle export button click."""
        if not self.project.is_synced:
            self._show_warning("Please run sync first")
            return

        # Get export path
        file_path, filter_used = QFileDialog.getSaveFileName(
            self,
            "Export Timeline",
            "syncfox_output",
            "Premiere XML (*.xml);;Final Cut Pro X (*.fcpxml);;OpenTimelineIO (*.otio)",
        )

        if not file_path:
            return

        # Determine format from filter
        if "fcpxml" in filter_used.lower():
            export_format = ExportFormat.FCPX
        elif "otio" in filter_used.lower():
            export_format = ExportFormat.OTIO
        else:
            export_format = ExportFormat.PREMIERE

        try:
            # Build timeline
            results = self.engine.get_sync_results(self.project)

            fps = self._settings.value("fps", 24.0, type=float)
            timeline_settings = TimelineSettings(fps=fps, name=self.project.name)

            builder = TimelineBuilder(timeline_settings)
            timeline = builder.build(results)

            # Export
            exporter = TimelineExporter()
            exported_path = exporter.export(timeline, Path(file_path), export_format)

            self._show_success(f"Exported to {exported_path}")

        except Exception as e:
            self._show_error(f"Export failed: {e}")

    def _on_clip_selected(self, clip_index: int):
        """Handle clip selection in project panel."""
        if 0 <= clip_index < len(self.project.clips):
            clip = self.project.clips[clip_index]

            # Update waveform display
            if clip.audio is not None:
                self.waveform_widget.set_audio(clip.audio, clip.sample_rate)
            else:
                self.waveform_widget.clear()

    def _on_set_reference(self, clip_index: int):
        """Handle set reference request."""
        if 0 <= clip_index < len(self.project.clips):
            clip = self.project.clips[clip_index]
            self.engine.set_reference(self.project, clip)
            self._update_project_view()
            self._show_info(f"Set {clip.path.name} as reference")

    def _update_project_view(self):
        """Update project panel with current clips."""
        self.project_panel.set_clips(self.project.clips)

    def _update_timeline_view(self):
        """Update timeline with sync results."""
        if self.project.is_synced:
            results = self.engine.get_sync_results(self.project)
            self.timeline_widget.set_sync_results(results)

    def _show_info(self, message: str):
        """Show info message."""
        if HAS_FLUENT:
            InfoBar.info(
                title="Info",
                content=message,
                parent=self,
                position=InfoBarPosition.TOP,
                duration=3000,
            )
        else:
            self.statusBar().showMessage(message, 3000)

    def _show_success(self, message: str):
        """Show success message."""
        if HAS_FLUENT:
            InfoBar.success(
                title="Success",
                content=message,
                parent=self,
                position=InfoBarPosition.TOP,
                duration=3000,
            )
        else:
            self.statusBar().showMessage(f"Success: {message}", 3000)

    def _show_warning(self, message: str):
        """Show warning message."""
        if HAS_FLUENT:
            InfoBar.warning(
                title="Warning",
                content=message,
                parent=self,
                position=InfoBarPosition.TOP,
                duration=3000,
            )
        else:
            QMessageBox.warning(self, "Warning", message)

    def _show_error(self, message: str):
        """Show error message."""
        if HAS_FLUENT:
            InfoBar.error(
                title="Error",
                content=message,
                parent=self,
                position=InfoBarPosition.TOP,
                duration=5000,
            )
        else:
            QMessageBox.critical(self, "Error", message)
