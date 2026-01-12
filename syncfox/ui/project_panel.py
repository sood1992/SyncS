"""Project panel for file browser and status display."""

from pathlib import Path
from typing import Optional
import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QMenu,
    QHeaderView,
)
from PySide6.QtGui import QColor, QIcon, QAction

try:
    from qfluentwidgets import TreeWidget, SubtitleLabel
    HAS_FLUENT = True
except ImportError:
    HAS_FLUENT = False
    TreeWidget = QTreeWidget
    SubtitleLabel = QLabel

from ..core.engine import MediaClip, ClipStatus

logger = logging.getLogger(__name__)


class ProjectPanel(QWidget):
    """Project panel showing all media clips and their status."""

    # Signals
    clip_selected = Signal(int)  # Emits clip index
    set_reference_requested = Signal(int)  # Emits clip index

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._clips: list[MediaClip] = []
        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Title
        title = SubtitleLabel("PROJECT") if HAS_FLUENT else QLabel("PROJECT")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Tree widget for clips
        if HAS_FLUENT:
            self.tree = TreeWidget()
        else:
            self.tree = QTreeWidget()

        self.tree.setHeaderLabels(["Name", "Status", "Offset"])
        self.tree.setColumnCount(3)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.tree.setRootIsDecorated(True)

        # Column widths
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self.tree)

        # Connect signals
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)

    def set_clips(self, clips: list[MediaClip]):
        """Set clips to display.

        Args:
            clips: List of MediaClip objects.
        """
        self._clips = clips
        self._update_tree()

    def _update_tree(self):
        """Update tree widget with clips."""
        self.tree.clear()

        # Group clips by track
        tracks: dict[str, list[tuple[int, MediaClip]]] = {}

        for i, clip in enumerate(self._clips):
            track_name = clip.track_name or "Unassigned"
            if track_name not in tracks:
                tracks[track_name] = []
            tracks[track_name].append((i, clip))

        # Create tree items
        for track_name, track_clips in tracks.items():
            # Track item
            track_item = QTreeWidgetItem([track_name, "", ""])
            track_item.setData(0, Qt.ItemDataRole.UserRole, None)

            # Count synced clips
            synced = sum(1 for _, c in track_clips if c.status == ClipStatus.SYNCED)
            total = len(track_clips)

            if synced == total:
                track_item.setText(1, "✓")
                track_item.setForeground(1, QColor("#4CAF50"))
            elif synced > 0:
                track_item.setText(1, f"{synced}/{total}")
            else:
                track_item.setText(1, "")

            # Clip items
            for idx, clip in track_clips:
                clip_item = self._create_clip_item(idx, clip)
                track_item.addChild(clip_item)

            self.tree.addTopLevelItem(track_item)
            track_item.setExpanded(True)

    def _create_clip_item(self, index: int, clip: MediaClip) -> QTreeWidgetItem:
        """Create tree item for a clip."""
        name = clip.path.name

        # Reference indicator
        if clip.is_reference:
            name = f"★ {name}"

        # Status
        status_text, status_color = self._get_status_display(clip.status)

        # Offset
        if clip.status == ClipStatus.SYNCED:
            offset_text = f"{clip.sync_offset:+.3f}s"
        else:
            offset_text = ""

        item = QTreeWidgetItem([name, status_text, offset_text])
        item.setData(0, Qt.ItemDataRole.UserRole, index)

        if status_color:
            item.setForeground(1, QColor(status_color))

        # Tooltip with details
        tooltip = f"Path: {clip.path}\n"
        tooltip += f"Duration: {clip.duration:.1f}s\n"
        tooltip += f"Status: {clip.status.value}\n"

        if clip.sync_confidence > 0:
            tooltip += f"Confidence: {clip.sync_confidence:.0%}\n"

        if clip.drift_rate and abs(clip.drift_rate) > 1e-7:
            ppm = clip.drift_rate * 1_000_000
            tooltip += f"Drift: {ppm:.1f} ppm\n"

        if clip.error:
            tooltip += f"Error: {clip.error}\n"

        item.setToolTip(0, tooltip)

        return item

    def _get_status_display(self, status: ClipStatus) -> tuple[str, Optional[str]]:
        """Get status text and color."""
        mapping = {
            ClipStatus.PENDING: ("⏳", "#9E9E9E"),
            ClipStatus.EXTRACTING: ("⏳", "#2196F3"),
            ClipStatus.SYNCING: ("⏳", "#2196F3"),
            ClipStatus.SYNCED: ("✓", "#4CAF50"),
            ClipStatus.FAILED: ("✗", "#F44336"),
            ClipStatus.UNSYNCED: ("?", "#FF9800"),
        }
        return mapping.get(status, ("", None))

    def _on_selection_changed(self):
        """Handle selection change."""
        items = self.tree.selectedItems()
        if items:
            item = items[0]
            index = item.data(0, Qt.ItemDataRole.UserRole)
            if index is not None:
                self.clip_selected.emit(index)

    def _on_context_menu(self, pos):
        """Show context menu."""
        item = self.tree.itemAt(pos)
        if not item:
            return

        index = item.data(0, Qt.ItemDataRole.UserRole)
        if index is None:
            return

        menu = QMenu(self)

        # Set as reference action
        set_ref_action = QAction("Set as Reference", self)
        set_ref_action.triggered.connect(lambda: self.set_reference_requested.emit(index))
        menu.addAction(set_ref_action)

        # Show in explorer action
        show_action = QAction("Show in Explorer", self)
        show_action.triggered.connect(lambda: self._show_in_explorer(index))
        menu.addAction(show_action)

        menu.exec(self.tree.mapToGlobal(pos))

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle double-click on item."""
        index = item.data(0, Qt.ItemDataRole.UserRole)
        if index is not None:
            # Set as reference on double-click
            self.set_reference_requested.emit(index)

    def _show_in_explorer(self, index: int):
        """Show clip in file explorer."""
        if 0 <= index < len(self._clips):
            clip = self._clips[index]
            import subprocess
            import sys

            if sys.platform == "win32":
                subprocess.run(["explorer", "/select,", str(clip.path)])
            elif sys.platform == "darwin":
                subprocess.run(["open", "-R", str(clip.path)])
            else:
                subprocess.run(["xdg-open", str(clip.path.parent)])

    def get_selected_clip_index(self) -> Optional[int]:
        """Get index of currently selected clip."""
        items = self.tree.selectedItems()
        if items:
            return items[0].data(0, Qt.ItemDataRole.UserRole)
        return None
