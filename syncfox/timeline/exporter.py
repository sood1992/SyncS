"""Timeline export to NLE formats.

Exports OTIO timelines to:
- Adobe Premiere Pro (FCP7 XML)
- Final Cut Pro X (FCPXML)
- DaVinci Resolve (FCP7 XML)
"""

from pathlib import Path
from enum import Enum
from typing import Optional
import logging

import opentimelineio as otio

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported export formats."""

    PREMIERE = "premiere"  # FCP7 XML
    FCPX = "fcpx"  # FCPXML
    RESOLVE = "resolve"  # FCP7 XML
    OTIO = "otio"  # Native OTIO


# Adapter names for each format
ADAPTER_MAP = {
    ExportFormat.PREMIERE: "fcp_xml",
    ExportFormat.FCPX: "fcpx_xml",
    ExportFormat.RESOLVE: "fcp_xml",  # Resolve uses FCP7 XML
    ExportFormat.OTIO: None,  # Native OTIO
}

# File extensions for each format
EXTENSION_MAP = {
    ExportFormat.PREMIERE: ".xml",
    ExportFormat.FCPX: ".fcpxml",
    ExportFormat.RESOLVE: ".xml",
    ExportFormat.OTIO: ".otio",
}


class TimelineExporter:
    """Exports OTIO timelines to various NLE formats."""

    def export(
        self,
        timeline: otio.schema.Timeline,
        path: Path,
        format: ExportFormat = ExportFormat.PREMIERE,
    ) -> Path:
        """Export timeline to file.

        Args:
            timeline: OTIO timeline to export.
            path: Output path (extension will be adjusted if needed).
            format: Export format.

        Returns:
            Path to exported file.

        Raises:
            RuntimeError: If export fails.
        """
        path = Path(path)

        # Ensure correct extension
        expected_ext = EXTENSION_MAP[format]
        if path.suffix.lower() != expected_ext:
            path = path.with_suffix(expected_ext)

        # Create parent directory
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            adapter = ADAPTER_MAP[format]

            if adapter is None:
                # Native OTIO
                otio.adapters.write_to_file(timeline, str(path))
            else:
                otio.adapters.write_to_file(
                    timeline,
                    str(path),
                    adapter_name=adapter,
                )

            logger.info(f"Exported timeline to {path}")
            return path

        except Exception as e:
            logger.error(f"Failed to export timeline: {e}")
            raise RuntimeError(f"Export failed: {e}") from e

    def export_premiere(
        self,
        timeline: otio.schema.Timeline,
        path: Path,
    ) -> Path:
        """Export to Adobe Premiere Pro (FCP7 XML).

        Args:
            timeline: OTIO timeline.
            path: Output path.

        Returns:
            Path to exported file.
        """
        return self.export(timeline, path, ExportFormat.PREMIERE)

    def export_fcpx(
        self,
        timeline: otio.schema.Timeline,
        path: Path,
    ) -> Path:
        """Export to Final Cut Pro X (FCPXML).

        Args:
            timeline: OTIO timeline.
            path: Output path.

        Returns:
            Path to exported file.
        """
        return self.export(timeline, path, ExportFormat.FCPX)

    def export_resolve(
        self,
        timeline: otio.schema.Timeline,
        path: Path,
    ) -> Path:
        """Export to DaVinci Resolve (FCP7 XML).

        Args:
            timeline: OTIO timeline.
            path: Output path.

        Returns:
            Path to exported file.
        """
        return self.export(timeline, path, ExportFormat.RESOLVE)

    def export_all(
        self,
        timeline: otio.schema.Timeline,
        output_dir: Path,
        base_name: str = "syncfox_output",
    ) -> dict[ExportFormat, Path]:
        """Export to all supported formats.

        Args:
            timeline: OTIO timeline.
            output_dir: Output directory.
            base_name: Base filename without extension.

        Returns:
            Dictionary mapping formats to exported paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for format in ExportFormat:
            ext = EXTENSION_MAP[format]
            path = output_dir / f"{base_name}{ext}"

            try:
                exported_path = self.export(timeline, path, format)
                results[format] = exported_path
            except Exception as e:
                logger.warning(f"Failed to export {format.value}: {e}")

        return results


def export_premiere(timeline: otio.schema.Timeline, path: Path) -> Path:
    """Convenience function to export to Premiere.

    Args:
        timeline: OTIO timeline.
        path: Output path.

    Returns:
        Path to exported file.
    """
    exporter = TimelineExporter()
    return exporter.export_premiere(timeline, path)


def export_fcpx(timeline: otio.schema.Timeline, path: Path) -> Path:
    """Convenience function to export to Final Cut Pro X.

    Args:
        timeline: OTIO timeline.
        path: Output path.

    Returns:
        Path to exported file.
    """
    exporter = TimelineExporter()
    return exporter.export_fcpx(timeline, path)


def export_resolve(timeline: otio.schema.Timeline, path: Path) -> Path:
    """Convenience function to export to DaVinci Resolve.

    Args:
        timeline: OTIO timeline.
        path: Output path.

    Returns:
        Path to exported file.
    """
    exporter = TimelineExporter()
    return exporter.export_resolve(timeline, path)


class ImportError(Exception):
    """Error importing timeline."""

    pass


class TimelineImporter:
    """Imports timelines from various formats."""

    def import_timeline(self, path: Path) -> otio.schema.Timeline:
        """Import timeline from file.

        Automatically detects format based on extension.

        Args:
            path: Path to timeline file.

        Returns:
            OTIO timeline.

        Raises:
            ImportError: If import fails.
        """
        path = Path(path)

        if not path.exists():
            raise ImportError(f"File not found: {path}")

        try:
            timeline = otio.adapters.read_from_file(str(path))

            if isinstance(timeline, otio.schema.Timeline):
                return timeline
            elif hasattr(timeline, "tracks"):
                # Might be a SerializableCollection
                return timeline
            else:
                raise ImportError(f"Unexpected object type: {type(timeline)}")

        except Exception as e:
            logger.error(f"Failed to import timeline: {e}")
            raise ImportError(f"Import failed: {e}") from e

    def import_clips(self, path: Path) -> list[dict]:
        """Import timeline and extract clip information.

        Args:
            path: Path to timeline file.

        Returns:
            List of clip dictionaries.
        """
        timeline = self.import_timeline(path)

        clips = []

        for track in timeline.tracks:
            current_time = 0.0

            for item in track:
                if isinstance(item, otio.schema.Gap):
                    # Accumulate gap time
                    duration = item.duration()
                    if duration:
                        current_time += duration.to_seconds()
                elif isinstance(item, otio.schema.Clip):
                    # Extract clip info
                    media_ref = item.media_reference

                    if isinstance(media_ref, otio.schema.ExternalReference):
                        url = media_ref.target_url
                        if url.startswith("file://"):
                            path = url[7:]
                        else:
                            path = url
                    else:
                        path = None

                    duration = item.duration()
                    dur_seconds = duration.to_seconds() if duration else 0.0

                    clips.append({
                        "name": item.name,
                        "path": path,
                        "track": track.name,
                        "start": current_time,
                        "duration": dur_seconds,
                    })

                    current_time += dur_seconds

        return clips
