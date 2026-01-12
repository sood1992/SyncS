"""OTIO timeline construction from sync results.

Uses OpenTimelineIO to build timelines that can be exported
to various NLE formats.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import logging

import opentimelineio as otio

logger = logging.getLogger(__name__)

# Default frame rate
DEFAULT_FPS = 24.0


@dataclass
class TimelineSettings:
    """Settings for timeline building."""

    fps: float = DEFAULT_FPS
    name: str = "SyncFox Output"
    include_unsynced: bool = True
    group_by_track: bool = True


class TimelineBuilder:
    """Builds OTIO timelines from sync results."""

    def __init__(self, settings: Optional[TimelineSettings] = None):
        """Initialize builder.

        Args:
            settings: Timeline settings.
        """
        self.settings = settings or TimelineSettings()

    def build(self, sync_results: list[dict]) -> otio.schema.Timeline:
        """Build OTIO timeline from sync results.

        Args:
            sync_results: List of sync result dictionaries with keys:
                - path: Path to media file
                - track: Track name
                - start: Start time in seconds
                - duration: Duration in seconds
                - confidence: Sync confidence
                - status: Sync status

        Returns:
            OpenTimelineIO Timeline object.
        """
        fps = self.settings.fps
        timeline = otio.schema.Timeline(name=self.settings.name)

        # Group results by track
        tracks_data: dict[str, list[dict]] = {}

        for item in sync_results:
            if item["status"] == "failed":
                continue

            if not self.settings.include_unsynced and item["status"] == "unsynced":
                continue

            track_name = item["track"]
            if track_name not in tracks_data:
                tracks_data[track_name] = []
            tracks_data[track_name].append(item)

        # Sort items within each track by start time
        for track_items in tracks_data.values():
            track_items.sort(key=lambda x: x["start"])

        # Find the minimum start time to normalize
        all_starts = [
            item["start"]
            for items in tracks_data.values()
            for item in items
        ]
        min_start = min(all_starts) if all_starts else 0.0

        # Build tracks
        for track_name, items in tracks_data.items():
            track = self._build_track(track_name, items, min_start, fps)
            timeline.tracks.append(track)

        logger.info(
            f"Built timeline with {len(timeline.tracks)} tracks, "
            f"{sum(len(t) for t in timeline.tracks)} clips"
        )

        return timeline

    def _build_track(
        self,
        track_name: str,
        items: list[dict],
        min_start: float,
        fps: float,
    ) -> otio.schema.Track:
        """Build a single track with clips.

        Args:
            track_name: Name of track.
            items: List of clip items for this track.
            min_start: Minimum start time (for normalization).
            fps: Frame rate.

        Returns:
            Track object.
        """
        # Determine track kind based on name
        if "audio" in track_name.lower():
            kind = otio.schema.TrackKind.Audio
        else:
            kind = otio.schema.TrackKind.Video

        track = otio.schema.Track(name=track_name, kind=kind)

        current_end = 0.0

        for item in items:
            # Normalize start time
            start = item["start"] - min_start

            # Add gap if needed
            gap_duration = start - current_end
            if gap_duration > 0.001:  # > 1ms gap
                gap = otio.schema.Gap(
                    duration=otio.opentime.RationalTime(
                        gap_duration * fps, fps
                    )
                )
                track.append(gap)
                current_end = start

            # Create clip
            clip = self._create_clip(item, fps)
            track.append(clip)

            current_end = start + item["duration"]

        return track

    def _create_clip(self, item: dict, fps: float) -> otio.schema.Clip:
        """Create a clip from sync result item.

        Args:
            item: Sync result dictionary.
            fps: Frame rate.

        Returns:
            Clip object.
        """
        path = Path(item["path"])

        # Create media reference
        media_ref = otio.schema.ExternalReference(
            target_url=f"file://{path.resolve()}"
        )

        # Source range - entire clip from beginning
        source_range = otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, fps),
            duration=otio.opentime.RationalTime(item["duration"] * fps, fps),
        )

        clip = otio.schema.Clip(
            name=path.stem,
            media_reference=media_ref,
            source_range=source_range,
        )

        # Add metadata
        clip.metadata["syncfox"] = {
            "confidence": item.get("confidence", 0.0),
            "drift_rate": item.get("drift_rate", 0.0),
            "drift_corrected": item.get("drift_corrected", False),
            "original_path": str(item.get("original_path", path)),
            "status": item.get("status", "unknown"),
            "is_reference": item.get("is_reference", False),
        }

        return clip

    def build_multicam(
        self,
        sync_results: list[dict],
        name: str = "Multicam",
    ) -> otio.schema.Timeline:
        """Build multicam timeline with stacked clips.

        Creates a timeline suitable for multicam editing where
        all synced clips are stacked at the same time.

        Args:
            sync_results: List of sync result dictionaries.
            name: Timeline name.

        Returns:
            Timeline object.
        """
        fps = self.settings.fps
        timeline = otio.schema.Timeline(name=name)

        # Filter to synced clips
        synced = [
            r for r in sync_results
            if r["status"] in ("synced",) or r.get("is_reference")
        ]

        if not synced:
            logger.warning("No synced clips for multicam")
            return timeline

        # Find reference clip
        reference = next((r for r in synced if r.get("is_reference")), synced[0])

        # Normalize to reference start
        ref_start = reference["start"]

        # Create track for each clip
        for item in synced:
            normalized_start = item["start"] - ref_start
            track = self._build_single_clip_track(item, normalized_start, fps)
            timeline.tracks.append(track)

        return timeline

    def _build_single_clip_track(
        self,
        item: dict,
        start: float,
        fps: float,
    ) -> otio.schema.Track:
        """Build track with single clip at specified start."""
        path = Path(item["path"])
        track_name = item.get("track", path.stem)

        if "audio" in track_name.lower():
            kind = otio.schema.TrackKind.Audio
        else:
            kind = otio.schema.TrackKind.Video

        track = otio.schema.Track(name=track_name, kind=kind)

        # Add gap if start > 0
        if start > 0.001:
            gap = otio.schema.Gap(
                duration=otio.opentime.RationalTime(start * fps, fps)
            )
            track.append(gap)

        # Add clip
        clip = self._create_clip(item, fps)
        track.append(clip)

        return track


def build_timeline(
    sync_results: list[dict],
    fps: float = DEFAULT_FPS,
    name: str = "SyncFox Output",
) -> otio.schema.Timeline:
    """Convenience function to build timeline.

    Args:
        sync_results: List of sync result dictionaries.
        fps: Frame rate.
        name: Timeline name.

    Returns:
        Timeline object.
    """
    settings = TimelineSettings(fps=fps, name=name)
    builder = TimelineBuilder(settings)
    return builder.build(sync_results)
