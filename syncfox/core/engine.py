"""Sync pipeline orchestration.

Main engine that coordinates the full synchronization workflow:
1. Audio extraction from all media files
2. Coarse alignment using fingerprints
3. Fine alignment using GCC-PHAT
4. Drift detection and correction
5. Output generation
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Any
import logging
import tempfile

import numpy as np

from .extractor import extract_audio, get_media_duration, get_media_info
from .fingerprint import Fingerprinter
from .correlator import gcc_phat_interp, sync_clip, find_best_alignment, SyncResult
from .drift import detect_drift, analyze_drift, correct_drift_file, DriftAnalysis
from .cache import AudioCache, CachedExtractor

logger = logging.getLogger(__name__)

# Default sample rate for sync processing
DEFAULT_SAMPLE_RATE = 8000


class SyncAccuracy(Enum):
    """Sync accuracy presets."""

    QUICK = "quick"  # Fast, lower accuracy
    STANDARD = "standard"  # Balanced
    THOROUGH = "thorough"  # Higher accuracy
    MAXIMUM = "maximum"  # Maximum accuracy


class ClipStatus(Enum):
    """Status of a clip in the project."""

    PENDING = "pending"
    EXTRACTING = "extracting"
    SYNCING = "syncing"
    SYNCED = "synced"
    FAILED = "failed"
    UNSYNCED = "unsynced"  # Could not find sync point


@dataclass
class MediaClip:
    """Represents a media clip in the project."""

    path: Path
    track_name: str = ""
    is_reference: bool = False
    status: ClipStatus = ClipStatus.PENDING

    # Extracted data
    audio: Optional[np.ndarray] = None
    duration: float = 0.0
    sample_rate: int = DEFAULT_SAMPLE_RATE

    # Sync results
    sync_offset: float = 0.0  # Offset relative to reference
    sync_confidence: float = 0.0
    drift_rate: float = 0.0
    drift_corrected_path: Optional[Path] = None

    # Media info
    media_info: dict = field(default_factory=dict)

    # Error info
    error: Optional[str] = None


@dataclass
class SyncProject:
    """Represents a sync project with all clips."""

    name: str = "Untitled Project"
    clips: list[MediaClip] = field(default_factory=list)
    reference_clip: Optional[MediaClip] = None
    output_dir: Optional[Path] = None

    # Settings
    accuracy: SyncAccuracy = SyncAccuracy.STANDARD
    enable_drift_correction: bool = True
    confidence_threshold: float = 0.6

    # State
    is_synced: bool = False


@dataclass
class SyncEngineSettings:
    """Settings for the sync engine."""

    accuracy: SyncAccuracy = SyncAccuracy.STANDARD
    enable_drift_correction: bool = True
    confidence_threshold: float = 0.6
    sample_rate: int = DEFAULT_SAMPLE_RATE
    cache_enabled: bool = True

    # Accuracy-specific settings
    @property
    def coarse_window(self) -> float:
        """Window size for coarse matching."""
        return {
            SyncAccuracy.QUICK: 60.0,
            SyncAccuracy.STANDARD: 30.0,
            SyncAccuracy.THOROUGH: 20.0,
            SyncAccuracy.MAXIMUM: 10.0,
        }[self.accuracy]

    @property
    def fine_window(self) -> float:
        """Window size for fine matching."""
        return {
            SyncAccuracy.QUICK: 10.0,
            SyncAccuracy.STANDARD: 5.0,
            SyncAccuracy.THOROUGH: 3.0,
            SyncAccuracy.MAXIMUM: 2.0,
        }[self.accuracy]

    @property
    def drift_segment_interval(self) -> int:
        """Interval between drift measurement segments."""
        return {
            SyncAccuracy.QUICK: 600,  # 10 min
            SyncAccuracy.STANDARD: 300,  # 5 min
            SyncAccuracy.THOROUGH: 120,  # 2 min
            SyncAccuracy.MAXIMUM: 60,  # 1 min
        }[self.accuracy]


# Type alias for progress callback
ProgressCallback = Callable[[str, float, Optional[str]], None]


class SyncEngine:
    """Main sync engine orchestrating the full pipeline."""

    def __init__(
        self,
        settings: Optional[SyncEngineSettings] = None,
        cache: Optional[AudioCache] = None,
    ):
        """Initialize sync engine.

        Args:
            settings: Engine settings.
            cache: Audio cache instance.
        """
        self.settings = settings or SyncEngineSettings()
        self.cache = cache or AudioCache() if self.settings.cache_enabled else None
        self.fingerprinter = Fingerprinter()
        self._progress_callback: Optional[ProgressCallback] = None

    def set_progress_callback(self, callback: ProgressCallback) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    def _report_progress(
        self,
        stage: str,
        progress: float,
        message: Optional[str] = None,
    ) -> None:
        """Report progress to callback."""
        if self._progress_callback:
            self._progress_callback(stage, progress, message)
        if message:
            logger.info(f"[{stage}] {progress:.0%} - {message}")

    def add_media(self, project: SyncProject, file_path: Path) -> MediaClip:
        """Add media file to project.

        Args:
            project: Project to add to.
            file_path: Path to media file.

        Returns:
            Created MediaClip.
        """
        file_path = Path(file_path).resolve()

        # Check if already added
        for clip in project.clips:
            if clip.path == file_path:
                return clip

        # Get media info
        media_info = get_media_info(file_path)
        duration = get_media_duration(file_path)

        # Determine track name from file
        track_name = self._suggest_track_name(file_path, media_info)

        clip = MediaClip(
            path=file_path,
            track_name=track_name,
            duration=duration,
            media_info=media_info,
        )

        project.clips.append(clip)
        logger.info(f"Added clip: {file_path.name} ({duration:.1f}s)")

        return clip

    def _suggest_track_name(self, path: Path, media_info: dict) -> str:
        """Suggest track name based on file/metadata."""
        name = path.stem

        # Check for common camera patterns
        lower_name = name.lower()

        if any(x in lower_name for x in ["cam", "camera", "video"]):
            return f"Camera - {name}"

        if any(x in lower_name for x in ["audio", "recorder", "zoom", "tascam"]):
            return f"Audio - {name}"

        # Check media info for streams
        streams = media_info.get("streams", [])
        has_video = any(s.get("codec_type") == "video" for s in streams)
        has_audio = any(s.get("codec_type") == "audio" for s in streams)

        if has_video:
            return f"Camera - {name}"
        elif has_audio:
            return f"Audio - {name}"

        return name

    def set_reference(self, project: SyncProject, clip: MediaClip) -> None:
        """Set reference clip for sync.

        Args:
            project: Project.
            clip: Clip to use as reference.
        """
        # Clear existing reference
        for c in project.clips:
            c.is_reference = False

        clip.is_reference = True
        project.reference_clip = clip
        logger.info(f"Set reference clip: {clip.path.name}")

    def auto_select_reference(self, project: SyncProject) -> Optional[MediaClip]:
        """Automatically select best reference clip.

        Prefers:
        1. Longest audio-only file
        2. Longest file with audio

        Args:
            project: Project.

        Returns:
            Selected reference clip.
        """
        audio_only = []
        with_audio = []

        for clip in project.clips:
            streams = clip.media_info.get("streams", [])
            has_video = any(s.get("codec_type") == "video" for s in streams)
            has_audio = any(s.get("codec_type") == "audio" for s in streams)

            if has_audio and not has_video:
                audio_only.append(clip)
            elif has_audio:
                with_audio.append(clip)

        # Sort by duration
        audio_only.sort(key=lambda c: c.duration, reverse=True)
        with_audio.sort(key=lambda c: c.duration, reverse=True)

        if audio_only:
            reference = audio_only[0]
        elif with_audio:
            reference = with_audio[0]
        else:
            logger.warning("No suitable reference clip found")
            return None

        self.set_reference(project, reference)
        return reference

    def extract_all(self, project: SyncProject) -> None:
        """Extract audio from all clips.

        Args:
            project: Project with clips.
        """
        total = len(project.clips)

        for i, clip in enumerate(project.clips):
            clip.status = ClipStatus.EXTRACTING
            self._report_progress(
                "extract",
                i / total,
                f"Extracting {clip.path.name}",
            )

            try:
                if self.cache:
                    cached = self.cache.get(clip.path, self.settings.sample_rate)
                    if cached is not None:
                        clip.audio = cached
                        clip.sample_rate = self.settings.sample_rate
                        clip.status = ClipStatus.PENDING
                        continue

                audio = extract_audio(clip.path, self.settings.sample_rate)
                clip.audio = audio
                clip.sample_rate = self.settings.sample_rate
                clip.status = ClipStatus.PENDING

                if self.cache:
                    self.cache.put(clip.path, audio, self.settings.sample_rate)

            except Exception as e:
                logger.error(f"Failed to extract {clip.path}: {e}")
                clip.status = ClipStatus.FAILED
                clip.error = str(e)

        self._report_progress("extract", 1.0, "Extraction complete")

    def sync_all(self, project: SyncProject) -> None:
        """Sync all clips to reference.

        Args:
            project: Project with extracted audio.
        """
        if project.reference_clip is None:
            self.auto_select_reference(project)

        if project.reference_clip is None:
            raise ValueError("No reference clip set")

        if project.reference_clip.audio is None:
            raise ValueError("Reference clip audio not extracted")

        ref_audio = project.reference_clip.audio
        clips_to_sync = [c for c in project.clips if c != project.reference_clip]
        total = len(clips_to_sync)

        for i, clip in enumerate(clips_to_sync):
            if clip.audio is None:
                clip.status = ClipStatus.FAILED
                clip.error = "Audio not extracted"
                continue

            clip.status = ClipStatus.SYNCING
            self._report_progress(
                "sync",
                i / total,
                f"Syncing {clip.path.name}",
            )

            try:
                result = self._sync_clip(clip.audio, ref_audio)

                clip.sync_offset = result.offset_seconds
                clip.sync_confidence = result.confidence

                if result.confidence >= self.settings.confidence_threshold:
                    clip.status = ClipStatus.SYNCED
                else:
                    clip.status = ClipStatus.UNSYNCED
                    logger.warning(
                        f"Low confidence sync for {clip.path.name}: "
                        f"{result.confidence:.2f}"
                    )

            except Exception as e:
                logger.error(f"Failed to sync {clip.path}: {e}")
                clip.status = ClipStatus.FAILED
                clip.error = str(e)

        # Set reference offset to 0
        project.reference_clip.sync_offset = 0.0
        project.reference_clip.sync_confidence = 1.0
        project.reference_clip.status = ClipStatus.SYNCED

        self._report_progress("sync", 1.0, "Sync complete")

    def _sync_clip(
        self,
        clip_audio: np.ndarray,
        ref_audio: np.ndarray,
    ) -> SyncResult:
        """Sync a single clip to reference.

        Uses two-stage approach:
        1. Coarse alignment with fingerprints
        2. Fine alignment with GCC-PHAT
        """
        fs = self.settings.sample_rate

        # Stage 1: Coarse alignment
        coarse_match = self.fingerprinter.find_coarse_offset(
            clip_audio,
            ref_audio,
            fs,
            self.settings.coarse_window,
        )

        coarse_offset = coarse_match.offset_seconds if coarse_match else None

        # Stage 2: Fine alignment
        result = find_best_alignment(
            clip_audio,
            ref_audio,
            fs,
            coarse_offset,
            self.settings.fine_window,
        )

        return result

    def detect_drift_all(self, project: SyncProject) -> None:
        """Detect drift for all synced clips.

        Args:
            project: Project with synced clips.
        """
        if not self.settings.enable_drift_correction:
            return

        if project.reference_clip is None or project.reference_clip.audio is None:
            return

        ref_audio = project.reference_clip.audio
        synced_clips = [
            c for c in project.clips
            if c.status == ClipStatus.SYNCED and c != project.reference_clip
        ]
        total = len(synced_clips)

        for i, clip in enumerate(synced_clips):
            if clip.audio is None:
                continue

            self._report_progress(
                "drift",
                i / total,
                f"Analyzing drift: {clip.path.name}",
            )

            # Align audio for drift analysis
            offset_samples = int(clip.sync_offset * self.settings.sample_rate)

            if offset_samples >= 0:
                clip_aligned = clip.audio
                ref_aligned = ref_audio[offset_samples:]
            else:
                clip_aligned = clip.audio[-offset_samples:]
                ref_aligned = ref_audio

            # Trim to same length
            min_len = min(len(clip_aligned), len(ref_aligned))
            clip_aligned = clip_aligned[:min_len]
            ref_aligned = ref_aligned[:min_len]

            # Detect drift
            drift_analysis = analyze_drift(
                clip_aligned,
                ref_aligned,
                self.settings.sample_rate,
                segment_sec=30,
                interval_sec=self.settings.drift_segment_interval,
            )

            clip.drift_rate = drift_analysis.drift_rate

            if drift_analysis.is_significant:
                logger.info(
                    f"Significant drift detected for {clip.path.name}: "
                    f"{drift_analysis.drift_ppm:.1f} ppm "
                    f"({drift_analysis.total_drift_seconds*1000:.1f} ms total)"
                )

        self._report_progress("drift", 1.0, "Drift analysis complete")

    def correct_drift_all(
        self,
        project: SyncProject,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Generate drift-corrected files for clips with significant drift.

        Args:
            project: Project with drift analysis.
            output_dir: Directory for corrected files.
        """
        if not self.settings.enable_drift_correction:
            return

        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="syncfox_"))

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        clips_to_correct = [
            c for c in project.clips
            if abs(c.drift_rate) > 1e-6 and c.status == ClipStatus.SYNCED
        ]
        total = len(clips_to_correct)

        for i, clip in enumerate(clips_to_correct):
            self._report_progress(
                "drift_correct",
                i / total,
                f"Correcting drift: {clip.path.name}",
            )

            output_path = output_dir / f"{clip.path.stem}_drift_corrected{clip.path.suffix}"

            success = correct_drift_file(
                clip.path,
                output_path,
                clip.drift_rate,
            )

            if success:
                clip.drift_corrected_path = output_path
                logger.info(f"Created drift-corrected file: {output_path}")

        self._report_progress("drift_correct", 1.0, "Drift correction complete")

    def run_full_sync(
        self,
        project: SyncProject,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Run full sync pipeline.

        Args:
            project: Project with clips.
            output_dir: Output directory for corrected files.
        """
        logger.info("Starting full sync pipeline")

        # Auto-select reference if not set
        if project.reference_clip is None:
            self.auto_select_reference(project)

        # Extract all audio
        self.extract_all(project)

        # Sync all clips
        self.sync_all(project)

        # Detect drift
        if self.settings.enable_drift_correction:
            self.detect_drift_all(project)
            self.correct_drift_all(project, output_dir)

        project.is_synced = True
        project.output_dir = output_dir

        logger.info("Full sync pipeline complete")

        # Log summary
        synced = sum(1 for c in project.clips if c.status == ClipStatus.SYNCED)
        failed = sum(1 for c in project.clips if c.status == ClipStatus.FAILED)
        unsynced = sum(1 for c in project.clips if c.status == ClipStatus.UNSYNCED)

        logger.info(f"Summary: {synced} synced, {unsynced} unsynced, {failed} failed")

    def get_sync_results(self, project: SyncProject) -> list[dict]:
        """Get sync results for timeline building.

        Args:
            project: Synced project.

        Returns:
            List of sync result dictionaries.
        """
        results = []

        for clip in project.clips:
            result = {
                "path": clip.drift_corrected_path or clip.path,
                "original_path": clip.path,
                "track": clip.track_name,
                "start": clip.sync_offset,
                "duration": clip.duration,
                "confidence": clip.sync_confidence,
                "drift_rate": clip.drift_rate,
                "drift_corrected": clip.drift_corrected_path is not None,
                "status": clip.status.value,
                "is_reference": clip.is_reference,
            }
            results.append(result)

        # Sort by start time
        results.sort(key=lambda r: r["start"])

        return results
