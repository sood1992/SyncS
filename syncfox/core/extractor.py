"""Audio extraction from media files using FFmpeg.

This module handles extracting audio from video/audio files and converting
to 8kHz mono float32 format for efficient sync processing.

8kHz provides 0.125ms resolution - 300x finer than 24fps frames.
6x faster than 48kHz processing.
"""

import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Callable
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Default sample rate for sync processing
DEFAULT_SAMPLE_RATE = 8000


@dataclass
class ExtractionResult:
    """Result of audio extraction."""

    path: Path
    audio: Optional[np.ndarray]
    duration: float  # Duration in seconds
    success: bool
    error: Optional[str] = None


def get_ffmpeg_path() -> str:
    """Get path to FFmpeg executable."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg and ensure it's in your PATH."
        )
    return ffmpeg


def get_ffprobe_path() -> str:
    """Get path to FFprobe executable."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError(
            "FFprobe not found. Please install FFmpeg and ensure it's in your PATH."
        )
    return ffprobe


def get_media_duration(file_path: Path) -> float:
    """Get duration of media file in seconds using FFprobe."""
    ffprobe = get_ffprobe_path()
    cmd = [
        ffprobe,
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning(f"Could not get duration for {file_path}: {e}")
        return 0.0


def get_media_info(file_path: Path) -> dict:
    """Get detailed media info using FFprobe."""
    ffprobe = get_ffprobe_path()
    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]
    try:
        import json
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning(f"Could not get media info for {file_path}: {e}")
        return {}


def extract_audio(
    file_path: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
) -> np.ndarray:
    """Extract audio to mono float32 via FFmpeg.

    Args:
        file_path: Path to media file (video or audio).
        sample_rate: Target sample rate (default 8kHz for sync).
        start_time: Optional start time in seconds.
        duration: Optional duration in seconds.

    Returns:
        Audio data as float32 numpy array.

    Raises:
        RuntimeError: If FFmpeg fails or file cannot be read.
    """
    ffmpeg = get_ffmpeg_path()

    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error"]

    # Add seek if start_time specified
    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])

    cmd.extend(["-i", str(file_path)])

    # Add duration limit if specified
    if duration is not None:
        cmd.extend(["-t", str(duration)])

    # Output format: mono float32 at target sample rate
    cmd.extend([
        "-vn",  # No video
        "-acodec", "pcm_f32le",  # 32-bit float little-endian
        "-ar", str(sample_rate),  # Target sample rate
        "-ac", "1",  # Mono
        "-f", "f32le",  # Raw float32 format
        "-",  # Output to stdout
    ])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg failed for {file_path}: {error_msg}")

    if len(proc.stdout) == 0:
        raise RuntimeError(f"No audio extracted from {file_path}")

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    logger.debug(f"Extracted {len(audio)} samples from {file_path}")
    return audio


def extract_audio_safe(
    file_path: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> ExtractionResult:
    """Extract audio with error handling, returning ExtractionResult.

    This version catches exceptions and returns an ExtractionResult
    with success=False on failure.
    """
    try:
        audio = extract_audio(file_path, sample_rate)
        duration = len(audio) / sample_rate
        return ExtractionResult(
            path=file_path,
            audio=audio,
            duration=duration,
            success=True,
        )
    except Exception as e:
        logger.error(f"Failed to extract audio from {file_path}: {e}")
        return ExtractionResult(
            path=file_path,
            audio=None,
            duration=0.0,
            success=False,
            error=str(e),
        )


def extract_batch(
    files: list[Path],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, Path], None]] = None,
) -> dict[Path, ExtractionResult]:
    """Parallel extraction using all CPU cores.

    Args:
        files: List of media file paths to extract.
        sample_rate: Target sample rate.
        max_workers: Max parallel workers (default: CPU count).
        progress_callback: Called with (completed, total, current_path).

    Returns:
        Dictionary mapping file paths to ExtractionResults.
    """
    results = {}
    total = len(files)
    completed = 0

    # Use ProcessPoolExecutor to bypass GIL
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        # Submit all tasks
        futures = {
            pool.submit(extract_audio_safe, f, sample_rate): f
            for f in files
        }

        # Collect results as they complete
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                result = future.result()
                results[file_path] = result
            except Exception as e:
                logger.error(f"Unexpected error extracting {file_path}: {e}")
                results[file_path] = ExtractionResult(
                    path=file_path,
                    audio=None,
                    duration=0.0,
                    success=False,
                    error=str(e),
                )

            completed += 1
            if progress_callback:
                progress_callback(completed, total, file_path)

    return results


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def compute_rms(audio: np.ndarray, frame_length: int = 2048) -> np.ndarray:
    """Compute RMS energy of audio signal."""
    n_frames = len(audio) // frame_length
    frames = audio[: n_frames * frame_length].reshape(n_frames, frame_length)
    return np.sqrt(np.mean(frames ** 2, axis=1))


def detect_silence(
    audio: np.ndarray,
    threshold_db: float = -50,
    min_duration_sec: float = 0.5,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> list[tuple[float, float]]:
    """Detect silent regions in audio.

    Args:
        audio: Audio signal.
        threshold_db: Silence threshold in dB.
        min_duration_sec: Minimum silence duration to report.
        sample_rate: Audio sample rate.

    Returns:
        List of (start_sec, end_sec) tuples for silent regions.
    """
    frame_length = int(sample_rate * 0.025)  # 25ms frames
    rms = compute_rms(audio, frame_length)

    # Convert threshold to linear
    threshold = 10 ** (threshold_db / 20)

    # Find silent frames
    is_silent = rms < threshold

    # Find contiguous silent regions
    silent_regions = []
    start = None

    for i, silent in enumerate(is_silent):
        if silent and start is None:
            start = i
        elif not silent and start is not None:
            end = i
            duration = (end - start) * frame_length / sample_rate
            if duration >= min_duration_sec:
                silent_regions.append((
                    start * frame_length / sample_rate,
                    end * frame_length / sample_rate,
                ))
            start = None

    # Handle trailing silence
    if start is not None:
        end = len(is_silent)
        duration = (end - start) * frame_length / sample_rate
        if duration >= min_duration_sec:
            silent_regions.append((
                start * frame_length / sample_rate,
                end * frame_length / sample_rate,
            ))

    return silent_regions
