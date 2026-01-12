"""Drift detection and correction.

Detects clock drift between recording devices and generates
corrected audio files. Uses linear regression on sync offsets
measured at multiple points.

Important: Generates NEW files for drift correction rather than
relying on NLE speed adjustment (which causes audio artifacts).
"""

import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Default sample rate
DEFAULT_SAMPLE_RATE = 8000


@dataclass
class DriftAnalysis:
    """Results of drift analysis."""

    drift_rate: float  # Seconds per second (e.g., 0.0001 = 0.01% slower)
    drift_ppm: float  # Parts per million
    r_squared: float  # Fit quality
    total_drift_seconds: float  # Total drift over duration
    measurement_points: int
    is_significant: bool  # Whether drift correction is recommended


def detect_drift(
    audio1: np.ndarray,
    audio2: np.ndarray,
    fs: int = DEFAULT_SAMPLE_RATE,
    segment_sec: int = 60,
    interval_sec: int = 300,
    min_confidence: float = 0.3,
) -> float:
    """Detect clock drift via linear regression on segment offsets.

    Measures sync offset at multiple points throughout the audio
    and fits a linear regression to detect systematic drift.

    Args:
        audio1: First audio signal (e.g., camera audio).
        audio2: Second audio signal (e.g., external recorder).
        fs: Sample rate.
        segment_sec: Duration of each segment to correlate.
        interval_sec: Interval between measurement points.
        min_confidence: Minimum correlation confidence to use point.

    Returns:
        Drift rate (seconds per second).
        E.g., 0.0001 = audio1 is 0.01% slower than audio2.
    """
    from .correlator import gcc_phat

    duration = min(len(audio1), len(audio2)) / fs
    offsets = []
    times = []

    for t in range(0, int(duration), interval_sec):
        s = int(t * fs)
        e = int((t + segment_sec) * fs)

        if e > min(len(audio1), len(audio2)):
            break

        seg1 = audio1[s:e]
        seg2 = audio2[s:e]

        offset, conf = gcc_phat(seg1, seg2, fs)

        if conf > min_confidence:
            offsets.append(offset)
            times.append(t)
            logger.debug(f"Drift measurement at t={t}s: offset={offset:.6f}s, conf={conf:.3f}")

    if len(offsets) >= 3:
        slope, intercept, r_value, p_value, std_err = stats.linregress(times, offsets)
        logger.info(f"Drift analysis: slope={slope:.9f}, r²={r_value**2:.3f}, n={len(offsets)}")
        return slope

    logger.warning(f"Insufficient measurements for drift analysis: {len(offsets)}")
    return 0.0


def analyze_drift(
    audio1: np.ndarray,
    audio2: np.ndarray,
    fs: int = DEFAULT_SAMPLE_RATE,
    segment_sec: int = 30,
    interval_sec: int = 120,
    min_confidence: float = 0.3,
) -> DriftAnalysis:
    """Comprehensive drift analysis with quality metrics.

    Args:
        audio1: First audio signal.
        audio2: Second audio signal.
        fs: Sample rate.
        segment_sec: Duration of each segment to correlate.
        interval_sec: Interval between measurement points.
        min_confidence: Minimum correlation confidence.

    Returns:
        DriftAnalysis with detailed results.
    """
    from .correlator import gcc_phat

    duration = min(len(audio1), len(audio2)) / fs
    offsets = []
    times = []

    for t in range(0, int(duration), interval_sec):
        s = int(t * fs)
        e = int((t + segment_sec) * fs)

        if e > min(len(audio1), len(audio2)):
            break

        seg1 = audio1[s:e]
        seg2 = audio2[s:e]

        offset, conf = gcc_phat(seg1, seg2, fs)

        if conf > min_confidence:
            offsets.append(offset)
            times.append(t)

    if len(offsets) >= 3:
        slope, intercept, r_value, p_value, std_err = stats.linregress(times, offsets)
        r_squared = r_value ** 2
    else:
        slope = 0.0
        r_squared = 0.0

    drift_ppm = slope * 1_000_000
    total_drift = slope * duration

    # Drift is significant if > 1ms over duration and good fit
    is_significant = abs(total_drift) > 0.001 and r_squared > 0.7

    return DriftAnalysis(
        drift_rate=slope,
        drift_ppm=drift_ppm,
        r_squared=r_squared,
        total_drift_seconds=total_drift,
        measurement_points=len(offsets),
        is_significant=is_significant,
    )


def correct_drift(
    audio: np.ndarray,
    drift_rate: float,
    fs: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """Time-stretch audio to correct drift.

    Uses high-quality resampling to correct clock drift.

    Args:
        audio: Audio to correct.
        drift_rate: Drift rate (seconds per second).
        fs: Sample rate.

    Returns:
        Corrected audio array.
    """
    if abs(drift_rate) < 1e-8:
        logger.debug("Drift rate negligible, skipping correction")
        return audio

    import librosa

    # Calculate target sample rate to correct drift
    # If drift_rate > 0, audio is slow, so we need to speed it up
    # by resampling to a slightly higher sample rate
    correction_factor = 1.0 + drift_rate
    target_sr = int(fs * correction_factor)

    logger.info(f"Correcting drift: rate={drift_rate:.9f}, factor={correction_factor:.9f}")

    # Resample to correct drift
    corrected = librosa.resample(
        audio.astype(np.float32),
        orig_sr=fs,
        target_sr=target_sr,
    )

    # Resample back to original sample rate
    corrected = librosa.resample(
        corrected,
        orig_sr=target_sr,
        target_sr=fs,
    )

    return corrected


def correct_drift_file(
    input_path: Path,
    output_path: Path,
    drift_rate: float,
    preserve_video: bool = True,
) -> bool:
    """Correct drift in media file using FFmpeg atempo filter.

    Generates a new file with drift-corrected audio.

    Args:
        input_path: Input media file.
        output_path: Output file path.
        drift_rate: Drift rate (seconds per second).
        preserve_video: Whether to copy video stream.

    Returns:
        True if successful.
    """
    from .extractor import get_ffmpeg_path

    if abs(drift_rate) < 1e-8:
        logger.debug("Drift rate negligible, copying file directly")
        import shutil
        shutil.copy2(input_path, output_path)
        return True

    ffmpeg = get_ffmpeg_path()

    # Calculate tempo adjustment
    # atempo range is 0.5 to 2.0
    # For small corrections, we need to chain multiple atempo filters
    tempo = 1.0 + drift_rate

    # FFmpeg atempo has range 0.5 to 2.0
    if not (0.5 <= tempo <= 2.0):
        logger.warning(f"Drift correction {tempo} outside atempo range, clamping")
        tempo = max(0.5, min(2.0, tempo))

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "warning",
        "-i", str(input_path),
    ]

    if preserve_video:
        cmd.extend(["-c:v", "copy"])

    # Apply atempo filter for audio
    cmd.extend([
        "-af", f"atempo={tempo:.10f}",
        "-c:a", "aac",  # Re-encode audio
        "-b:a", "256k",
        "-y",  # Overwrite output
        str(output_path),
    ])

    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
        logger.info(f"Drift correction complete: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg drift correction failed: {e.stderr.decode()}")
        return False


def correct_drift_audio_file(
    input_path: Path,
    output_path: Path,
    drift_rate: float,
    high_quality: bool = True,
) -> bool:
    """Correct drift in audio-only file using librosa.

    Uses librosa's high-quality resampling for best results.

    Args:
        input_path: Input audio file.
        output_path: Output file path.
        drift_rate: Drift rate (seconds per second).
        high_quality: Use high-quality resampling.

    Returns:
        True if successful.
    """
    import librosa
    import soundfile as sf

    try:
        # Load audio
        audio, sr = librosa.load(input_path, sr=None, mono=False)

        if abs(drift_rate) < 1e-8:
            # No correction needed
            sf.write(output_path, audio.T if audio.ndim > 1 else audio, sr)
            return True

        # Correct each channel
        if audio.ndim == 1:
            corrected = _correct_drift_channel(audio, drift_rate, sr, high_quality)
        else:
            corrected = np.array([
                _correct_drift_channel(ch, drift_rate, sr, high_quality)
                for ch in audio
            ])

        # Write output
        sf.write(
            output_path,
            corrected.T if corrected.ndim > 1 else corrected,
            sr,
        )

        logger.info(f"Audio drift correction complete: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Drift correction failed: {e}")
        return False


def _correct_drift_channel(
    audio: np.ndarray,
    drift_rate: float,
    sr: int,
    high_quality: bool,
) -> np.ndarray:
    """Correct drift for a single audio channel."""
    import librosa

    correction_factor = 1.0 + drift_rate
    target_sr = int(sr * correction_factor)

    res_type = "soxr_hq" if high_quality else "kaiser_fast"

    # Resample to correct drift
    corrected = librosa.resample(
        audio.astype(np.float32),
        orig_sr=sr,
        target_sr=target_sr,
        res_type=res_type,
    )

    # Resample back to original
    corrected = librosa.resample(
        corrected,
        orig_sr=target_sr,
        target_sr=sr,
        res_type=res_type,
    )

    return corrected


def estimate_drift_from_sync_points(
    sync_points: list[Tuple[float, float]],
) -> DriftAnalysis:
    """Estimate drift from a list of sync point measurements.

    Args:
        sync_points: List of (time_seconds, offset_seconds) tuples.

    Returns:
        DriftAnalysis object.
    """
    if len(sync_points) < 3:
        return DriftAnalysis(
            drift_rate=0.0,
            drift_ppm=0.0,
            r_squared=0.0,
            total_drift_seconds=0.0,
            measurement_points=len(sync_points),
            is_significant=False,
        )

    times = [p[0] for p in sync_points]
    offsets = [p[1] for p in sync_points]

    slope, intercept, r_value, p_value, std_err = stats.linregress(times, offsets)

    duration = max(times) - min(times)
    drift_ppm = slope * 1_000_000
    total_drift = slope * duration
    r_squared = r_value ** 2

    is_significant = abs(total_drift) > 0.001 and r_squared > 0.7

    return DriftAnalysis(
        drift_rate=slope,
        drift_ppm=drift_ppm,
        r_squared=r_squared,
        total_drift_seconds=total_drift,
        measurement_points=len(sync_points),
        is_significant=is_significant,
    )


def visualize_drift(
    sync_points: list[Tuple[float, float]],
    drift_analysis: DriftAnalysis,
) -> dict:
    """Generate data for drift visualization.

    Returns data suitable for plotting drift over time.
    """
    times = [p[0] for p in sync_points]
    offsets = [p[1] for p in sync_points]

    # Generate regression line
    if drift_analysis.drift_rate != 0:
        x_fit = np.linspace(min(times), max(times), 100)
        y_fit = drift_analysis.drift_rate * x_fit + (offsets[0] - drift_analysis.drift_rate * times[0])
    else:
        x_fit = times
        y_fit = offsets

    return {
        "times": times,
        "offsets": offsets,
        "fit_times": list(x_fit),
        "fit_offsets": list(y_fit),
        "drift_ppm": drift_analysis.drift_ppm,
        "r_squared": drift_analysis.r_squared,
    }
