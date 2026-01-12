"""GCC-PHAT cross-correlation for sample-accurate sync.

Stage 2 of sync: Fine-tune offset to sample accuracy using
Generalized Cross-Correlation with Phase Transform (GCC-PHAT).

PHAT weighting normalizes magnitude and uses only phase information,
making it robust to noise, reverb, and amplitude differences.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import logging

import numpy as np
from scipy import fft
from scipy import signal

logger = logging.getLogger(__name__)

# Default sample rate
DEFAULT_SAMPLE_RATE = 8000


@dataclass
class SyncResult:
    """Result of synchronization."""

    offset_seconds: float
    offset_samples: int
    confidence: float
    method: str = "gcc_phat"


def gcc_phat(
    sig: np.ndarray,
    ref: np.ndarray,
    fs: int = DEFAULT_SAMPLE_RATE,
    max_delay: Optional[float] = None,
) -> Tuple[float, float]:
    """GCC-PHAT cross-correlation.

    Generalized Cross-Correlation with Phase Transform is robust to
    noise and reverberation because it normalizes the magnitude spectrum
    and relies only on phase information for time delay estimation.

    Args:
        sig: Signal to align (e.g., camera audio).
        ref: Reference signal (e.g., external recorder).
        fs: Sample rate in Hz.
        max_delay: Maximum delay to search (seconds). None = full range.

    Returns:
        Tuple of (offset_seconds, confidence).
        Positive offset means sig starts after ref.
    """
    # Ensure float64 for precision
    sig = np.asarray(sig, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)

    # Pad to sum of lengths for full correlation
    n = len(sig) + len(ref)

    # Compute FFTs
    SIG = fft.rfft(sig, n=n)
    REF = fft.rfft(ref, n=n)

    # Cross-power spectrum
    R = SIG * np.conj(REF)

    # PHAT weighting: normalize by magnitude (use only phase)
    R_phat = R / (np.abs(R) + 1e-15)

    # Inverse FFT to get cross-correlation
    cc = np.real(fft.fftshift(fft.irfft(R_phat, n=n)))

    # Limit search range if max_delay specified
    if max_delay is not None:
        max_samples = int(max_delay * fs)
        center = n // 2
        start = max(0, center - max_samples)
        end = min(n, center + max_samples)
        search_cc = cc[start:end]
        peak_idx = np.argmax(np.abs(search_cc)) + start
    else:
        peak_idx = np.argmax(np.abs(cc))

    # Convert to time offset
    offset_samples = peak_idx - n // 2
    offset_seconds = offset_samples / fs

    # Compute confidence as ratio of peak to mean
    peak_value = np.abs(cc[peak_idx])
    mean_value = np.mean(np.abs(cc))
    confidence = peak_value / (mean_value + 1e-10)

    # Normalize confidence to 0-1 range (empirically, good matches > 5)
    confidence = min(confidence / 5.0, 1.0)

    return offset_seconds, confidence


def gcc_phat_interp(
    sig: np.ndarray,
    ref: np.ndarray,
    fs: int = DEFAULT_SAMPLE_RATE,
    max_delay: Optional[float] = None,
) -> Tuple[float, float]:
    """GCC-PHAT with parabolic interpolation for sub-sample accuracy.

    Uses parabolic interpolation around the peak for improved
    precision beyond integer sample resolution.
    """
    # Get integer sample result first
    offset_seconds, confidence = gcc_phat(sig, ref, fs, max_delay)

    if confidence < 0.1:
        return offset_seconds, confidence

    # Refine with parabolic interpolation
    n = len(sig) + len(ref)
    SIG = fft.rfft(sig, n=n)
    REF = fft.rfft(ref, n=n)
    R = SIG * np.conj(REF)
    R_phat = R / (np.abs(R) + 1e-15)
    cc = np.real(fft.fftshift(fft.irfft(R_phat, n=n)))

    peak_idx = np.argmax(np.abs(cc))

    # Parabolic interpolation if not at edges
    if 0 < peak_idx < len(cc) - 1:
        y0 = cc[peak_idx - 1]
        y1 = cc[peak_idx]
        y2 = cc[peak_idx + 1]

        # Parabolic fit: delta = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-10:
            delta = 0.5 * (y0 - y2) / denom
            refined_idx = peak_idx + delta
            offset_samples = refined_idx - n // 2
            offset_seconds = offset_samples / fs

    return offset_seconds, confidence


def sync_clip(
    clip_audio: np.ndarray,
    ref_audio: np.ndarray,
    coarse_offset: float,
    window: float = 5.0,
    fs: int = DEFAULT_SAMPLE_RATE,
) -> Tuple[float, float]:
    """Fine sync within window around coarse offset.

    Uses GCC-PHAT in a narrow window around the coarse offset
    from fingerprint matching for efficient and accurate sync.

    Args:
        clip_audio: Audio from clip to sync.
        ref_audio: Reference audio.
        coarse_offset: Approximate offset from fingerprint matching.
        window: Search window size in seconds.
        fs: Sample rate.

    Returns:
        Tuple of (refined_offset_seconds, confidence).
    """
    margin = int(window * fs)
    center = int(coarse_offset * fs)

    # Extract window from reference
    start = max(0, center - margin)
    end = min(len(ref_audio), center + margin)

    if end <= start:
        logger.warning(f"Invalid window for coarse_offset={coarse_offset}")
        return coarse_offset, 0.0

    ref_window = ref_audio[start:end]

    # Use portion of clip audio that should overlap
    clip_len = min(len(clip_audio), len(ref_window))
    clip_portion = clip_audio[:clip_len]

    # Fine sync within window
    offset, conf = gcc_phat_interp(clip_portion, ref_window, fs)

    # Convert window-relative offset to absolute
    absolute_offset = (start / fs) + offset

    return absolute_offset, conf


def multi_point_sync(
    clip_audio: np.ndarray,
    ref_audio: np.ndarray,
    n_points: int = 5,
    segment_duration: float = 10.0,
    fs: int = DEFAULT_SAMPLE_RATE,
) -> Tuple[float, float, float]:
    """Sync using multiple points along the audio for validation.

    Computes sync offset at multiple points and returns
    the median offset, overall confidence, and detected drift.

    Args:
        clip_audio: Audio from clip to sync.
        ref_audio: Reference audio.
        n_points: Number of points to sample.
        segment_duration: Duration of each segment.
        fs: Sample rate.

    Returns:
        Tuple of (offset_seconds, confidence, drift_rate).
    """
    clip_duration = len(clip_audio) / fs
    ref_duration = len(ref_audio) / fs
    min_duration = min(clip_duration, ref_duration)

    if min_duration < segment_duration * 2:
        # Not enough audio for multi-point, use single sync
        offset, conf = gcc_phat_interp(clip_audio, ref_audio, fs)
        return offset, conf, 0.0

    segment_samples = int(segment_duration * fs)
    offsets = []
    confidences = []
    times = []

    # Sample at evenly spaced points
    step = int((min_duration - segment_duration) / (n_points - 1) * fs)

    for i in range(n_points):
        start = i * step
        end = start + segment_samples

        if end > len(clip_audio) or end > len(ref_audio):
            break

        clip_seg = clip_audio[start:end]
        ref_seg = ref_audio[start:end]

        offset, conf = gcc_phat_interp(clip_seg, ref_seg, fs)

        if conf > 0.3:  # Only use high-confidence matches
            offsets.append(offset)
            confidences.append(conf)
            times.append(start / fs)

    if len(offsets) < 2:
        # Fall back to single sync
        offset, conf = gcc_phat_interp(clip_audio, ref_audio, fs)
        return offset, conf, 0.0

    # Compute median offset
    median_offset = np.median(offsets)
    mean_confidence = np.mean(confidences)

    # Detect drift via linear regression
    if len(offsets) >= 3:
        from scipy import stats

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            times, offsets
        )
        drift_rate = slope
    else:
        drift_rate = 0.0

    return median_offset, mean_confidence, drift_rate


def compute_correlation_peak_quality(
    sig: np.ndarray,
    ref: np.ndarray,
    fs: int = DEFAULT_SAMPLE_RATE,
) -> dict:
    """Compute detailed quality metrics for correlation.

    Returns metrics about the correlation peak quality
    for diagnostic purposes.
    """
    n = len(sig) + len(ref)
    SIG = fft.rfft(sig, n=n)
    REF = fft.rfft(ref, n=n)
    R = SIG * np.conj(REF)
    R_phat = R / (np.abs(R) + 1e-15)
    cc = np.real(fft.fftshift(fft.irfft(R_phat, n=n)))

    peak_idx = np.argmax(np.abs(cc))
    peak_value = np.abs(cc[peak_idx])
    mean_value = np.mean(np.abs(cc))
    std_value = np.std(np.abs(cc))

    # Find secondary peaks
    cc_abs = np.abs(cc)
    peaks, properties = signal.find_peaks(cc_abs, height=peak_value * 0.5)

    return {
        "peak_value": peak_value,
        "mean_value": mean_value,
        "std_value": std_value,
        "peak_to_mean_ratio": peak_value / (mean_value + 1e-10),
        "peak_to_std_ratio": peak_value / (std_value + 1e-10),
        "n_secondary_peaks": len(peaks) - 1,
        "peak_width": _estimate_peak_width(cc_abs, peak_idx),
    }


def _estimate_peak_width(cc: np.ndarray, peak_idx: int) -> int:
    """Estimate width of correlation peak at half maximum."""
    half_max = cc[peak_idx] / 2

    # Search left
    left = peak_idx
    while left > 0 and cc[left] > half_max:
        left -= 1

    # Search right
    right = peak_idx
    while right < len(cc) - 1 and cc[right] > half_max:
        right += 1

    return right - left


def cross_correlate_sliding(
    short_sig: np.ndarray,
    long_sig: np.ndarray,
    fs: int = DEFAULT_SAMPLE_RATE,
    step_seconds: float = 1.0,
) -> list[Tuple[float, float, float]]:
    """Sliding window correlation for finding best match position.

    Slides short_sig along long_sig to find where it best matches.
    Useful when coarse matching fails.

    Args:
        short_sig: Shorter signal to find.
        long_sig: Longer signal to search in.
        fs: Sample rate.
        step_seconds: Step size for sliding.

    Returns:
        List of (position_seconds, offset_seconds, confidence) tuples.
    """
    results = []
    step = int(step_seconds * fs)
    window = len(short_sig)

    for pos in range(0, len(long_sig) - window, step):
        ref_window = long_sig[pos : pos + window]
        offset, conf = gcc_phat(short_sig, ref_window, fs)

        results.append((
            pos / fs,
            offset,
            conf,
        ))

    return results


def find_best_alignment(
    clip_audio: np.ndarray,
    ref_audio: np.ndarray,
    fs: int = DEFAULT_SAMPLE_RATE,
    coarse_offset: Optional[float] = None,
    fine_window: float = 5.0,
) -> SyncResult:
    """Find best alignment between clip and reference.

    Two-stage approach:
    1. If coarse_offset provided, refine within window
    2. Otherwise, search full range

    Args:
        clip_audio: Audio from clip to sync.
        ref_audio: Reference audio.
        fs: Sample rate.
        coarse_offset: Optional coarse offset from fingerprinting.
        fine_window: Window size for fine sync.

    Returns:
        SyncResult with offset and confidence.
    """
    if coarse_offset is not None:
        # Two-stage: refine around coarse offset
        offset, conf = sync_clip(
            clip_audio, ref_audio, coarse_offset, fine_window, fs
        )
        method = "two_stage"
    else:
        # Direct GCC-PHAT on full signals
        offset, conf = gcc_phat_interp(clip_audio, ref_audio, fs)
        method = "gcc_phat_direct"

    return SyncResult(
        offset_seconds=offset,
        offset_samples=int(offset * fs),
        confidence=conf,
        method=method,
    )
