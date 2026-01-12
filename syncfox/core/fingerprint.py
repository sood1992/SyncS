"""Audio fingerprinting using Chromaprint for coarse matching.

Stage 1 of sync: Find approximate match region (~100ms accuracy)
using audio fingerprints before fine-tuning with GCC-PHAT.
"""

import subprocess
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Chromaprint sample rate requirement
CHROMAPRINT_SAMPLE_RATE = 11025


@dataclass
class FingerprintMatch:
    """Result of fingerprint matching."""

    offset_seconds: float
    confidence: float
    matched_duration: float


@dataclass
class Fingerprint:
    """Audio fingerprint data."""

    raw_fingerprint: bytes
    duration: float
    compressed: Optional[str] = None


def get_fpcalc_path() -> str:
    """Get path to fpcalc executable (Chromaprint)."""
    fpcalc = shutil.which("fpcalc")
    if fpcalc is None:
        raise RuntimeError(
            "fpcalc (Chromaprint) not found. "
            "Please install Chromaprint and ensure fpcalc is in your PATH."
        )
    return fpcalc


def has_chromaprint() -> bool:
    """Check if Chromaprint is available."""
    return shutil.which("fpcalc") is not None


class Fingerprinter:
    """Audio fingerprinting using Chromaprint.

    Provides coarse matching capability for finding approximate
    sync points between audio sources.
    """

    def __init__(self, chunk_duration: float = 10.0):
        """Initialize fingerprinter.

        Args:
            chunk_duration: Duration of chunks to fingerprint (seconds).
        """
        self.chunk_duration = chunk_duration
        self._fpcalc = None

    @property
    def fpcalc(self) -> str:
        """Get fpcalc path, caching result."""
        if self._fpcalc is None:
            self._fpcalc = get_fpcalc_path()
        return self._fpcalc

    def fingerprint_file(
        self,
        file_path: Path,
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> Fingerprint:
        """Generate fingerprint from media file.

        Args:
            file_path: Path to media file.
            start_time: Optional start time in seconds.
            duration: Optional duration to fingerprint.

        Returns:
            Fingerprint object.
        """
        cmd = [
            self.fpcalc,
            "-raw",  # Output raw fingerprint
            "-plain",  # Plain output format
        ]

        if duration:
            cmd.extend(["-length", str(int(duration))])

        cmd.append(str(file_path))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"fpcalc failed for {file_path}: {e.stderr}")

        # Parse output
        lines = result.stdout.strip().split("\n")
        fp_duration = 0.0
        raw_fp = b""

        for line in lines:
            if line.startswith("DURATION="):
                fp_duration = float(line.split("=")[1])
            elif line.startswith("FINGERPRINT="):
                # Raw fingerprint is comma-separated integers
                fp_str = line.split("=")[1]
                if fp_str:
                    fp_ints = [int(x) for x in fp_str.split(",")]
                    raw_fp = np.array(fp_ints, dtype=np.int32).tobytes()

        return Fingerprint(
            raw_fingerprint=raw_fp,
            duration=fp_duration,
        )

    def fingerprint_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 8000,
    ) -> Fingerprint:
        """Generate fingerprint from audio array.

        Args:
            audio: Audio data as numpy array.
            sample_rate: Sample rate of audio.

        Returns:
            Fingerprint object.
        """
        # fpcalc needs a file, so we write to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Write audio to temp WAV file
            self._write_wav(audio, sample_rate, temp_path)
            return self.fingerprint_file(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _write_wav(
        self,
        audio: np.ndarray,
        sample_rate: int,
        path: Path,
    ) -> None:
        """Write audio array to WAV file."""
        import wave

        # Convert to 16-bit PCM
        audio_16 = (audio * 32767).astype(np.int16)

        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)
            wav.writeframes(audio_16.tobytes())

    def find_coarse_offset(
        self,
        clip_audio: np.ndarray,
        ref_audio: np.ndarray,
        sample_rate: int = 8000,
        search_window: float = 30.0,
    ) -> Optional[FingerprintMatch]:
        """Find coarse offset between clip and reference using fingerprints.

        This uses a sliding window approach with fingerprint comparison
        to find approximate alignment.

        Args:
            clip_audio: Audio from clip to sync.
            ref_audio: Reference audio to sync against.
            sample_rate: Sample rate of audio.
            search_window: Max offset to search (seconds).

        Returns:
            FingerprintMatch with approximate offset, or None if no match.
        """
        # Use cross-correlation of fingerprints for coarse matching
        # This is a simplified approach - real Chromaprint matching
        # would use the acoustid web service or local comparison

        # Fall back to spectral fingerprinting
        return self._spectral_coarse_match(
            clip_audio, ref_audio, sample_rate, search_window
        )

    def _spectral_coarse_match(
        self,
        clip_audio: np.ndarray,
        ref_audio: np.ndarray,
        sample_rate: int,
        search_window: float,
    ) -> Optional[FingerprintMatch]:
        """Coarse matching using spectral features.

        Computes spectral centroids and uses cross-correlation
        for faster initial alignment before GCC-PHAT refinement.
        """
        from scipy import signal

        # Compute spectral features at lower resolution
        hop_length = int(sample_rate * 0.1)  # 100ms hops
        n_fft = 2048

        def spectral_centroid(audio: np.ndarray) -> np.ndarray:
            """Compute spectral centroid over time."""
            n_frames = (len(audio) - n_fft) // hop_length + 1
            if n_frames <= 0:
                return np.array([])

            centroids = []
            for i in range(n_frames):
                start = i * hop_length
                frame = audio[start : start + n_fft]
                if len(frame) < n_fft:
                    break

                spectrum = np.abs(np.fft.rfft(frame))
                freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)

                # Compute centroid
                centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
                centroids.append(centroid)

            return np.array(centroids)

        # Compute features
        clip_features = spectral_centroid(clip_audio)
        ref_features = spectral_centroid(ref_audio)

        if len(clip_features) < 10 or len(ref_features) < 10:
            return None

        # Normalize features
        clip_features = (clip_features - np.mean(clip_features)) / (
            np.std(clip_features) + 1e-10
        )
        ref_features = (ref_features - np.mean(ref_features)) / (
            np.std(ref_features) + 1e-10
        )

        # Cross-correlate
        correlation = signal.correlate(ref_features, clip_features, mode="full")
        lags = signal.correlation_lags(len(ref_features), len(clip_features), mode="full")

        # Find peak
        max_idx = np.argmax(correlation)
        lag = lags[max_idx]

        # Convert lag to time
        offset_seconds = lag * hop_length / sample_rate

        # Compute confidence from correlation peak
        peak_value = correlation[max_idx]
        mean_value = np.mean(np.abs(correlation))
        confidence = min(peak_value / (mean_value * 10 + 1e-10), 1.0)

        if confidence < 0.2:
            return None

        return FingerprintMatch(
            offset_seconds=offset_seconds,
            confidence=confidence,
            matched_duration=len(clip_features) * hop_length / sample_rate,
        )

    def match_fingerprints(
        self,
        fp1: Fingerprint,
        fp2: Fingerprint,
    ) -> float:
        """Compute similarity between two fingerprints.

        Args:
            fp1: First fingerprint.
            fp2: Second fingerprint.

        Returns:
            Similarity score between 0 and 1.
        """
        if not fp1.raw_fingerprint or not fp2.raw_fingerprint:
            return 0.0

        # Convert to int32 arrays
        arr1 = np.frombuffer(fp1.raw_fingerprint, dtype=np.int32)
        arr2 = np.frombuffer(fp2.raw_fingerprint, dtype=np.int32)

        if len(arr1) == 0 or len(arr2) == 0:
            return 0.0

        # Compare using bit error rate
        min_len = min(len(arr1), len(arr2))
        arr1 = arr1[:min_len]
        arr2 = arr2[:min_len]

        # XOR and count different bits
        xor = np.bitwise_xor(arr1, arr2)
        total_bits = min_len * 32

        # Count set bits
        bit_errors = 0
        for val in xor:
            bit_errors += bin(val & 0xFFFFFFFF).count("1")

        similarity = 1.0 - (bit_errors / total_bits)
        return max(0.0, similarity)


def compute_audio_hash(audio: np.ndarray, sample_rate: int = 8000) -> str:
    """Compute simple hash of audio for caching purposes."""
    import hashlib

    if len(audio) == 0:
        return "empty"

    # Use first and last samples plus length
    head = audio[:min(100, len(audio))]
    tail = audio[-min(100, len(audio)):]
    data = f"{len(audio)}:{head.tobytes()}:{tail.tobytes()}"
    return hashlib.md5(data.encode()).hexdigest()[:16]
