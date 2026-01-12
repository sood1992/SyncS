"""Disk cache for extracted audio data.

Caches extracted 8kHz audio to disk to enable instant re-sync
without re-extracting audio from source files.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import logging

import numpy as np
import appdirs

logger = logging.getLogger(__name__)

# Cache version - bump when format changes
CACHE_VERSION = 1


@dataclass
class CacheEntry:
    """Metadata for a cached audio file."""

    source_path: str
    source_mtime: float
    source_size: int
    sample_rate: int
    duration: float
    n_samples: int
    cache_time: float
    version: int = CACHE_VERSION


class AudioCache:
    """Disk cache for extracted audio.

    Stores extracted 8kHz audio arrays to disk using a hash-based
    naming scheme. Validates cache entries against source file
    modification time and size.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache.

        Args:
            cache_dir: Cache directory path. Defaults to app data directory.
        """
        if cache_dir is None:
            cache_dir = Path(appdirs.user_cache_dir("SyncFox", "SyncFox"))

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self._metadata: dict[str, CacheEntry] = {}
        self._load_metadata()

        logger.info(f"Audio cache initialized at {self.cache_dir}")

    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)

                for key, entry_data in data.items():
                    self._metadata[key] = CacheEntry(**entry_data)

                logger.debug(f"Loaded {len(self._metadata)} cache entries")
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self._metadata = {}

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            data = {k: asdict(v) for k, v in self._metadata.items()}
            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")

    def _compute_key(self, source_path: Path, sample_rate: int) -> str:
        """Compute cache key for source file."""
        # Include sample rate in key since we might cache at different rates
        key_data = f"{source_path.resolve()}:{sample_rate}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        # Use subdirectories to avoid too many files in one directory
        subdir = self.cache_dir / key[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key}.npy"

    def _is_valid(self, entry: CacheEntry, source_path: Path) -> bool:
        """Check if cache entry is still valid."""
        if entry.version != CACHE_VERSION:
            return False

        if not source_path.exists():
            return False

        stat = source_path.stat()

        # Check modification time and size
        if abs(stat.st_mtime - entry.source_mtime) > 1.0:  # 1 second tolerance
            return False

        if stat.st_size != entry.source_size:
            return False

        return True

    def get(
        self,
        source_path: Path,
        sample_rate: int = 8000,
    ) -> Optional[np.ndarray]:
        """Get cached audio for source file.

        Args:
            source_path: Path to source media file.
            sample_rate: Sample rate of cached audio.

        Returns:
            Cached audio array, or None if not cached/invalid.
        """
        source_path = Path(source_path).resolve()
        key = self._compute_key(source_path, sample_rate)

        if key not in self._metadata:
            return None

        entry = self._metadata[key]

        if not self._is_valid(entry, source_path):
            logger.debug(f"Cache invalid for {source_path}")
            self._remove(key)
            return None

        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            self._metadata.pop(key, None)
            return None

        try:
            audio = np.load(cache_path)
            logger.debug(f"Cache hit for {source_path}")
            return audio
        except Exception as e:
            logger.warning(f"Failed to load cached audio: {e}")
            self._remove(key)
            return None

    def put(
        self,
        source_path: Path,
        audio: np.ndarray,
        sample_rate: int = 8000,
    ) -> None:
        """Store audio in cache.

        Args:
            source_path: Path to source media file.
            audio: Extracted audio array.
            sample_rate: Sample rate of audio.
        """
        source_path = Path(source_path).resolve()
        key = self._compute_key(source_path, sample_rate)
        cache_path = self._get_cache_path(key)

        try:
            stat = source_path.stat()

            # Save audio
            np.save(cache_path, audio)

            # Update metadata
            self._metadata[key] = CacheEntry(
                source_path=str(source_path),
                source_mtime=stat.st_mtime,
                source_size=stat.st_size,
                sample_rate=sample_rate,
                duration=len(audio) / sample_rate,
                n_samples=len(audio),
                cache_time=time.time(),
            )

            self._save_metadata()
            logger.debug(f"Cached audio for {source_path}")

        except Exception as e:
            logger.warning(f"Failed to cache audio: {e}")

    def _remove(self, key: str) -> None:
        """Remove cache entry."""
        self._metadata.pop(key, None)
        cache_path = self._get_cache_path(key)
        cache_path.unlink(missing_ok=True)
        self._save_metadata()

    def remove(self, source_path: Path, sample_rate: int = 8000) -> None:
        """Remove cache entry for source file.

        Args:
            source_path: Path to source media file.
            sample_rate: Sample rate of cached audio.
        """
        source_path = Path(source_path).resolve()
        key = self._compute_key(source_path, sample_rate)
        self._remove(key)

    def clear(self) -> None:
        """Clear all cached data."""
        for key in list(self._metadata.keys()):
            self._remove(key)

        # Remove any orphaned files
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir() and len(subdir.name) == 2:
                for f in subdir.iterdir():
                    if f.suffix == ".npy":
                        f.unlink()
                subdir.rmdir()

        self._metadata = {}
        self._save_metadata()
        logger.info("Cache cleared")

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        total_size = 0
        valid_entries = 0
        invalid_entries = 0
        total_duration = 0.0

        for key, entry in self._metadata.items():
            cache_path = self._get_cache_path(key)

            if cache_path.exists():
                total_size += cache_path.stat().st_size
                source_path = Path(entry.source_path)

                if self._is_valid(entry, source_path):
                    valid_entries += 1
                    total_duration += entry.duration
                else:
                    invalid_entries += 1
            else:
                invalid_entries += 1

        return {
            "cache_dir": str(self.cache_dir),
            "total_entries": len(self._metadata),
            "valid_entries": valid_entries,
            "invalid_entries": invalid_entries,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "total_duration_seconds": total_duration,
            "total_duration_hours": total_duration / 3600,
        }

    def cleanup(self, max_age_days: int = 30) -> int:
        """Remove old cache entries.

        Args:
            max_age_days: Maximum age of entries to keep.

        Returns:
            Number of entries removed.
        """
        cutoff = time.time() - (max_age_days * 24 * 3600)
        removed = 0

        for key, entry in list(self._metadata.items()):
            if entry.cache_time < cutoff:
                self._remove(key)
                removed += 1

        logger.info(f"Cleaned up {removed} old cache entries")
        return removed

    def verify(self) -> dict:
        """Verify cache integrity.

        Returns:
            Dictionary with verification results.
        """
        results = {
            "total": len(self._metadata),
            "valid": 0,
            "invalid": 0,
            "missing_files": 0,
            "invalid_sources": 0,
        }

        for key, entry in list(self._metadata.items()):
            cache_path = self._get_cache_path(key)
            source_path = Path(entry.source_path)

            if not cache_path.exists():
                results["missing_files"] += 1
                results["invalid"] += 1
            elif not self._is_valid(entry, source_path):
                results["invalid_sources"] += 1
                results["invalid"] += 1
            else:
                results["valid"] += 1

        return results


def get_default_cache() -> AudioCache:
    """Get default cache instance."""
    return AudioCache()


class CachedExtractor:
    """Audio extractor with caching support.

    Wraps the standard extractor with automatic caching.
    """

    def __init__(self, cache: Optional[AudioCache] = None):
        """Initialize cached extractor.

        Args:
            cache: Cache instance. Creates default if not provided.
        """
        self.cache = cache or AudioCache()

    def extract(
        self,
        file_path: Path,
        sample_rate: int = 8000,
        force: bool = False,
    ) -> np.ndarray:
        """Extract audio with caching.

        Args:
            file_path: Path to media file.
            sample_rate: Target sample rate.
            force: Force re-extraction even if cached.

        Returns:
            Audio array.
        """
        from .extractor import extract_audio

        file_path = Path(file_path)

        if not force:
            cached = self.cache.get(file_path, sample_rate)
            if cached is not None:
                return cached

        # Extract and cache
        audio = extract_audio(file_path, sample_rate)
        self.cache.put(file_path, audio, sample_rate)

        return audio

    def extract_batch(
        self,
        files: list[Path],
        sample_rate: int = 8000,
        force: bool = False,
    ) -> dict[Path, np.ndarray]:
        """Extract multiple files with caching.

        Args:
            files: List of media file paths.
            sample_rate: Target sample rate.
            force: Force re-extraction.

        Returns:
            Dictionary mapping paths to audio arrays.
        """
        results = {}

        # Check cache first
        to_extract = []
        for f in files:
            if not force:
                cached = self.cache.get(f, sample_rate)
                if cached is not None:
                    results[f] = cached
                    continue
            to_extract.append(f)

        if to_extract:
            from .extractor import extract_batch

            # Extract uncached files
            extracted = extract_batch(to_extract, sample_rate)

            for path, result in extracted.items():
                if result.success and result.audio is not None:
                    results[path] = result.audio
                    self.cache.put(path, result.audio, sample_rate)

        return results
