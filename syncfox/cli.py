"""Command-line interface for SyncFox.

Provides CLI access to sync functionality for testing
and batch processing.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .core.engine import (
    SyncEngine,
    SyncEngineSettings,
    SyncProject,
    SyncAccuracy,
    ClipStatus,
)
from .core.cache import AudioCache
from .timeline.builder import TimelineBuilder, TimelineSettings
from .timeline.exporter import TimelineExporter, ExportFormat

logger = logging.getLogger(__name__)

# Supported media extensions
MEDIA_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".mxf", ".m4v",  # Video
    ".wav", ".mp3", ".aiff", ".aif", ".flac", ".m4a",  # Audio
}


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Configure logging."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def find_media_files(paths: list[Path], recursive: bool = False) -> list[Path]:
    """Find all media files in given paths.

    Args:
        paths: List of file or directory paths.
        recursive: Search directories recursively.

    Returns:
        List of media file paths.
    """
    files = []

    for path in paths:
        path = Path(path)

        if path.is_file():
            if path.suffix.lower() in MEDIA_EXTENSIONS:
                files.append(path)
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            for f in path.glob(pattern):
                if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS:
                    files.append(f)

    return sorted(set(files))


class CLIProgressBar:
    """Progress bar for CLI output."""

    def __init__(self):
        self.bars: dict[str, tqdm] = {}
        self.current_stage: Optional[str] = None

    def update(self, stage: str, progress: float, message: Optional[str] = None) -> None:
        """Update progress."""
        if stage != self.current_stage:
            # Close previous bar
            if self.current_stage and self.current_stage in self.bars:
                self.bars[self.current_stage].close()

            # Create new bar
            self.bars[stage] = tqdm(
                total=100,
                desc=stage.capitalize(),
                unit="%",
                leave=True,
            )
            self.current_stage = stage

        if stage in self.bars:
            bar = self.bars[stage]
            bar.n = int(progress * 100)
            if message:
                bar.set_postfix_str(message[:40])
            bar.refresh()

    def close(self) -> None:
        """Close all progress bars."""
        for bar in self.bars.values():
            bar.close()


def cmd_sync(args: argparse.Namespace) -> int:
    """Run sync command.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    # Find media files
    input_paths = [Path(p) for p in args.input]
    files = find_media_files(input_paths, args.recursive)

    if not files:
        print("Error: No media files found")
        return 1

    print(f"Found {len(files)} media files")

    # Setup
    accuracy = SyncAccuracy(args.accuracy)
    settings = SyncEngineSettings(
        accuracy=accuracy,
        enable_drift_correction=args.drift_correction,
        confidence_threshold=args.confidence,
    )

    engine = SyncEngine(settings)
    project = SyncProject(name="CLI Sync")

    # Progress bar
    progress = CLIProgressBar()
    engine.set_progress_callback(progress.update)

    # Add files
    for f in files:
        engine.add_media(project, f)

    # Set reference if specified
    if args.reference:
        ref_path = Path(args.reference).resolve()
        for clip in project.clips:
            if clip.path == ref_path:
                engine.set_reference(project, clip)
                break
        else:
            print(f"Warning: Reference file not in project: {args.reference}")

    # Run sync
    print("\nStarting sync...")

    output_dir = Path(args.output) if args.output else None

    try:
        engine.run_full_sync(project, output_dir)
    except Exception as e:
        progress.close()
        print(f"\nError during sync: {e}")
        return 1

    progress.close()

    # Print results
    print("\n" + "=" * 60)
    print("SYNC RESULTS")
    print("=" * 60)

    for clip in project.clips:
        status_symbol = {
            ClipStatus.SYNCED: "✓",
            ClipStatus.FAILED: "✗",
            ClipStatus.UNSYNCED: "?",
        }.get(clip.status, " ")

        if clip.is_reference:
            status_symbol = "★"

        offset_str = f"{clip.sync_offset:+.3f}s" if clip.status == ClipStatus.SYNCED else "N/A"
        conf_str = f"{clip.sync_confidence:.0%}" if clip.sync_confidence > 0 else "N/A"

        print(f"[{status_symbol}] {clip.path.name}")
        print(f"    Track: {clip.track_name}")
        print(f"    Offset: {offset_str}, Confidence: {conf_str}")

        if clip.drift_rate and abs(clip.drift_rate) > 1e-7:
            ppm = clip.drift_rate * 1_000_000
            print(f"    Drift: {ppm:.1f} ppm")

        if clip.error:
            print(f"    Error: {clip.error}")

    # Export timeline if output specified
    if args.output:
        output_path = Path(args.output)

        # Build timeline
        results = engine.get_sync_results(project)
        timeline_settings = TimelineSettings(
            fps=args.fps,
            name=project.name,
        )
        builder = TimelineBuilder(timeline_settings)
        timeline = builder.build(results)

        # Export
        exporter = TimelineExporter()
        export_format = ExportFormat(args.format)
        exported_path = exporter.export(timeline, output_path, export_format)

        print(f"\nTimeline exported to: {exported_path}")

    # Summary
    synced = sum(1 for c in project.clips if c.status == ClipStatus.SYNCED)
    failed = sum(1 for c in project.clips if c.status == ClipStatus.FAILED)
    unsynced = sum(1 for c in project.clips if c.status == ClipStatus.UNSYNCED)

    print(f"\nSummary: {synced} synced, {unsynced} uncertain, {failed} failed")

    return 0 if failed == 0 else 1


def cmd_cache(args: argparse.Namespace) -> int:
    """Manage cache.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    cache = AudioCache()

    if args.action == "stats":
        stats = cache.get_stats()
        print(f"Cache directory: {stats['cache_dir']}")
        print(f"Total entries: {stats['total_entries']}")
        print(f"Valid entries: {stats['valid_entries']}")
        print(f"Invalid entries: {stats['invalid_entries']}")
        print(f"Total size: {stats['total_size_mb']:.1f} MB")
        print(f"Total duration: {stats['total_duration_hours']:.1f} hours")

    elif args.action == "clear":
        cache.clear()
        print("Cache cleared")

    elif args.action == "cleanup":
        removed = cache.cleanup(args.max_age)
        print(f"Removed {removed} old entries")

    elif args.action == "verify":
        results = cache.verify()
        print(f"Total: {results['total']}")
        print(f"Valid: {results['valid']}")
        print(f"Invalid: {results['invalid']}")
        print(f"Missing files: {results['missing_files']}")
        print(f"Invalid sources: {results['invalid_sources']}")

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show media file info.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    from .core.extractor import get_media_info, get_media_duration

    path = Path(args.file)

    if not path.exists():
        print(f"Error: File not found: {path}")
        return 1

    info = get_media_info(path)
    duration = get_media_duration(path)

    print(f"File: {path.name}")
    print(f"Duration: {duration:.2f}s ({duration/60:.1f} min)")

    if "format" in info:
        fmt = info["format"]
        print(f"Format: {fmt.get('format_long_name', 'Unknown')}")
        print(f"Size: {int(fmt.get('size', 0)) / (1024*1024):.1f} MB")

    if "streams" in info:
        print("\nStreams:")
        for i, stream in enumerate(info["streams"]):
            codec_type = stream.get("codec_type", "unknown")
            codec_name = stream.get("codec_name", "unknown")

            if codec_type == "video":
                width = stream.get("width", "?")
                height = stream.get("height", "?")
                fps = stream.get("r_frame_rate", "?")
                print(f"  [{i}] Video: {codec_name}, {width}x{height}, {fps} fps")
            elif codec_type == "audio":
                channels = stream.get("channels", "?")
                sample_rate = stream.get("sample_rate", "?")
                print(f"  [{i}] Audio: {codec_name}, {sample_rate} Hz, {channels} ch")
            else:
                print(f"  [{i}] {codec_type}: {codec_name}")

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="syncfox",
        description="SyncFox - Audio-Video Sync Tool",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync media files")
    sync_parser.add_argument(
        "input",
        nargs="+",
        help="Input files or directories",
    )
    sync_parser.add_argument(
        "-o", "--output",
        help="Output timeline path",
    )
    sync_parser.add_argument(
        "-r", "--reference",
        help="Reference file for sync",
    )
    sync_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directories recursively",
    )
    sync_parser.add_argument(
        "-a", "--accuracy",
        choices=["quick", "standard", "thorough", "maximum"],
        default="standard",
        help="Sync accuracy (default: standard)",
    )
    sync_parser.add_argument(
        "--no-drift-correction",
        dest="drift_correction",
        action="store_false",
        default=True,
        help="Disable drift correction",
    )
    sync_parser.add_argument(
        "-c", "--confidence",
        type=float,
        default=0.6,
        help="Confidence threshold (default: 0.6)",
    )
    sync_parser.add_argument(
        "-f", "--format",
        choices=["premiere", "fcpx", "resolve", "otio"],
        default="premiere",
        help="Export format (default: premiere)",
    )
    sync_parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Timeline frame rate (default: 24)",
    )
    sync_parser.set_defaults(func=cmd_sync)

    # Cache command
    cache_parser = subparsers.add_parser("cache", help="Manage audio cache")
    cache_parser.add_argument(
        "action",
        choices=["stats", "clear", "cleanup", "verify"],
        help="Cache action",
    )
    cache_parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Max age in days for cleanup (default: 30)",
    )
    cache_parser.set_defaults(func=cmd_cache)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show media file info")
    info_parser.add_argument("file", help="Media file")
    info_parser.set_defaults(func=cmd_info)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point.

    Args:
        argv: Command line arguments.

    Returns:
        Exit code.
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    setup_logging(args.verbose, args.debug)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
