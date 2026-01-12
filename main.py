#!/usr/bin/env python3
"""SyncFox - Audio-Video Sync Tool

Main entry point for both CLI and GUI modes.

Usage:
    # Launch GUI
    python main.py

    # CLI mode
    python main.py sync input_files -o output.xml
    python main.py cache stats
    python main.py info media_file.mp4

    # With options
    python main.py sync *.mp4 *.wav -r reference.wav -a thorough -o output.xml
"""

import argparse
import logging
import sys
from pathlib import Path


def setup_logging(level: int = logging.WARNING) -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_gui() -> int:
    """Launch the GUI application.

    Returns:
        Exit code.
    """
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
    except ImportError:
        print("Error: PySide6 is required for GUI mode.")
        print("Install with: pip install PySide6")
        return 1

    # Try to import qfluentwidgets for Fluent Design
    try:
        from qfluentwidgets import setTheme, Theme
        has_fluent = True
    except ImportError:
        has_fluent = False

    from syncfox.ui.main_window import MainWindow

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("SyncFox")
    app.setOrganizationName("SyncFox")
    app.setOrganizationDomain("syncfox.app")

    # Set theme if qfluentwidgets available
    if has_fluent:
        setTheme(Theme.DARK)

    # Enable high DPI scaling
    app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    # Create and show main window
    window = MainWindow()
    window.show()

    return app.exec()


def run_cli(argv: list[str]) -> int:
    """Run CLI mode.

    Args:
        argv: Command line arguments.

    Returns:
        Exit code.
    """
    from syncfox.cli import main as cli_main
    return cli_main(argv)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="syncfox",
        description="SyncFox - Audio-Video Sync Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Launch GUI
    python main.py

    # Sync files via CLI
    python main.py sync video1.mp4 video2.mp4 audio.wav -o output.xml

    # Sync with options
    python main.py sync *.mp4 -r reference.wav -a thorough --fps 30

    # View cache stats
    python main.py cache stats

    # Clear cache
    python main.py cache clear

    # Get media file info
    python main.py info video.mp4
        """,
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
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Force GUI mode even with other arguments",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command")

    # Sync command
    sync_parser = subparsers.add_parser(
        "sync",
        help="Sync media files",
        description="Synchronize multi-camera footage with external audio",
    )
    sync_parser.add_argument(
        "input",
        nargs="+",
        help="Input media files or directories",
    )
    sync_parser.add_argument(
        "-o", "--output",
        help="Output timeline file path",
    )
    sync_parser.add_argument(
        "-r", "--reference",
        help="Reference file for sync (usually the external audio)",
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
        help="Sync accuracy level (default: standard)",
    )
    sync_parser.add_argument(
        "--no-drift-correction",
        dest="drift_correction",
        action="store_false",
        default=True,
        help="Disable automatic drift correction",
    )
    sync_parser.add_argument(
        "-c", "--confidence",
        type=float,
        default=0.6,
        help="Minimum confidence threshold (default: 0.6)",
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

    # Cache command
    cache_parser = subparsers.add_parser(
        "cache",
        help="Manage audio extraction cache",
        description="View, clear, or manage the audio extraction cache",
    )
    cache_parser.add_argument(
        "action",
        choices=["stats", "clear", "cleanup", "verify"],
        help="Cache action to perform",
    )
    cache_parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Max age in days for cleanup (default: 30)",
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show media file information",
        description="Display detailed information about a media file",
    )
    info_parser.add_argument(
        "file",
        help="Media file to inspect",
    )

    return parser


def main() -> int:
    """Main entry point.

    Returns:
        Exit code.
    """
    parser = create_parser()

    # Parse known args to check for command
    args, remaining = parser.parse_known_args()

    # Configure logging
    if args.debug:
        setup_logging(logging.DEBUG)
    elif args.verbose:
        setup_logging(logging.INFO)
    else:
        setup_logging(logging.WARNING)

    # Decide mode
    if args.gui:
        # Explicit GUI request
        return run_gui()

    if args.command:
        # CLI command specified
        # Rebuild argv for CLI parser
        cli_argv = [args.command] + remaining
        if args.verbose:
            cli_argv.insert(0, "-v")
        if args.debug:
            cli_argv.insert(0, "--debug")
        return run_cli(cli_argv)

    # No command - launch GUI
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
