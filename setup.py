#!/usr/bin/env python3
"""Setup script for SyncFox."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text()

setup(
    name="syncfox",
    version="1.0.0",
    description="Audio-Video Sync Tool for multi-camera footage",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SyncFox Team",
    python_requires=">=3.11",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "librosa>=0.10",
        "PySide6>=6.5",
        "PyQt-Fluent-Widgets>=1.4",
        "opentimelineio>=0.15",
        "pyacoustid>=1.2",
        "tqdm>=4.65",
        "appdirs>=1.4",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "pyinstaller>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "syncfox=main:main",
        ],
        "gui_scripts": [
            "syncfox-gui=main:run_gui",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: X11 Applications :: Qt",
        "Environment :: Win32 (MS Windows)",
        "Environment :: MacOS X",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    keywords="audio video sync multicam timeline nle premiere finalcut",
)
