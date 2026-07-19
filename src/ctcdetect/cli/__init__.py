"""CLI package for CTC-Detect."""

from ctcdetect.cli.app import app
from ctcdetect.cli.utils import console, print_banner, validate_input_path, validate_output_path

__all__ = [
    "app",
    "validate_input_path",
    "validate_output_path",
    "print_banner",
    "console",
]
