"""CLI package for CTC-Detect."""

from ctcdetect.cli.app import app as cli_app
from ctcdetect.cli.utils import console, print_banner, validate_input_path, validate_output_path

# Re-export the app and utilities
__all__ = [
    "cli_app",
    "console",
    "print_banner",
    "validate_input_path",
    "validate_output_path",
]
