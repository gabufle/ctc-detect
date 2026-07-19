"""CLI commands package for CTC-Detect."""

from ctcdetect.cli.commands.run import run_app
from ctcdetect.cli.commands.validate import validate_app
from ctcdetect.cli.commands.batch import batch_app
from ctcdetect.cli.commands.evaluate import evaluate_app
from ctcdetect.cli.commands.info import info_app
from ctcdetect.cli.commands.model import model_app
from ctcdetect.cli.commands.onboard import onboard_app

__all__ = [
    "run_app",
    "validate_app",
    "batch_app",
    "evaluate_app",
    "info_app",
    "model_app",
    "onboard_app",
]