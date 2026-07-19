"""Validate command for CTC-Detect CLI."""


import typer

from ctcdetect.cli.utils import console, print_banner, validate_input_path
from ctcdetect.core.preprocess import validate_input

validate_app = typer.Typer(
    help="Validate input data without running detection."
)


@validate_app.command()
def validate(
    input: str = typer.Option(
        ...,
        "--input", "-i",
        help=(
            "Path to Cell Ranger output directory or .h5ad file.\n"
            "CTC-Detect will check that the data is valid and readable."
        ),
        rich_help_panel="Input/Output",
    ),
):
    """Validate input data without running detection.

    Use this to check that your Cell Ranger output is complete
    and in the expected format before running the full pipeline.
    """
    print_banner()
    input_path = validate_input_path(input, "Input path")

    validate_input(input_path)

    console.print(f"\n[green]✓[/green] Input data looks good: {input_path}")
