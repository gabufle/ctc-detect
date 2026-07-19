"""CTC-Detect CLI application factory."""

import typer
from rich.console import Console

from ctcdetect.cli.commands import (
    run_app,
    validate_app,
    batch_app,
    multi_app,
    evaluate_app,
    onboard_app,
    info_app,
    model_app,
)

console = Console()

# Main app
app = typer.Typer(
    name="ctc-detect",
    help=(
        "Detect circulating tumor cells (CTCs) from single-cell RNA-seq data.\n\n"
        "CTC-Detect uses a fine-tuned Geneformer model to score each cell\n"
        "in your Cell Ranger output for likelihood of being a CTC."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Add sub-apps
app.add_typer(run_app, name="run")
app.add_typer(validate_app, name="validate")
app.add_typer(batch_app, name="batch")
app.add_typer(multi_app, name="multi")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(onboard_app, name="onboard")
app.add_typer(info_app, name="info")
app.add_typer(model_app, name="model")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()