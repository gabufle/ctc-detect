"""Onboard command for CTC-Detect CLI.

Interactive onboarding: turn ONE raw dataset download into standardized
data.h5ad + ground_truth.csv by chaining existing prep scripts with
human confirmation at every judgment call.
"""

import subprocess
import sys
from pathlib import Path

import typer

from ctcdetect.cli.utils import console, print_banner, validate_input_path, validate_output_path

onboard_app = typer.Typer(
    name="onboard",
    help=(
        "Interactive onboarding: turn ONE raw dataset into standardized data.h5ad + ground_truth.csv.\n\n"
        "Walks you through 7 confirmation steps (input shape, delimiter, orientation,\n"
        "normalization state, label source, run prepare_external_dataset.py, patient ID pattern).\n"
        "Nothing proceeds without explicit [y/n/e] confirmation at each step."
    ),
    rich_markup_mode="rich",
)


@onboard_app.command()
def onboard(
    input_path: str = typer.Option(
        ...,
        "--input-path", "-i",
        help=(
            "Path to raw dataset file or directory.\n"
            "Can be a single file (.txt, .csv, .tsv, .h5ad, .txt.gz) or a directory\n"
            "of per-cell files (e.g., GSM*.txt.gz) or a .tar.gz archive."
        ),
        rich_help_panel="Input/Output",
    ),
    output_dir: str = typer.Option(
        ...,
        "--output-dir", "-o",
        help=(
            "Output directory for standardized dataset.\n"
            "Will create data.h5ad + ground_truth.csv + patient_id_pattern.json."
        ),
        rich_help_panel="Input/Output",
    ),
    skip_merge: bool = typer.Option(
        False,
        "--skip-merge",
        help="Skip merge_per_cell_files.py even if input is a directory.",
        rich_help_panel="Input/Output",
    ),
):
    """Interactive onboarding: turn ONE raw dataset into standardized data.h5ad + ground_truth.csv.

    Walks you through 7 confirmation steps (input shape, delimiter, orientation,
    normalization state, label source, run prepare_external_dataset.py, patient ID pattern).
    Nothing proceeds without explicit [y/n/e] confirmation at each step.
    """
    print_banner()
    input_p = validate_input_path(input_path, "Input path")
    output_p = validate_output_path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)

    # Run the orchestrator script
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    orchestrator = scripts_dir / "onboard_new_dataset.py"

    if not orchestrator.exists():
        console.print(f"[red]Error:[/red] Orchestrator not found at {orchestrator}")
        raise typer.Exit(1)

    cmd = [
        sys.executable, str(orchestrator),
        "--input-path", str(input_p),
        "--output-dir", str(output_p),
    ]
    if skip_merge:
        cmd.append("--skip-merge")

    console.print("[bold]Launching interactive onboarding...[/bold]")
    console.print(f"Command: {' '.join(cmd)}\n")

    # Run interactively (inherit stdin/stdout/tty)
    result = subprocess.run(cmd)

    if result.returncode == 0:
        console.print(f"\n[green]✓[/green] Onboarding complete. Results in {output_p}")
        console.print("  data.h5ad")
        console.print("  ground_truth.csv")
        console.print("  patient_id_pattern.json")
    else:
        console.print(f"\n[red]✗[/red] Onboarding exited with code {result.returncode}")
        raise typer.Exit(result.returncode)


__all__ = ["onboard_app"]
