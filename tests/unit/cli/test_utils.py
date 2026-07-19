"""Tests for shared utilities."""

import pytest

from ctcdetect.cli.utils import validate_input_path, validate_output_path, print_banner


def test_validate_input_path_exists(tmp_path):
    """validate_input_path should return the Path for an existing file."""
    p = tmp_path / "test.txt"
    p.write_text("hello")
    result = validate_input_path(str(p))
    assert result == p


def test_validate_input_path_missing():
    """validate_input_path should SystemExit for a non-existent path."""
    with pytest.raises(SystemExit):
        validate_input_path("/nonexistent/path/file.txt")


def test_validate_output_path(tmp_path):
    """validate_output_path should create parent dirs and return the Path."""
    p = tmp_path / "new_dir" / "output.txt"
    result = validate_output_path(str(p))
    assert result == p
    assert (tmp_path / "new_dir").exists()


def test_print_banner(capsys):
    """print_banner should print without error."""
    print_banner()
    captured = capsys.readouterr()
    assert "CTC-Detect" in captured.out
