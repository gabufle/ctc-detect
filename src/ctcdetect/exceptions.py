"""Exception hierarchy for CTC-Detect.

All custom exceptions inherit from CTCDetectError, which provides
structured error information and suggested remediation.
"""

from typing import Optional, Any
from pathlib import Path


class CTCDetectError(Exception):
    """Base exception for all CTC-Detect errors.

    Attributes:
        message: Human-readable error description.
        hint: Optional remediation suggestion.
        details: Optional dict of structured context (paths, values, etc.).
    """

    def __init__(
        self,
        message: str,
        hint: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details = details or {}

    def __str__(self) -> str:
        parts = [self.message]
        if self.hint:
            parts.append(f"Hint: {self.hint}")
        if self.details:
            parts.append(f"Details: {self.details}")
        return "\n".join(parts)

    def to_rich_text(self) -> str:
        """Format for Rich console output."""
        lines = [f"[red]Error:[/red] {self.message}"]
        if self.hint:
            lines.append(f"[yellow]Hint:[/yellow] {self.hint}")
        if self.details:
            lines.append(f"[dim]Details:[/dim] {self.details}")
        return "\n".join(lines)


class ConfigurationError(CTCDetectError):
    """Raised when configuration is invalid or missing."""

    pass


class ModelError(CTCDetectError):
    """Raised when a model cannot be loaded, validated, or used."""

    def __init__(
        self,
        message: str,
        model_path: Optional[Path] = None,
        hint: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, hint, details)
        self.model_path = model_path
        if model_path:
            self.details.setdefault("model_path", str(model_path))


class InputError(CTCDetectError):
    """Raised when input data is invalid, missing, or unreadable."""

    def __init__(
        self,
        message: str,
        input_path: Optional[Path] = None,
        hint: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, hint, details)
        self.input_path = input_path
        if input_path:
            self.details.setdefault("input_path", str(input_path))


class ValidationError(CTCDetectError):
    """Raised when data validation fails (format, shape, content)."""

    def __init__(
        self,
        message: str,
        failed_check: Optional[str] = None,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None,
        hint: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, hint, details)
        self.failed_check = failed_check
        self.expected = expected
        self.actual = actual
        if failed_check:
            self.details.setdefault("failed_check", failed_check)
        if expected is not None:
            self.details.setdefault("expected", expected)
        if actual is not None:
            self.details.setdefault("actual", actual)


# Alias for clarity in user-facing code
InputValidationError = ValidationError


class GeneMappingError(CTCDetectError):
    """Raised when gene symbol to Ensembl ID mapping fails or is insufficient."""

    def __init__(
        self,
        message: str,
        mapped_count: Optional[int] = None,
        total_count: Optional[int] = None,
        hint: Optional[str] = None,
    ):
        super().__init__(message, hint)
        self.mapped_count = mapped_count
        self.total_count = total_count
        if mapped_count is not None:
            self.details["mapped_genes"] = mapped_count
        if total_count is not None:
            self.details["total_genes"] = total_count
        if mapped_count is not None and total_count is not None:
            self.details["mapping_rate"] = mapped_count / total_count


class TokenizationError(CTCDetectError):
    """Raised when Geneformer tokenization fails."""

    pass


class InferenceError(CTCDetectError):
    """Raised when model inference fails."""

    pass


class OutputError(CTCDetectError):
    """Raised when writing results fails."""

    def __init__(
        self,
        message: str,
        output_path: Optional[Path] = None,
        hint: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, hint, details)
        self.output_path = output_path
        if output_path:
            self.details.setdefault("output_path", str(output_path))


class DependencyError(CTCDetectError):
    """Raised when a required external dependency is missing or incompatible."""

    def __init__(
        self,
        message: str,
        package: Optional[str] = None,
        required_version: Optional[str] = None,
        hint: Optional[str] = None,
    ):
        super().__init__(message, hint)
        self.package = package
        self.required_version = required_version
        if package:
            self.details["package"] = package
        if required_version:
            self.details["required_version"] = required_version


# Convenience function for CLI error handling
def handle_error(err: Exception, console=None) -> int:
    """Convert exception to exit code and optional console output.

    Args:
        err: The exception to handle.
        console: Optional Rich Console for pretty printing.

    Returns:
        Exit code (1 for CTCDetectError, 2 for unexpected errors).
    """
    if isinstance(err, CTCDetectError):
        if console:
            console.print(err.to_rich_text())
        else:
            print(err)
        return 1
    else:
        if console:
            console.print(f"[red]Unexpected error:[/red] {err}")
            console.print("[dim]Please report this as a bug.[/dim]")
        else:
            print(f"Unexpected error: {err}")
        return 2