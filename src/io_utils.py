"""Utilities for loading, validating, and writing the project's JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from src.models import FunctionDefinition, TestPrompt


class InputLoadError(Exception):
    """Raised when a JSON input file cannot be loaded, parsed, or validated."""


def _read_json_file(path: Path) -> object:
    """Read and parse a JSON file into plain Python objects.

    Args:
        path: Path to the JSON file to read.

    Returns:
        The parsed JSON content (typically a list or dict).

    Raises:
        InputLoadError: If the file is missing, unreadable, or not valid JSON.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise InputLoadError(f"Input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InputLoadError(
            f"Invalid JSON in {path} (line {exc.lineno}, col {exc.colno}): {exc.msg}"
        ) from exc
    except OSError as exc:
        raise InputLoadError(f"Could not read {path}: {exc}") from exc


def load_function_definitions(path: Path) -> list[FunctionDefinition]:
    """Load and validate the function definitions schema file.

    Args:
        path: Path to `functions_definition.json`.

    Returns:
        A list of validated `FunctionDefinition` objects.

    Raises:
        InputLoadError: If the file is missing, is not a JSON array, or any
            entry fails schema validation.
    """
    raw = _read_json_file(path)

    if not isinstance(raw, list):
        raise InputLoadError(
            f"{path} must contain a JSON array of function definitions."
        )

    try:
        adapter = TypeAdapter(list[FunctionDefinition])
        return adapter.validate_python(raw)
    except ValidationError as exc:
        raise InputLoadError(
            f"{path} does not match the expected function definition schema:\n{exc}"
        ) from exc


def load_test_prompts(path: Path) -> list[TestPrompt]:
    """Load and validate the natural-language test prompts file.

    Args:
        path: Path to `function_calling_tests.json`.

    Returns:
        A list of validated `TestPrompt` objects.

    Raises:
        InputLoadError: If the file is missing, is not a JSON array, or any
            entry fails schema validation.
    """
    raw = _read_json_file(path)

    if not isinstance(raw, list):
        raise InputLoadError(f"{path} must contain a JSON array of prompts.")

    try:
        adapter = TypeAdapter(list[TestPrompt])
        return adapter.validate_python(raw)
    except ValidationError as exc:
        raise InputLoadError(
            f"{path} does not match the expected prompt schema:\n{exc}"
        ) from exc


def write_results(path: Path, results: list[dict[str, object]]) -> None:
    """Write the final function-call results to disk as pretty-printed JSON.

    Args:
        path: Destination path for function_calling_results.json.
        results: A list of plain dicts, one per resolved function call.

    Raises:
        InputLoadError: If the output directory cannot be created or the
            file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise InputLoadError(f"Could not write output file {path}: {exc}") from exc
