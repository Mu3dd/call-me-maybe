"""Entry point for the function-calling CLI tool.

Pipeline: load + validate the two input files -> load the LLM SDK and its
vocabulary -> resolve each prompt into a function call via constrained
decoding -> write the results file.

Run it with:
    uv run python -m src
or with explicit paths:
    uv run python -m src \\
        --functions_definition data/input/functions_definition.json \\
        --input data/input/function_calling_tests.json \\
        --output data/output/function_calling_results.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.io_utils import (
    InputLoadError,
    load_function_definitions,
    load_test_prompts,
    write_results,
)
from src.models import FunctionDefinition, TestPrompt
from src.resolver import resolve_prompt
from src.vocab import Vocabulary, VocabFormatError

DEFAULT_FUNCTIONS_DEFINITION = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional explicit argument list (used by tests). When None,
            arguments are read from `sys.argv`.

    Returns:
        The parsed argparse Namespace.
    """
    parser = argparse.ArgumentParser(
        prog="function-caller",
        description=(
            "Translate natural language prompts into structured function calls."
        ),
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS_DEFINITION,
        help="Path to the functions_definition.json schema file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the function_calling_tests.json prompts file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path where function_calling_results.json will be written.",
    )
    return parser.parse_args(argv)


def _load_sdk_and_vocab() -> tuple[object, Vocabulary]:
    """Import and instantiate the LLM SDK, then load its vocabulary.

    Kept as its own function so import/instantiation failures produce one
    clear error message instead of a raw traceback.

    Returns:
        A tuple of (sdk instance, loaded Vocabulary).

    Raises:
        InputLoadError: If llm_sdk cannot be imported/instantiated, or if
            its vocabulary file cannot be parsed.
    """
    try:
        from llm_sdk import Small_LLM_Model
    except ImportError as exc:
        raise InputLoadError(
            "Could not import llm_sdk.Small_LLM_Model. Make sure the "
            "llm_sdk/ directory sits next to src/ (see README)."
        ) from exc

    try:
        sdk = Small_LLM_Model()
    except Exception as exc:  # noqa: BLE001 - any SDK init failure is fatal here
        raise InputLoadError(f"Could not initialize the LLM SDK: {exc}") from exc

    try:
        vocab = Vocabulary(sdk)
    except VocabFormatError as exc:
        raise InputLoadError(str(exc)) from exc

    return sdk, vocab


def _resolve_all(
    prompts: list[TestPrompt],
    functions: list[FunctionDefinition],
    sdk: object,
    vocab: Vocabulary,
) -> list[dict[str, object]]:
    """Resolve every prompt, isolating failures so one bad prompt doesn't
    stop the whole batch.

    Args:
        prompts: All prompts to resolve.
        functions: All available function definitions.
        sdk: The LLM SDK.
        vocab: The loaded vocabulary.

    Returns:
        A list of plain dicts, one per successfully resolved prompt.
        Prompts that fail to resolve are skipped, with a warning printed
        to stderr, so the output file always stays schema-valid.
    """
    results: list[dict[str, object]] = []
    for prompt in prompts:
        try:
            result = resolve_prompt(prompt, functions, sdk, vocab)
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            print(
                f"Warning: could not resolve prompt {prompt.prompt!r}: {exc}",
                file=sys.stderr,
            )
            continue
        results.append(result.model_dump())
    return results


def main(argv: list[str] | None = None) -> int:
    """Run the CLI tool.

    Args:
        argv: Optional explicit argument list (used by tests).

    Returns:
        Process exit code: 0 on success, 1 on any fatal setup error.
    """
    args = parse_args(argv)

    try:
        functions = load_function_definitions(args.functions_definition)
        prompts = load_test_prompts(args.input)
        sdk, vocab = _load_sdk_and_vocab()
    except InputLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Loaded {len(functions)} function definition(s) "
        f"from {args.functions_definition}"
    )
    print(f"Loaded {len(prompts)} prompt(s) from {args.input}")

    results = _resolve_all(prompts, functions, sdk, vocab)

    try:
        write_results(args.output, results)
    except InputLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(results)} result(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
