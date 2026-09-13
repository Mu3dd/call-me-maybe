"""Resolves a single natural-language prompt into a structured function call.

This is the module that ties `constrained_decoder.py` (the mechanism) to
the actual task (pick a function, fill its arguments) for one prompt at
a time.
"""

from __future__ import annotations

from src.constrained_decoder import choose_function_name, generate_parameters
from src.models import FunctionCallResult, FunctionDefinition, TestPrompt
from src.vocab import Vocabulary


def _encode_prompt(sdk: object, prompt_text: str) -> list[int]:
    """Encode a prompt into a list of token ids, tolerating tensor-like returns.

    The subject specifies `encode(text: str) -> Tensor` for the real SDK,
    but this project only ever needs a plain list of ints to append to.
    This helper converts defensively so it works whether the SDK returns
    a list, a tuple, or a tensor-like object exposing `.tolist()`.

    Args:
        sdk: The LLM SDK.
        prompt_text: The natural-language prompt to encode.

    Returns:
        A plain list of integer token ids.
    """
    encoded = sdk.encode(prompt_text)  # type: ignore[attr-defined]
    if hasattr(encoded, "tolist"):
        return list(encoded.tolist())
    return list(encoded)


def resolve_prompt(
    prompt: TestPrompt,
    functions: list[FunctionDefinition],
    sdk: object,
    vocab: Vocabulary,
) -> FunctionCallResult:
    """Resolve one prompt into a function name and its arguments.

    Two constrained-decoding phases happen here, sharing one running
    context (`context_ids`) so the second phase "sees" the first phase's
    choice as part of its input, the same way a real autoregressive model
    would:

        1. Choose which function to call, constrained to the known names.
        2. Fill in that function's parameters, constrained per-type.

    Args:
        prompt: The natural-language prompt to resolve.
        functions: All available function definitions.
        sdk: The LLM SDK (real or mock) used for logits.
        vocab: The loaded vocabulary for the SDK's tokenizer.

    Returns:
        A validated `FunctionCallResult`.

    Raises:
        ValueError: If `functions` is empty (nothing to choose from).
    """
    if not functions:
        raise ValueError("No function definitions available to choose from.")

    context_text = f"Prompt: {prompt.prompt}\nFunction call: "
    context_ids = _encode_prompt(sdk, context_text)

    function_names = [f.name for f in functions]
    chosen_name = choose_function_name(sdk, vocab, context_ids, function_names)

    chosen_function = next(f for f in functions if f.name == chosen_name)
    _, parameters = generate_parameters(sdk, vocab, context_ids, chosen_function)

    return FunctionCallResult(
        prompt=prompt.prompt,
        name=chosen_name,
        parameters=parameters,
    )
