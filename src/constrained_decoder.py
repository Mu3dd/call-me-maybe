"""The constrained decoding engine.

Everything in this file builds on ONE primitive: `constrained_step`. At
every single generation step -- whether we're forcing a fixed character
like `{`, choosing which function to call, or filling in a digit of a
number -- we go through the same path:

    1. Ask the model for logits over the next token (`get_logits_from_input_ids`).
    2. Compute the set of tokens that are legal right now.
    3. Set every other token's logit to -infinity.
    4. Take the highest-scoring token that's left.

The only thing that changes between "forced" steps and "real decision"
steps is the SIZE of the allowed set: 1 token for something fully
determined by the JSON grammar (like a closing brace), or several tokens
when the model is actually choosing something (a function name, a digit,
a letter). This is a deliberate simplification/design choice: rather than
writing separate code paths for "literal" text and "free choice" text,
everything is expressed as "the set of tokens allowed at this position",
which keeps the logic in one place and easy to reason about (and to
defend/explain).

This module has NO knowledge of files, prompts, or the CLI -- it only
knows how to turn "a function's parameter schema" into "a syntactically
and semantically valid JSON string", token by token.
"""

from __future__ import annotations

from src.models import FunctionDefinition
from src.vocab import Vocabulary

DIGITS = set("0123456789")
LETTERS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

MAX_NUMBER_DIGITS = 10
MAX_STRING_CHARS = 24


class ConstrainedDecodingError(Exception):
    """Raised when constrained decoding cannot proceed.

    This happens if the vocabulary is missing a token that the grammar
    requires (for example, a function name that isn't a single vocab
    token in the mock SDK), which signals a vocabulary/tokenizer mismatch
    rather than a bug in the decoding logic itself.
    """


def constrained_step(
    sdk: object,
    vocab: Vocabulary,
    context_ids: list[int],
    allowed_tokens: set[str],
) -> str:
    """Generate exactly one token, restricted to `allowed_tokens`.

    This is the single point in the whole project where the model is
    asked for logits and where masking happens. Every other function in
    this module is built by calling this one repeatedly with different
    allowed sets.

    Args:
        sdk: An object exposing `get_logits_from_input_ids(input_ids)`.
        vocab: The loaded vocabulary for id<->token lookups.
        context_ids: The token ids generated/seen so far. Mutated in
            place: the chosen token's id is appended before returning.
        allowed_tokens: The set of token strings that are legal right now.

    Returns:
        The chosen token's string value.

    Raises:
        ConstrainedDecodingError: If `allowed_tokens` is empty, or if one
            of the allowed tokens has no corresponding id in the
            vocabulary (a schema/tokenizer mismatch).
    """
    if not allowed_tokens:
        raise ConstrainedDecodingError(
            "No allowed tokens at this generation step -- the grammar "
            "reached an impossible state."
        )

    allowed_ids: set[int] = set()
    for token in allowed_tokens:
        token_id = vocab.id_for_token(token)
        if token_id is None:
            raise ConstrainedDecodingError(
                f"Token {token!r} is required by the grammar but is not "
                "in the vocabulary. If you're using the real llm_sdk, "
                "check how it tokenizes this string -- it may be split "
                "into multiple sub-word tokens rather than one."
            )
        allowed_ids.add(token_id)

    logits = sdk.get_logits_from_input_ids(context_ids)  # type: ignore[attr-defined]
    masked = [
        logit if token_id in allowed_ids else float("-inf")
        for token_id, logit in enumerate(logits)
    ]
    chosen_id = max(range(len(masked)), key=lambda i: masked[i])
    context_ids.append(chosen_id)
    return vocab.id_to_token[chosen_id]


def emit_literal(
    sdk: object, vocab: Vocabulary, context_ids: list[int], text: str
) -> str:
    """Force-generate a fixed, known string, one character at a time.

    Each character still goes through `constrained_step` with an
    allowed set of size 1 -- so it's technically "chosen" by the model,
    but there is genuinely only one legal option, which is exactly the
    point: the grammar removes all ambiguity here.

    Args:
        sdk: The LLM SDK.
        vocab: The loaded vocabulary.
        context_ids: Running context, mutated in place.
        text: The exact string to force-generate.

    Returns:
        `text`, unchanged (returned for symmetry with the other
        generate_* functions, so callers can always do
        `output += generate_xxx(...)`).
    """
    for char in text:
        constrained_step(sdk, vocab, context_ids, {char})
    return text


def choose_function_name(
    sdk: object,
    vocab: Vocabulary,
    context_ids: list[int],
    function_names: list[str],
) -> str:
    """Pick which function to call, constrained to the known function names.

    Args:
        sdk: The LLM SDK.
        vocab: The loaded vocabulary.
        context_ids: Running context, mutated in place.
        function_names: The exact set of valid function names.

    Returns:
        The chosen function name (guaranteed to be one of `function_names`).
    """
    return constrained_step(sdk, vocab, context_ids, set(function_names))


def generate_number_field(
    sdk: object,
    vocab: Vocabulary,
    context_ids: list[int],
    stop_char: str,
) -> tuple[str, float]:
    """Generate a numeric value, stopping when the field's terminator is chosen.

    At each step the model may add another digit, add a decimal point
    (once), add a leading minus sign (only as the very first character),
    or -- once at least one digit exists -- choose the terminator
    character that naturally follows this field in the JSON (a comma or
    a closing brace). Choosing the terminator ends the field.

    Args:
        sdk: The LLM SDK.
        vocab: The loaded vocabulary.
        context_ids: Running context, mutated in place.
        stop_char: The single character that legally follows this field
            (",", "}", etc.), used both as a valid choice and as what
            gets appended to the output when the field ends.

    Returns:
        A tuple of (the raw digit text produced, the parsed float value).
    """
    digits = ""
    for _ in range(MAX_NUMBER_DIGITS):
        allowed = set(DIGITS)
        if digits == "":
            allowed.add("-")
        has_digit = any(char in DIGITS for char in digits)
        if "." not in digits and has_digit:
            allowed.add(".")
        if has_digit:
            allowed.add(stop_char)

        token = constrained_step(sdk, vocab, context_ids, allowed)
        if token == stop_char:
            break
        digits += token
    else:
        # Safety valve: hit the max length without the model choosing to
        # stop. Force the terminator so we never loop forever or produce
        # an unterminated field.
        constrained_step(sdk, vocab, context_ids, {stop_char})

    try:
        value = float(digits)
    except ValueError:
        # Defensive fallback: should be unreachable given the mask above
        # (which never allows an empty or sign-only field to terminate),
        # but we never want a crash here.
        value = 0.0

    return digits + stop_char, value


def generate_string_field(
    sdk: object,
    vocab: Vocabulary,
    context_ids: list[int],
    stop_char: str,
) -> tuple[str, str]:
    """Generate a string value, stopping when the closing quote is chosen.

    Args:
        sdk: The LLM SDK.
        vocab: The loaded vocabulary.
        context_ids: Running context, mutated in place.
        stop_char: The character that follows this field once closed.

    Returns:
        A tuple of (the full quoted text including the terminator that
        was appended, the parsed string value without quotes).
    """
    emit_literal(sdk, vocab, context_ids, '"')
    content = ""
    for _ in range(MAX_STRING_CHARS):
        allowed = set(LETTERS) | {" "}
        if content:
            allowed.add('"')
        token = constrained_step(sdk, vocab, context_ids, allowed)
        if token == '"':
            break
        content += token
    else:
        constrained_step(sdk, vocab, context_ids, {'"'})

    emit_literal(sdk, vocab, context_ids, stop_char)
    return f'"{content}"' + stop_char, content


def generate_boolean_field(
    sdk: object,
    vocab: Vocabulary,
    context_ids: list[int],
    stop_char: str,
) -> tuple[str, bool]:
    """Generate a boolean value: a forced choice between true and false.

    Args:
        sdk: The LLM SDK.
        vocab: The loaded vocabulary.
        context_ids: Running context, mutated in place.
        stop_char: The character that follows this field.

    Returns:
        A tuple of (the literal text produced including the terminator,
        the parsed bool value).
    """
    token = constrained_step(sdk, vocab, context_ids, {"true", "false"})
    emit_literal(sdk, vocab, context_ids, stop_char)
    return token + stop_char, token == "true"


def generate_parameters(
    sdk: object,
    vocab: Vocabulary,
    context_ids: list[int],
    function_def: FunctionDefinition,
) -> tuple[str, dict[str, object]]:
    """Generate a full `parameters` JSON object matching a function's schema.

    Walks the function's declared parameters in order, emitting each key
    (forced literal) followed by a value generated according to that
    parameter's declared type.

    Args:
        sdk: The LLM SDK.
        vocab: The loaded vocabulary.
        context_ids: Running context, mutated in place.
        function_def: The chosen function's schema.

    Returns:
        A tuple of (the raw JSON text for the parameters object, a plain
        dict of parameter name -> parsed Python value).

    Raises:
        ConstrainedDecodingError: If a parameter's declared type is not
            one of number/string/boolean.
    """
    output = emit_literal(sdk, vocab, context_ids, "{")
    parsed: dict[str, object] = {}

    items = list(function_def.parameters.items())
    for index, (param_name, param_spec) in enumerate(items):
        is_last = index == len(items) - 1
        stop_char = "}" if is_last else ","

        key_literal = f'"{param_name}": '
        output += emit_literal(sdk, vocab, context_ids, key_literal)

        field_text: str
        value: float | str | bool
        if param_spec.type == "number":
            field_text, value = generate_number_field(sdk, vocab, context_ids, stop_char)
        elif param_spec.type == "string":
            field_text, value = generate_string_field(sdk, vocab, context_ids, stop_char)
        elif param_spec.type == "boolean":
            field_text, value = generate_boolean_field(sdk, vocab, context_ids, stop_char)
        else:  # pragma: no cover - unreachable given the pydantic Literal type
            raise ConstrainedDecodingError(
                f"Unsupported parameter type: {param_spec.type!r}"
            )

        output += field_text
        parsed[param_name] = value

    if not items:
        output += emit_literal(sdk, vocab, context_ids, "}")

    return output, parsed
