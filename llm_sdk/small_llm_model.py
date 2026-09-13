"""A lightweight stand-in for the real `llm_sdk` package.

IMPORTANT: This is NOT the real Qwen/Qwen3-0.6B SDK. The subject says the
real `llm_sdk` package (with `Small_LLM_Model`) is provided separately and
must be copied next to `src/`. Since that package isn't available in this
environment, this mock implements the exact same public interface so the
rest of the project (constrained decoding, generation loop, CLI) can be
written and tested end to end.

Behavioural difference you must know about: this mock has NO real language
understanding. `get_logits_from_input_ids` returns pseudo-random numbers,
not a trained model's predictions. That means constrained decoding will
still guarantee 100% *valid, schema-compliant* JSON (that part is real and
correctly demonstrated), but the *values* it picks (which digits, which
letters) will look arbitrary rather than semantically correct, since there
is no real reasoning happening. When you swap in the real llm_sdk, the
same masking logic will select values a real trained model actually
believes are correct.

Format assumption: this mock's vocab file (`vocab.json`) is a plain JSON
array of strings, where the token's id is simply its index in the array.
The REAL llm_sdk's vocab file may use a different format (e.g. a
token->id dict, or a BPE merges file) -- inspect
`get_path_to_vocab_file()`'s actual output before assuming `src/vocab.py`
parses it correctly.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


class Small_LLM_Model:
    """Mock wrapper around a "small" language model.

    Mirrors the interface described in the subject:
    - get_logits_from_input_ids(input_ids) -> list[float]
    - get_path_to_vocab_file() -> str
    - encode(text) -> list[int]
    - decode(token_ids) -> str
    """

    def __init__(self, vocab_path: str | None = None) -> None:
        """Load the (mock) vocabulary.

        Args:
            vocab_path: Optional override path to a vocab JSON file. If
                omitted, the vocab.json shipped alongside this module is
                used.
        """
        self._vocab_path = vocab_path or str(Path(__file__).parent / "vocab.json")
        with open(self._vocab_path, "r", encoding="utf-8") as handle:
            self._vocab: list[str] = json.load(handle)

    def get_path_to_vocab_file(self) -> str:
        """Return the path to the vocabulary file.

        Returns:
            Absolute or relative path to the JSON vocab file in use.
        """
        return self._vocab_path

    def encode(self, text: str) -> list[int]:
        """Tokenize text into a list of token ids using greedy longest-match.

        Note: the real SDK returns a Tensor; this mock returns a plain
        list[int] for simplicity. Calling code should not assume either
        way and should convert defensively (see src/vocab.py).

        Args:
            text: The text to tokenize.

        Returns:
            A list of token ids.

        Raises:
            ValueError: If a character in `text` has no matching token in
                the vocabulary.
        """
        ids: list[int] = []
        i = 0
        # Sort tokens by length (longest first) so multi-character tokens
        # like "true" or a function name are preferred over spelling them
        # out one character at a time.
        sorted_tokens = sorted(
            enumerate(self._vocab), key=lambda pair: len(pair[1]), reverse=True
        )
        while i < len(text):
            matched = False
            for token_id, token in sorted_tokens:
                if text.startswith(token, i):
                    ids.append(token_id)
                    i += len(token)
                    matched = True
                    break
            if not matched:
                raise ValueError(
                    f"Character {text[i]!r} at position {i} has no matching "
                    "token in the mock vocabulary. Extend llm_sdk/vocab.json."
                )
        return ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token ids back into text.

        Args:
            token_ids: The token ids to decode.

        Returns:
            The concatenated string these tokens represent.
        """
        return "".join(self._vocab[token_id] for token_id in token_ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Produce logits for the next token, given the context so far.

        This mock has no real neural network: it returns deterministic
        pseudo-random numbers seeded by the input sequence, purely so that
        running the same prompt twice gives the same (arbitrary) result.
        This is intentionally NOT a trained model's predictions -- see the
        module docstring.

        Args:
            input_ids: The token ids generated/seen so far.

        Returns:
            A list of floats, one per vocabulary entry.
        """
        seed = sum((idx + 1) * tid for idx, tid in enumerate(input_ids)) or 1
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(len(self._vocab))]
