"""Loads a model's vocabulary and exposes id<->token lookups.

Constrained decoding needs to answer one question at every generation
step: "which token id(s) correspond to the character(s) I'm willing to
accept right now?" This module is the piece that makes that lookup
possible, by turning the SDK's vocab file into two dictionaries.

FORMAT ASSUMPTION: this implementation expects the vocab file to be a
JSON array of strings, where a token's id is simply its position in the
array (this is what llm_sdk/vocab.json, the mock, provides). The REAL
llm_sdk package's vocab file may use a different format entirely (for
example a `{"token": id}` dict, as produced by Hugging Face
tokenizers). Before trusting this against the real SDK, call
`sdk.get_path_to_vocab_file()` yourself and inspect the file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol


class VocabFormatError(Exception):
    """Raised when the vocab file does not match the expected format."""


class SupportsVocabPath(Protocol):
    """Structural type for any SDK object exposing a vocab file path."""

    def get_path_to_vocab_file(self) -> str: ...  # noqa: E704


class Vocabulary:
    """Bidirectional mapping between token ids and their string values."""

    def __init__(self, sdk: SupportsVocabPath) -> None:
        """Load the vocabulary from the path the SDK reports.

        Args:
            sdk: Any object exposing `get_path_to_vocab_file()`.

        Raises:
            VocabFormatError: If the vocab file cannot be read or does not
                parse into a JSON array of strings.
        """
        path = Path(sdk.get_path_to_vocab_file())
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise VocabFormatError(f"Could not read vocab file {path}: {exc}") from exc

        if not isinstance(raw, list) or not all(isinstance(t, str) for t in raw):
            raise VocabFormatError(
                f"Expected {path} to contain a JSON array of strings; "
                "the real llm_sdk's vocab format may differ -- adapt this "
                "loader if so."
            )

        self.id_to_token: dict[int, str] = dict(enumerate(raw))
        self.token_to_id: dict[str, int] = {}
        for token_id, token in self.id_to_token.items():
            # If a token string appears twice, keep the first (lowest) id.
            self.token_to_id.setdefault(token, token_id)

    def id_for_token(self, token: str) -> int | None:
        """Look up the id for an exact token string.

        Args:
            token: The exact token string to look up.

        Returns:
            The token's id, or None if it is not in the vocabulary.
        """
        return self.token_to_id.get(token)
