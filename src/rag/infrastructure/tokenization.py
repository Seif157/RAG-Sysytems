"""Token counting.

Chunk token counts drive the context budget, so a wrong count means either a
wasted prompt or a truncated one. ``tiktoken`` gives the real number for OpenAI
models and is close enough for others.

Falls back to a character heuristic rather than failing: an unknown model name
should not stop ingestion, and a slightly wrong budget is a far smaller problem
than a crash.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from rag.core.logging import get_logger

__all__ = ["TokenCounter"]

_logger = get_logger(__name__)

#: Used when no tokenizer is available. Roughly four characters per token.
_CHARS_PER_TOKEN = 4


@lru_cache(maxsize=8)
def _encoding_for(model: str) -> object | None:
    """Load and cache a tokenizer, or return ``None`` if there is not one.

    ``tiktoken`` downloads its vocabulary on first use, so this can fail with no
    network. Every failure degrades to the character heuristic rather than
    propagating: a slightly wrong context budget is a far smaller problem than
    ingestion refusing to run offline.
    """
    try:
        import tiktoken
    except ImportError:  # pragma: no cover - tiktoken is a declared dependency
        return None

    loaders: tuple[Callable[[], object], ...] = (
        lambda: tiktoken.encoding_for_model(model),
        lambda: tiktoken.get_encoding("cl100k_base"),
    )
    for load in loaders:
        try:
            return load()
        except Exception:
            continue
    return None


class TokenCounter:
    """Counts tokens for a specific model."""

    def __init__(self, model: str = "gpt-4o") -> None:
        """Initialise the counter.

        Args:
            model: Model whose tokenizer to use. An unrecognised name falls back
                to a general-purpose encoding.
        """
        self._model = model
        self._encoding = _encoding_for(model)
        if self._encoding is None:
            _logger.warning("tokenizer.unavailable", model=model, fallback="characters")

    def __call__(self, text: str) -> int:
        """Count the tokens in a string.

        Args:
            text: The text to measure.

        Returns:
            The token count, at least 1 for non-empty text.
        """
        if not text:
            return 0
        if self._encoding is None:
            return max(1, len(text) // _CHARS_PER_TOKEN)
        return max(1, len(self._encoding.encode(text)))  # type: ignore[attr-defined]
