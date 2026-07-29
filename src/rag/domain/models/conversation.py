"""Conversation history.

Deliberately minimal. Memory feeds the query rewriter and the prompt builder and
nothing else -- retrieval never sees raw history, because concatenating a
five-turn history onto a six-word question produces a query vector dominated by
the history and collapses recall (ADR-018).

Keeping this model small is also what keeps the memory backend replaceable: an
in-memory store, Redis, and a summarising decorator all satisfy the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Self

__all__ = ["Conversation", "Role", "Turn"]


class Role(StrEnum):
    """Who produced a turn."""

    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Turn:
    """One message in a conversation.

    Attributes:
        role: Who produced it.
        content: The message text.
        created_at: When it was recorded.
    """

    role: Role
    content: str
    created_at: datetime

    def __post_init__(self) -> None:
        """Reject an empty turn, which carries no context for a follow-up."""
        if not self.content or not self.content.strip():
            raise ValueError("turn content must be a non-empty string")


@dataclass(frozen=True, slots=True)
class Conversation:
    """An ordered history of turns for one conversation.

    Immutable: appending returns a new conversation. That keeps history safe to
    share across concurrent requests without copying defensively.

    Attributes:
        conversation_id: Stable identifier.
        turns: Turns in chronological order, oldest first.
    """

    conversation_id: str
    turns: tuple[Turn, ...] = ()

    def __post_init__(self) -> None:
        """Validate identifiers."""
        if not self.conversation_id.strip():
            raise ValueError("conversation_id must be a non-empty string")

    def with_turn(self, turn: Turn) -> Self:
        """Return a copy with one more turn appended.

        Args:
            turn: The turn to append.

        Returns:
            A new :class:`Conversation`; the original is unchanged.
        """
        return replace(self, turns=(*self.turns, turn))

    def recent(self, limit: int) -> tuple[Turn, ...]:
        """Return the most recent turns, oldest first.

        The window is applied at read time rather than on write, so changing
        ``MEMORY_WINDOW_TURNS`` takes effect immediately and without discarding
        history that a longer window would still want.

        Args:
            limit: Maximum number of turns to return. ``0`` returns nothing.

        Returns:
            Up to ``limit`` turns, in chronological order.

        Raises:
            ValueError: If ``limit`` is negative.
        """
        if limit < 0:
            raise ValueError("limit must be >= 0")
        if limit == 0:
            return ()
        return self.turns[-limit:]
