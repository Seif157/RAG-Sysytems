"""Behaviour of conversation memory models.

Memory is deliberately narrow and deliberately isolated from retrieval: it feeds
the query rewriter and the prompt builder only (ADR-018).
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from rag.domain.models import Conversation, Role, Turn

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)


def _turn(content: str = "Who wrote this?", role: Role = Role.USER) -> Turn:
    return Turn(role=role, content=content, created_at=_NOW)


class TestTurn:
    def test_carries_role_content_and_time(self):
        turn = _turn("Who wrote this?")

        assert turn.role is Role.USER
        assert turn.content == "Who wrote this?"
        assert turn.created_at == _NOW

    def test_the_two_conversational_roles_exist(self):
        assert {r.value for r in Role} == {"user", "assistant"}

    def test_empty_content_is_rejected(self):
        with pytest.raises(ValueError, match="content"):
            _turn("   ")

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            _turn().content = "changed"  # type: ignore[misc]


class TestConversation:
    def test_starts_empty(self):
        conversation = Conversation(conversation_id="c-1")

        assert conversation.turns == ()

    def test_empty_conversation_id_is_rejected(self):
        with pytest.raises(ValueError, match="conversation_id"):
            Conversation(conversation_id="")

    def test_appending_returns_a_new_conversation(self):
        original = Conversation(conversation_id="c-1")

        updated = original.with_turn(_turn("Who wrote this?"))

        assert len(updated.turns) == 1
        assert original.turns == ()

    def test_turns_are_kept_in_order(self):
        conversation = (
            Conversation(conversation_id="c-1")
            .with_turn(_turn("Who wrote this?", Role.USER))
            .with_turn(_turn("Ada Lovelace.", Role.ASSISTANT))
            .with_turn(_turn("When was she born?", Role.USER))
        )

        assert [t.content for t in conversation.turns] == [
            "Who wrote this?",
            "Ada Lovelace.",
            "When was she born?",
        ]

    def test_recent_returns_the_most_recent_turns_in_order(self):
        conversation = (
            Conversation(conversation_id="c-1")
            .with_turn(_turn("one"))
            .with_turn(_turn("two"))
            .with_turn(_turn("three"))
        )

        assert [t.content for t in conversation.recent(2)] == ["two", "three"]

    def test_recent_with_a_window_larger_than_the_history_returns_everything(self):
        conversation = Conversation(conversation_id="c-1").with_turn(_turn("one"))

        assert len(conversation.recent(10)) == 1

    def test_recent_zero_returns_nothing(self):
        conversation = Conversation(conversation_id="c-1").with_turn(_turn("one"))

        assert conversation.recent(0) == ()

    def test_negative_window_is_rejected(self):
        conversation = Conversation(conversation_id="c-1")

        with pytest.raises(ValueError, match="limit"):
            conversation.recent(-1)

    def test_is_immutable(self):
        conversation = Conversation(conversation_id="c-1")

        with pytest.raises(dataclasses.FrozenInstanceError):
            conversation.conversation_id = "other"  # type: ignore[misc]
