"""The assembled context block handed to the prompt builder.

The marker mapping is the load-bearing part. An LLM cites by emitting a marker
such as ``[2]``; turning that back into a :class:`~rag.domain.models.answer.Citation`
requires knowing which chunk marker ``2`` referred to *in this particular
prompt*. Markers are per-request, so the mapping travels with the context rather
than being reconstructed later.

Resolution is deliberately strict: an unknown marker returns ``None`` and the
citation is dropped. A model that invents ``[7]`` when six chunks were supplied
must not produce a citation pointing at an arbitrary chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.domain.models.chunk import ScoredChunk

__all__ = ["ContextBlock"]


@dataclass(frozen=True, slots=True)
class ContextBlock:
    """Retrieved context, rendered and budgeted, ready for prompting.

    Attributes:
        chunks: The chunks included, in presentation order.
        rendered: The context as it will appear in the prompt, with markers.
        token_count: Tokens the rendered block occupies.
        marker_to_chunk_id: Citation marker to chunk id, for this request only.
        dropped_count: Chunks excluded by the token budget. Non-zero means the
            answer was produced with less context than retrieval found, which is
            worth surfacing when quality is investigated.
    """

    chunks: tuple[ScoredChunk, ...]
    rendered: str
    token_count: int
    marker_to_chunk_id: tuple[tuple[str, str], ...]
    dropped_count: int = 0

    def __post_init__(self) -> None:
        """Validate the block is internally consistent."""
        if self.token_count < 0:
            raise ValueError("token_count must be >= 0")
        if self.dropped_count < 0:
            raise ValueError("dropped_count must be >= 0")

        present = {scored.chunk_id for scored in self.chunks}
        for marker, chunk_id in self.marker_to_chunk_id:
            if chunk_id not in present:
                raise ValueError(
                    f"citation marker {marker!r} maps to chunk {chunk_id!r}, "
                    f"which is not part of this context block"
                )

    @property
    def is_empty(self) -> bool:
        """Whether no chunk was included.

        A valid outcome: retrieval may legitimately find nothing. The prompt is
        still built, and the model is instructed to say so rather than guess.
        """
        return not self.chunks

    def chunk_id_for_marker(self, marker: str) -> str | None:
        """Resolve a citation marker emitted by the model back to a chunk.

        Args:
            marker: The marker text, e.g. ``"2"``.

        Returns:
            The chunk id, or ``None`` if the marker was not one this context
            supplied -- in which case the citation must be dropped.
        """
        for candidate, chunk_id in self.marker_to_chunk_id:
            if candidate == marker:
                return chunk_id
        return None
