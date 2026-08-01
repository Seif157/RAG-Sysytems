"""Pure business rules: no I/O, no clock, no randomness.

Chunk identity, context budgeting and citation assembly. Rank fusion joins them
when hybrid retrieval arrives.

Because these are pure functions over domain types, they are the densest and
cheapest tests in the codebase -- no mocks, no setup, exhaustive edge cases.
"""

from rag.domain.policies.chunk_identity import derive_chunk_id
from rag.domain.policies.citation_assembly import (
    assemble_citations,
    extract_citation_markers,
    remove_unknown_markers,
)
from rag.domain.policies.context_budget import select_within_budget
from rag.domain.policies.context_dedup import (
    DEFAULT_DUPLICATE_THRESHOLD,
    deduplicate_chunks,
)
from rag.domain.policies.context_ordering import order_for_attention
from rag.domain.policies.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion

__all__ = [
    "DEFAULT_DUPLICATE_THRESHOLD",
    "DEFAULT_RRF_K",
    "assemble_citations",
    "deduplicate_chunks",
    "derive_chunk_id",
    "extract_citation_markers",
    "order_for_attention",
    "reciprocal_rank_fusion",
    "remove_unknown_markers",
    "select_within_budget",
]
