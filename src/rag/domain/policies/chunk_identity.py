"""Deterministic chunk identity (ADR-014).

A chunk's id is derived from its content and position rather than generated, so
re-ingesting a document overwrites its chunks instead of duplicating them. That
one property is what makes the whole ingestion pipeline safe to retry.
"""

from __future__ import annotations

import hashlib

__all__ = ["derive_chunk_id"]

_ID_LENGTH = 32


def derive_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    """Derive a stable identifier for a chunk.

    The three parts are hashed individually before being combined, rather than
    joined with a separator. Naive concatenation lets content forge a boundary:
    ``("doc|0", "text")`` and ``("doc", "0|text")`` would produce the same
    string, and therefore the same id for two genuinely different chunks.

    Args:
        document_id: The document the chunk belongs to.
        chunk_index: Zero-based position within the document. Included so two
            identical paragraphs in one document do not collide.
        text: The chunk text.

    Returns:
        A 32-character hexadecimal identifier.
    """
    digest = hashlib.sha256()
    for part in (document_id.encode(), str(chunk_index).encode(), text.encode()):
        digest.update(hashlib.sha256(part).digest())
    return digest.hexdigest()[:_ID_LENGTH]
