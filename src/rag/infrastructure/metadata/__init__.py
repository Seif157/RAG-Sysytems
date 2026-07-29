"""Metadata extractors, composed into a chain.

Intrinsic (filename, hash), structural (page, heading path), document properties
(author, title) and language detection. Extractor failures are non-fatal: the
chain logs and continues with an empty fragment.

Populated from Phase 3.
"""
