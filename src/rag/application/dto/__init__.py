"""Data transfer objects crossing use-case boundaries internally.

Distinct from :mod:`rag.presentation.schemas`, which are the HTTP wire contracts.
Keeping them separate means an API field rename never reaches the core, and a
domain refactor never breaks a client.

Populated from Phase 2 onward.
"""
