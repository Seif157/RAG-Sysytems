"""Streamlit front end -- a thin client.

``api_client`` is the only module permitted to call the backend; components
never import :mod:`rag.application` or :mod:`rag.infrastructure`. That is what
keeps the front end replaceable (ADR-022).

Populated from Phase 2.
"""
