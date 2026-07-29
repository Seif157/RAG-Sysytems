"""Reusable orchestration across several ports.

A service exists when logic is shared by two use cases, or is too large to sit
comfortably inside one.
"""

from rag.application.services.context_builder import ContextBuilder

__all__ = ["ContextBuilder"]
