"""Adapters implementing the domain ports.

One subpackage per port family. Every module here implements an interface from
:mod:`rag.domain.ports` and translates vendor exceptions into
:mod:`rag.domain.errors` types at its own boundary.

This is the only layer permitted to import a vendor SDK.

Populated from Phase 2 onward.
"""
