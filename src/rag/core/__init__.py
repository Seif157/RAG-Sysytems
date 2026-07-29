"""Composition root and cross-cutting concerns.

The only package that legitimately knows every concrete class. ``container``
reads configuration and builds the object graph -- including decorator stacking,
which is how optional behaviour such as caching is expressed without a single
feature flag reaching a call site.

Deliberately thin. ``core`` is a composition root, not a utility bucket.
"""

__all__: list[str] = []
