"""Tracking the heading breadcrumb while walking a document.

Shared by the DOCX and Markdown parsers, which differ only in how they *detect*
a heading -- a paragraph style versus a run of ``#`` characters. The rule for
what a heading does to the breadcrumb is identical, and duplicating it would let
the two formats drift apart in a way no test would obviously catch.

The rule: a heading at level *n* replaces everything at level *n* and deeper. So
a new ``## Costs`` under ``# Financials`` gives ``Financials > Costs``, and a new
``# Operations`` discards the whole previous subtree.
"""

from __future__ import annotations

__all__ = ["HeadingStack"]

#: Separator used when the breadcrumb is rendered as a single string. Chosen to
#: be readable in a prompt and unlikely to appear in a real heading.
_SEPARATOR = " > "


class HeadingStack:
    """Maintains the current heading breadcrumb."""

    def __init__(self) -> None:
        """Start with no headings seen."""
        self._levels: list[tuple[int, str]] = []

    def push(self, level: int, text: str) -> None:
        """Record a heading, discarding any deeper or equal levels.

        Args:
            level: Heading depth, 1 for the top level.
            text: The heading text.
        """
        self._levels = [(depth, title) for depth, title in self._levels if depth < level]
        self._levels.append((level, text))

    @property
    def path(self) -> tuple[str, ...] | None:
        """The breadcrumb, or ``None`` before any heading has been seen.

        ``None`` rather than an empty tuple: text appearing before the first
        heading genuinely has no section, and that should stay distinguishable
        from a section whose title is blank.
        """
        return tuple(title for _, title in self._levels) or None

    @property
    def nearest(self) -> str | None:
        """The most recent heading, at whatever level."""
        return self._levels[-1][1] if self._levels else None

    @property
    def section(self) -> str | None:
        """The breadcrumb rendered for display and for prompting."""
        path = self.path
        return _SEPARATOR.join(path) if path else None
