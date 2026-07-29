"""Selecting the parser for a document's format.

Adding a format edits exactly two existing things: the ``DocumentType`` enum and
the registry's construction in the container. Everything else is a new file.
That is the Open/Closed Principle made concrete, and the test of whether the
ingestion design worked.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag.domain.errors import UnsupportedFormatError
from rag.domain.models import DocumentType
from rag.domain.ports import DocumentParser

__all__ = ["ParserRegistry"]


class ParserRegistry:
    """Maps a document format to the parser that handles it."""

    def __init__(self, parsers: Sequence[DocumentParser]) -> None:
        """Build the registry.

        Args:
            parsers: The available parsers, one per format.

        Raises:
            ValueError: If two parsers claim the same format. Silent
                replacement would make which parser runs depend on argument
                order.
        """
        self._parsers: dict[DocumentType, DocumentParser] = {}
        for parser in parsers:
            if parser.supported_type in self._parsers:
                raise ValueError(f"two parsers registered for {parser.supported_type.value}")
            self._parsers[parser.supported_type] = parser

    def for_type(self, document_type: DocumentType) -> DocumentParser:
        """Return the parser for a format.

        Args:
            document_type: The format to parse.

        Returns:
            The registered parser.

        Raises:
            UnsupportedFormatError: If no parser handles that format. The
                message lists what is supported, so the next step is not reading
                source code.
        """
        parser = self._parsers.get(document_type)
        if parser is None:
            supported = ", ".join(sorted(t.value for t in self._parsers))
            raise UnsupportedFormatError(
                f"no parser for {document_type.value}; supported: {supported}",
                context={"requested": document_type.value},
            )
        return parser

    def supported_types(self) -> tuple[DocumentType, ...]:
        """Every format this registry can parse."""
        return tuple(self._parsers)
