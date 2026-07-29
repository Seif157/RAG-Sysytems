"""Port: extracting text and structure from a document."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.domain.models import DocumentType, ParsedDocument, RawDocument

__all__ = ["DocumentParser"]


class DocumentParser(ABC):
    """Extracts text *and structural position* from one document format.

    One implementation per format, selected by a registry. Adding a format edits
    exactly two existing files -- the :class:`~rag.domain.models.DocumentType`
    enum and the registry -- which is the Open/Closed Principle made concrete
    and the test of whether the ingestion design succeeded.

    A parser's responsibility is deliberately narrow:

    * It extracts text and where that text came from -- page, heading path,
      character offsets. Structural position is what makes a citation useful.
    * It does **not** chunk. Choosing boundaries is the chunking strategy's
      decision, and a parser that chunks has taken it away.
    * It does **not** clean beyond whitespace normalisation. Aggressive cleaning
      destroys information the chunker or the reader may need.

    Synchronous: parsing is CPU-bound work running in the ingestion worker.
    """

    @property
    @abstractmethod
    def supported_type(self) -> DocumentType:
        """The single format this parser handles."""

    @abstractmethod
    def parse(self, document: RawDocument) -> ParsedDocument:
        """Extract ordered content blocks and document properties.

        An empty result is valid, not an error: a scanned PDF contains no
        extractable text, and OCR is explicitly out of scope. The caller decides
        whether an empty parse should fail ingestion.

        Args:
            document: The raw document to parse.

        Returns:
            Ordered content blocks plus document-level properties.

        Raises:
            DocumentParsingError: If extraction fails.
            CorruptDocumentError: If the file is malformed.
            EncryptedDocumentError: If the document is password-protected.
        """
