"""Turning uploaded bytes into a typed, validated raw document.

A plain class, not a port: there is one implementation and no prospect of a
second. What it does is still worth isolating, because it is where every
untrusted input is checked.

Format detection uses the extension *and* the bytes. When they disagree the
upload is rejected rather than resolved -- a ``.txt`` file whose content is a
PDF is either a mistake or an attack, and guessing turns both into a confusing
parse failure much further downstream.
"""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath

from rag.domain.errors import DocumentTooLargeError, UnsupportedFormatError
from rag.domain.models import DocumentType, RawDocument

__all__ = ["DocumentLoader"]

_EXTENSIONS: dict[str, DocumentType] = {
    ".pdf": DocumentType.PDF,
    ".docx": DocumentType.DOCX,
    ".txt": DocumentType.TXT,
    ".md": DocumentType.MARKDOWN,
    ".markdown": DocumentType.MARKDOWN,
}

#: Leading bytes that identify a format regardless of what the name claims.
_MAGIC: tuple[tuple[bytes, DocumentType], ...] = (
    (b"%PDF-", DocumentType.PDF),
    (b"PK\x03\x04", DocumentType.DOCX),  # DOCX is a zip container
)


class DocumentLoader:
    """Validates an upload and determines its format."""

    def __init__(self, max_upload_bytes: int) -> None:
        """Initialise the loader.

        Args:
            max_upload_bytes: Largest file accepted.
        """
        self._max_upload_bytes = max_upload_bytes

    def load(self, filename: str, content: bytes) -> RawDocument:
        """Validate an upload and resolve its type.

        Args:
            filename: Original filename as uploaded.
            content: The raw bytes.

        Returns:
            A raw document with a resolved type and a content hash.

        Raises:
            DocumentTooLargeError: If the content exceeds the configured limit.
            UnsupportedFormatError: If the format is unsupported, the file is
                empty, or the extension and the content disagree.
        """
        if len(content) > self._max_upload_bytes:
            raise DocumentTooLargeError(
                f"{filename} is {len(content)} bytes, limit is {self._max_upload_bytes}",
                context={"filename": filename, "size_bytes": len(content)},
            )
        if not content:
            raise UnsupportedFormatError(f"{filename} is empty", context={"filename": filename})

        declared = self._from_extension(filename)
        sniffed = self._from_content(content)

        if sniffed is not None and sniffed is not declared:
            raise UnsupportedFormatError(
                f"{filename} claims to be {declared.value} but its content is {sniffed.value}",
                context={"filename": filename, "declared": declared.value},
            )

        return RawDocument(
            filename=filename,
            content=content,
            document_type=declared,
            content_hash=hashlib.sha256(content).hexdigest(),
        )

    @staticmethod
    def _from_extension(filename: str) -> DocumentType:
        """Resolve the format from the filename extension."""
        suffix = PurePosixPath(filename.replace("\\", "/")).suffix.lower()
        document_type = _EXTENSIONS.get(suffix)
        if document_type is None:
            supported = ", ".join(sorted(_EXTENSIONS))
            raise UnsupportedFormatError(
                f"{filename} has no supported extension; supported: {supported}",
                context={"filename": filename, "extension": suffix},
            )
        return document_type

    @staticmethod
    def _from_content(content: bytes) -> DocumentType | None:
        """Resolve the format from leading bytes, where they are distinctive.

        Text and Markdown have no magic number, so an absent result means
        "nothing contradicts the extension" rather than "unknown".
        """
        for magic, document_type in _MAGIC:
            if content.startswith(magic):
                return document_type
        return None
