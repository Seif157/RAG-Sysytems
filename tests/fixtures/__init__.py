"""Generated document fixtures.

Built in code rather than committed as binaries so that what each file contains
is readable in the diff, and so a fixture cannot quietly drift from the
assertions that depend on it.
"""

from tests.fixtures.documents import make_docx, make_markdown, make_pdf

__all__ = ["make_docx", "make_markdown", "make_pdf"]
