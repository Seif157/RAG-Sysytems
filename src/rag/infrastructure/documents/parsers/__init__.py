"""One parser per supported format, selected by the parser registry.

Parsers extract text *and structure* -- pages, headings, character offsets --
and nothing else. A parser that chunks has taken a decision belonging to the
chunking strategy.

Adding a format edits exactly two existing files: the ``DocumentType`` enum and
the registry (architecture spec section 15.6).
"""
