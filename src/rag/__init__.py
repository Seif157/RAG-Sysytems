"""Document RAG Platform.

A Clean Architecture implementation of a document question-answering system.

Layering (dependencies point inward only)::

    presentation  ->  application  ->  domain  <-  infrastructure
                                         ^
                                core (composition root)

The rule is enforced by ``import-linter``; see ``importlinter.ini`` and section 4
of the architecture specification.
"""

__version__ = "0.1.0"
