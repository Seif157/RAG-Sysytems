"""Prompt *contracts*, not prompt text.

Declares what a prompt must contain -- a grounding instruction, a context slot,
a citation instruction, the question -- and the invariants a prompt builder must
satisfy. The requirement is a business rule; the Jinja template that renders it
is an infrastructure detail living in :mod:`rag.infrastructure.prompts`.
"""

from rag.domain.prompts.contracts import NOT_FOUND_PHRASE, PromptSpec, is_refusal

__all__ = ["NOT_FOUND_PHRASE", "PromptSpec", "is_refusal"]
