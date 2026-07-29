"""One class per user-visible operation.

A use case exposes a single public method and forms the error boundary for that
operation. Collaborators are injected as port interfaces, so every one of these
is executable against in-memory doubles with no network access.
"""

from rag.application.use_cases.answer_question import AnswerQuestionUseCase
from rag.application.use_cases.ingest_document import IngestDocumentUseCase

__all__ = ["AnswerQuestionUseCase", "IngestDocumentUseCase"]
