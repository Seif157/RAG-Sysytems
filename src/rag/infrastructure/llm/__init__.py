"""LLM provider clients.

Each translates its SDK's exceptions into the ``LLMError`` family, so retry
policy reads ``retryable`` as data rather than matching on message strings.

``OpenRouterLLMClient`` is the default and reaches many vendors' models through
one OpenAI-compatible endpoint; ``GeminiLLMClient`` remains for a deployment
that would rather talk to Google directly.
"""

from rag.infrastructure.llm.gemini_client import GeminiLLMClient
from rag.infrastructure.llm.openrouter_client import OpenRouterLLMClient

__all__ = ["GeminiLLMClient", "OpenRouterLLMClient"]
