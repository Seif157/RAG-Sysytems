"""Cross-encoder reranking (ADR-008).

Retrieval uses *bi-encoders*: the query and each passage are turned into vectors
independently and compared by geometry. That independence is what makes search
fast -- passages are embedded once, at ingest -- and it is also the ceiling,
because the model never sees the query and the passage together. It can only ask
"are these two things similar", never "does this passage answer this question".

A cross-encoder scores the **pair jointly**, attending across both at once. It
is far too slow to run over a corpus and ideal over twenty candidates. Hence the
shape of the pipeline: retrieve wide with something cheap, narrow with something
expensive.

Two implementation details that are not incidental:

*Scoring runs off the event loop.* ``CrossEncoder.predict`` is synchronous and
CPU-bound. Called inline it would stall every other coroutine for its whole
duration, which on a shared process means one user's rerank blocks everyone's.

*The model loads lazily.* A cross-encoder is hundreds of megabytes to gigabytes.
Loading it when the container is built would make start-up slow even for
deployments that never rerank, and would make ``--check-store`` download a model
to answer a question about Qdrant.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Sequence
from typing import Any

from rag.core.logging import get_logger
from rag.domain.errors import RerankingError
from rag.domain.models import ScoredChunk, ScoreSource
from rag.domain.ports import Reranker

__all__ = ["BgeReranker"]

_logger = get_logger(__name__)

#: Scores a batch of (query, passage) pairs.
type PairScorer = Callable[[list[tuple[str, str]]], Sequence[float]]


def _sigmoid(logit: float) -> float:
    """Squash a raw cross-encoder logit into ``[0, 1]``.

    Cross-encoders emit unbounded logits, and
    :class:`~rag.domain.models.Citation` requires a relevance score in the unit
    interval -- an unsquashed score would fail validation at the very last step
    of the pipeline, after all the work was done.

    Monotonic, so it never changes the ranking. The result is a *relative*
    relevance, not a calibrated probability, and is presented as such.
    """
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    # Rearranged for large negative logits, where exp(-logit) overflows.
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


class BgeReranker(Reranker):
    """Re-scores retrieved candidates with a cross-encoder.

    Defaults to a BGE reranker, but works with any Sentence-Transformers
    cross-encoder -- the class holds no BGE-specific logic, so switching models
    is a configuration change.
    """

    def __init__(
        self,
        model: str,
        *,
        batch_size: int = 16,
        max_length: int = 512,
        scorer: PairScorer | None = None,
    ) -> None:
        """Initialise the reranker.

        Args:
            model: Cross-encoder model identifier.
            batch_size: Pairs scored per forward pass.
            max_length: Token ceiling per pair. Longer passages are truncated by
                the tokenizer, which is why chunk size and this value should be
                chosen together.
            scorer: Overrides the model with a scoring callable. Tests pass one
                so the contract can be exercised without downloading gigabytes;
                production leaves it ``None``.
        """
        self._model_name = model
        self._batch_size = batch_size
        self._max_length = max_length
        self._scorer = scorer

    def _load(self) -> PairScorer:
        """Build the scoring callable, loading the model on first use."""
        if self._scorer is not None:
            return self._scorer

        from sentence_transformers import CrossEncoder

        started = time.perf_counter()
        encoder: Any = CrossEncoder(self._model_name, max_length=self._max_length)
        _logger.info(
            "reranker.model_loaded",
            model=self._model_name,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

        def score(pairs: list[tuple[str, str]]) -> Sequence[float]:
            scores: Sequence[float] = encoder.predict(pairs, batch_size=self._batch_size)
            return scores

        self._scorer = score
        return score

    async def warm_up(self) -> None:
        """Load the model ahead of the first query.

        Optional. Without it the first question of a session pays the load time,
        which for a multi-gigabyte model is conspicuous.
        """
        await asyncio.to_thread(self._load)

    async def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        top_n: int,
    ) -> tuple[ScoredChunk, ...]:
        """Re-score candidates against the query and keep the best.

        Args:
            query: The question to judge relevance against.
            candidates: Chunks from retrieval, in fusion order.
            top_n: Maximum number to return.

        Returns:
            At most ``top_n`` of the input chunks, most relevant first, each
            re-scored and marked
            :attr:`~rag.domain.models.ScoreSource.RERANKED`.

        Raises:
            ValueError: If ``top_n`` is not positive.
            RerankingError: If scoring fails. Callers degrade to fusion order
                rather than failing the question.
        """
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if not candidates:
            # Nothing to order, and no reason to pay for a model load.
            return ()

        pairs = [(query, candidate.text) for candidate in candidates]

        try:
            scores = await asyncio.to_thread(self._score, pairs)
        except RerankingError:
            raise
        except Exception as exc:
            raise RerankingError(
                f"cross-encoder scoring failed: {exc}",
                context={"model": self._model_name, "candidates": len(candidates)},
            ) from exc

        if len(scores) != len(candidates):
            # Zipping mismatched lists would attach each score to the wrong
            # passage, producing a confident and completely wrong ranking.
            raise RerankingError(
                f"model returned {len(scores)} scores for {len(candidates)} candidates",
                context={"model": self._model_name},
            )

        ranked = sorted(
            (
                candidate.rescored(_sigmoid(float(score)), ScoreSource.RERANKED)
                for candidate, score in zip(candidates, scores, strict=True)
            ),
            key=lambda scored: scored.score,
            reverse=True,
        )
        return tuple(ranked[:top_n])

    def _score(self, pairs: list[tuple[str, str]]) -> Sequence[float]:
        """Score pairs on a worker thread."""
        return self._load()(pairs)
