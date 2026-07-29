"""Translating domain filters into Qdrant's filter language (ADR-013).

The single place in the codebase where a vendor filter type exists. Everything
above this file speaks the domain's filter tree, which is what allows the
in-memory test double to honour exactly the same semantics as Qdrant.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import models as qmodels

from rag.domain.models import (
    And,
    FieldFilter,
    FilterExpression,
    FilterOperator,
    Not,
    Or,
)

__all__ = ["to_qdrant_filter"]

_RANGE_OPERATORS: dict[FilterOperator, str] = {
    FilterOperator.GT: "gt",
    FilterOperator.GTE: "gte",
    FilterOperator.LT: "lt",
    FilterOperator.LTE: "lte",
}


def to_qdrant_filter(expression: FilterExpression | None) -> qmodels.Filter | None:
    """Convert a domain filter expression into a Qdrant filter.

    Args:
        expression: The domain filter, or ``None`` for no filtering.

    Returns:
        The equivalent Qdrant filter, or ``None``.

    Raises:
        NotImplementedError: If an expression type is unhandled. The hierarchy
            is sealed, so this can only fire if a node type is added without
            updating this translator -- which is exactly when a loud failure is
            wanted.
    """
    if expression is None:
        return None

    match expression:
        case FieldFilter():
            return _field(expression)
        case And():
            return qmodels.Filter(must=[_as_condition(clause) for clause in expression.clauses])
        case Or():
            return qmodels.Filter(should=[_as_condition(clause) for clause in expression.clauses])
        case Not():
            return qmodels.Filter(must_not=[_as_condition(expression.clause)])
        case _:  # pragma: no cover - the hierarchy is sealed
            raise NotImplementedError(type(expression))


def _as_condition(expression: FilterExpression) -> Any:
    """Render a sub-expression as something Qdrant accepts in a clause list."""
    translated = to_qdrant_filter(expression)
    assert translated is not None
    return translated


def _field(clause: FieldFilter) -> qmodels.Filter:
    """Translate a single predicate."""
    key = clause.field.value

    match clause.operator:
        case FilterOperator.EQ:
            return qmodels.Filter(
                must=[qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=clause.value))]
            )
        case FilterOperator.NE:
            return qmodels.Filter(
                must_not=[
                    qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=clause.value))
                ]
            )
        case FilterOperator.IN:
            return qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=list(clause.value)))
                ]
            )
        case FilterOperator.NOT_IN:
            return qmodels.Filter(
                must_not=[
                    qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=list(clause.value)))
                ]
            )
        case FilterOperator.IS_NULL:
            return qmodels.Filter(
                must=[qmodels.IsNullCondition(is_null=qmodels.PayloadField(key=key))]
            )
        case FilterOperator.IS_NOT_NULL:
            return qmodels.Filter(
                must_not=[qmodels.IsNullCondition(is_null=qmodels.PayloadField(key=key))]
            )
        case FilterOperator.CONTAINS:
            return qmodels.Filter(
                must=[qmodels.FieldCondition(key=key, match=qmodels.MatchText(text=clause.value))]
            )

    bound = _RANGE_OPERATORS.get(clause.operator)
    if bound is None:  # pragma: no cover - every operator is handled above
        raise NotImplementedError(clause.operator)
    return qmodels.Filter(
        must=[qmodels.FieldCondition(key=key, range=qmodels.Range(**{bound: clause.value}))]
    )
