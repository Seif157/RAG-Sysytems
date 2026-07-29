"""A vendor-neutral filter expression tree for metadata filtering (ADR-013).

Metadata filtering is a user-facing feature, so it is a domain concept. The
expression built here is translated into a vendor's own filter type by an adapter
(``QdrantFilterTranslator`` and friends), which is the single place a vendor
filter type is allowed to exist.

The trade-off is deliberate: this language is a *subset* of what any one store
can express. Vendor-specific predicates are unavailable until added here and to
every translator. That constraint is what keeps adapters substitutable.

Example:
    ``document_type = PDF AND page_number > 10 AND author = "Mohamed"``::

        (
            FieldFilter(MetadataField.DOCUMENT_TYPE, FilterOperator.EQ, "PDF")
            & FieldFilter(MetadataField.PAGE_NUMBER, FilterOperator.GT, 10)
            & FieldFilter(MetadataField.AUTHOR, FilterOperator.EQ, "Mohamed")
        )
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rag.domain.models.metadata import FILTERABLE_FIELDS, MetadataField

__all__ = [
    "And",
    "FieldFilter",
    "FilterExpression",
    "FilterOperator",
    "Not",
    "Or",
]


class FilterOperator(StrEnum):
    """Comparison operators supported by the filter language."""

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


#: Operators that assert on presence and therefore take no operand.
_NULLARY_OPERATORS = frozenset({FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL})

#: Operators whose operand is a collection of candidate values.
_SEQUENCE_OPERATORS = frozenset({FilterOperator.IN, FilterOperator.NOT_IN})


class FilterExpression(ABC):  # noqa: B024 -- sealed marker base, see below
    """Base class for every node in a filter expression tree.

    Abstract by intent rather than by having abstract members: it exists to seal
    the hierarchy so a translator can exhaustively match on node type, and to
    give every node the boolean operators. There is no operation a node must
    implement, because evaluation belongs to the translator, not to the tree.

    Subclasses are immutable value objects. The boolean operators build
    composites rather than evaluating anything: evaluation is the translator's
    job, in whatever the target store's dialect happens to be.
    """

    def __and__(self, other: FilterExpression) -> And:
        """Combine with another expression as a conjunction.

        Chained conjunctions are flattened, so ``(a & b) & c`` yields a single
        three-clause :class:`And` rather than a nested tree. Translators then
        emit one flat clause list instead of redundant nesting.
        """
        return And(self._as_clauses(And) + other._as_clauses(And))

    def __or__(self, other: FilterExpression) -> Or:
        """Combine with another expression as a disjunction, flattening chains."""
        return Or(self._as_clauses(Or) + other._as_clauses(Or))

    def __invert__(self) -> Not:
        """Negate this expression.

        Double negation is preserved rather than simplified: a translator may
        render ``NOT NOT x`` more efficiently than we could guess, and silent
        simplification hides caller mistakes.
        """
        return Not(self)

    def _as_clauses(self, composite: type[And] | type[Or]) -> tuple[FilterExpression, ...]:
        """Return this node's clauses if it is the given composite, else itself."""
        if isinstance(self, composite):
            return self.clauses
        return (self,)


@dataclass(frozen=True, slots=True)
class FieldFilter(FilterExpression):
    """A single predicate over one metadata field.

    Attributes:
        field: The metadata field being filtered. Must be in
            :data:`~rag.domain.models.metadata.FILTERABLE_FIELDS`.
        operator: The comparison to apply.
        value: The operand. ``None`` for the nullary operators; a tuple for the
            sequence operators; a scalar otherwise.
    """

    field: MetadataField
    operator: FilterOperator
    value: Any = None

    def __post_init__(self) -> None:
        """Validate the field is filterable and the operand suits the operator."""
        if self.field not in FILTERABLE_FIELDS:
            raise ValueError(
                f"{self.field.value!r} is not filterable; it has no payload index. "
                f"Filterable fields: {sorted(f.value for f in FILTERABLE_FIELDS)}"
            )

        if self.operator in _NULLARY_OPERATORS:
            if self.value is not None:
                raise ValueError(f"{self.operator.name} takes no value, got {self.value!r}")
            return

        if self.value is None:
            raise ValueError(f"{self.operator.name} requires a value")

        if self.operator in _SEQUENCE_OPERATORS:
            if isinstance(self.value, str | bytes) or not isinstance(self.value, Sequence):
                raise ValueError(f"{self.operator.name} requires a sequence of values")
            if len(self.value) == 0:
                raise ValueError(f"{self.operator.name} requires a non-empty sequence")
            object.__setattr__(self, "value", tuple(self.value))


@dataclass(frozen=True, slots=True)
class And(FilterExpression):
    """Conjunction: every clause must match.

    Attributes:
        clauses: The conjoined expressions. At least one.
    """

    clauses: tuple[FilterExpression, ...]

    def __post_init__(self) -> None:
        """Reject an empty conjunction, which would match everything."""
        if not self.clauses:
            raise ValueError("And requires at least one clause")


@dataclass(frozen=True, slots=True)
class Or(FilterExpression):
    """Disjunction: at least one clause must match.

    Attributes:
        clauses: The alternative expressions. At least one.
    """

    clauses: tuple[FilterExpression, ...]

    def __post_init__(self) -> None:
        """Reject an empty disjunction, which would match nothing."""
        if not self.clauses:
            raise ValueError("Or requires at least one clause")


@dataclass(frozen=True, slots=True)
class Not(FilterExpression):
    """Negation of a single expression.

    Attributes:
        clause: The expression being negated.
    """

    clause: FilterExpression
