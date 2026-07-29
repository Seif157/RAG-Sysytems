"""Behaviour of the vendor-neutral metadata filter expression tree (ADR-013).

Filters are a domain concept because metadata filtering is a user-facing
feature. Translation into a vendor's filter type happens at the adapter
boundary, so the expression built here works unchanged against any store.
"""

from __future__ import annotations

import dataclasses

import pytest

from rag.domain.models import (
    And,
    FieldFilter,
    FilterOperator,
    MetadataField,
    Not,
    Or,
)

pytestmark = pytest.mark.unit


class TestFieldFilter:
    def test_carries_field_operator_and_value(self):
        clause = FieldFilter(MetadataField.PAGE_NUMBER, FilterOperator.GT, 10)

        assert clause.field is MetadataField.PAGE_NUMBER
        assert clause.operator is FilterOperator.GT
        assert clause.value == 10

    def test_non_filterable_fields_are_rejected(self):
        # Filtering on a field with no payload index would silently scan.
        with pytest.raises(ValueError, match="not filterable"):
            FieldFilter(MetadataField.TOKEN_COUNT, FilterOperator.GT, 10)

    def test_is_immutable(self):
        clause = FieldFilter(MetadataField.AUTHOR, FilterOperator.EQ, "Mohamed")

        with pytest.raises(dataclasses.FrozenInstanceError):
            clause.value = "someone else"  # type: ignore[misc]

    def test_compares_by_value(self):
        left = FieldFilter(MetadataField.AUTHOR, FilterOperator.EQ, "Mohamed")
        right = FieldFilter(MetadataField.AUTHOR, FilterOperator.EQ, "Mohamed")

        assert left == right


class TestNullOperatorsTakeNoValue:
    def test_is_null_requires_no_value(self):
        clause = FieldFilter(MetadataField.AUTHOR, FilterOperator.IS_NULL)

        assert clause.value is None

    def test_supplying_a_value_to_is_null_is_rejected(self):
        with pytest.raises(ValueError, match="IS_NULL"):
            FieldFilter(MetadataField.AUTHOR, FilterOperator.IS_NULL, "x")

    def test_comparison_operators_require_a_value(self):
        with pytest.raises(ValueError, match="requires a value"):
            FieldFilter(MetadataField.PAGE_NUMBER, FilterOperator.GT)


class TestMembershipOperators:
    def test_in_accepts_a_sequence(self):
        clause = FieldFilter(MetadataField.DOCUMENT_TYPE, FilterOperator.IN, ["PDF", "DOCX"])

        assert clause.value == ("PDF", "DOCX")

    def test_in_rejects_a_scalar(self):
        with pytest.raises(ValueError, match="sequence"):
            FieldFilter(MetadataField.DOCUMENT_TYPE, FilterOperator.IN, "PDF")

    def test_in_rejects_an_empty_sequence(self):
        # An empty IN matches nothing; it is always a caller bug.
        with pytest.raises(ValueError, match="empty"):
            FieldFilter(MetadataField.DOCUMENT_TYPE, FilterOperator.IN, [])


class TestComposition:
    def test_and_combines_clauses(self):
        left = FieldFilter(MetadataField.DOCUMENT_TYPE, FilterOperator.EQ, "PDF")
        right = FieldFilter(MetadataField.PAGE_NUMBER, FilterOperator.GT, 10)

        combined = And((left, right))

        assert combined.clauses == (left, right)

    def test_ampersand_operator_builds_a_conjunction(self):
        left = FieldFilter(MetadataField.DOCUMENT_TYPE, FilterOperator.EQ, "PDF")
        right = FieldFilter(MetadataField.PAGE_NUMBER, FilterOperator.GT, 10)

        assert left & right == And((left, right))

    def test_pipe_operator_builds_a_disjunction(self):
        left = FieldFilter(MetadataField.AUTHOR, FilterOperator.EQ, "Mohamed")
        right = FieldFilter(MetadataField.AUTHOR, FilterOperator.EQ, "Sara")

        assert left | right == Or((left, right))

    def test_invert_operator_builds_a_negation(self):
        clause = FieldFilter(MetadataField.LANGUAGE, FilterOperator.EQ, "en")

        assert ~clause == Not(clause)

    def test_the_spec_example_composes(self):
        # document_type = PDF AND page_number > 10 AND author = "Mohamed"
        expression = (
            FieldFilter(MetadataField.DOCUMENT_TYPE, FilterOperator.EQ, "PDF")
            & FieldFilter(MetadataField.PAGE_NUMBER, FilterOperator.GT, 10)
            & FieldFilter(MetadataField.AUTHOR, FilterOperator.EQ, "Mohamed")
        )

        assert isinstance(expression, And)
        assert len(expression.clauses) == 3

    def test_chained_conjunction_is_flattened_rather_than_nested(self):
        a = FieldFilter(MetadataField.DOCUMENT_TYPE, FilterOperator.EQ, "PDF")
        b = FieldFilter(MetadataField.PAGE_NUMBER, FilterOperator.GT, 10)
        c = FieldFilter(MetadataField.AUTHOR, FilterOperator.EQ, "Mohamed")

        assert (a & b) & c == And((a, b, c))

    def test_conjunction_requires_at_least_one_clause(self):
        with pytest.raises(ValueError, match="at least one"):
            And(())

    def test_disjunction_requires_at_least_one_clause(self):
        with pytest.raises(ValueError, match="at least one"):
            Or(())

    def test_double_negation_is_preserved_rather_than_silently_simplified(self):
        clause = FieldFilter(MetadataField.LANGUAGE, FilterOperator.EQ, "en")

        assert ~~clause == Not(Not(clause))
