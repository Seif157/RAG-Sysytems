"""The domain layer: the centre of the architecture.

Contains entities, value objects, port interfaces, pure policies and the error
hierarchy. This package imports **nothing but the Python standard library** -- no
framework, no vendor SDK, no configuration, no I/O.

That constraint is what allows every provider in the system to be replaced
without touching business rules, and it is enforced by two ``import-linter``
contracts rather than by convention (ADR-001, ADR-002).
"""
