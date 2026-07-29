"""The application layer: orchestration.

Answers *what the system does*, never *how it is done*. Depends on
:mod:`rag.domain` and on nothing else -- no vendor SDK, no configuration object,
no infrastructure. Every use case here is executable against in-memory fakes
with zero network access, which is what makes the test suite fast and hermetic.
"""
