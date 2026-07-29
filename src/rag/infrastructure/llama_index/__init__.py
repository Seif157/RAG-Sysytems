"""LlamaIndex quarantine.

LlamaIndex is used as a *library* inside adapters, never as an architectural
layer (ADR-002). Conversions between LlamaIndex types and domain types happen
here and nowhere else, so removing the dependency touches a bounded set of files
(architecture spec section 15.1).

Populated from Phase 2.
"""
