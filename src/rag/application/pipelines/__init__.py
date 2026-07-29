"""Ordered composition of stages.

A pipeline owns *sequence*, timing, logging and per-stage error semantics.
Stages own *behaviour* and are always ports. Inserting or reordering a stage is
therefore a change to one composition function.

Populated from Phase 2 onward.
"""
