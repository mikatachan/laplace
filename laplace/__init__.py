"""Laplace — priority-aware admission + idle/external-tenant reaping for a shared local LLM pool.

Public API (settling during extraction — see EXTRACTION.md):
    ContentionBroker, Reaper, PriorityTier, resolve_priority, InferenceAdapter

The core (broker/reaper/priority) is engine-agnostic and depends only on an
``InferenceAdapter``. Concrete adapters live under ``laplace.adapters`` — the
LM Studio adapter is the reference implementation.
"""

__version__ = "0.1.0"
