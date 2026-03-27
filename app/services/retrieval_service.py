"""
Backward-compatible facade for retrieval.

Actual logic has been moved to app/retrieval/ submodules.
This file re-exports retrieve_chunks for existing import points.
"""
from __future__ import annotations

# Re-export retrieve_chunks directly from the service module to avoid
# triggering app/retrieval/__init__.py (which imports app.services.common
# and would create a circular import chain).
from app.retrieval.service import retrieve_chunks

__all__ = ["retrieve_chunks"]
