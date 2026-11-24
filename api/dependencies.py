"""
api/dependencies.py
===================
Dependency injection for the central RestorationEngine instance.
Ensures the neural model is loaded once at server startup and reused across requests.
"""

from typing import Optional
from inference.restoration_engine import RestorationEngine

_engine_instance: Optional[RestorationEngine] = None


def get_restoration_engine() -> RestorationEngine:
    """Provides singleton access to the loaded RestorationEngine."""
    global _engine_instance
    if _engine_instance is None:
        # Fallback lazy initialization if accessed before lifespan handler
        _engine_instance = RestorationEngine()
    return _engine_instance


def set_restoration_engine(engine: RestorationEngine) -> None:
    """Sets the global engine instance (typically during application lifespan startup)."""
    global _engine_instance
    _engine_instance = engine

