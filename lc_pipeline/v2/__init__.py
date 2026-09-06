"""Shared catalog contracts, data utilities, and benchmark implementations.

Lazy catalog exports keep provenance tooling independent of numerical-model
imports and installed model artifacts.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "CATALOG_SCHEMA",
    "COORDINATE_FRAME",
    "DIRECTIONALITY",
    "PINNED_DUMP_ID",
    "CatalogError",
    "build_catalog",
    "build_grouped_splits",
]


def __getattr__(name: str) -> Any:
    """Lazily expose catalog contracts without eager numerical imports."""
    if name in __all__:
        module = import_module(".catalog", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
