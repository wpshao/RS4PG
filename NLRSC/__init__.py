# type: ignore[attr-defined]
"""The core implementation of RS4PG"""

from importlib import metadata as importlib_metadata

from .solver import quadratic_flow
from .utils import feats_sampling


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()

__all__ = ["quadratic_flow", "feats_sampling", "version"]
