"""Windhover — self-hosted LangGraph observability."""
from importlib.metadata import PackageNotFoundError, version as _pkg_version

from .tracer import WindhoverTracer, SpanBuilder

try:
    __version__ = _pkg_version("windhover")
except PackageNotFoundError:  # source tree without an installed dist
    __version__ = "0.0.0.dev0"
__all__ = ["WindhoverTracer", "SpanBuilder", "__version__"]
