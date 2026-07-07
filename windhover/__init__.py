"""Windhover — self-hosted LangGraph observability."""
from .tracer import WindhoverTracer, SpanBuilder

__version__ = "0.3.0"
__all__ = ["WindhoverTracer", "SpanBuilder", "__version__"]
