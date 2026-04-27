"""This module is adapt from https://github.com/zeno-ml/zeno-build"""

from .providers.hf_utils import generate_from_huggingface_completion
from .providers.openai_utils import (
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
)
from .utils import call_llm

__all__ = [
    "generate_from_openai_completion",
    "generate_from_openai_chat_completion",
    "generate_from_huggingface_completion",
    "generate_from_gemini_completion",
    "call_llm",
]


def __getattr__(name: str):
    # Lazy-load Gemini so Vertex AI is not imported on OpenAI-only runs (avoids asyncio
    # side effects that break Playwright's sync API).
    if name == "generate_from_gemini_completion":
        from .providers.gemini_utils import generate_from_gemini_completion

        return generate_from_gemini_completion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
