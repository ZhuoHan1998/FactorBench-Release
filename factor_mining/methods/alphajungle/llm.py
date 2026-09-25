"""Alpha Jungle's view of the shared LLM client.

The client moved to ``factor_mining.llm`` when CogAlpha needed the same
plumbing. Re-exported here so this method's imports stay stable.
"""

from factor_mining.llm import (  # noqa: F401
    TEMPERATURE_CORRECT,
    TEMPERATURE_GENERATE,
    TEMPERATURE_SCORE,
    ChatClient,
    LLMError,
    OpenAIChatClient,
    UsageStats,
    complete_json,
    extract_json,
    load_dotenv,
)

__all__ = [
    "TEMPERATURE_CORRECT",
    "TEMPERATURE_GENERATE",
    "TEMPERATURE_SCORE",
    "ChatClient",
    "LLMError",
    "OpenAIChatClient",
    "UsageStats",
    "complete_json",
    "extract_json",
    "load_dotenv",
]
