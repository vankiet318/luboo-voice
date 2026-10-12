"""Connection to Groq.

The key is taken in this order: key entered by the user in Settings -> GROQ_API_KEY environment variable
-> key embedded in the build sent to friends for testing (`packaging/build.ps1 -EmbedKey`).
"""
import os

import groq

from . import config


class RateLimited(Exception):
    """Groq is temporarily refusing because of too many calls — try another model or wait a moment."""


class NotConfigured(Exception):
    """No API key yet."""


def chat_options(search: bool) -> dict:
    """Parameters sent to the gpt-oss models; search=True enables Groq's web search tool."""
    options = {"temperature": 0.7, "reasoning_effort": config.REASONING_EFFORT,
               "max_completion_tokens": config.MAX_COMPLETION_TOKENS,
               "timeout": config.SEARCH_TIMEOUT if search else config.CHAT_TIMEOUT}
    if search:
        # "auto": the model decides whether to search. "required" makes Groq return a 400 whenever the model decides
        # no search is needed (7/30 sentences in the P0 run) -> another call is needed, ~0.5–1s slower.
        options.update(tools=[{"type": "browser_search"}], tool_choice="auto")
    return options


class GroqClient:
    def __init__(self, api_key: str):
        self.client = groq.Groq(api_key=api_key, max_retries=1)

    def chat(self, model: str, messages: list[dict], search: bool = False) -> str:
        try:
            r = self.client.chat.completions.create(model=model, messages=messages, **chat_options(search))
        except groq.RateLimitError as e:
            raise RateLimited(str(e)) from e
        return r.choices[0].message.content or ""

    def chat_stream(self, model: str, messages: list[dict], search: bool = False):
        """Like chat() but yields text pieces as soon as the model writes them (to speak the first sentence early)."""
        try:
            stream = self.client.chat.completions.create(model=model, messages=messages, stream=True,
                                                         **chat_options(search))
        except groq.RateLimitError as e:
            raise RateLimited(str(e)) from e
        try:
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except groq.RateLimitError as e:
            raise RateLimited(str(e)) from e
        finally:
            stream.close()                       # stopped early (interrupted) -> close the connection, stop reading

    def transcribe(self, wav: bytes, language: str | None = None) -> tuple[str, str | None]:
        kwargs = {"language": language} if language else {"response_format": "verbose_json"}
        try:
            r = self.client.audio.transcriptions.create(
                file=("speech.wav", wav), model=config.STT_MODEL, timeout=config.STT_TIMEOUT, **kwargs)
        except groq.RateLimitError as e:
            raise RateLimited(str(e)) from e
        return r.text, getattr(r, "language", None)


def embedded_key() -> str:
    """Key embedded in the build sent to friends for testing; regular builds have none."""
    try:
        from . import _embedded_key
        return _embedded_key.load()
    except ImportError:
        return ""


def key_source(settings: dict) -> str:
    """'own' (entered in Settings), 'env' (GROQ_API_KEY), 'embedded' (test build) or '' (no key)."""
    if settings["api_key"].strip():
        return "own"
    if os.environ.get("GROQ_API_KEY"):
        return "env"
    return "embedded" if embedded_key() else ""


_client: GroqClient | None = None


def configure(settings: dict):
    """Create the connection from the current settings. Call again whenever the user changes settings."""
    global _client
    key = settings["api_key"].strip() or os.environ.get("GROQ_API_KEY", "") or embedded_key()
    _client = GroqClient(key) if key else None


def current() -> GroqClient:
    if _client is None:
        raise NotConfigured()
    return _client
