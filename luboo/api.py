"""Kết nối tới Groq.

Key được lấy theo thứ tự: key người dùng nhập trong Settings -> biến môi trường GROQ_API_KEY
-> key nhúng sẵn trong bản build gửi bạn bè test (`packaging/build.ps1 -EmbedKey`).
"""
import os

import groq

from . import config


class RateLimited(Exception):
    """Groq tạm thời từ chối vì gọi quá nhanh — thử model khác hoặc chờ một lát."""


class NotConfigured(Exception):
    """Chưa có API key."""


def chat_options(search: bool) -> dict:
    """Tham số gửi kèm cho model gpt-oss; search=True bật công cụ tra web của Groq."""
    options = {"temperature": 0.7, "reasoning_effort": config.REASONING_EFFORT}
    if search:
        options.update(tools=[{"type": "browser_search"}], tool_choice="required")
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

    def transcribe(self, wav: bytes, language: str | None = None) -> tuple[str, str | None]:
        kwargs = {"language": language} if language else {"response_format": "verbose_json"}
        try:
            r = self.client.audio.transcriptions.create(
                file=("speech.wav", wav), model=config.STT_MODEL, **kwargs)
        except groq.RateLimitError as e:
            raise RateLimited(str(e)) from e
        return r.text, getattr(r, "language", None)


def embedded_key() -> str:
    """Key nhúng sẵn trong bản build gửi bạn bè test; bản thường không có."""
    try:
        from . import _embedded_key
        return _embedded_key.load()
    except ImportError:
        return ""


def key_source(settings: dict) -> str:
    """'own' (nhập trong Settings), 'env' (GROQ_API_KEY), 'embedded' (bản test) hoặc '' (chưa có key)."""
    if settings["api_key"].strip():
        return "own"
    if os.environ.get("GROQ_API_KEY"):
        return "env"
    return "embedded" if embedded_key() else ""


_client: GroqClient | None = None


def configure(settings: dict):
    """Tạo kết nối theo cài đặt hiện tại. Gọi lại mỗi khi người dùng đổi cài đặt."""
    global _client
    key = settings["api_key"].strip() or os.environ.get("GROQ_API_KEY", "") or embedded_key()
    _client = GroqClient(key) if key else None


def current() -> GroqClient:
    if _client is None:
        raise NotConfigured()
    return _client
