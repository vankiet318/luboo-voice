"""STT (giọng nói -> chữ) và LLM (suy nghĩ, trả lời), có dự phòng nhiều tầng."""
import io
import json
import re
import urllib.request
import wave

import numpy as np

from . import api, config

# Whisper hay "bịa" ra mấy câu này khi gặp đoạn âm thanh gần như im lặng
HALLUCINATION_EXACT = ["you", "thank you", "thanks", "bye"]
HALLUCINATION_CONTAINS = [
    "thanks for watching", "subscribe", "like and subscribe",
]

EMOTIONS = ["happy", "sad", "surprised", "angry", "thinking", "neutral"]


def _has(text: str, phrases: list[str]) -> bool:
    """Có cụm từ nào xuất hiện như một từ/cụm từ trọn vẹn không (không khớp nửa chữ)."""
    return any(re.search(rf"(?<!\w){re.escape(p)}(?!\w)", text) for p in phrases)


# ======================= STT =======================

def to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(config.SAMPLE_RATE)
        w.writeframes(audio.astype(np.int16).tobytes())
    return buf.getvalue()


def _transcribe_groq(audio: np.ndarray) -> str:
    text, _ = api.current().transcribe(to_wav_bytes(audio), language="en")
    return text.strip()


_local_model = None


def _transcribe_local(audio: np.ndarray) -> str:
    global _local_model
    from faster_whisper import WhisperModel  # chỉ cần khi dùng dự phòng

    if _local_model is None:
        _local_model = WhisperModel(config.LOCAL_STT_MODEL, device="cpu", compute_type="int8")
    segments, _ = _local_model.transcribe(audio.astype(np.float32) / 32768, beam_size=1, language="en")
    return " ".join(s.text for s in segments).strip()


def transcribe(audio: np.ndarray, log) -> str:
    """Trả về câu đã chép (tiếng Anh). Chuỗi rỗng nếu không nghe được gì."""
    try:
        text = _transcribe_groq(audio)
    except api.NotConfigured:
        raise                                   # để assistant báo cho người dùng
    except Exception as e:
        log(f"Groq STT failed ({e.__class__.__name__}: {e})")
        try:
            text = _transcribe_local(audio)
        except ImportError:
            return ""                           # bản đóng gói không kèm Whisper trên máy
        except Exception as e2:
            log(f"Local STT failed too: {e2}")
            return ""

    cleaned = re.sub(r"[^\w\s]", "", text.lower()).strip()
    if not cleaned or cleaned in HALLUCINATION_EXACT or _has(cleaned, HALLUCINATION_CONTAINS):
        return ""
    return text


# ======================= Định tuyến =======================

SEARCH_HINTS = [
    # English
    "how do", "how to", "how can", "where", "what is", "what's the", "who is", "when",
    "which", "best", "build", "guide", "beat", "unlock", "find", "location", "recipe",
    "weakness", "patch", "latest", "news", "weather", "look up", "search",
]
SMALL_TALK = [
    "how are you", "what's up", "who are you", "what is your name", "what's your name",
]


def needs_search(text: str) -> bool:
    t = text.lower()
    if _has(t, SMALL_TALK):
        return False
    return _has(t, SEARCH_HINTS)


# ======================= LLM =======================

def _ask_ollama(messages: list[dict]) -> str:
    body = json.dumps({"model": config.OLLAMA_MODEL, "messages": messages, "stream": False,
                       "think": False}).encode()
    req = urllib.request.Request(config.OLLAMA_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["message"]["content"]


def ask(question: str, history: list[dict], log) -> str:
    messages = ([{"role": "system", "content": config.SYSTEM_PROMPT}]
                + history[-config.HISTORY_TURNS * 2:]
                + [{"role": "user", "content": question}])

    chain = []                                   # (model, có tra web không)
    if needs_search(question):
        chain.append((config.SEARCH_MODEL, True))
    chain += [(config.CHAT_MODEL, False), (config.FALLBACK_MODEL, False)]
    if config.OLLAMA_MODEL:
        chain.append(("ollama", False))

    for model, search in chain:
        try:
            log(f"LLM {model}" + (" + web search" if search else ""))
            if model == "ollama":
                return _ask_ollama(messages)
            return api.current().chat(model, messages, search)
        except api.NotConfigured:
            if not config.OLLAMA_MODEL:
                raise
            log("LLM ollama")                    # chưa có key -> chỉ còn Ollama trên máy
            return _ask_ollama(messages)
        except api.RateLimited:
            log(f"{model} rate-limited, trying the next model")
        except Exception as e:
            log(f"{model} failed: {e.__class__.__name__}: {e}")

    return "[sad] Sorry, I can't reach any model right now."


def split_emotion(reply: str) -> tuple[str, str]:
    """Tách nhãn cảm xúc ở đầu câu trả lời. Trả về (cảm xúc, câu sạch để đọc)."""
    emotion = "neutral"
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
    m = re.match(r"\s*\[(\w+)\]\s*", reply)
    if m and m.group(1).lower() in EMOTIONS:
        emotion = m.group(1).lower()
        reply = reply[m.end():]
    return emotion, clean_for_speech(reply)


def clean_for_speech(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)   # model có chế độ suy nghĩ
    text = re.sub(r"https?://\S+", "", text)                     # link
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", "", text)              # link markdown
    text = re.sub(r"\[(\d+|\w+)\]|【[^】]*】", "", text)           # trích dẫn, nhãn thừa
    text = re.sub(r"[*_#`>|]", "", text)                         # ký hiệu markdown
    return re.sub(r"\s+", " ", text).strip()
