"""STT (speech -> text) and LLM (thinking, replying), with multi-level fallbacks."""
import io
import json
import re
import urllib.request
import wave

import numpy as np

from . import api, config

# Whisper often "hallucinates" these phrases on near-silent audio
HALLUCINATION_EXACT = ["you", "thank you", "thanks", "bye"]
HALLUCINATION_CONTAINS = [
    "thanks for watching", "subscribe", "like and subscribe",
]

EMOTIONS = ["happy", "sad", "surprised", "angry", "thinking", "neutral"]


def _has(text: str, phrases: list[str]) -> bool:
    """Whether any phrase appears as a whole word/phrase (no partial-word matches)."""
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
    from faster_whisper import WhisperModel  # only needed for the fallback

    if _local_model is None:
        _local_model = WhisperModel(config.LOCAL_STT_MODEL, device="cpu", compute_type="int8")
    segments, _ = _local_model.transcribe(audio.astype(np.float32) / 32768, beam_size=1, language="en")
    return " ".join(s.text for s in segments).strip()


def transcribe(audio: np.ndarray, log) -> str:
    """Return the transcribed sentence (English). Empty string if nothing was heard."""
    try:
        text = _transcribe_groq(audio)
    except api.NotConfigured:
        raise                                   # let the assistant notify the user
    except Exception as e:
        log(f"Groq STT failed ({e.__class__.__name__}: {e})")
        try:
            text = _transcribe_local(audio)
        except ImportError:
            return ""                           # packaged build does not bundle local Whisper
        except Exception as e2:
            log(f"Local STT failed too: {e2}")
            return ""

    cleaned = re.sub(r"[^\w\s]", "", text.lower()).strip()
    if not cleaned or cleaned in HALLUCINATION_EXACT or _has(cleaned, HALLUCINATION_CONTAINS):
        return ""
    return text


# ======================= Routing =======================

SEARCH_HINTS = [
    # English
    "how do", "how to", "how can", "where", "what is", "what's the", "who is", "when",
    "which", "best", "find", "recipe", "price", "how much", "open", "score", "who won", "near",
    "latest", "news", "weather", "look up", "search",
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


def ask_stream(question: str, history: list[dict], log, info: dict | None = None):
    """Yield the reply piece by piece. If a model fails before writing anything -> try the next model;
    if it fails midway, stop at what we have. info (if given) records which model replied and whether it searched."""
    info = {} if info is None else info
    messages = ([{"role": "system", "content": config.SYSTEM_PROMPT}]
                + history[-config.HISTORY_TURNS * 2:]
                + [{"role": "user", "content": question}])

    chain = []                                   # (model, web search or not)
    if needs_search(question):
        chain.append((config.SEARCH_MODEL, True))
    chain += [(config.CHAT_MODEL, False), (config.FALLBACK_MODEL, False)]
    if config.OLLAMA_MODEL:
        chain.append(("ollama", False))

    for model, search in chain:
        info.update(model=model, search=search)
        wrote = False
        try:
            log(f"LLM {model}" + (" + web search" if search else ""))
            if model == "ollama":
                yield _ask_ollama(messages)
                return
            for piece in api.current().chat_stream(model, messages, search):
                wrote = True
                yield piece
            return
        except api.NotConfigured:
            if not config.OLLAMA_MODEL:
                raise
            log("LLM ollama")                    # no key yet -> only local Ollama left
            info.update(model="ollama", search=False)
            yield _ask_ollama(messages)
            return
        except api.RateLimited:
            log(f"{model} rate-limited" + ("" if wrote else ", trying the next model"))
        except Exception as e:
            log(f"{model} failed: {e.__class__.__name__}: {e}")
        if wrote:                                # already spoke part of it to the user -> don't switch models midway
            return

    info.update(model="none", search=False)
    yield "[sad] Sorry, I can't reach any model right now."


def ask(question: str, history: list[dict], log, info: dict | None = None) -> str:
    """The full reply (non-streaming)."""
    return "".join(ask_stream(question, history, log, info))


# ======================= Split the reply into chunks for speaking =======================

_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "e.g", "i.e", "no", "approx", "jr", "sr"}
_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*\s")
_CLAUSE_END = re.compile(r"[,;:—–]\s")


class ReplyStream:
    """Takes text pieces streamed from the LLM; strips the leading emotion tag and returns chunks ready to speak.

        rs = ReplyStream()
        for piece in ask_stream(...):
            for chunk in rs.feed(piece): speak(chunk)
        for chunk in rs.flush(): speak(chunk)
        rs.emotion, rs.text  -> emotion, the full cleaned reply
    """

    def __init__(self):
        self.buf = ""
        self.emotion: str | None = None
        self.parts: list[str] = []               # chunks already emitted (cleaned)

    @property
    def text(self) -> str:
        return " ".join(self.parts)

    def feed(self, piece: str) -> list[str]:
        self.buf += piece
        if self.emotion is None and not self._take_emotion(final=False):
            return []
        out = []
        while (chunk := self._next_chunk()) is not None:
            out += self._emit(chunk)
        return out

    def flush(self) -> list[str]:
        if self.emotion is None:
            self._take_emotion(final=True)
        rest, self.buf = self.buf, ""
        return self._emit(rest)

    # ---------- internals ----------
    def _take_emotion(self, final: bool) -> bool:
        """Parse the leading [happy]... tag. False if there isn't enough text yet to tell."""
        s = self.buf.lstrip()
        if not s and not final:
            return False
        if s.startswith("[") and "]" not in s and len(s) < 20 and not final:
            return False
        m = re.match(r"\[(\w+)\]\s*", s)
        if m and m.group(1).lower() in EMOTIONS:
            self.emotion, self.buf = m.group(1).lower(), s[m.end():]
        else:
            self.emotion, self.buf = "neutral", s
        return True

    def _next_chunk(self) -> str | None:
        first = not self.parts
        min_chars = config.FIRST_CHUNK_MIN_CHARS if first else config.CHUNK_MIN_CHARS
        for m in _SENTENCE_END.finditer(self.buf):
            if m.end() < min_chars and not first:
                continue                                   # sentence too short -> merge with the next one
            word = self.buf[:m.start()].split()[-1:] or [""]
            if word[0].lower().rstrip(".") in _ABBREVIATIONS:
                continue                                   # "Dr. Smith" is not a sentence end
            return self._cut(m.end())
        if first:                                          # first chunk: also cut at commas to get audio sooner
            for m in _CLAUSE_END.finditer(self.buf):
                if m.start() >= config.FIRST_CHUNK_MIN_CHARS:
                    return self._cut(m.end())
        if len(self.buf) > config.CHUNK_MAX_CHARS:         # no punctuation -> cut at whitespace
            cut = self.buf.rfind(" ", 0, config.CHUNK_MAX_CHARS)
            return self._cut(cut if cut > 0 else config.CHUNK_MAX_CHARS)
        return None

    def _cut(self, n: int) -> str:
        chunk, self.buf = self.buf[:n], self.buf[n:]
        return chunk

    def _emit(self, chunk: str) -> list[str]:
        clean = clean_for_speech(chunk)
        if not re.search(r"\w", clean):                    # only punctuation / citations left -> drop
            return []
        self.parts.append(clean)
        return [clean]


def clean_for_speech(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)   # models with a thinking mode
    text = re.sub(r"https?://\S+", "", text)                     # link
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", "", text)              # markdown links
    text = re.sub(r"\[(\d+|\w+)\]|【[^】]*】", "", text)           # citations, stray tags
    text = re.sub(r"[*_#`>|]", "", text)                         # markdown symbols
    return re.sub(r"\s+", " ", text).strip()
