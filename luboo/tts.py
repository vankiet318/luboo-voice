"""Text -> speech. Every TTS engine shares the same "shape" synth(text) -> (int16 24 kHz audio, subtitle cues),
so switching to another engine (e.g. Kokoro running locally) only needs one more class like EdgeTTS."""
import asyncio
import threading
import time

import edge_tts
import miniaudio
import numpy as np

from . import config

SAMPLE_RATE = 24000


class EdgeTTS:
    """Microsoft Edge online voices (edge-tts library, unofficial)."""
    SAMPLE_RATE = SAMPLE_RATE

    def __init__(self, voice: str | None = None):
        self.voice = voice or config.VOICE
        self._cache: dict[str, tuple[np.ndarray, list]] = {}    # sentence -> pre-synthesized (audio, subtitle cues)
        self._lock = threading.Lock()
        self.retries = 0                                       # number of resends (for measurement)

    # ---------- synthesis ----------
    async def _once(self, text: str) -> tuple[bytes, list[tuple[float, str]]]:
        """One edge-tts call. No audio within TTS_FIRST_AUDIO_TIMEOUT -> TimeoutError."""
        stream = edge_tts.Communicate(text, self.voice, rate=config.TTS_RATE, pitch=config.TTS_PITCH,
                                      boundary="WordBoundary").stream()
        loop = asyncio.get_running_loop()
        t0, data, words = loop.time(), b"", []
        try:
            while True:
                limit = config.TTS_FIRST_AUDIO_TIMEOUT if not data else config.TTS_CHUNK_TIMEOUT
                remaining = limit - (loop.time() - t0)
                if remaining <= 0:
                    raise asyncio.TimeoutError
                try:
                    chunk = await asyncio.wait_for(stream.__anext__(), remaining)
                except StopAsyncIteration:
                    return data, words
                if chunk["type"] == "audio":
                    data += chunk["data"]
                elif chunk["type"] == "WordBoundary":
                    words.append((chunk["offset"] / 10_000_000, chunk["text"]))   # 100 ns units -> seconds
        finally:
            await stream.aclose()

    async def _synthesize(self, text: str):
        # The Edge service occasionally spikes in latency or returns nothing -> cancel and resend (new connection)
        for attempt in range(config.TTS_ATTEMPTS):
            try:
                return await self._once(text)
            except (asyncio.TimeoutError, edge_tts.exceptions.NoAudioReceived):
                if attempt == config.TTS_ATTEMPTS - 1:
                    raise
                with self._lock:
                    self.retries += 1
                await asyncio.sleep(0.2 * attempt)

    def synth(self, text: str) -> tuple[np.ndarray, list[tuple[float, int]]]:
        """(int16 24 kHz audio, [(start seconds, chars spoken so far)]). Pre-synthesized sentences return at once."""
        if text in self._cache:
            return self._cache[text]
        mp3, words = asyncio.run(self._synthesize(text))
        if not mp3:
            return np.zeros(0, dtype=np.int16), []
        decoded = miniaudio.decode(mp3, output_format=miniaudio.SampleFormat.SIGNED16,
                                   nchannels=1, sample_rate=SAMPLE_RATE)
        return np.frombuffer(decoded.samples, dtype=np.int16), subtitle_cues(text, words)

    # ---------- pre-synthesized sentences (greetings) ----------
    def preload(self, texts: list[str]):
        """Pre-synthesize common sentences — runs in the background at app start."""
        for text in texts:
            if text in self._cache:
                continue
            try:
                self._cache[text] = self.synth(text)
            except Exception:
                pass                                            # network error: will fetch again when needed
            time.sleep(0.8)                                     # avoid Microsoft rate limiting

    def is_ready(self, text: str) -> bool:
        return text in self._cache


def subtitle_cues(text: str, words: list[tuple[float, str]]) -> list[tuple[float, int]]:
    """Match each word edge-tts returns to its position in the sentence -> [(start seconds, chars shown so far)].
    Punctuation attached after a word ("tough," "left!") is shown together with that word."""
    cues, pos, lower = [], 0, text.lower()
    for start, word in words:
        i = lower.find(word.lower(), pos)
        if i < 0:
            continue
        end = i + len(word)
        while end < len(text) and not text[end].isspace() and not text[end].isalnum():
            end += 1
        cues.append((start, end))
        pos = end
    return cues
