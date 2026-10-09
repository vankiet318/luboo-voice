"""Mic, từ kích hoạt, thu âm câu nói và phát giọng đọc."""
import asyncio
import sys
import threading
import time

import edge_tts
import miniaudio
import numpy as np
import sounddevice as sd

from . import config, wakeword


def rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))


class Microphone:
    def __init__(self):
        self.stream = sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1,
                                     dtype="int16", blocksize=config.FRAME)
        self.threshold = config.MIN_SPEECH_THRESHOLD

    def start(self):
        if not self.stream.active:
            self.stream.start()

    def pause(self):
        """Tạm dừng mic khi đang xử lý / đang nói, để bot không tự nghe chính mình."""
        if self.stream.active:
            self.stream.stop()

    def read(self) -> np.ndarray:
        audio, _ = self.stream.read(config.FRAME)
        return audio.flatten()

    def calibrate(self, seconds=1.5):
        """Đo tiếng ồn nền để tự chọn ngưỡng 'có tiếng nói'."""
        self.start()
        levels = [rms(self.read()) for _ in range(int(seconds * config.SAMPLE_RATE / config.FRAME))]
        noise = float(np.median(levels))
        self.threshold = max(noise * 3, config.MIN_SPEECH_THRESHOLD)
        return noise, self.threshold

    def record_utterance(self, wait_sec: float, stop_event: threading.Event | None = None):
        """Chờ bạn bắt đầu nói (tối đa wait_sec), ghi đến khi im lặng SILENCE_SEC.
        Trả về mảng int16, hoặc None nếu không ai nói gì."""
        frame_sec = config.FRAME / config.SAMPLE_RATE
        chunks, start_idx, silent, elapsed = [], None, 0.0, 0.0
        while elapsed < config.MAX_RECORD_SEC:
            if stop_event is not None and stop_event.is_set():
                return None
            frame = self.read()
            chunks.append(frame)
            elapsed += frame_sec
            if rms(frame) > self.threshold:
                if start_idx is None:
                    start_idx = len(chunks) - 1
                silent = 0.0
            elif start_idx is not None:
                silent += frame_sec
                if silent >= config.SILENCE_SEC:
                    break
            elif elapsed > wait_sec:
                return None
        if start_idx is None:
            return None
        pre_roll = int(0.4 / frame_sec)          # giữ lại 0.4s trước khi bắt đầu nói
        speech = np.concatenate(chunks[max(0, start_idx - pre_roll):])
        if len(speech) < 0.4 * config.SAMPLE_RATE:   # quá ngắn -> có thể là tiếng động
            return None
        return speech


class WakeWordDetector:
    def __init__(self):
        if not getattr(sys, "frozen", False):          # bản đóng gói đã kèm sẵn models/
            pretrained = [] if config.USE_CUSTOM_WAKE_WORDS else list(config.WAKE_WORDS)
            wakeword.download_models(config.MODELS_DIR, pretrained)
        self.model = wakeword.WakeWordEngine(
            config.MODELS_DIR, {n: config.wake_word_path(n) for n in config.WAKE_WORDS})
        self.streak = {n: 0 for n in config.WAKE_WORDS}       # số khung liên tiếp đạt ngưỡng

    def detect(self, frame: np.ndarray) -> str | None:
        """Trả về tên từ kích hoạt nếu nghe thấy, ngược lại None."""
        scores = self.model.predict(frame)
        for name in self.streak:
            self.streak[name] = self.streak[name] + 1 if scores.get(name, 0) >= config.WAKE_THRESHOLD else 0
        ready = [n for n, k in self.streak.items() if k >= config.WAKE_PATIENCE]
        if ready:
            best = max(ready, key=lambda name: scores.get(name, 0))
            self.reset()
            return best
        return None

    def reset(self):
        self.model.reset()
        self.streak = {n: 0 for n in config.WAKE_WORDS}


def beep(freq=880, duration=0.12, volume=0.25):
    """Tiếng 'ting' báo bot đã nghe thấy."""
    sr = 24000
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = np.sin(2 * np.pi * freq * t) * volume * np.hanning(len(t))
    sd.play(tone.astype(np.float32), sr)
    sd.wait()


class Speaker:
    SAMPLE_RATE = 24000

    def __init__(self):
        self._stop = threading.Event()
        self._cache: dict[str, tuple[np.ndarray, list]] = {}    # câu -> (âm thanh, mốc phụ đề) đã tạo sẵn

    async def _synthesize(self, text: str, voice: str) -> tuple[bytes, list[tuple[float, str]]]:
        """Trả về (mp3, [(giây bắt đầu, từ)...]) — thời điểm từng từ dùng để chạy phụ đề khớp giọng."""
        # Dịch vụ của Edge thỉnh thoảng trả về rỗng (bị giới hạn tạm thời) -> thử lại vài lần
        for attempt in range(3):
            try:
                data, words = b"", []
                async for chunk in edge_tts.Communicate(text, voice, rate=config.TTS_RATE,
                                                        boundary="WordBoundary").stream():
                    if chunk["type"] == "audio":
                        data += chunk["data"]
                    elif chunk["type"] == "WordBoundary":
                        words.append((chunk["offset"] / 10_000_000, chunk["text"]))   # đơn vị 100 ns -> giây
                return data, words
            except edge_tts.exceptions.NoAudioReceived:
                if attempt == 2 or self._stop.is_set():
                    raise
                await asyncio.sleep(1.0 * (attempt + 1))

    def stop(self):
        self._stop.set()
        sd.stop()

    def _prepare(self, text: str) -> tuple[np.ndarray, list]:
        """Tải giọng đọc + giải mã. Câu đã tạo sẵn (preload) thì lấy ngay, không cần mạng."""
        if text in self._cache:
            return self._cache[text]
        mp3, words = asyncio.run(self._synthesize(text, config.VOICE))
        if not mp3:
            return np.zeros(0, dtype=np.int16), []
        decoded = miniaudio.decode(mp3, output_format=miniaudio.SampleFormat.SIGNED16,
                                   nchannels=1, sample_rate=self.SAMPLE_RATE)
        return np.frombuffer(decoded.samples, dtype=np.int16), subtitle_cues(text, words)

    def preload(self, texts: list[str]):
        """Tạo sẵn giọng đọc cho các câu hay dùng (lời chào) — chạy nền lúc mở app."""
        for text in texts:
            if text in self._cache:
                continue
            try:
                self._cache[text] = self._prepare(text)
            except Exception:
                pass                                            # lỗi mạng: lúc cần sẽ tải lại
            time.sleep(0.8)                                     # tránh bị Microsoft giới hạn

    def is_ready(self, text: str) -> bool:
        return text in self._cache

    def say(self, text: str, on_level=None, on_subtitle=None):
        """Đọc câu trả lời. on_level(0..1): âm lượng để mấp máy miệng.
        on_subtitle(chữ): phần câu đã được đọc tới — hiện dần đúng lúc từng từ phát ra."""
        self._stop.clear()
        samples, cues = self._prepare(text)
        if not len(samples) or self._stop.is_set():
            return

        sd.play(samples, self.SAMPLE_RATE)
        started = time.time()
        duration = len(samples) / self.SAMPLE_RATE
        window = int(0.05 * self.SAMPLE_RATE)
        shown = None
        while not self._stop.is_set():
            elapsed = time.time() - started
            pos = int(elapsed * self.SAMPLE_RATE)
            if pos >= len(samples):
                break
            if on_level:
                on_level(min(1.0, rms(samples[pos:pos + window]) / 6000))
            if on_subtitle:
                end = max((e for t, e in cues if t <= elapsed), default=0)
                if end != shown:
                    shown = end
                    on_subtitle(text[:end] if cues else text)
            time.sleep(0.03)
        if not self._stop.is_set():
            time.sleep(max(0.0, duration - (time.time() - started)))
            if on_subtitle:
                on_subtitle(text)
        if on_level:
            on_level(0.0)


def subtitle_cues(text: str, words: list[tuple[float, str]]) -> list[tuple[float, int]]:
    """Khớp từng từ edge-tts trả về với vị trí trong câu -> [(giây bắt đầu, số ký tự hiện tới)].
    Dấu câu dính liền sau từ ("tough," "left!") được hiện cùng từ đó."""
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
