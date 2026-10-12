"""Mic (always on) + VAD, wake word, utterance recording. (Speech playback: tts.py, speech.py, player.py.)"""
import collections
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd

from . import config, wakeword
from .vad import BLOCK, BLOCK_SEC, Endpointer, SileroVAD


def rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))


class AudioInput:
    """Mic runs continuously (not muted while Luboo thinks / speaks). Each 32ms block is scored by Silero VAD on a
    separate thread, then queued for the coordinator thread. Always keeps the last ~MIC_BUFFER_SEC (for barge-in)."""

    def __init__(self):
        self.vad = SileroVAD()
        self._raw: queue.Queue = queue.Queue()
        self._out: queue.Queue = queue.Queue()
        self.recent = collections.deque(maxlen=round(config.MIC_BUFFER_SEC / BLOCK_SEC))   # [(block, probability)]
        self.last_speech_end = 0.0      # perf_counter() at the last voiced block (for latency measurement)
        self.last_speech_sec = 0.0      # length in seconds of the last recorded utterance (voiced part only)
        self.last_voiced = np.zeros(0, dtype=np.int16)
        self.overflows = 0
        self._running = True
        self._worker = threading.Thread(target=self._score, daemon=True)
        self._worker.start()
        self.stream = sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1, dtype="int16",
                                     blocksize=BLOCK, callback=self._callback)
        self.stream.start()

    def _callback(self, data, _frames, _time, status):
        if status.input_overflow:
            self.overflows += 1
        self._raw.put(data[:, 0].copy())

    def _score(self):
        while self._running:
            try:
                block = self._raw.get(timeout=0.2)
            except queue.Empty:
                continue
            item = (block, self.vad(block))
            self.recent.append(item)
            self._out.put(item)

    # ---------- reading ----------
    def read(self, timeout: float = 0.2):
        """(512-sample int16 block, speech probability) — or None on timeout."""
        try:
            return self._out.get(timeout=timeout)
        except queue.Empty:
            return None

    def flush(self):
        """Discard audio captured while Luboo was thinking / speaking (no barge-in at this stage)."""
        while True:
            try:
                self._out.get_nowait()
            except queue.Empty:
                return

    def level(self, seconds: float = 1.0) -> float:
        """Average mic loudness over the last `seconds` — to check whether the mic is muted."""
        blocks = [b for b, _ in list(self.recent)[-round(seconds / BLOCK_SEC):]]
        return float(np.median([rms(b) for b in blocks])) if blocks else 0.0

    def record_utterance(self, wait_sec: float, stop_event: threading.Event | None = None,
                         ep: Endpointer | None = None):
        """Wait for you to start speaking (up to wait_sec), record until SILENCE_SEC of silence (per Silero VAD).
        ep: continue an utterance already in progress (when you talk over Luboo). Returns an int16 array, or None."""
        ep = ep or Endpointer()
        waited, total = 0.0, len(ep.blocks) * BLOCK_SEC
        while total < config.MAX_RECORD_SEC:
            if stop_event is not None and stop_event.is_set():
                return None
            item = self.read()
            if item is None:
                continue
            total += BLOCK_SEC
            if ep.feed(*item):
                break
            if not ep.started:
                waited += BLOCK_SEC
                if waited > wait_sec:
                    return None
        if not ep.started:
            return None
        self.last_speech_end = time.perf_counter() - ep.silence_after_speech_sec()
        self.last_speech_sec = ep.speech_sec()
        self.last_voiced = ep.speech_audio()             # voiced part only (for speaker recognition / enrollment)
        if ep.speech_sec() < config.MIN_UTTERANCE_SEC:     # too short -> probably just noise
            return None
        return ep.audio()

    def close(self):
        self._running = False
        self.stream.close()


class WakeWordDetector:
    def __init__(self):
        if not getattr(sys, "frozen", False):          # the packaged build already bundles models/
            pretrained = [] if config.USE_CUSTOM_WAKE_WORDS else list(config.WAKE_WORDS)
            wakeword.download_models(config.MODELS_DIR, pretrained)
        self.model = wakeword.WakeWordEngine(
            config.MODELS_DIR, {n: config.wake_word_path(n) for n in config.WAKE_WORDS})
        self.streak = {n: 0 for n in config.WAKE_WORDS}       # consecutive frames above threshold
        self.pending = np.zeros(0, dtype=np.int16)            # gathers 32ms mic blocks into 80ms frames

    def feed(self, block: np.ndarray) -> str | None:
        """Takes mic audio of any size; returns the wake word name if heard."""
        self.pending = np.concatenate([self.pending, block])
        found = None
        while len(self.pending) >= config.FRAME and not found:
            frame, self.pending = self.pending[:config.FRAME], self.pending[config.FRAME:]
            found = self.detect(frame)
        return found

    def detect(self, frame: np.ndarray) -> str | None:
        """frame: exactly 1280 samples. Returns the wake word name if heard, otherwise None."""
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
        self.pending = np.zeros(0, dtype=np.int16)
