"""Player: manages its own audio output stream (instead of sd.play).

One OutputStream stays open and the callback pulls audio from a queue. This allows:
  - stopping instantly and clearing everything not yet played (stop),
  - changing volume while playing (set_gain — used to "duck" when interrupted in a later stage),
  - knowing exactly how far playback has got (elapsed) to drive subtitles and mouth movement.
"""
import collections
import itertools
import threading

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 24000                  # edge-tts decodes to 24 kHz
BLOCK = 480                          # 20ms per callback


class Player:
    def __init__(self):
        self._lock = threading.Lock()
        self._queue: collections.deque = collections.deque()      # [(id, samples)]
        self._current = None                                      # (id, samples)
        self._pos = 0                                             # samples played of the current segment
        self._played: dict[int, int] = {}                         # id -> samples played
        self._done: set[int] = set()
        self._ids = itertools.count(1)
        self.gain = 1.0
        self.level = 0.0                                          # 0..1 loudness of the last played block
        self.stream = None
        self._open()

    def _open(self):
        # latency="low": Windows default is ~200ms (audio reaches the speaker that much later and stops that much later)
        self.stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                      blocksize=BLOCK, latency="low", callback=self._callback)
        self.stream.start()

    def _callback(self, out, frames, _time, _status):
        buf = np.zeros(frames, dtype=np.int16)
        filled = 0
        with self._lock:
            while filled < frames:
                if self._current is None:
                    if not self._queue:
                        break
                    self._current, self._pos = self._queue.popleft(), 0
                seg_id, samples = self._current
                n = min(frames - filled, len(samples) - self._pos)
                buf[filled:filled + n] = samples[self._pos:self._pos + n]
                filled += n
                self._pos += n
                self._played[seg_id] = self._pos
                if self._pos >= len(samples):
                    self._done.add(seg_id)
                    self._current = None
            gain = self.gain
        if gain != 1.0:
            buf = np.clip(buf.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
        self.level = min(1.0, float(np.sqrt(np.mean(buf.astype(np.float32) ** 2))) / 6000) if filled else 0.0
        out[:, 0] = buf

    # ---------- control ----------
    def play(self, samples: np.ndarray) -> int:
        """Queue an audio segment (int16, 24 kHz). Returns the segment id."""
        if self.stream is None or not self.stream.active:      # device changed / turned off -> reopen
            try:
                self._open()
            except Exception:
                pass
        seg_id = next(self._ids)
        with self._lock:
            self._queue.append((seg_id, np.ascontiguousarray(samples, dtype=np.int16)))
        return seg_id

    def stop(self):
        """Stop immediately, dropping the segment being played and all queued segments."""
        with self._lock:
            for seg_id, _ in self._queue:
                self._done.add(seg_id)
            if self._current is not None:
                self._done.add(self._current[0])
            self._queue.clear()
            self._current = None
        self.level = 0.0

    def set_gain(self, gain: float):
        self.gain = gain

    # ---------- state ----------
    def finished(self, seg_id: int) -> bool:
        with self._lock:
            return seg_id in self._done

    def elapsed(self, seg_id: int) -> float:
        """Seconds of segment seg_id played so far (minus device latency, to match when it is heard)."""
        with self._lock:
            played = self._played.get(seg_id, 0)
        latency = self.stream.latency if self.stream is not None else 0.0
        return max(0.0, played / SAMPLE_RATE - latency)

    def forget(self, seg_id: int):
        with self._lock:
            self._played.pop(seg_id, None)
            self._done.discard(seg_id)

    def close(self):
        self.stop()
        if self.stream is not None:
            self.stream.close()
