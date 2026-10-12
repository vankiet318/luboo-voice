"""Speaks a reply chunk by chunk; each turn has a turn_id.

    turn = voice.new_turn(...)      # cancel the previous turn (if any), open a new one
    turn.say("Ooh, nice!")          # call as the LLM streams each chunk — the first chunk is spoken right away
    turn.say("It's sunny today.")
    turn.finish()
    spoken = turn.wait()            # what was actually spoken (if interrupted, only the part that was heard)

When a turn is cancelled (voice.cancel() or opening a new turn), all of its audio — playing, queued, or arriving
late from TTS — is dropped. Checking turn_id and feeding the speaker share one lock with cancellation, so stale
audio can never reach the speaker after a cancel.
"""
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config
from .player import Player


class Voice:
    def __init__(self, tts, player: Player):
        self.tts, self.player = tts, player
        self.lock = threading.Lock()
        self.turn_id = 0
        self.current: SpeechTurn | None = None              # turn currently (or most recently) speaking
        self.pool = ThreadPoolExecutor(max_workers=config.TTS_PARALLEL, thread_name_prefix="tts")

    def new_turn(self, on_level=None, on_subtitle=None, on_first_audio=None, log=None) -> "SpeechTurn":
        with self.lock:
            self.turn_id += 1
            self.player.stop()
            self.player.set_gain(1.0)
            tid = self.turn_id
        self.current = SpeechTurn(self, tid, on_level, on_subtitle, on_first_audio, log)
        return self.current

    def speaking(self) -> bool:
        """Luboo is playing audio for the current turn (audio started, not cancelled, not finished)."""
        turn = self.current
        return bool(turn and not turn.cancelled and turn.started_audio and not turn.done)

    def cancel(self):
        """Cancel the current turn: stop the speaker now, drop all unplayed chunks and any late results."""
        with self.lock:
            self.turn_id += 1
            self.player.stop()
            self.player.set_gain(1.0)

    def is_current(self, tid: int) -> bool:
        return tid == self.turn_id

    def beep(self, freq=880, duration=0.12, volume=0.25):
        """A 'ding' to signal the bot heard you."""
        sr = self.tts.SAMPLE_RATE
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        tone = np.sin(2 * np.pi * freq * t) * volume * np.hanning(len(t)) * 32767
        seg = self.player.play(tone.astype(np.int16))
        while not self.player.finished(seg):
            time.sleep(0.01)
        self.player.forget(seg)


class SpeechTurn:
    def __init__(self, voice: Voice, tid: int, on_level, on_subtitle, on_first_audio, log):
        self.voice, self.tid = voice, tid
        self.on_level, self.on_subtitle, self.on_first_audio = on_level, on_subtitle, on_first_audio
        self.log = log or (lambda _msg: None)
        self._futures: queue.Queue = queue.Queue()        # (text chunk, audio future) in order
        self._segments: queue.Queue = queue.Queue()       # (Player segment id, text, subtitle cues)
        self.spoken: list[str] = []                       # chunks fully spoken (or partly, if interrupted)
        self.started_audio = False
        self.done = False                                 # finished speaking (or stopped for good)
        self.dropped = 0                                  # chunks dropped because the turn was cancelled
        self.failed = 0                                   # chunks where TTS failed
        self._feeder = threading.Thread(target=self._feed_player, daemon=True)
        self._tracker = threading.Thread(target=self._track, daemon=True)
        self._feeder.start()
        self._tracker.start()

    @property
    def cancelled(self) -> bool:
        return not self.voice.is_current(self.tid)

    # ---------- receive text ----------
    def say(self, text: str):
        """Add a chunk to speak. Synthesis starts right away (up to TTS_PARALLEL chunks in parallel)."""
        if text and not self.cancelled:
            self._futures.put((text, self.voice.pool.submit(self.voice.tts.synth, text)))

    def finish(self):
        self._futures.put(None)

    def wait(self) -> str:
        self._feeder.join()
        self._tracker.join()
        return " ".join(self.spoken)

    # ---------- feed audio to the speaker in order ----------
    def _feed_player(self):
        while True:
            item = self._futures.get()
            if item is None:
                break
            text, future = item
            try:
                while not future.done() and not self.cancelled:    # wait for TTS but stop at once if cancelled
                    time.sleep(0.02)
                if self.cancelled:
                    self.dropped += 1
                    continue
                samples, cues = future.result()
            except Exception as e:
                self.failed += 1
                self.log(f"TTS failed for a chunk: {e.__class__.__name__}: {e}")
                continue
            if not len(samples):
                continue
            with self.voice.lock:
                if self.cancelled:                             # cancelled while waiting for TTS
                    self.dropped += 1
                    continue
                seg = self.voice.player.play(samples)
                first = not self.started_audio
                self.started_audio = True
            if first and self.on_first_audio:
                self.on_first_audio()
            self._segments.put((seg, text, cues))
        self._segments.put(None)

    # ---------- subtitles + mouth movement for the chunk being played ----------
    def _track(self):
        player = self.voice.player
        while True:
            item = self._segments.get()
            if item is None:
                break
            seg, text, cues = item
            prefix = " ".join(self.spoken)
            shown, end = None, 0
            while not self.cancelled and not player.finished(seg):
                if self.on_level:
                    self.on_level(player.level)
                if self.on_subtitle:
                    elapsed = player.elapsed(seg)
                    end = max((e for t, e in cues if t <= elapsed), default=0) if cues else len(text)
                    if end != shown:
                        shown = end
                        self.on_subtitle((prefix + " " + text[:end]).strip())
                time.sleep(0.03)
            player.forget(seg)
            if self.cancelled:
                if end:
                    self.spoken.append(text[:end].rstrip() + "…")   # only the part that was heard
                break
            self.spoken.append(text)
        if not self.cancelled and self.started_audio:
            time.sleep(player.stream.latency if player.stream is not None else 0)   # tail still in the speaker
            if self.on_subtitle:
                self.on_subtitle(" ".join(self.spoken))
        if self.on_level:
            self.on_level(0.0)
        self.done = True
