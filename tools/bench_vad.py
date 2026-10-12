"""Benchmark the "done speaking" detector (the app's Silero VAD + Endpointer) on audio files, no mic needed.

    python tools/bench_vad.py

For each pre-recorded question (see bench_pipeline.py), append silence with light noise like a real mic, then feed
it in 32ms blocks through the same Endpointer the app uses:
  - endpoint: from when the utterance actually ends to when the VAD reports "done"  (before P1: fixed 1.0s)
  - hesitation: insert a mid-sentence pause (0.3s / 0.45s) to see whether the sentence gets cut off early
"""
import os
import statistics
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from luboo import config  # noqa: E402
from luboo.vad import BLOCK, BLOCK_SEC, Endpointer, SileroVAD  # noqa: E402

NOISE_RMS = 30            # light background noise (headset mic in a quiet room)
_rng = np.random.default_rng(0)


def noise(sec: float) -> np.ndarray:
    return _rng.normal(0, NOISE_RMS, int(sec * config.SAMPLE_RATE)).astype(np.int16)


def with_noise(audio: np.ndarray) -> np.ndarray:
    hiss = _rng.normal(0, NOISE_RMS, len(audio))
    return np.clip(audio.astype(np.int32) + hiss, -32768, 32767).astype(np.int16)


def run_endpointer(audio: np.ndarray, vad: SileroVAD, speech_end: float):
    """Feed audio through VAD + Endpointer like the app. Returns (seconds from speech_end to "done", captured audio)
    or (None, None) if no utterance was detected. A negative value = cut off before speech finished."""
    vad.reset()
    ep = Endpointer()
    for i in range(0, len(audio) - BLOCK + 1, BLOCK):
        block = audio[i:i + BLOCK]
        if ep.feed(block, vad(block)):
            t_detect = (i + BLOCK) / config.SAMPLE_RATE
            return t_detect - speech_end, ep.audio()
    return None, None


def trim(audio: np.ndarray, win_sec: float = 0.01, level: float = 300) -> np.ndarray:
    """Strip leading/trailing silence (edge-tts files have ~0.5s of trailing silence) to know when speech ends."""
    win = int(win_sec * config.SAMPLE_RATE)
    loud = [i for i in range(0, len(audio) - win, win)
            if np.sqrt(np.mean(audio[i:i + win].astype(np.float32) ** 2)) > level]
    return audio[loud[0]:loud[-1] + win] if loud else audio


def measure(audio: np.ndarray, vad: SileroVAD, lead: float = 0.5, tail: float = 2.0):
    """The utterance sits between `lead` seconds and `tail` seconds of noisy silence."""
    audio = trim(audio)
    clip = np.concatenate([noise(lead), with_noise(audio), noise(tail)])
    return run_endpointer(clip, vad, lead + len(audio) / config.SAMPLE_RATE)


def with_pause(audio: np.ndarray, pause: float) -> np.ndarray:
    """Insert silence at the quietest spot near the middle of the sentence (like hesitating mid-sentence)."""
    n = len(audio)
    lo, hi = int(n * 0.35), int(n * 0.65)
    win = BLOCK
    energies = [np.abs(audio[i:i + win].astype(np.int32)).mean() for i in range(lo, hi - win, win)]
    cut = lo + int(np.argmin(energies)) * win
    return np.concatenate([audio[:cut], np.zeros(int(pause * config.SAMPLE_RATE), dtype=np.int16), audio[cut:]])


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from bench_pipeline import AUDIO_DIR, QUESTIONS, prepare_audio, wav_path
    prepare_audio()
    vad = SileroVAD()
    ends, missed, early = [], 0, 0
    pauses = {0.3: [0, 0], 0.45: [0, 0]}       # pause -> [cut off early, total]
    for i, q in enumerate(QUESTIONS):
        audio = np.load(wav_path(i))
        d, _ = measure(audio, vad)
        if d is None:
            missed += 1
            continue
        if d < 0:
            early += 1
        ends.append(d)
        for p in pauses:
            d2, _ = measure(with_pause(trim(audio), p), vad)
            pauses[p][1] += 1
            if d2 is None or d2 < 0:
                pauses[p][0] += 1

    print(f"SILENCE_SEC = {config.SILENCE_SEC}s, VAD_THRESHOLD = {config.VAD_THRESHOLD}, test clips: {len(QUESTIONS)}")
    print(f"Endpoint (speech end -> VAD reports): median {statistics.median(ends)*1000:.0f} ms, "
          f"max {max(ends)*1000:.0f} ms   (before P1: fixed 1000 ms)")
    print(f"Utterances not detected: {missed}   |   reported done before speech finished: {early}")
    for p, (cut, total) in pauses.items():
        print(f"Hesitation of {p:.2f}s mid-sentence: cut off early {cut}/{total}")
    print(f"(Test audio: {AUDIO_DIR})")


if __name__ == "__main__":
    main()
