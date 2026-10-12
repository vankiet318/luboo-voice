"""Measure Luboo's latency without speaking: feed pre-recorded questions (edge-tts voices) through the same
real processing path — Groq Whisper -> Groq LLM (+ web search) -> edge-tts — just without the mic and speakers.

    python tools/bench_pipeline.py --label P0           # run the question set, write tools/bench_results/P0.jsonl
    python tools/bench_pipeline.py --label P0 --limit 5 # trial run with 5 questions
    python tools/timing_report.py tools/bench_results/P0.jsonl

Needs a Groq API key (Luboo Settings or the GROQ_API_KEY env var). Each question costs 1 STT call + 1–2 LLM calls.
The "endpoint" step (wait for silence): feed the question + noisy silence through the app's Silero VAD (bench_vad.py).
(The P0 run — before the VAD existed — used config.SILENCE_SEC = 1.0s.)
"""
import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import time
import types

import edge_tts
import miniaudio
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from luboo import api, brain, config, settings, timing  # noqa: E402
from luboo.assistant import Assistant  # noqa: E402
from luboo.speech import Voice  # noqa: E402
from luboo.vad import SileroVAD  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_vad import measure as measure_endpoint  # noqa: E402

AUDIO_DIR = os.path.join(tempfile.gettempdir(), "luboo_bench_audio")   # outside the project so the repo stays small
RESULTS_DIR = os.path.join(ROOT, "tools", "bench_results")
VOICES = ["en-US-GuyNeural", "en-US-JennyNeural", "en-GB-RyanNeural", "en-US-AndrewMultilingualNeural"]

# Same question set for every run so before/after can be compared. Don't edit — if you add questions,
# rename the results file.
QUESTIONS = [
    # casual chat (no web search)
    "Hey, how are you doing?", "Tell me a joke.", "I feel a bit tired today.",
    "What's your name?", "Can you say something nice to me?", "I just finished my work, finally!",
    "Do you like music?", "Give me a fun fact.", "I'm bored. Any ideas?",
    "Say good night to me.", "Why is the sky blue?", "How many legs does a spider have?",
    "Can you help me think of a name for my cat?", "I'm learning to cook.", "Thanks, that's all for now.",
    # needs web search (per the current SEARCH_HINTS)
    "What's the weather in Hanoi today?", "What is the latest news about AI?",
    "How much is an iPhone 17 Pro?", "Who won the football match last night?",
    "When is the next full moon?", "Where is the nearest airport to Da Lat?",
    "What is the best laptop for students?", "How do I make pho at home?",
    "Who is the president of France?", "What's the price of gold today?",
    "Which phone has the best camera?", "How to fix a slow Windows laptop?",
    "Find me a good recipe for pancakes.", "What time does the Louvre open?",
    "What's the score of the Real Madrid game?",
]


def wav_path(i):
    return os.path.join(AUDIO_DIR, f"q{i:02d}.npy")


async def synth(text, voice) -> np.ndarray:
    mp3 = b""
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk["type"] == "audio":
            mp3 += chunk["data"]
    pcm = miniaudio.decode(mp3, output_format=miniaudio.SampleFormat.SIGNED16, nchannels=1,
                           sample_rate=config.SAMPLE_RATE)
    return np.frombuffer(pcm.samples, dtype=np.int16)


def prepare_audio():
    """Record the questions once (16 kHz mono like the mic) and reuse them later so runs are identical."""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    for i, q in enumerate(QUESTIONS):
        if not os.path.exists(wav_path(i)):
            audio = asyncio.run(synth(q, VOICES[i % len(VOICES)]))
            np.save(wav_path(i), audio)
            time.sleep(0.5)
    print(f"Audio ready: {len(QUESTIONS)} questions in {AUDIO_DIR}")


def words(t):
    return re.sub(r"[^\w\s]", "", t.lower()).split()


class SilentPlayer:
    """Fake player: opens no speaker, treats playback as finished instantly. Tracks total seconds of audio "played"."""
    stream = None
    level = 0.0

    def __init__(self):
        self.n, self.audio_sec = 0, 0.0

    def play(self, samples):
        self.n += 1
        self.audio_sec += len(samples) / 24000
        return self.n

    def finished(self, _seg): return True
    def elapsed(self, _seg): return 1e9
    def forget(self, _seg): pass
    def stop(self): pass
    def set_gain(self, _gain): pass
    def close(self): pass


def run(label, limit, pause):
    api.configure(settings.load())
    try:
        api.current()
    except api.NotConfigured:
        sys.exit("No Groq API key. Add one in Luboo Settings or set GROQ_API_KEY, then run again.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timing.PATH = os.path.join(RESULTS_DIR, f"{label}.jsonl")
    if os.path.exists(timing.PATH):
        sys.exit(f"{timing.PATH} already exists — pick another --label or delete it.")
    log = lambda msg: print("   ", msg)  # noqa: E731
    noop = lambda *_a: None  # noqa: E731

    # Run the app's real orchestration code (Assistant._handle: STT -> LLM stream -> chunking -> parallel TTS),
    # only replacing the mic with audio files and the speaker with SilentPlayer.
    extra = {}
    replies = []
    bot = Assistant(on_state=noop, on_message=lambda role, t: replies.append(t) if role == "bot" else None,
                    on_level=noop, on_log=log, on_subtitle=noop)
    player = SilentPlayer()
    bot.voice = Voice(bot.tts, player)
    real_transcribe = brain.transcribe

    def transcribe(audio, log_):
        text = real_transcribe(audio, log_)
        extra.update(stt_text=text, stt_exact=words(text) == words(extra["expected"]))
        return text

    brain.transcribe = transcribe
    real_save = timing.TurnTimer.save

    def save(timer, log_):
        timer.data.update(extra, reply=replies[-1] if replies else "",
                          reply_audio_sec=round(player.audio_sec, 2), tts_retries=bot.tts.retries)
        real_save(timer, log_)

    timing.TurnTimer.save = save
    vad = SileroVAD()

    for i, q in enumerate(QUESTIONS[:limit]):
        print(f"[{i + 1}/{min(limit, len(QUESTIONS))}] {q}")
        # Feed the question (plus noisy silence) through the app's real VAD + Endpointer: learn how long the app
        # takes to notice you've finished, and get exactly the audio the app would send for transcription.
        endpoint_sec, audio = measure_endpoint(np.load(wav_path(i)), vad)
        if audio is None:
            print("    VAD did not detect speech — skipped")
            continue
        extra.clear()
        extra.update(expected=q, label=label)
        replies.clear()
        player.audio_sec = 0.0
        bot.tts.retries = 0
        bot.history.clear()                                   # each question is independent, like earlier runs
        bot.mic = types.SimpleNamespace(last_speech_end=time.perf_counter() - endpoint_sec)
        bot._handle(audio)
        time.sleep(pause)                                     # avoid hitting the Groq free-tier rate limit

    print(f"\nSaved: {timing.PATH}\nReport: python tools/timing_report.py {os.path.relpath(timing.PATH, ROOT)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--label", help="run name, e.g. P0 — results go to tools/bench_results/<label>.jsonl")
    p.add_argument("--limit", type=int, default=len(QUESTIONS))
    p.add_argument("--pause", type=float, default=3.0, help="pause between questions (seconds)")
    p.add_argument("--prepare-only", action="store_true", help="only pre-record the questions, don't call Groq")
    a = p.parse_args()
    prepare_audio()
    if not a.prepare_only:
        if not a.label:
            p.error("--label is required")
        run(a.label, a.limit, a.pause)
