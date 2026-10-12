"""Benchmark barge-in and sentence merging with the app's real orchestration code (Assistant._run_turn / _watch),
no mic, speaker or Groq needed. A fake mic feeds audio in real time (32ms blocks through the real Silero VAD),
a fake speaker simulates playback time, and speaker ID uses the real WeSpeaker. STT / LLM / TTS are delayed fakes.

    python tools/bench_barge.py            # ~3–4 min; test voices (edge-tts) are generated once in %TEMP%

Scenarios:
  owner      the owner cuts in while Luboo is speaking            -> must stop
  other      someone else talks while Luboo is speaking           -> must not stop
  short      short ~0.2s sound (cough, "um") while Luboo speaks   -> must not stop
  merge      you pause mid-sentence, then continue while Luboo thinks -> merged into one sentence, old turn silent
  no_enroll  the owner cuts in but no voice is enrolled           -> no stop (barge-in disabled)
Metrics: correct-handling rate, time to duck / fully stop measured from speech start, count of stale audio played.
"""
import asyncio
import hashlib
import os
import statistics
import sys
import tempfile
import threading
import time

import edge_tts
import miniaudio
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from luboo import assistant as A  # noqa: E402
from luboo import brain, config, speech, timing  # noqa: E402
from luboo.audio import AudioInput  # noqa: E402
from luboo.speaker_id import SpeakerID  # noqa: E402
from luboo.vad import BLOCK, BLOCK_SEC, SileroVAD  # noqa: E402

DATA = os.path.join(tempfile.gettempdir(), "luboo_barge_bench")
OWNER, OTHERS = "en-US-AndrewNeural", ["en-US-JennyNeural", "en-US-BrianNeural", "en-GB-RyanNeural"]
LINES = ["Wait, I meant tomorrow, not today.", "No, stop, that's not what I asked.",
         "Actually, tell me something else instead.", "Hold on, let me think about that.",
         "Okay, but what about the weekend?", "Sorry, can you say that again slower?"]
RNG = np.random.default_rng(1)


# ---------------- test voice data ----------------
async def _tts(text, voice):
    for attempt in range(5):
        try:
            mp3 = b""
            async for c in edge_tts.Communicate(text, voice).stream():
                if c["type"] == "audio":
                    mp3 += c["data"]
            pcm = miniaudio.decode(mp3, output_format=miniaudio.SampleFormat.SIGNED16, nchannels=1,
                                   sample_rate=config.SAMPLE_RATE)
            return np.frombuffer(pcm.samples, dtype=np.int16)
        except edge_tts.exceptions.NoAudioReceived:
            await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError("edge-tts failed")


def trim(a, level=300, win=160):
    loud = [i for i in range(0, len(a) - win, win) if np.sqrt(np.mean(a[i:i + win].astype(float) ** 2)) > level]
    return a[loud[0]:loud[-1] + win]


def clip(voice, text):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, f"{voice}_{hashlib.md5(text.encode()).hexdigest()[:10]}.npy")
    if not os.path.exists(path):
        np.save(path, trim(asyncio.run(_tts(text, voice))))
        time.sleep(0.5)
    return np.load(path)


# ---------------- fake mic / speaker / STT / LLM / TTS ----------------
class FakeMic(AudioInput):
    """Fake mic: every 32ms returns a block (background noise, or audio "spoken" via say()), scored by the real VAD."""

    def __init__(self):                                     # does not open a real mic
        self.vad = SileroVAD()
        self.pending = np.zeros(0, dtype=np.int16)
        self.lock = threading.Lock()
        self.next_t = time.perf_counter()
        self.last_speech_end, self.last_speech_sec = time.perf_counter(), 0.0
        self.last_voiced = np.zeros(0, dtype=np.int16)
        self.said_at = None                                 # perf_counter when the utterance's first block was fed in

    def say(self, audio):
        with self.lock:
            self.pending = np.concatenate([self.pending, audio])
            self.said_at = None

    def read(self, timeout=0.2):
        now = time.perf_counter()
        if self.next_t > now:
            if self.next_t - now > timeout:
                time.sleep(timeout)
                return None
            time.sleep(self.next_t - now)
        self.next_t = max(self.next_t + BLOCK_SEC, time.perf_counter() - 0.1)
        noise = RNG.normal(0, 30, BLOCK)
        with self.lock:
            take, self.pending = self.pending[:BLOCK], self.pending[BLOCK:]
            if len(take) and self.said_at is None:
                self.said_at = time.perf_counter()
        block = noise.copy()
        block[:len(take)] += take
        block = np.clip(block, -32768, 32767).astype(np.int16)
        return block, self.vad(block)

    def flush(self):
        pass


class SimPlayer:
    """Fake speaker: plays segments in order for their real duration; records duck / stop / play events."""
    stream = None
    level = 0.2

    def __init__(self, voice_ref):
        self.voice_ref = voice_ref
        self.lock = threading.Lock()
        self.queue, self.n, self.busy_until = [], 0, time.perf_counter()
        self.ends, self.stopped = {}, set()
        self.events = []                                    # (time, kind, turn_id)

    def play(self, samples):
        with self.lock:
            self.n += 1
            start = max(time.perf_counter(), self.busy_until)
            self.busy_until = start + len(samples) / 24000
            self.ends[self.n] = self.busy_until
            self.events.append((time.perf_counter(), "play", self.voice_ref().turn_id))
            return self.n

    def stop(self):
        with self.lock:
            now = time.perf_counter()
            for seg, end in self.ends.items():
                if end > now:
                    self.stopped.add(seg)
            self.busy_until = now
            self.events.append((now, "stop", self.voice_ref().turn_id))

    def set_gain(self, g):
        self.events.append((time.perf_counter(), f"gain{g}", self.voice_ref().turn_id))

    def finished(self, seg):
        return seg in self.stopped or time.perf_counter() >= self.ends.get(seg, 0)

    def elapsed(self, seg):
        return 1e9

    def forget(self, seg):
        pass

    def close(self):
        pass


class FakeTTS:
    SAMPLE_RATE = 24000
    retries = 0

    def synth(self, text):
        time.sleep(0.35)
        return np.ones(int(24000 * 0.04 * len(text)), np.int16), [(0.0, len(text))]   # ~0.04s per character

    def is_ready(self, _):
        return True

    def preload(self, _):
        pass


LONG_REPLY = ("[happy] Ooh, sure! So here's the plan for today. It's going to be sunny in the morning, "
              "a bit cloudy after lunch, and there might be a little rain in the evening, so bring an umbrella. "
              "Also, the temperature will drop quite a bit tonight, so a light jacket is a good idea. Anything else?")


def fake_stream(_q, _h, _log, info):
    info.update(model="fake", search=False)
    time.sleep(0.5)
    for i in range(0, len(LONG_REPLY), 8):
        yield LONG_REPLY[i:i + 8]
        time.sleep(0.01)


# ---------------- run one scenario ----------------
def make_bot(sid, enrolled):
    bot = A.Assistant(lambda *a: None, lambda *a: None, lambda *a: None, lambda *a: None, lambda *a: None)
    bot.tts = FakeTTS()
    bot.voice = speech.Voice(bot.tts, None)
    bot.voice.player = SimPlayer(lambda: bot.voice)
    bot.mic = FakeMic()
    bot.speaker_id = sid if enrolled else SpeakerID(profile_path=os.path.join(DATA, "none.npy"))
    return bot


def scenario_speaking(bot, interrupt_audio, delay=1.0):
    """Luboo is answering a question; `delay` seconds after audio starts, speak interrupt_audio into the mic."""
    question = clip(OWNER, "What's the weather going to be like today?")
    bot.mic.last_speech_end = time.perf_counter()
    out = {}

    def inject():
        while bot.voice.current is None or not bot.voice.current.started_audio:
            time.sleep(0.01)
        time.sleep(delay)
        bot.mic.say(interrupt_audio)
        while bot.mic.said_at is None:
            time.sleep(0.002)
        out["said_at"] = bot.mic.said_at

    threading.Thread(target=inject, daemon=True).start()
    t0 = time.perf_counter()
    result = bot._run_turn(question)
    out.update(result=result, wall=time.perf_counter() - t0)
    return out


def analyse(bot, out):
    ev = bot.voice.player.events
    said = out.get("said_at")
    duck = next((t for t, k, _ in ev if k == f"gain{config.BARGE_DUCK_GAIN}" and said and t >= said), None)
    stop = next((t for t, k, _ in ev if k == "stop" and said and t >= said), None)
    stale = 0
    if stop is not None:
        old_tid = next(tid for t, k, tid in ev if k == "stop" and t == stop) - 1
        stale = sum(1 for t, k, tid in ev if k == "play" and t > stop and tid == old_tid)
    return {"duck_ms": None if duck is None else (duck - said) * 1000,
            "stop_ms": None if stop is None else (stop - said) * 1000,
            "stopped": stop is not None, "stale": stale}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    timing.PATH = os.path.join(DATA, "events.jsonl")
    brain.transcribe = lambda audio, log: (time.sleep(0.3), "what's the weather")[1]
    brain.ask_stream = fake_stream
    print("Preparing voices (first run only)…")
    enroll = [clip(OWNER, s) for s in config.ENROLL_SENTENCES]
    owner_lines = [clip(OWNER, s) for s in LINES]
    other_lines = [clip(v, s) for v in OTHERS for s in LINES[:3]]
    sid = SpeakerID(profile_path=os.path.join(DATA, "owner.npy"))
    sid.enroll(enroll)

    results = {k: [] for k in ("owner", "other", "short", "merge", "no_enroll")}
    for a in owner_lines:
        bot = make_bot(sid, True)
        out = scenario_speaking(bot, a)
        results["owner"].append({**analyse(bot, out), "got_new": out["result"] is not None})
    for a in other_lines:
        bot = make_bot(sid, True)
        out = scenario_speaking(bot, a)
        results["other"].append(analyse(bot, out))
    for a in owner_lines:
        bot = make_bot(sid, True)
        out = scenario_speaking(bot, a[:int(0.2 * config.SAMPLE_RATE)])
        results["short"].append(analyse(bot, out))
    for a in owner_lines[:3]:
        bot = make_bot(sid, False)
        out = scenario_speaking(bot, a)
        results["no_enroll"].append(analyse(bot, out))
    for i in range(4):                                       # merge: keep talking 0.5s after Luboo starts thinking
        bot = make_bot(sid, True)
        first, second = clip(OWNER, "What's the weather"), clip(OWNER, "in Hanoi tomorrow afternoon?")
        bot.mic.last_speech_end = time.perf_counter()
        threading.Timer(0.5, lambda b=bot, s=second: b.mic.say(s)).start()
        res = bot._run_turn(first)
        played = [e for e in bot.voice.player.events if e[1] == "play"]
        merged_ok = res is not None and len(res) >= len(first) + len(second) - 2 * BLOCK
        results["merge"].append({"merged": merged_ok, "old_played": len(played)})

    def rate(rows, key):
        return f"{sum(bool(r[key]) for r in rows)}/{len(rows)}"

    def med(rows, key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return f"{statistics.median(v):.0f} ms (max {max(v):.0f})" if v else "-"

    o, ot, sh, ne, mg = (results[k] for k in ("owner", "other", "short", "no_enroll", "merge"))
    print(f"\nOwner cuts in          : stopped {rate(o, 'stopped')} | heard the new sentence {rate(o, 'got_new')}")
    print(f"   ducked after        : {med(o, 'duck_ms')}   (from speech start)")
    print(f"   fully stopped after : {med(o, 'stop_ms')}")
    print(f"Someone else talks     : false stops {rate(ot, 'stopped')}")
    print(f"Short sound (~0.2s)    : false stops {rate(sh, 'stopped')}")
    print(f"Voice not enrolled     : stopped {rate(ne, 'stopped')} (must be 0 — barge-in disabled)")
    print(f"Merge while thinking   : merged correctly {rate(mg, 'merged')} | old turn played audio "
          f"{sum(r['old_played'] for r in mg)} times")
    print(f"Stale audio after stop : {sum(r['stale'] for rows in results.values() for r in rows if 'stale' in r)}")
    os.remove(timing.PATH) if os.path.exists(timing.PATH) else None


if __name__ == "__main__":
    main()
