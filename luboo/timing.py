"""Time each stage of a conversation turn -> %APPDATA%\\Luboo\\timing.jsonl (one JSON line per turn).
See the summary: python tools/timing_report.py"""
import json
import os
import time

from . import config

PATH = os.path.join(config.DATA_DIR, "timing.jsonl")


class TurnTimer:
    """Time 0 = when you stopped speaking (last frame with sound). Each mark() records ms since that point."""

    def __init__(self, speech_end: float, **info):
        self.t0 = speech_end
        self.data = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **info}

    def mark(self, name: str):
        self.data[f"{name}_ms"] = round((time.perf_counter() - self.t0) * 1000)

    def note(self, **info):
        self.data.update(info)

    def save(self, log):
        d = self.data
        stages = [k for k in ("endpoint_ms", "stt_ms", "first_token_ms", "first_chunk_ms", "first_audio_ms", "llm_ms")
                  if k in d]
        log("TIMING " + " ".join(f"{k[:-3]}={d[k]}ms" for k in stages)
            + f" ({d.get('result', '?')}, {d.get('model', '-')}, search={d.get('search', '-')})")
        save_event(d)


def save_event(data: dict):
    """Append one line to timing.jsonl (a Q&A turn, or another event with a "kind", e.g. an interruption)."""
    try:
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        with open(PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
    except OSError:
        pass
