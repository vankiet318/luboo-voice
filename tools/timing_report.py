"""Summarize latency from %APPDATA%\\Luboo\\timing.jsonl (written by Luboo for every conversation turn).

    python tools/timing_report.py              # everything
    python tools/timing_report.py --last 20    # the 20 most recent turns
    python tools/timing_report.py path\\to\\timing.jsonl

All milestones are measured from when you stop speaking:
  endpoint     wait for silence to know you have finished (SILENCE_SEC)
  stt          + transcription done
  first_token  + LLM produces its first token          (from P2 — read chunk by chunk)
  first_chunk  + first chunk long enough to speak      (from P2)
  llm          + LLM finished                           (P0/P1: before speaking; from P2: usually after audio starts)
  first_audio  + audio starts playing  ->  this is the wait you actually perceive
"""
import json
import os
import statistics
import sys

STAGES = ["endpoint", "stt", "first_token", "first_chunk", "llm", "first_audio"]


def default_path() -> str:
    return os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "Luboo", "timing.jsonl")


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def table(rows, title):
    print(f"\n{title} — {len(rows)} turns")
    print(f"  {'stage':<20}{'median':>9}{'p90':>9}{'max':>9}")
    for i, s in enumerate(STAGES):
        key = f"{s}_ms"
        vals = [r[key] for r in rows if key in r]
        if not vals:
            continue
        print(f"  {'to ' + s:<20}{statistics.median(vals):>8.0f}ms{pct(vals, 90):>7.0f}ms{max(vals):>7.0f}ms")
        own = []                       # own time: relative to the latest earlier milestone present in the turn
        for r in rows:
            if key not in r:
                continue
            earlier = [r[f"{p}_ms"] for p in STAGES[:i] if f"{p}_ms" in r and r[f"{p}_ms"] <= r[key]]
            if earlier:
                own.append(r[key] - max(earlier))
        if own and i:
            print(f"  {'  own ' + s:<20}{statistics.median(own):>8.0f}ms{pct(own, 90):>7.0f}ms{max(own):>7.0f}ms")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")    # default Windows console can't print non-ASCII text
    args = sys.argv[1:]
    last = None
    if "--last" in args:
        i = args.index("--last")
        last = int(args[i + 1])
        del args[i:i + 2]
    path = args[0] if args else default_path()
    if not os.path.exists(path):
        sys.exit(f"No data yet: {path}\nTalk to Luboo for a few turns, then run this again.")
    with open(path, encoding="utf-8") as f:
        allrows = [json.loads(line) for line in f if line.strip()]
    barges = [r for r in allrows if r.get("kind") == "barge"]
    rows = [r for r in allrows if "kind" not in r]
    if last:
        rows = rows[-last:]
        barges = barges[-last:]
    if barges:
        outcomes = {}
        for b in barges:
            outcomes[b["outcome"]] = outcomes.get(b["outcome"], 0) + 1
        stops = [b["react_ms"] for b in barges if b["outcome"] == "interrupted"]
        print("Barge-in: " + ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items()))
              + (f" | full stop after ducking started: median {statistics.median(stops):.0f} ms" if stops else ""))

    results = {}
    for r in rows:
        results[r.get("result", "?")] = results.get(r.get("result", "?"), 0) + 1
    print(f"File: {path}")
    print("Turn results: " + ", ".join(f"{k}={v}" for k, v in sorted(results.items())))

    ok = [r for r in rows if r.get("result") == "ok"]
    if not ok:
        return
    table(ok, "All successful turns")
    for search, name in ((False, "No web search"), (True, "With web search")):
        group = [r for r in ok if r.get("search") is search]
        if group:
            table(group, name)
    models = {}
    for r in ok:
        models[r.get("model", "?")] = models.get(r.get("model", "?"), 0) + 1
    print("\nModels that answered: " + ", ".join(f"{k}={v}" for k, v in models.items()))


if __name__ == "__main__":
    main()
