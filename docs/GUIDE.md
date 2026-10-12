# Luboo developer guide

1. [Run from source](#1-run-from-source)
2. [Build the `.exe` installer](#2-build-the-exe-installer)
3. [Wake word model](#3-wake-word-model)
4. [Customization](#4-customization)
5. [Troubleshooting](#5-troubleshooting)
6. [Benchmarks](#6-benchmarks)

All commands below run in PowerShell from the repo root.

---

## 1. Run from source

Requires Windows 10/11 64-bit, Python 3.11 (`py -3.11`), a microphone and a Groq API key (free at
https://console.groq.com/keys).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m luboo
```

- On first run the app downloads the 2 openWakeWord feature models into `resources/models/` if they are missing.
- Enter the key in **Settings**, or set the `GROQ_API_KEY` environment variable.
- At startup the log shows `Mic level: …`; a value near 0 means the mic is muted or the wrong input device is selected.
- `requirements-dev.txt` adds local Whisper (fallback when Groq fails) — optional.
- `python -m luboo --selftest` checks the bundled models and exits.

Log: `%APPDATA%\Luboo\luboo.log` · Settings: `%APPDATA%\Luboo\settings.json` ·
Latency log: `%APPDATA%\Luboo\timing.jsonl` · Voice ID: `%APPDATA%\Luboo\voiceprint.npy`.

## 2. Build the `.exe` installer

Also requires Inno Setup 6: `winget install JRSoftware.InnoSetup`

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build.ps1
```

Output: `dist\Luboo-Setup-<version>.exe`. Change the version via `APP_VERSION` in `luboo/config.py`.

`packaging/build.ps1` creates a clean `.venv-build` environment, downloads the models, creates the icon, bundles
the app with PyInstaller (`packaging/luboo.spec`) and wraps it into an installer with Inno Setup
(`packaging/installer.iss`).

**Build with a Groq key embedded (to share with friends for testing):**

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build.ps1 -EmbedKey
```

- The key is taken from the `GROQ_API_KEY` environment variable, or else from the key saved in Luboo's Settings.
- Produces `dist\Luboo-Setup-<version>-PRIVATE.exe` — recipients can use it right after installing.
- The key is only **obfuscated**, not encrypted: a technical person can still extract it. Only share it with
  people you trust; if the key leaks, delete it at console.groq.com/keys and rebuild. Everyone shares the key's
  free quota.
- The temporary file `luboo/_embedded_key.py` is git-ignored and deleted automatically after the build.

The exe isn't code-signed, so Windows SmartScreen warns on first run: click **More info → Run anyway**.

## 3. Wake word model

`resources/wakewords/hey_luboo.onnx` ("Luboo" is pronounced "Loo-boo") is a small network (3 layers, ~0.8 MB)
running on openWakeWord's audio features (`melspectrogram.onnx` + `embedding_model.onnx`, see
`luboo/wakeword.py`).

The model was trained separately on Kaggle (GPU) with:
- ~14,000 synthetic voice samples (Piper, edge-tts) plus real recordings of the app's author;
- negative samples: similar-sounding phrases, ordinary conversation, ~700,000 everyday audio clips (ACAV100M);
- an 80 ms sliding window identical to how the app runs, plus data augmentation (room reverb, noise, voice
  changes).

Results with the current settings (threshold 0.99, 3 consecutive frames): ~1–2 false activations per hour,
~95% of calls recognized.

To replace the model: copy a new `.onnx` file (input `[1, 16, 96]`, output 0..1) into `resources/wakewords/`
and rebuild. If the file is missing, the app falls back to openWakeWord's built-in "Hey Jarvis".

## 4. Customization

In `luboo/config.py`:

| Setting | Meaning |
|---|---|
| `WAKE_THRESHOLD`, `WAKE_PATIENCE` | Score threshold and number of consecutive 80 ms frames that must exceed it (lower = easier to call but more false triggers) |
| `VOICE`, `TTS_RATE`, `TTS_PITCH` | Voice, speed, pitch. Pick a fast voice: Emma/Ava Multilingual are ~10x slower for new text |
| `SILENCE_SEC`, `FOLLOWUP_SEC` | How long a silence counts as "done speaking" (measured with `tools/bench_vad.py`); how long Luboo keeps listening after answering |
| `VAD_THRESHOLD` | Silero VAD threshold (0..1) for "someone is speaking" — raise it if noise is treated as speech |
| `FIRST_CHUNK_MIN_CHARS`, `CHUNK_MIN_CHARS` | Luboo speaks a reply chunk by chunk; the first chunk is cut early (even at a comma) so audio starts sooner |
| `TTS_PARALLEL`, `TTS_FIRST_AUDIO_TIMEOUT` | Number of chunks synthesized at once; if edge-tts is slower than this timeout the request is retried |
| `BARGE_IN`, `BARGE_DUCK_GAIN`, `BARGE_VERIFY_SEC` | Interrupting by voice (only on once Voice ID is set up): volume level while someone is talking, how long to listen before checking the voice |
| `SPEAKER_THRESHOLD` | Threshold for recognizing the owner's voice. Other people interrupt Luboo too often -> raise it; you can't interrupt -> lower it or record your voice again |
| `MERGE_WINDOW_SEC` | If you pause mid-sentence and continue within this time (while Luboo is still thinking), both parts are joined into one question |
| `GREETINGS`, `GREETING_GRACE_SEC` | Lines Luboo greets you with when called; if you say the question in the same breath as the wake word, the greeting is skipped |
| `SYSTEM_PROMPT` | Luboo's personality |
| `CHAT_MODEL`, `SEARCH_MODEL`, `FALLBACK_MODEL` | Groq models |
| `OLLAMA_MODEL` | Last-resort fallback using an LLM running locally |

## 5. Troubleshooting

| Symptom | What to do |
|---|---|
| Doesn't respond when called | Check the mic isn't muted; Windows Settings → Privacy → Microphone must allow desktop apps. A log line `Mic level: 0 (very low…)` means the mic isn't picking anything up |
| Activates by mistake too often | Raise `WAKE_PATIENCE` (3 → 4) or `WAKE_THRESHOLD` |
| Hard to call | Lower `WAKE_PATIENCE` (3 → 2) |
| Cuts you off too early | Raise `SILENCE_SEC` (if you keep talking while Luboo is still thinking, both parts are joined) |
| Talking over Luboo doesn't stop it | Settings → Voice ID → record your voice (voice interruption is only on after setup); if the log shows `Voice while speaking ignored (score …)`, record again or lower `SPEAKER_THRESHOLD` |
| No spoken reply | edge-tts needs internet; see `%APPDATA%\Luboo\luboo.log` |
| "Add a Groq API key in Settings…" | Settings → enter your Groq API key → Save |
| Error 404 "model does not exist" | Groq retired that model — change `CHAT_MODEL`/`SEARCH_MODEL` per console.groq.com/docs/models |

## 6. Benchmarks

Scripts in `tools/` measure latency and voice handling without a microphone. Test audio is generated once with
edge-tts into `%TEMP%` (not into the project).

| Command | Measures |
|---|---|
| `python tools/bench_pipeline.py --label NAME` | 30 fixed questions through the real pipeline (VAD → Groq STT → Groq LLM → edge-tts); results in `tools/bench_results/NAME.jsonl`. Needs a Groq key |
| `python tools/timing_report.py [file]` | Median / p90 / max per stage; with no argument it reads the app's own `%APPDATA%\Luboo\timing.jsonl` |
| `python tools/bench_vad.py` | End-of-speech delay and early cut-offs when hesitating mid-sentence |
| `python tools/bench_barge.py` | Interruptions (owner / other people / short noises), merging, stop delay, stale audio played |
