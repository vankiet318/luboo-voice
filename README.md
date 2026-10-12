# Luboo

An **English** voice assistant for Windows, with an expressive character named Luboo and a minimal interface.
Call Luboo by voice, ask anything, and hear the answer.

## Features

- **"Hey Luboo" wake word** ("Luboo" is pronounced "Loo-boo") — a custom-trained model that runs locally.
- **Web search** for knowledge questions (facts, news, weather, prices); casual chat gets a fast direct reply.
- **Fast replies** — Luboo starts speaking as soon as the first sentence is written (~2.6 s from the end of your question for casual chat), and you can click Luboo to cut it off at any time.
- **Talk over Luboo** — after a 30-second Voice ID setup (Settings), just start talking to interrupt it; other people's voices are ignored. If you pause mid-sentence and keep going while Luboo is still thinking, both parts are joined into one question.
- **Luboo the character** changes expression to match each reply, blinks, looks around, and moves its mouth with the voice.
- **Minimal** — just the character, one line of text and two buttons, *Talk* · *Settings*; click Luboo to talk or to interrupt.
- **Free Groq API key** — enter it in Settings; you can also build a version with a key embedded to share with friends and family for testing.
- **Lightweight** — runs on the CPU; the installer is about 34 MB.

## How it works

```
Mic ─► Wake word ─► Groq Whisper ─► Groq LLM ─► edge-tts ─► Speaker
       (local)      (speech → text)  (+ web search)  (text → speech)
                                          └─► Luboo changes expression
```

| Component | Technology |
|---|---|
| Wake word | Custom-trained model ("Hey Luboo") on openWakeWord features, run with onnxruntime |
| End of speech | Silero VAD (local, onnxruntime) |
| Voice ID (who is speaking) | WeSpeaker ResNet34 (local, onnxruntime) + kaldi-native-fbank |
| Speech → text | Groq `whisper-large-v3-turbo` |
| Replies | Groq `openai/gpt-oss-120b` (+ `browser_search` web tool), falling back to `openai/gpt-oss-20b` |
| Text → speech | edge-tts (`en-US-MichelleNeural`) |
| Interface | Tkinter |

## Privacy

- Audio is sent out only **after** the wake word is heard (or when you press *Talk* / click Luboo).
- Your own API key is stored only on your computer and is sent only to Groq.
- While Luboo thinks or talks, the mic keeps listening **locally** (so you can interrupt or finish your sentence).
- Voice ID stores a voiceprint (numbers, not a recording) only on your computer; remove it in Settings any time.

## Project layout

```
luboo/          app source — run with: python -m luboo
resources/      custom wake word model (wakewords/), downloaded models, icon
packaging/      builds the .exe installer (build.ps1, PyInstaller, Inno Setup)
docs/           guides
tools/          dev scripts — latency / VAD / barge-in benchmarks, timing_report.py
```

| File in `luboo/` | Role |
|---|---|
| `app.py` | Interface: the Luboo character (drawn with Pillow), Settings |
| `assistant.py` | Orchestration (UI-independent): sleep → listen → greet → think → speak, one `turn_id` per turn |
| `audio.py`, `vad.py` | Always-on mic, wake word, Silero VAD + end-of-speech detection |
| `speech.py` | Speaks a reply chunk by chunk as the LLM streams it; cancelling a turn drops all of its audio |
| `speaker_id.py` | Voice ID: enroll your voice, check whether the person talking is you |
| `tts.py`, `player.py` | Pluggable text-to-speech (edge-tts, retries slow requests); audio output queue (instant stop, volume) |
| `timing.py` | Per-turn latency log (`%APPDATA%\Luboo\timing.jsonl`) |
| `wakeword.py` | Wake word detection |
| `brain.py`, `api.py` | Transcription, model selection, Groq calls |
| `config.py`, `settings.py` | Configuration, user settings |

How to run, build, train the wake word model and customize the app: [docs/GUIDE.md](docs/GUIDE.md).

## Third-party notices

Luboo is free software for personal, non-commercial use. It relies on the following third-party
components and services; the full notice text shipped with the installer is in
[THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).

**Services**

- [Groq](https://groq.com) — speech recognition and replies. Your recorded voice and questions are sent to Groq.
- Microsoft Edge online text-to-speech (via edge-tts) — reads replies aloud. Reply text is sent to Microsoft.

**Models**

| Model | Source | License |
|---|---|---|
| "Hey Luboo" wake word (`hey_luboo`) | Trained for Luboo using openWakeWord's pre-computed ACAV100M features and validation set, MIT environmental impulse responses, and synthetic speech from Piper voices and edge-tts | — |
| openWakeWord feature models (`melspectrogram`, `embedding_model`) | © David Scripka — [openWakeWord](https://github.com/dscripka/openWakeWord); the embedding model is derived from Google's speech_embedding model (Apache 2.0) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — **non-commercial use only** |

**Libraries**

| Component | Copyright | License |
|---|---|---|
| ONNX Runtime | Microsoft Corporation | MIT |
| NumPy | NumPy Developers | BSD 3-Clause |
| python-sounddevice | Matthias Geier | MIT |
| PortAudio | Ross Bencina and Phil Burk | MIT-style |
| pyminiaudio / miniaudio | Irmen de Jong / David Reid | MIT |
| Pillow | Jeffrey 'Alex' Clark and contributors; Secret Labs AB; Fredrik Lundh and contributors | MIT-CMU (HPND) |
| edge-tts | rany2 | LGPL-3.0 |
| groq-python | Groq | Apache 2.0 |
| httpx | Encode OSS Ltd. | BSD 3-Clause |
| Python | Python Software Foundation | PSF License |
| Tcl/Tk | The Regents of the University of California and others | BSD-style |

Because the openWakeWord feature models are licensed CC BY-NC-SA 4.0, Luboo as a whole may not be
used commercially.
