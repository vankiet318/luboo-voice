"""Orchestrator: wait for wake word -> listen -> think -> speak -> keep listening -> sleep.

UI-independent: reports out only through callbacks (Tkinter today, a small screen / LEDs later).
The reply is spoken chunk by chunk as soon as the LLM writes it (speech.SpeechTurn); each turn has a turn_id,
and clicking Luboo (stop_speaking) cancels the running turn immediately — even while thinking.
While Luboo thinks / speaks, the mic keeps listening (_watch): merges when you keep talking, barges in when the
owner cuts in.
"""
import datetime
import queue
import random
import threading
import time
import traceback

import numpy as np

from . import api, brain, config, timing
from .audio import AudioInput, WakeWordDetector
from .player import Player
from .speaker_id import SpeakerID
from .speech import Voice
from .timing import TurnTimer
from .tts import EdgeTTS
from .vad import Endpointer

NOT_CONNECTED_REPLY = "[sad] I'm not connected yet. Please open Settings and add a Groq API key."


def all_greetings() -> list[str]:
    return [g for group in config.GREETINGS.values() for g in group]


def pick_greeting(recent: list[str]) -> str:
    hour = datetime.datetime.now().hour
    part = "morning" if 5 <= hour < 12 else "afternoon" if hour < 18 else "evening" if hour < 22 else "night"
    fresh = lambda pool: [g for g in pool if g not in recent]   # noqa: E731
    timely = fresh(config.GREETINGS[part]) if random.random() < 0.3 else []
    return random.choice(timely or fresh(config.GREETINGS["any"]) or config.GREETINGS["any"])


def wake_words_hint() -> str:
    return " · ".join(f"“{config.WAKE_LABELS.get(w, w)}”" for w in config.WAKE_WORDS)


_END = object()


def _cancellable(pieces, cancelled):
    """Iterate over text pieces (usually the LLM stream) on a separate thread so we can stop IMMEDIATELY on cancel —
    even while the LLM is still searching the web and hasn't written anything. Stream errors (e.g. no key yet) are
    re-raised here."""
    q: queue.Queue = queue.Queue()

    def pump():
        try:
            for p in pieces:
                if cancelled():
                    break
                q.put(p)
        except BaseException as e:   # noqa: BLE001 — hand the error to the orchestrator thread
            q.put(e)
        finally:
            if hasattr(pieces, "close"):
                pieces.close()       # close the LLM stream (close the connection, stop reading)
            q.put(_END)

    threading.Thread(target=pump, daemon=True).start()
    while True:
        try:
            item = q.get(timeout=0.05)
        except queue.Empty:
            if cancelled():
                return               # the pump thread stops on its own at the next piece
            continue
        if item is _END:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


class Assistant(threading.Thread):
    """Runs on its own thread. Reports status to the UI through callbacks:
    on_state(state, emotion), on_message(role, text), on_level(0..1), on_subtitle(text being spoken),
    on_enroll(voice enrollment event) — see enroll_voice()."""

    def __init__(self, on_state, on_message, on_level, on_log, on_subtitle, on_enroll=None):
        super().__init__(daemon=True)
        self.on_state, self.on_message = on_state, on_message
        self.on_level, self.log = on_level, on_log
        self.on_subtitle = on_subtitle
        self.on_enroll = on_enroll or (lambda *_: None)
        self.history: list[dict] = []
        self.running = True
        self.push_to_talk = threading.Event()
        self.end_conversation = threading.Event()
        self.enroll_request = threading.Event()
        self.enroll_cancel = threading.Event()
        self.tts = EdgeTTS()
        self.voice: Voice | None = None          # created in run() (needs the speaker open)
        self.speaker_id: SpeakerID | None = None
        self.recent_greetings: list[str] = []
        self._merge_pending = False              # running turn was cancelled to merge with the next utterance

    # ---------- controls from the UI ----------
    def talk_now(self):
        self.push_to_talk.set()

    def stop_speaking(self):
        """Clicking Luboo while listening / thinking / speaking: stop immediately, cancel the running turn."""
        self.end_conversation.set()
        if self.voice:
            self.voice.cancel()

    def clear_history(self):
        self.history.clear()

    def shutdown(self):
        self.running = False
        self.enroll_cancel.set()
        self.stop_speaking()

    def voice_enrolled(self) -> bool:
        return bool(self.speaker_id and self.speaker_id.enrolled)

    def barge_in_enabled(self) -> bool:
        return config.BARGE_IN and self.voice_enrolled()

    def enroll_voice(self):
        """Start voice enrollment (runs on the Assistant thread while Luboo sleeps). Progress is reported via on_enroll:
        ("prompt", i, sentence), ("heard", i), ("retry", i), ("done", scores), ("failed", reason)."""
        self.enroll_cancel.clear()
        self.stop_speaking()                     # in a conversation -> go back to sleep to enroll
        self.enroll_request.set()

    def cancel_enroll(self):
        self.enroll_cancel.set()

    def forget_voice(self):
        if self.speaker_id:
            self.speaker_id.forget()

    # ---------- main loop ----------
    def run(self):
        try:
            self.on_state("loading", "neutral")
            self.log("Loading wake word model...")
            self.voice = Voice(self.tts, Player())
            self.wake = WakeWordDetector()
            self.speaker_id = SpeakerID()
            self.mic = AudioInput()
            time.sleep(1.0)
            level = self.mic.level()
            self.log(f"Mic level: {level:.0f}" + ("  (very low — is the mic muted or the wrong device?)"
                                                  if level < 5 else ""))
            self.log("Voice ID: " + ("enrolled — your voice can interrupt Luboo" if self.voice_enrolled()
                                     else "not set up — interrupting by voice is off"))
            self.log(f"Ready. Say {wake_words_hint()}")
            threading.Thread(target=self.tts.preload, args=(all_greetings(),), daemon=True).start()
        except Exception as e:
            self.log(f"Failed to start: {e}")
            self.on_state("error", "sad")
            traceback.print_exc()
            return

        while self.running:
            try:
                if not self._wait_for_trigger():
                    break
                self._conversation()
            except Exception as e:
                if not self.running:
                    break
                self.log(f"Error: {e}")
                traceback.print_exc()
            finally:
                self.wake.reset()
        self.mic.close()
        self.voice.player.close()

    def _wait_for_trigger(self) -> bool:
        """Wait for the wake word or the Talk button. False if the app is shutting down."""
        self.on_state("sleeping", "neutral")
        self.mic.flush()                                 # drop audio captured while Luboo was speaking
        while self.running:
            if self.enroll_request.is_set():
                self.enroll_request.clear()
                self._enroll()
                self.on_state("sleeping", "neutral")
                self.mic.flush()
                self.wake.reset()
            item = self.mic.read()
            if self.push_to_talk.is_set():
                self.push_to_talk.clear()
                return True
            if item is not None and self.wake.feed(item[0]):
                self.log("Wake word detected")
                return True
        return False

    def _conversation(self):
        self.end_conversation.clear()
        # "Hey Luboo, what's the weather?" in one go -> answer right away; call then pause -> Luboo greets first.
        # No flush here: whatever is said right after the wake word is the question.
        self.on_state("listening", "neutral")
        speech = self.mic.record_utterance(config.GREETING_GRACE_SEC, self.end_conversation)
        if speech is None or self.mic.last_speech_sec < 0.6:          # shorter: may just be the "...boo" tail
            speech = self._run_turn(None)                              # greet; talking over the greeting = question
            wait = config.FIRST_WAIT_SEC
        else:
            wait = config.FOLLOWUP_SEC
        while not self.end_conversation.is_set():
            # Speech caught while Luboo thinks / speaks (merge, barge-in) is handled right away, no waiting
            while speech is not None and not self.end_conversation.is_set():
                speech = self._run_turn(speech)
                wait = config.FOLLOWUP_SEC                             # keep talking without calling again
            if self.end_conversation.is_set():
                return
            self.on_state("listening", "neutral")
            speech = self.mic.record_utterance(wait, self.end_conversation)
            if speech is None:
                return                                                 # silence -> go back to sleep

    def _greet(self):
        text = pick_greeting(self.recent_greetings)
        self.recent_greetings = (self.recent_greetings + [text])[-3:]
        if self.tts.is_ready(text):
            self._speak([f"[happy] {text}"])
        else:
            self.voice.beep()            # audio not ready (app just opened / offline) -> old "ting"

    # ---------- one turn: runs on a worker thread, this thread keeps listening to the mic ----------
    def _run_turn(self, speech):
        """speech=None: greeting. While Luboo thinks / speaks, the mic is still monitored:
          - thinking (nothing spoken yet) and you keep talking within MERGE_WINDOW_SEC -> cancel, merge both into one;
          - speaking and the owner cuts in (voice enrolled) -> duck immediately, verify voice, stop, hear the new one;
          - greeting: anyone may speak (that is the question).
        Returns the new utterance to handle right away (or None)."""
        greeting = speech is None
        speech_end = self.mic.last_speech_end
        self._merge_pending = False
        first_tid = self.voice.turn_id + 1           # speech turns of this run have turn_id >= this
        worker = threading.Thread(target=self._greet if greeting else self._handle,
                                  args=() if greeting else (speech, speech_end), daemon=True)
        worker.start()
        try:
            return self._watch(worker, speech, speech_end, greeting, first_tid)
        finally:
            worker.join()

    def _phase(self, first_tid) -> str:
        """'thinking' (no audio yet) / 'speaking' (playing) / 'done' (finished speaking) for the current run."""
        turn = self.voice.current
        if turn is None or turn.tid < first_tid or not turn.started_audio:
            return "thinking"
        return "done" if turn.done or turn.cancelled else "speaking"

    def _watch(self, worker, speech, speech_end, greeting, first_tid):
        ep, ducked, ignoring, verified_at = Endpointer(), False, None, 0.0
        player = self.voice.player
        while True:
            if self.end_conversation.is_set():
                return None
            item = self.mic.read(timeout=0.03)
            if not worker.is_alive() and (item is None or not ep.started):
                if ep.started:                                         # Luboo finished while you were talking
                    return self._finish(ep, ducked)
                return None
            if item is None:
                continue
            block, prob = item
            if ignoring is not None:                                   # currently ignoring a non-owner segment
                if ignoring.feed(block, prob):
                    ignoring = None
                continue
            ended = ep.feed(block, prob)
            if not ep.started:
                continue
            phase = "greeting" if greeting else self._phase(first_tid)

            # 0) Luboo finished speaking (worker is only saving history): this is a normal next utterance
            if phase == "done":
                return self._finish(ep, ducked)

            # 1) Thinking (no audio yet): you continue the previous sentence -> merge. Past MERGE_WINDOW_SEC, only merge
            #    if a voice is enrolled (voice checked at the end); if not enrolled, ignore to be safe.
            if phase == "thinking":
                if time.perf_counter() - speech_end > config.MERGE_WINDOW_SEC and not self.barge_in_enabled():
                    ignoring, ep = ep, Endpointer()
                    continue
                self.log("You kept talking while Luboo was thinking — merging")
                self._merge_pending = True
                self.voice.cancel()
                return self._finish(ep, False, merge_with=speech, owner_check=True)

            # 2) Greeting: anyone speaking is the question -> stop greeting, keep listening
            if phase == "greeting":
                self.voice.cancel()
                return self._finish(ep, False)

            # 3) Speaking: only the owner (voice enrolled) can interrupt
            if not self.barge_in_enabled():
                ignoring, ep = ep, Endpointer()
                continue
            t = time.perf_counter()
            if not ducked:
                player.set_gain(config.BARGE_DUCK_GAIN)                # duck as soon as a human voice is heard
                ducked, t_onset = True, t
                event = {"kind": "barge", "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            sec = ep.speech_sec()
            if ended and sec < config.BARGE_MIN_SPEECH_SEC:            # cough, "umm"... -> keep speaking
                player.set_gain(1.0)
                self._log_barge(event, t_onset, "short_noise", None)
                ducked, ep = False, Endpointer()
                continue
            if sec < config.BARGE_VERIFY_SEC and not ended:
                continue
            if sec - verified_at < 0.3 and not ended:                   # checked; wait for more speech, then re-check
                continue
            verified_at = sec
            owner, score = self.speaker_id.is_owner(ep.speech_audio())
            if owner:
                self.voice.cancel()                                    # stop now, drop all unplayed chunks
                self._log_barge(event, t_onset, "interrupted", score, sec)
                self.log(f"Barge-in: your voice (score {score:.2f}) — Luboo stopped")
                return self._finish(ep, False)
            if score >= config.SPEAKER_GRAY and sec < config.BARGE_VERIFY_MAX_SEC and not ended:
                continue                                               # borderline -> listen more, then re-check
            player.set_gain(1.0)                                       # not the owner -> keep speaking
            self._log_barge(event, t_onset, "not_owner", score, sec)
            self.log(f"Voice while speaking ignored (score {score:.2f})")
            ignoring = None if ended else ep
            ducked, verified_at, ep = False, 0.0, Endpointer()

    def _finish(self, ep, ducked, merge_with=None, owner_check=False):
        """Finish recording the utterance already started. merge_with: append after the previous segment (merge)."""
        if ducked:
            self.voice.player.set_gain(1.0)
        self.on_state("listening", "neutral")
        audio = self.mic.record_utterance(config.MAX_RECORD_SEC, self.end_conversation, ep=ep)
        if merge_with is None:
            return audio
        valid = audio is not None and self.mic.last_speech_sec >= config.BARGE_MIN_SPEECH_SEC
        if valid and owner_check and self.voice_enrolled():
            valid, score = self.speaker_id.is_owner(self.mic.last_voiced)
            if not valid:
                self.log(f"Continuation was not your voice (score {score:.2f}) — answering the first part")
        if not valid:
            return merge_with                                          # noise / someone else: re-ask the old one
        gap = np.zeros(int(0.3 * config.SAMPLE_RATE), dtype=np.int16)
        return np.concatenate([merge_with, gap, audio])

    def _log_barge(self, event, t_onset, outcome, score, speech_sec=None):
        event.update(outcome=outcome, react_ms=round((time.perf_counter() - t_onset) * 1000),
                     score=None if score is None else round(score, 3),
                     speech_sec=None if speech_sec is None else round(speech_sec, 2))
        timing.save_event(event)

    # ---------- voice enrollment ----------
    def _enroll(self):
        self.log("Voice ID enrollment started")
        self.on_state("listening", "neutral")
        clips = []
        for i, sentence in enumerate(config.ENROLL_SENTENCES):
            for attempt in range(3):
                if self.enroll_cancel.is_set() or not self.running:
                    self.on_enroll("failed", "cancelled")
                    return
                self.on_enroll("prompt", i, sentence)
                self.mic.flush()
                audio = self.mic.record_utterance(10, self.enroll_cancel)
                if audio is not None and self.mic.last_speech_sec >= 1.0:
                    clips.append(self.mic.last_voiced)
                    self.on_enroll("heard", i)
                    break
                self.on_enroll("retry", i)
            else:
                self.on_enroll("failed", "no_speech")
                return
        scores = self.speaker_id.enroll(clips)
        self.log("Voice ID saved (consistency " + ", ".join(f"{s:.2f}" for s in scores) + ")")
        self.on_enroll("done", [float(s) for s in scores])

    # ---------- one Q&A turn ----------
    def _handle(self, speech, speech_end=None):
        timer = TurnTimer(speech_end if speech_end is not None else self.mic.last_speech_end,
                          speech_sec=round(len(speech) / config.SAMPLE_RATE, 2))
        timer.mark("endpoint")                   # silence wait needed to know you finished speaking
        timer.note(result="error")
        self.on_state("thinking", "thinking")
        try:
            self._answer(speech, timer)
        except api.NotConfigured:
            timer.note(result="not_configured")
            self.end_conversation.set()
            self._speak([NOT_CONNECTED_REPLY])
        finally:
            timer.save(self.log)

    def _answer(self, speech, timer: TurnTimer):
        tid = self.voice.turn_id
        text = brain.transcribe(speech, self.log)
        timer.mark("stt")
        if self.voice.turn_id != tid:            # cancelled during transcription (merge / Luboo clicked)
            timer.note(result="merged" if self._merge_pending else "cancelled")
            return
        if not text:
            timer.note(result="nothing_heard")
            self.log("(nothing heard)")
            return
        self.on_message("user", text)

        info = {}
        reply, spoken, ok, cancelled = self._speak(brain.ask_stream(text, self.history, self.log, info), timer)
        timer.note(**info, reply_chars=len(reply))
        if cancelled:
            timer.note(result="merged" if self._merge_pending else "cancelled")
            if self._merge_pending:
                return                           # the full sentence will be re-asked -> don't log the half
        else:
            timer.note(result="ok" if ok else "tts_failed")
        # History records only what Luboo actually said: if interrupted, the LLM knows you didn't hear it all
        said = reply if not cancelled else (spoken + " (interrupted)" if spoken else "(interrupted before answering)")
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": said}]

    def _speak(self, pieces, timer: TurnTimer | None = None):
        """Speak the reply from text pieces (LLM stream or canned text); the first chunk is spoken right away.
        Returns (full reply, part spoken, whether any audio played, whether it was cancelled)."""
        rs = brain.ReplyStream()

        def first_audio():
            if timer:
                timer.mark("first_audio")
            self.on_state("speaking", rs.emotion or "neutral")

        turn = self.voice.new_turn(self.on_level, self.on_subtitle, first_audio, self.log)
        first_piece = first_chunk = True
        try:
            for piece in _cancellable(pieces, lambda: turn.cancelled):
                if first_piece and timer:
                    timer.mark("first_token")
                first_piece = False
                for chunk in rs.feed(piece):
                    if first_chunk and timer:
                        timer.mark("first_chunk")
                    first_chunk = False
                    turn.say(chunk)
            if not turn.cancelled:
                for chunk in rs.flush():
                    if first_chunk and timer:
                        timer.mark("first_chunk")
                    first_chunk = False
                    turn.say(chunk)
            if timer:
                timer.mark("llm")                # LLM finished writing (usually while the first chunk plays)
        finally:
            turn.finish()
            spoken = turn.wait()
        if timer:
            timer.note(chunks=len(rs.parts), tts_failed_chunks=turn.failed, stale_dropped=turn.dropped)
        reply = rs.text
        if reply:
            self.on_message("bot", reply)
        if not turn.started_audio and not turn.cancelled and reply:
            self.on_subtitle(reply)              # no audio (TTS error) -> still show the text
        if not turn.cancelled:
            self.on_state("idle", rs.emotion or "neutral")
        return reply, spoken, turn.started_audio or not reply, turn.cancelled
