"""Bộ não điều phối: chờ từ kích hoạt -> nghe -> nghĩ -> nói -> nghe tiếp -> ngủ."""
import datetime
import random
import threading
import traceback

from . import api, brain, config
from .audio import Microphone, Speaker, WakeWordDetector, beep

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


class Assistant(threading.Thread):
    """Chạy ở luồng riêng. Báo trạng thái cho giao diện qua các hàm callback:
    on_state(state, emotion), on_message(role, text), on_level(0..1), on_subtitle(chữ đang đọc tới)."""

    def __init__(self, on_state, on_message, on_level, on_log, on_subtitle):
        super().__init__(daemon=True)
        self.on_state, self.on_message = on_state, on_message
        self.on_level, self.log = on_level, on_log
        self.on_subtitle = on_subtitle
        self.history: list[dict] = []
        self.running = True
        self.push_to_talk = threading.Event()
        self.end_conversation = threading.Event()
        self.speaker = Speaker()
        self.recent_greetings: list[str] = []

    # ---------- điều khiển từ giao diện ----------
    def talk_now(self):
        self.push_to_talk.set()

    def stop_speaking(self):
        self.speaker.stop()
        self.end_conversation.set()

    def clear_history(self):
        self.history.clear()

    def shutdown(self):
        self.running = False
        self.end_conversation.set()
        self.speaker.stop()

    # ---------- vòng lặp chính ----------
    def run(self):
        try:
            self.on_state("loading", "neutral")
            self.log("Loading wake word model...")
            self.wake = WakeWordDetector()
            self.mic = Microphone()
            noise, threshold = self.mic.calibrate()
            self.log(f"Background noise: {noise:.0f} -> speech threshold: {threshold:.0f}")
            self.log(f"Ready. Say {wake_words_hint()}")
            threading.Thread(target=self.speaker.preload, args=(all_greetings(),), daemon=True).start()
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
        self.mic.pause()

    def _wait_for_trigger(self) -> bool:
        """Chờ từ kích hoạt hoặc nút 🎤. False nếu app đang tắt."""
        self.on_state("sleeping", "neutral")
        self.mic.start()
        while self.running:
            frame = self.mic.read()
            if self.push_to_talk.is_set():
                self.push_to_talk.clear()
                return True
            if self.wake.detect(frame):
                self.log("Wake word detected")
                return True
        return False

    def _conversation(self):
        self.end_conversation.clear()
        # Gọi "Hey Luboo, what's the weather?" một mạch -> trả lời luôn; gọi xong rồi ngừng -> Luboo chào trước
        self.on_state("listening", "neutral")
        self.mic.start()
        early = self.mic.record_utterance(config.GREETING_GRACE_SEC, self.end_conversation)
        self.mic.pause()
        if early is not None and len(early) >= 0.8 * config.SAMPLE_RATE:
            self._handle(early)
            wait = config.FOLLOWUP_SEC
        else:
            self._greet()
            wait = config.FIRST_WAIT_SEC
        while not self.end_conversation.is_set():
            self.on_state("listening", "neutral")
            self.mic.start()
            speech = self.mic.record_utterance(wait, self.end_conversation)
            self.mic.pause()
            if speech is None:
                return                                   # im lặng -> quay về ngủ
            self._handle(speech)
            wait = config.FOLLOWUP_SEC                   # nói tiếp không cần gọi lại

    def _greet(self):
        text = pick_greeting(self.recent_greetings)
        self.recent_greetings = (self.recent_greetings + [text])[-3:]
        if self.speaker.is_ready(text):
            self._speak(f"[happy] {text}")
        else:
            beep()                       # giọng chưa tạo xong (vừa mở app / mất mạng) -> kêu "ting" như cũ

    def _handle(self, speech):
        self.on_state("thinking", "thinking")
        try:
            self._answer(speech)
        except api.NotConfigured:
            self.end_conversation.set()
            self._speak(NOT_CONNECTED_REPLY)

    def _answer(self, speech):
        text = brain.transcribe(speech, self.log)
        if not text:
            self.log("(nothing heard)")
            return
        self.on_message("user", text)

        reply = brain.ask(text, self.history, self.log)
        self.history += [{"role": "user", "content": text},
                         {"role": "assistant", "content": brain.clean_for_speech(reply)}]
        self._speak(reply)

    def _speak(self, reply: str):
        emotion, spoken = brain.split_emotion(reply)
        if not spoken:
            return
        self.on_message("bot", spoken)
        self.on_state("speaking", emotion)
        try:
            self.speaker.say(spoken, self.on_level, self.on_subtitle)    # phụ đề hiện dần theo giọng đọc
        except Exception as e:
            self.log(f"TTS failed (edge-tts needs internet): {e}")
            self.on_subtitle(spoken)                                     # không có tiếng thì vẫn hiện chữ
        self.on_state("idle", emotion)
