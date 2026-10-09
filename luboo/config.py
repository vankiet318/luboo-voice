"""Cấu hình của app. Sửa các giá trị ở đây cho phù hợp với máy và sở thích của bạn."""
import os
import sys

APP_NAME = "Luboo"
APP_VERSION = "1.0.0"

# Thư mục gốc của repo khi chạy từ mã nguồn; khi đóng gói bằng PyInstaller là thư mục _internal
BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
MODELS_DIR = os.path.join(RESOURCES_DIR, "models")        # model tải về (melspectrogram, embedding...)
ICON_PATH = os.path.join(RESOURCES_DIR, "icon.ico")       # tạo bằng packaging/make_icon.py
# Nơi lưu cài đặt của người dùng (thư mục cài đặt app là chỉ đọc)
DATA_DIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), APP_NAME)

# ---------- Groq ----------
STT_MODEL = "whisper-large-v3-turbo"
# (Groq đã ngừng Llama 3.x và groq/compound cho gói miễn phí từ 08–09/2026 — xem console.groq.com/docs/deprecations)
SEARCH_MODEL = "openai/gpt-oss-120b"         # câu hỏi kiến thức: bật công cụ tra web browser_search
CHAT_MODEL = "openai/gpt-oss-120b"           # trò chuyện thường
FALLBACK_MODEL = "openai/gpt-oss-20b"        # dự phòng khi bị giới hạn — nhanh hơn, kém thông minh hơn
REASONING_EFFORT = "low"                     # gpt-oss "suy nghĩ" trước khi trả lời; thấp = trả lời nhanh

# Dự phòng cuối cùng bằng Ollama chạy trên máy (để trống "" nếu không dùng)
OLLAMA_MODEL = ""                            # ví dụ: "qwen3:1.7b"
OLLAMA_URL = "http://localhost:11434/api/chat"

# STT dự phòng chạy trên máy bằng faster-whisper (cần: pip install faster-whisper)
LOCAL_STT_MODEL = "small"                    # "base" nhanh hơn, "small" tiếng Việt tốt hơn

# ---------- Từ kích hoạt ----------
# Luboo chỉ nghe và nói tiếng Anh.
# Model tự train nằm trong resources/wakewords/ (xem docs/HUONG_DAN.md). Khi đủ file, app tự dùng chúng;
# nếu chưa có thì tạm dùng model có sẵn của openWakeWord (CC BY-NC-SA 4.0 — phi thương mại).
WAKEWORDS_DIR = os.path.join(RESOURCES_DIR, "wakewords")
CUSTOM_WAKE_WORDS = {"hey_luboo": "en"}
FALLBACK_WAKE_WORDS = {"hey_jarvis_v0.1": "en"}
WAKE_LABELS = {"hey_luboo": "Hey Luboo", "hey_jarvis_v0.1": "Hey Jarvis"}
USE_CUSTOM_WAKE_WORDS = all(os.path.exists(os.path.join(WAKEWORDS_DIR, f"{n}.onnx"))
                            for n in CUSTOM_WAKE_WORDS)
WAKE_WORDS = CUSTOM_WAKE_WORDS if USE_CUSTOM_WAKE_WORDS else FALLBACK_WAKE_WORDS
# Chỉ bật khi điểm >= ngưỡng trong WAKE_PATIENCE khung 80ms liên tiếp (bật nhầm thường chỉ "lóe" 1 khung).
# Model tự train (lần 2): 0.99 + 3 khung -> ~1–2 lần bật nhầm/giờ, nhận ~95% lần gọi.
WAKE_THRESHOLD = 0.99 if USE_CUSTOM_WAKE_WORDS else 0.5
WAKE_PATIENCE = 3 if USE_CUSTOM_WAKE_WORDS else 1


def wake_word_path(name: str) -> str:
    return os.path.join(WAKEWORDS_DIR if name in CUSTOM_WAKE_WORDS else MODELS_DIR, f"{name}.onnx")


# ---------- Thu âm ----------
SAMPLE_RATE = 16000
FRAME = 1280                 # 80ms mỗi khung
SILENCE_SEC = 1.0            # im lặng bao lâu thì coi là nói xong
FIRST_WAIT_SEC = 6           # sau từ kích hoạt, chờ bạn bắt đầu nói tối đa bao lâu
FOLLOWUP_SEC = 6             # sau khi bot trả lời, nghe tiếp bao lâu (không cần gọi lại)
MAX_RECORD_SEC = 20
MIN_SPEECH_THRESHOLD = 300   # ngưỡng âm lượng tối thiểu để coi là có tiếng nói

# ---------- Giọng đọc (edge-tts) ----------
VOICE = "en-US-AriaNeural"           # giọng khác: en-US-GuyNeural, en-US-AvaMultilingualNeural...
TTS_RATE = "+0%"                     # đọc chậm hơn: "-10%"

# ---------- Lời chào khi được gọi ----------
# Chọn ngẫu nhiên, không lặp lại 3 câu gần nhất; ~30% dùng câu theo giờ trong ngày.
# Giọng đọc của mọi câu được tạo sẵn lúc mở app nên Luboo chào ngay, không phải chờ mạng.
GREETING_GRACE_SEC = 0.7     # gọi xong mà nói tiếp luôn trong khoảng này -> bỏ qua lời chào, trả lời luôn
GREETINGS = {
    "any": [
        "Hey! What's up?", "Hi there! How can I help?", "Yep, I'm listening.",
        "Hey, what can I do for you?", "Hi! What do you need?", "I'm here. What's up?",
        "Yes? How can I help?", "Hey hey! What's going on?", "Hello! What can I help you with?",
        "Sure, go ahead.", "Hi! Need a hand?", "Hey, you called?", "What's on your mind?",
    ],
    "morning": ["Good morning! What can I do for you?", "Morning! What's up?"],
    "afternoon": ["Good afternoon! How can I help?", "Hey! How's your afternoon going?"],
    "evening": ["Good evening! What do you need?", "Evening! What's up?"],
    "night": ["Still up? What can I do for you?", "Hey, night owl! What's up?"],
}

# ---------- Hội thoại ----------
HISTORY_TURNS = 10                   # số lượt gần nhất gửi kèm cho LLM

SYSTEM_PROMPT = (
    "You are Luboo, a friendly voice assistant who helps the user while they play games "
    "and chats casually. Reply in English with 1-3 short spoken sentences. "
    "No markdown, no lists, no links. "
    "Start every reply with exactly one emotion tag: "
    "[happy], [sad], [surprised], [angry], [thinking] or [neutral]. "
    "Example: [happy] Nice! That boss is tough, well done."
)
