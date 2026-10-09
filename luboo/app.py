"""Luboo UI: an animated character + one line of text + two buttons. Run: python -m luboo"""
import math
import os
import queue
import random
import sys
import time
import tkinter as tk
import webbrowser

from . import config

# The packaged app has no console (sys.stderr is None) -> write logs/errors to %APPDATA%\Luboo\luboo.log
LOG_PATH = os.path.join(config.DATA_DIR, "luboo.log")
if sys.stderr is None or getattr(sys, "frozen", False):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 1_000_000:
        os.replace(LOG_PATH, LOG_PATH + ".old")
    sys.stdout = sys.stderr = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
else:
    for stream in (sys.stdout, sys.stderr):   # Windows console can't print every character -> '?'
        stream.reconfigure(errors="replace")

from PIL import Image, ImageDraw, ImageTk  # noqa: E402

from . import api, settings  # noqa: E402
from .assistant import Assistant, wake_words_hint  # noqa: E402

# ---------- palette: soft lilac — away from both the dark/neon "AI" look and Claude's cream/terracotta ----------
PAPER = "#F4F1F8"      # background
CLAY = "#A493D8"       # Luboo's body
CLAY_DARK = "#7E6CC0"  # links, focus
CLAY_LIGHT = "#B9ABE4" # body highlight
INK = "#2A2638"        # eyes, text
BLUSH = "#F3B3C3"
SHADOW = "#E3DDEE"
MUTED = "#948DA6"
LINE = "#E2DCEC"
FONT = "Segoe UI"


def ease(cur, target, k=0.18):
    return cur + (target - cur) * k


class Face:
    """Luboo: a soft lilac character. Rendered with Pillow at 2x and downsampled for smooth edges."""
    W, H, S = 340, 290, 2

    def __init__(self, parent):
        self.canvas = tk.Canvas(parent, width=self.W, height=self.H, bg=PAPER, highlightthickness=0, cursor="hand2")
        self.image_id = self.canvas.create_image(0, 0, anchor="nw")
        self.photo = None
        self.state, self.emotion, self.level = "loading", "neutral", 0.0
        self.cur = {"open": 0.2, "eye": 1.0, "lx": 0.0, "ly": 0.0, "mouth": 0.0, "squash": 1.0, "blush": 0.4}
        self.blink_at, self.blink_until = time.time() + 2.5, 0.0
        self.look, self.look_at = (0.0, 0.0), time.time() + 2

    # ---------- what the face should look like now ----------
    def target(self, now):
        state, emo = self.state, self.emotion if self.state in ("speaking", "idle") else "neutral"
        t = {"open": 1.0, "eye": 1.0, "lx": 0.0, "ly": 0.0, "mouth": 0.0, "squash": 1.0, "blush": 0.45}
        if state in ("loading", "sleeping"):
            t.update(open=0.0, ly=3, squash=1.0 + 0.018 * math.sin(now * 1.6), blush=0.3)
        elif state == "error":
            t.update(open=0.55, ly=4, blush=0.2)
        elif state == "listening":
            t.update(eye=1.14, squash=1.03, blush=0.6)
        elif state == "thinking":
            t.update(open=0.8, lx=9, ly=-8 + 1.5 * math.sin(now * 3), blush=0.4)
        if state in ("speaking", "idle"):
            t["lx"], t["ly"] = self.look
            t["mouth"] = self.level if state == "speaking" else 0.0
            if emo == "happy":
                t.update(blush=0.9, squash=1.0 + 0.012 * math.sin(now * 6))
            elif emo == "surprised":
                t.update(eye=1.3, ly=-3)
            elif emo == "sad":
                t.update(eye=0.9, ly=5, blush=0.25)
            elif emo == "thinking":
                t.update(lx=8, ly=-7)
        return t

    def update_motion(self, now):
        awake = self.state not in ("loading", "sleeping")
        if awake and now > self.blink_at:
            self.blink_until = now + 0.11
            self.blink_at = now + random.uniform(2.2, 5.5)
        if now > self.look_at:
            self.look = (random.uniform(-6, 6), random.uniform(-3, 3)) if random.random() < 0.6 else (0.0, 0.0)
            self.look_at = now + random.uniform(1.5, 3.5)
        tgt = self.target(now)
        for k in self.cur:
            self.cur[k] = ease(self.cur[k], tgt[k], 0.35 if k == "mouth" else 0.18)

    # ---------- drawing ----------
    def draw(self):
        now = time.time()
        self.update_motion(now)
        S, c = self.S, self.cur
        img = Image.new("RGB", (self.W * S, self.H * S), PAPER)
        d = ImageDraw.Draw(img)
        p = lambda *xy: [v * S for v in xy]   # noqa: E731  (logical -> 2x pixels)

        emo = self.emotion if self.state in ("speaking", "idle") else "neutral"
        cx = self.W / 2
        bob = 3 * math.sin(now * 1.8) if self.state != "sleeping" else 0
        bw, bh = 196 / c["squash"] ** 0.5, 176 * c["squash"]
        base = 238
        top = base - bh + bob

        # shadow + body
        sw = bw * (0.42 - 0.02 * bob / 3)
        d.ellipse(p(cx - sw, base + 10, cx + sw, base + 24), fill=SHADOW)
        d.rounded_rectangle(p(cx - bw / 2, top, cx + bw / 2, base + bob), radius=78 * S, fill=CLAY)
        hx, hy = cx - bw * 0.27, top + 26                          # soft highlight for a bit of volume
        d.ellipse(p(hx - 16, hy - 7, hx + 16, hy + 7), fill=CLAY_LIGHT)

        fy = top + bh * 0.46                                       # face center line
        ex, ey = 40 + c["lx"] * 0.4, fy + c["ly"]

        # cheeks
        if c["blush"] > 0.05:
            blush = Image.new("RGB", img.size, BLUSH)
            mask = Image.new("L", img.size, 0)
            md = ImageDraw.Draw(mask)
            for side in (-1, 1):
                bx = cx + side * 62 + c["lx"] * 0.3
                md.ellipse(p(bx - 16, ey + 22, bx + 16, ey + 34), fill=int(255 * min(1, c["blush"])))
            img.paste(blush, (0, 0), mask)

        # eyes
        blinking = now < self.blink_until
        openness = 0.08 if blinking else c["open"]
        for side in (-1, 1):
            x = cx + side * ex + c["lx"]
            w, h = 11 * c["eye"], 15 * c["eye"] * max(openness, 0.0)
            if emo == "happy" and self.state in ("speaking", "idle") and not blinking:
                d.arc(p(x - 12, ey - 8, x + 12, ey + 14), 200, 340, fill=INK, width=5 * S)      # ^ ^
            elif openness < 0.25:
                d.arc(p(x - 11, ey - 10, x + 11, ey + 6), 20, 160, fill=INK, width=4 * S)       # closed
            else:
                d.ellipse(p(x - w, ey - h, x + w, ey + h), fill=INK)
                g = 3.4 * c["eye"]
                d.ellipse(p(x - w * 0.35 - g, ey - h * 0.45 - g, x - w * 0.35 + g, ey - h * 0.45 + g), fill=PAPER)
            # brows only when they say something
            by = ey - 15 * c["eye"] - 8
            if emo == "sad":                                       # inner ends raised
                d.line(p(x + side * 12, by + 1, x - side * 6, by - 5), fill=INK, width=4 * S)
            elif emo == "angry":                                   # inner ends lowered
                d.line(p(x + side * 12, by - 5, x - side * 6, by + 2), fill=INK, width=4 * S)
            elif emo == "surprised":
                d.arc(p(x - 12, by - 9, x + 12, by + 5), 200, 340, fill=INK, width=3 * S)

        # mouth
        mx, my = cx + c["lx"] * 0.5, ey + 30
        if self.state == "speaking" and c["mouth"] > 0.04:
            mw, mh = 9 + 5 * c["mouth"], 3 + 13 * c["mouth"]
            d.rounded_rectangle(p(mx - mw, my - mh / 2, mx + mw, my + mh / 2), radius=int(min(mw, mh / 2) * S), fill=INK)
            if mh > 9:
                d.ellipse(p(mx - mw * 0.55, my + mh / 2 - 6, mx + mw * 0.55, my + mh / 2 - 1), fill=BLUSH)
        elif self.state in ("loading", "sleeping"):
            d.ellipse(p(mx - 4, my - 3, mx + 4, my + 3), fill=INK)
        elif emo == "angry":
            d.line(p(mx - 9, my + 3, mx + 9, my + 3), fill=INK, width=4 * S)
        elif emo == "sad" or self.state == "error":
            d.arc(p(mx - 11, my - 2, mx + 11, my + 12), 200, 340, fill=INK, width=4 * S)
        elif emo == "surprised":
            d.ellipse(p(mx - 6, my - 7, mx + 6, my + 7), fill=INK)
        elif self.state == "thinking" or emo == "thinking":
            d.line(p(mx - 6, my + 2, mx + 8, my - 1), fill=INK, width=4 * S)
        else:
            wide = 14 if emo == "happy" or self.state == "listening" else 10
            d.arc(p(mx - wide, my - 10, mx + wide, my + 6), 20, 160, fill=INK, width=4 * S)

        # extras
        if self.state == "thinking":
            for i in range(3):
                on = (now * 2.5 - i) % 3 < 1.6
                r = 4.5 if on else 3
                dx, dy = cx + 70 + i * 14, top - 6 - i * 9
                d.ellipse(p(dx - r, dy - r, dx + r, dy + r), fill=INK if on else LINE)
        if self.state == "sleeping":
            for i in range(2):
                phase = (now * 0.45 + i * 0.5) % 1
                size, zx, zy = 6 + 5 * phase, cx + 72 + 22 * phase, top + 8 - 46 * phase
                if phase < 0.9:
                    shade = MUTED if phase < 0.6 else LINE
                    d.line(p(zx, zy, zx + size, zy, zx, zy + size, zx + size, zy + size), fill=shade, width=2 * S)

        self.photo = ImageTk.PhotoImage(img.reduce(S))
        self.canvas.itemconfig(self.image_id, image=self.photo)


class SettingsDialog:
    def __init__(self, app: "App"):
        self.app = app
        s = app.settings
        win = self.win = tk.Toplevel(app.root)
        win.title("Settings")
        win.configure(bg=PAPER, padx=22, pady=18)
        win.resizable(False, False)
        win.transient(app.root)
        win.grab_set()

        text = dict(bg=PAPER, fg=INK, font=(FONT, 10), anchor="w", justify="left")
        small = {**text, "fg": MUTED, "font": (FONT, 9)}

        tk.Label(win, text="Connection", **{**text, "font": (FONT, 11, "bold")}).pack(fill="x")
        tk.Label(win, text="Groq API key", **text).pack(fill="x", pady=(8, 0))

        self.key = tk.Entry(win, show="•", width=40, bg="#FCFBFE", fg=INK, insertbackground=INK, relief="flat",
                            highlightthickness=1, highlightbackground=LINE, highlightcolor=CLAY, font=("Consolas", 10))
        self.key.insert(0, s["api_key"])
        self.key.pack(fill="x", pady=(6, 0), ipady=5)
        link = tk.Label(win, text="Get a free key at console.groq.com", cursor="hand2",
                        **{**small, "fg": CLAY_DARK, "font": (FONT, 9, "underline")})
        link.pack(fill="x", pady=(4, 0))
        link.bind("<Button-1>", lambda _: webbrowser.open("https://console.groq.com/keys"))
        if api.embedded_key():
            tk.Label(win, text="Leave empty to use the built-in test key.", **small).pack(fill="x")
        tk.Label(win, text=app.connection_text(), **small).pack(fill="x", pady=(10, 0))

        tk.Frame(win, bg=LINE, height=1).pack(fill="x", pady=14)
        self.on_top = tk.BooleanVar(value=s.get("on_top", False))
        tk.Checkbutton(win, text="Keep Luboo on top of other windows", variable=self.on_top, bg=PAPER, fg=INK,
                       selectcolor=PAPER, activebackground=PAPER, font=(FONT, 10), anchor="w",
                       highlightthickness=0).pack(fill="x")

        buttons = tk.Frame(win, bg=PAPER)
        buttons.pack(fill="x", pady=(18, 0))
        flat = dict(relief="flat", bd=0, padx=16, pady=5, font=(FONT, 10), cursor="hand2")
        tk.Button(buttons, text="Save", bg=INK, fg=PAPER, activebackground=CLAY, activeforeground=PAPER,
                  command=self.save, **flat).pack(side="right")
        tk.Button(buttons, text="Cancel", bg=PAPER, fg=MUTED, activebackground=PAPER, activeforeground=INK,
                  command=win.destroy, **flat).pack(side="right", padx=6)

    def save(self):
        s = self.app.settings
        s["api_key"] = self.key.get().strip()
        if not api.key_source(s):                            # chưa có key nào để dùng
            self.key.configure(highlightbackground=CLAY_DARK, highlightcolor=CLAY_DARK)
            return
        s["on_top"] = self.on_top.get()
        settings.save(s)
        self.app.apply_settings()
        self.win.destroy()


class App:
    def __init__(self):
        if sys.platform == "win32":
            # Báo Windows đây là app "Luboo" riêng -> taskbar hiện icon Luboo, kể cả khi chạy bằng python -m luboo
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Luboo.VoiceAssistant")
        self.root = tk.Tk()
        self.root.title(config.APP_NAME)
        self.root.configure(bg=PAPER)
        self.root.resizable(False, False)
        if os.path.exists(config.ICON_PATH):
            self.root.iconbitmap(default=config.ICON_PATH)          # mọi cửa sổ (kể cả Settings) dùng icon Luboo
        self.events: queue.Queue = queue.Queue()
        self.settings = settings.load()
        self.state = "loading"

        self.face = Face(self.root)
        self.face.canvas.pack(padx=10, pady=(14, 0))
        self.face.canvas.bind("<Button-1>", lambda _: self.face_clicked())

        self.caption = tk.Label(self.root, text="", bg=PAPER, fg=MUTED, font=(FONT, 10),
                                wraplength=300, justify="center", height=3)
        self.caption.pack(padx=20)

        bar = tk.Frame(self.root, bg=PAPER)
        bar.pack(pady=(4, 16))
        flat = dict(relief="flat", bd=0, bg=PAPER, activebackground=PAPER, font=(FONT, 10), cursor="hand2", padx=10)
        tk.Button(bar, text="Talk", fg=INK, activeforeground=CLAY_DARK,
                  command=lambda: self.assistant.talk_now(), **flat).pack(side="left")
        tk.Label(bar, text="·", bg=PAPER, fg=LINE, font=(FONT, 10)).pack(side="left")
        tk.Button(bar, text="Settings", fg=MUTED, activeforeground=INK,
                  command=lambda: SettingsDialog(self), **flat).pack(side="left")

        e = self.events.put
        self.assistant = Assistant(
            on_state=lambda s, emo: e(("state", s, emo)),
            on_message=lambda role, t: e(("msg", role, t)),
            on_level=lambda lvl: setattr(self.face, "level", lvl),
            on_log=lambda t: print(time.strftime("%Y-%m-%d %H:%M:%S"), t),   # log file only
            on_subtitle=lambda t: e(("subtitle", t)),
        )
        self.apply_settings()
        self.assistant.start()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.tick()

    # ---------- settings / connection ----------
    def apply_settings(self):
        api.configure(self.settings)
        self.root.attributes("-topmost", bool(self.settings.get("on_top")))
        self.show_idle_caption()

    def connected(self) -> bool:
        try:
            api.current()
            return True
        except api.NotConfigured:
            return False

    def connection_text(self) -> str:
        return {"own": "Using your Groq key.", "env": "Using the key from GROQ_API_KEY.",
                "embedded": "Using the built-in test key."}.get(api.key_source(self.settings), "Not connected yet.")

    # ---------- caption ----------
    def set_caption(self, text, color=MUTED):
        self.caption.configure(text=text, fg=color)

    def show_idle_caption(self):
        if self.state != "sleeping":
            return
        if not self.connected():
            self.set_caption("Add a Groq API key in Settings to get started.")
        else:
            self.set_caption(f"Say {wake_words_hint()}")

    def on_state(self, state, emotion):
        self.state = state
        self.face.state, self.face.emotion = state, emotion
        if state == "loading":
            self.set_caption("Waking up…")
        elif state == "sleeping":
            self.show_idle_caption()
        elif state == "listening":
            self.set_caption("Listening…")
        elif state == "thinking" and not self.caption.cget("text").startswith("“"):
            self.set_caption("Thinking…")
        elif state == "error":
            self.set_caption("Luboo couldn't start. Check that a microphone is connected.")

    def on_message(self, role, text):
        print(time.strftime("%Y-%m-%d %H:%M:%S"), {"user": "You:", "bot": "Luboo:"}[role], text)
        if role == "user":
            self.set_caption(f"“{text}”")
        # câu trả lời của Luboo hiện qua "subtitle" — đồng bộ với giọng đọc

    def face_clicked(self):
        if self.state in ("speaking", "listening", "thinking", "idle"):
            self.assistant.stop_speaking()
        else:
            self.assistant.talk_now()

    # ---------- loop ----------
    def close(self):
        self.assistant.shutdown()
        self.assistant.join(timeout=1)
        self.root.destroy()

    def tick(self):
        while not self.events.empty():
            ev = self.events.get()
            if ev[0] == "state":
                self.on_state(ev[1], ev[2])
            elif ev[0] == "msg":
                self.on_message(ev[1], ev[2])
            elif ev[0] == "subtitle":
                self.set_caption(ev[1], INK)
        self.face.draw()
        self.root.after(33, self.tick)        # ~30 fps

    def run(self):
        self.root.mainloop()


def selftest() -> int:
    """`Luboo.exe --selftest`: load every bundled model and run it once; result goes to the log."""
    import numpy as np
    from .audio import WakeWordDetector
    try:
        noise = (np.random.default_rng(0).normal(0, 800, 16000 * 3)).astype(np.int16)
        detector = WakeWordDetector()
        for i in range(0, len(noise) - 1280, 1280):
            detector.detect(noise[i:i + 1280])
        print(f"SELFTEST OK — wake words: {list(config.WAKE_WORDS)}")
        return 0
    except Exception:
        import traceback
        traceback.print_exc()
        print("SELFTEST FAILED")
        return 1


def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    App().run()
