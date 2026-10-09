"""Vẽ icon app (nhân vật Luboo) ra resources/icon.ico."""
import os

from PIL import Image, ImageDraw

CLAY, CLAY_LIGHT, INK, BLUSH, PAPER = "#A493D8", "#B9ABE4", "#2A2638", "#F3B3C3", "#F4F1F8"
OUT = os.path.join(os.path.dirname(__file__), "..", "resources", "icon.ico")

S = 1024                                   # vẽ lớn rồi thu nhỏ cho mượt
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle((64, 96, S - 64, S - 64), radius=380, fill=CLAY)
d.ellipse((250, 210, 400, 280), fill=CLAY_LIGHT)
for side in (-1, 1):
    cx = S / 2 + side * 170
    bx = cx + side * 30                     # má hồng lệch ra ngoài mắt một chút
    d.ellipse((bx - 80, 590, bx + 80, 660), fill=BLUSH)
    d.ellipse((cx - 52, 380, cx + 52, 520), fill=INK)
    d.ellipse((cx - 34, 402, cx - 4, 432), fill=PAPER)
d.arc((S / 2 - 70, 560, S / 2 + 70, 680), 20, 160, fill=INK, width=34)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
img.resize((256, 256), Image.LANCZOS).save(OUT, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("saved", os.path.abspath(OUT))
