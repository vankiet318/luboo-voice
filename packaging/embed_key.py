"""Tạo luboo/_embedded_key.py chứa Groq API key (đã làm rối) cho bản build gửi bạn bè test.

Chỉ dùng qua `packaging/build.ps1 -EmbedKey`. Key lấy từ biến môi trường LUBOO_EMBED_KEY (build.ps1 đặt).
LÀM RỐI ≠ MÃ HÓA: ai rành kỹ thuật vẫn lấy được key từ file exe — chỉ gửi cho người tin cậy,
lộ key thì thu hồi tại console.groq.com/keys rồi build lại.
"""
import os
import secrets
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "luboo", "_embedded_key.py")

key = os.environ.get("LUBOO_EMBED_KEY", "").strip()
if not key.startswith("gsk_"):
    sys.exit("Không tìm thấy Groq API key hợp lệ (bắt đầu bằng gsk_).")
data = key.encode()
pad = secrets.token_bytes(len(data))
with open(OUT, "w", encoding="utf-8") as f:
    f.write('"""TỰ SINH bởi packaging/embed_key.py — KHÔNG commit (đã có trong .gitignore)."""\n')
    f.write(f'_PAD = bytes.fromhex("{pad.hex()}")\n')
    f.write(f'_DATA = bytes.fromhex("{bytes(a ^ b for a, b in zip(data, pad)).hex()}")\n\n\n')
    f.write("def load() -> str:\n    return bytes(a ^ b for a, b in zip(_DATA, _PAD)).decode()\n")
print(f"Đã nhúng key ({key[:4]}…{len(key)} ký tự) vào luboo/_embedded_key.py")
