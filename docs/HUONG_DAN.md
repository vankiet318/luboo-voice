# Hướng dẫn phát triển Luboo

1. [Chạy từ mã nguồn](#1-chạy-từ-mã-nguồn)
2. [Build file cài đặt `.exe`](#2-build-file-cài-đặt-exe)
3. [Model từ kích hoạt](#3-model-từ-kích-hoạt)
4. [Tùy chỉnh](#4-tùy-chỉnh)
5. [Xử lý sự cố](#5-xử-lý-sự-cố)

Các lệnh dưới đây chạy trong PowerShell, tại thư mục gốc của repo.

---

## 1. Chạy từ mã nguồn

Cần Windows 10/11 64-bit, Python 3.11 (`py -3.11`), mic và một Groq API key (miễn phí tại
https://console.groq.com/keys).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m luboo
```

- Lần đầu app tự tải 2 model trích đặc trưng vào `resources/models/`.
- Nhập key trong **Settings**, hoặc đặt biến môi trường `GROQ_API_KEY`.
- Khi khởi động, giữ yên lặng ~2 giây để app đo tiếng ồn nền.
- `requirements-dev.txt` thêm Whisper chạy trên máy (dự phòng khi Groq lỗi) — không bắt buộc.
- `python -m luboo --selftest` kiểm tra các model đi kèm rồi thoát.

Log: `%APPDATA%\Luboo\luboo.log` · Cài đặt: `%APPDATA%\Luboo\settings.json`.

## 2. Build file cài đặt `.exe`

Cần thêm Inno Setup 6: `winget install JRSoftware.InnoSetup`

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build.ps1
```

Kết quả: `dist\Luboo-Setup-<phiên bản>.exe`. Đổi phiên bản ở `APP_VERSION` trong `luboo/config.py`.

`packaging/build.ps1` tạo môi trường `.venv-build` sạch, tải model, tạo icon, đóng gói bằng PyInstaller
(`packaging/luboo.spec`) rồi gói thành bộ cài bằng Inno Setup (`packaging/installer.iss`).

**Bản có nhúng sẵn Groq key (gửi bạn bè test):**

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build.ps1 -EmbedKey
```

- Key lấy từ biến môi trường `GROQ_API_KEY`, nếu không có thì lấy key đã lưu trong Settings của Luboo.
- Ra file `dist\Luboo-Setup-<phiên bản>-PRIVATE.exe` — người nhận cài xong dùng được ngay.
- Key chỉ được **làm rối**, không phải mã hóa: người rành kỹ thuật vẫn lấy ra được. Chỉ gửi người tin cậy;
  lộ key thì xóa tại console.groq.com/keys rồi build lại. Mọi người dùng chung hạn mức miễn phí của key.
- File tạm `luboo/_embedded_key.py` bị git bỏ qua và tự xóa sau khi build.

Exe chưa ký số nên Windows SmartScreen cảnh báo lần đầu: bấm **More info → Run anyway**.

## 3. Model từ kích hoạt

`resources/wakewords/hey_luboo.onnx` ("Luboo" đọc là "Lu-bu") là mạng nhỏ (3 lớp, ~0,8 MB) chạy trên đặc trưng
âm thanh của openWakeWord (`melspectrogram.onnx` + `embedding_model.onnx`, xem `luboo/wakeword.py`).

Model được train riêng trên Kaggle (GPU) với:
- ~14.000 mẫu giọng tổng hợp (Piper, edge-tts) và giọng thật của người tạo app;
- mẫu sai: câu nghe gần giống, nói chuyện thường, ~700.000 đoạn âm thanh đời thường (ACAV100M);
- cửa sổ trượt 80 ms giống hệt lúc app chạy, kèm tăng cường dữ liệu (tiếng vang phòng, tiếng ồn, đổi giọng).

Kết quả với cấu hình hiện tại (ngưỡng 0,99, 3 khung liên tiếp): ~1–2 lần bật nhầm/giờ, nhận ~95% lần gọi.

Thay model: chép file `.onnx` mới (đầu vào `[1, 16, 96]`, đầu ra 0..1) vào `resources/wakewords/` rồi build lại.
Thiếu file thì app tạm dùng "Hey Jarvis" có sẵn của openWakeWord.

## 4. Tùy chỉnh

Trong `luboo/config.py`:

| Thiết lập | Ý nghĩa |
|---|---|
| `WAKE_THRESHOLD`, `WAKE_PATIENCE` | Ngưỡng điểm và số khung 80ms liên tiếp phải vượt ngưỡng (thấp = dễ gọi nhưng dễ nhầm) |
| `VOICE`, `TTS_RATE` | Giọng đọc, tốc độ đọc |
| `SILENCE_SEC`, `FOLLOWUP_SEC` | Im lặng bao lâu thì coi là nói xong; nghe tiếp bao lâu sau khi trả lời |
| `GREETINGS`, `GREETING_GRACE_SEC` | Các câu Luboo chào khi được gọi; gọi liền một mạch với câu hỏi thì bỏ qua lời chào |
| `SYSTEM_PROMPT` | Tính cách của Luboo |
| `CHAT_MODEL`, `SEARCH_MODEL`, `FALLBACK_MODEL` | Model Groq |
| `OLLAMA_MODEL` | Dự phòng cuối bằng LLM chạy trên máy |

## 5. Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| Gọi không phản ứng | Kiểm tra mic không bị tắt; Windows Settings → Privacy → Microphone cho phép app desktop |
| Hay kích hoạt nhầm | Tăng `WAKE_PATIENCE` (3 → 4) hoặc `WAKE_THRESHOLD` |
| Gọi khó nhận | Giảm `WAKE_PATIENCE` (3 → 2) |
| Cắt câu sớm | Tăng `SILENCE_SEC` |
| Không có tiếng trả lời | edge-tts cần mạng; xem `%APPDATA%\Luboo\luboo.log` |
| "Add a Groq API key in Settings…" | Settings → nhập API key Groq → Save |
| Lỗi 404 "model does not exist" | Groq đã ngừng model đó — đổi `CHAT_MODEL`/`SEARCH_MODEL` theo console.groq.com/docs/models |
