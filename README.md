# Luboo

Trợ lý giọng nói **tiếng Anh** cho Windows, với nhân vật Luboo biểu cảm và giao diện tối giản.
Gọi Luboo bằng giọng nói, hỏi bất cứ điều gì — kể cả khi đang chơi game — và nghe câu trả lời.

## Tính năng

- **Từ kích hoạt "Hey Luboo"** ("Luboo" đọc là "Lu-bu") — model tự train, chạy ngay trên máy.
- **Tra cứu web** cho câu hỏi kiến thức (hướng dẫn game, thông tin mới), trò chuyện thường thì trả lời nhanh.
- **Nhân vật Luboo** đổi cảm xúc theo câu trả lời, chớp mắt, nhìn quanh, miệng mấp máy theo giọng nói.
- **Tối giản** — chỉ có nhân vật, một dòng chữ và hai nút *Talk* · *Settings*; bấm vào Luboo để nói hoặc ngắt lời.
- **Dùng Groq API key miễn phí** — nhập trong Settings; có thể build bản nhúng sẵn key để gửi người thân test.
- **Nhẹ** — chạy bằng CPU; bộ cài khoảng 34 MB.

## Cách hoạt động

```
Mic ─► Từ kích hoạt ─► Groq Whisper ─► Groq LLM ─► edge-tts ─► Loa
       (trên máy)      (giọng → chữ)    (+ tra web)  (chữ → giọng)
                                            └─► Luboo đổi cảm xúc
```

| Thành phần | Công nghệ |
|---|---|
| Từ kích hoạt | Model tự train ("Hey Luboo") trên đặc trưng của openWakeWord, chạy bằng onnxruntime |
| Giọng nói → chữ | Groq `whisper-large-v3-turbo` |
| Trả lời | Groq `openai/gpt-oss-120b` (+ công cụ tra web `browser_search`), dự phòng `openai/gpt-oss-20b` |
| Chữ → giọng nói | edge-tts |
| Giao diện | Tkinter |

## Quyền riêng tư

- Âm thanh chỉ được gửi đi **sau khi** nghe thấy từ kích hoạt (hoặc bấm *Talk* / bấm vào Luboo).
- API key riêng chỉ lưu trên máy và chỉ gửi tới Groq.

## Cấu trúc

```
luboo/          mã nguồn app — chạy: python -m luboo
resources/      model từ kích hoạt tự train (wakewords/), model tải về, icon
packaging/      build file cài đặt .exe (build.ps1, PyInstaller, Inno Setup)
docs/           hướng dẫn
```

| File trong `luboo/` | Vai trò |
|---|---|
| `app.py` | Giao diện: nhân vật Luboo (vẽ bằng Pillow), Settings |
| `assistant.py` | Điều phối: ngủ → nghe → chào → nghĩ → nói |
| `audio.py` | Mic, thu âm, phát giọng đọc, phụ đề khớp giọng |
| `wakeword.py` | Nhận diện từ kích hoạt |
| `brain.py`, `api.py` | Chép lời, chọn model, gọi Groq |
| `config.py`, `settings.py` | Cấu hình, cài đặt người dùng |

Hướng dẫn chạy, build, model từ kích hoạt, tùy chỉnh: [docs/HUONG_DAN.md](docs/HUONG_DAN.md).

## Giấy phép bên thứ ba

Xem [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt). Hai model trích đặc trưng của openWakeWord
(melspectrogram, embedding) dùng giấy phép CC BY-NC-SA 4.0 (phi thương mại).
