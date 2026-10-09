"""Nhận diện từ kích hoạt bằng các model ONNX của openWakeWord, chạy thẳng bằng onnxruntime.

Viết lại phần xử lý của thư viện openWakeWord để app không phải kéo theo scipy/scikit-learn
(~150 MB). Quy trình cho mỗi khung 80ms (1280 mẫu, 16kHz):

    âm thanh -> melspectrogram.onnx -> 76 khung mel gần nhất -> embedding_model.onnx
             -> vector 96 chiều -> 16 vector gần nhất -> <từ kích hoạt>.onnx -> điểm 0..1
"""
import os
import urllib.request

import numpy as np
import onnxruntime as ort

FRAME = 1280                    # 80ms @ 16kHz
MEL_WINDOW = 76                 # số khung mel đưa vào model embedding
MEL_STEP = 8                    # mỗi 80ms sinh thêm 8 khung mel
MEL_CONTEXT = 160 * 3           # mẫu âm thanh cũ cần thêm để tính mel liền mạch
MAX_FEATURES = 120              # ~10 giây lịch sử embedding
WARMUP_FRAMES = 5               # bỏ qua vài khung đầu sau khi reset

DOWNLOAD_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/{}"


def download_models(models_dir: str, names: list[str]):
    """Tải các file model còn thiếu (dùng khi chạy từ mã nguồn; bản đóng gói đã có sẵn)."""
    os.makedirs(models_dir, exist_ok=True)
    for name in ["melspectrogram", "embedding_model", *names]:
        path = os.path.join(models_dir, f"{name}.onnx")
        if os.path.exists(path):
            continue
        tmp = path + ".part"                        # tải xong mới đổi tên -> không bao giờ dính file dở
        urllib.request.urlretrieve(DOWNLOAD_URL.format(f"{name}.onnx"), tmp)
        os.replace(tmp, path)


class WakeWordEngine:
    def __init__(self, models_dir: str, wake_models: dict[str, str]):
        """models_dir: chứa melspectrogram/embedding_model. wake_models: {tên: đường dẫn .onnx}."""
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1

        def load(path):
            return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

        self.mel = load(os.path.join(models_dir, "melspectrogram.onnx"))
        self.embedding = load(os.path.join(models_dir, "embedding_model.onnx"))
        self.models = {n: load(p) for n, p in wake_models.items()}
        self.inputs = {n: (m.get_inputs()[0].name, m.get_inputs()[0].shape[1])
                       for n, m in self.models.items()}
        self.reset()

    def _melspec(self, audio: np.ndarray) -> np.ndarray:
        spec = self.mel.run(None, {"input": audio[None].astype(np.float32)})[0]
        return np.squeeze(spec) / 10 + 2            # đưa về thang giống bản TensorFlow gốc

    def _embed(self, mel_windows: np.ndarray) -> np.ndarray:
        out = self.embedding.run(None, {"input_1": mel_windows[..., None].astype(np.float32)})[0]
        return out.reshape(-1, 96)

    def reset(self):
        self.raw = np.zeros(0, dtype=np.int16)
        self.mel_buffer = np.ones((MEL_WINDOW, 32), dtype=np.float32)
        # Khởi tạo bộ nhớ embedding bằng tiếng ồn ngẫu nhiên (giống thư viện gốc)
        noise = np.random.default_rng().integers(-1000, 1000, 16000 * 4).astype(np.int16)
        spec = self._melspec(noise)
        windows = np.array([spec[i:i + MEL_WINDOW]
                            for i in range(0, len(spec) - MEL_WINDOW + 1, MEL_STEP)])
        self.features = self._embed(windows)
        self.frames = 0

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        """frame: đúng 1280 mẫu int16. Trả về điểm 0..1 cho từng từ kích hoạt."""
        self.raw = np.concatenate([self.raw, frame])[-(FRAME + MEL_CONTEXT):]
        self.mel_buffer = np.vstack([self.mel_buffer, self._melspec(self.raw)])[-MEL_WINDOW:]
        self.features = np.vstack([self.features, self._embed(self.mel_buffer[None])])[-MAX_FEATURES:]
        self.frames += 1

        scores = {}
        for name, model in self.models.items():
            input_name, n = self.inputs[name]
            x = self.features[-n:][None].astype(np.float32)
            score = float(model.run(None, {input_name: x})[0].reshape(-1)[0])
            scores[name] = 0.0 if self.frames <= WARMUP_FRAMES else score
        return scores
