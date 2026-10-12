"""Wake word detection using openWakeWord's ONNX models, run directly with onnxruntime.

Reimplements the openWakeWord library's processing so the app doesn't pull in scipy/scikit-learn
(~150 MB). Pipeline for each 80ms frame (1280 samples, 16kHz):

    audio -> melspectrogram.onnx -> last 76 mel frames -> embedding_model.onnx
          -> 96-dim vector -> last 16 vectors -> <wake word>.onnx -> score 0..1
"""
import os
import urllib.request

import numpy as np
import onnxruntime as ort

FRAME = 1280                    # 80ms @ 16kHz
MEL_WINDOW = 76                 # mel frames fed to the embedding model
MEL_STEP = 8                    # each 80ms yields 8 new mel frames
MEL_CONTEXT = 160 * 3           # extra past audio samples needed for seamless mel computation
MAX_FEATURES = 120              # ~10 seconds of embedding history
WARMUP_FRAMES = 5               # ignore the first few frames after a reset

DOWNLOAD_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/{}"


def download_models(models_dir: str, names: list[str]):
    """Download any missing model files (used when running from source; the packaged build has them)."""
    os.makedirs(models_dir, exist_ok=True)
    for name in ["melspectrogram", "embedding_model", *names]:
        path = os.path.join(models_dir, f"{name}.onnx")
        if os.path.exists(path):
            continue
        tmp = path + ".part"                        # rename only after download completes -> never a partial file
        urllib.request.urlretrieve(DOWNLOAD_URL.format(f"{name}.onnx"), tmp)
        os.replace(tmp, path)


class WakeWordEngine:
    def __init__(self, models_dir: str, wake_models: dict[str, str]):
        """models_dir: holds melspectrogram/embedding_model. wake_models: {name: .onnx path}."""
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
        return np.squeeze(spec) / 10 + 2            # rescale to match the original TensorFlow version

    def _embed(self, mel_windows: np.ndarray) -> np.ndarray:
        out = self.embedding.run(None, {"input_1": mel_windows[..., None].astype(np.float32)})[0]
        return out.reshape(-1, 96)

    def reset(self):
        self.raw = np.zeros(0, dtype=np.int16)
        self.mel_buffer = np.ones((MEL_WINDOW, 32), dtype=np.float32)
        # Initialize the embedding buffer with random noise (same as the original library)
        noise = np.random.default_rng().integers(-1000, 1000, 16000 * 4).astype(np.int16)
        spec = self._melspec(noise)
        windows = np.array([spec[i:i + MEL_WINDOW]
                            for i in range(0, len(spec) - MEL_WINDOW + 1, MEL_STEP)])
        self.features = self._embed(windows)
        self.frames = 0

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        """frame: exactly 1280 int16 samples. Returns a 0..1 score for each wake word."""
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
