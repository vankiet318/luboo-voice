"""Owner voice recognition (speaker verification) with WeSpeaker ResNet34 (ONNX, CC BY 4.0).

Enrollment: read a few sentences -> the model turns each into a "voiceprint" (256-number vector) -> the average is
saved to %APPDATA%\\Luboo\\voiceprint.npy (local only, never sent anywhere). The model is not retrained.
Verification: compute a vector for the new speech clip and compare its (cosine) similarity with the saved vector.
"""
import os

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort

from . import config

MODEL_PATH = os.path.join(config.MODELS_DIR, "wespeaker_resnet34.onnx")
PROFILE_PATH = os.path.join(config.DATA_DIR, "voiceprint.npy")


def fbank(audio: np.ndarray) -> np.ndarray:
    """80-dim fbank features as in WeSpeaker training (25ms / 10ms, no dither), mean-subtracted (CMN)."""
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = config.SAMPLE_RATE
    opts.frame_opts.dither = 0.0
    opts.mel_opts.num_bins = 80
    f = knf.OnlineFbank(opts)
    f.accept_waveform(config.SAMPLE_RATE, audio.astype(np.float32).tolist())   # int16 scale, like WeSpeaker
    f.input_finished()
    feats = np.array([f.get_frame(i) for i in range(f.num_frames_ready)], dtype=np.float32)
    return feats - feats.mean(axis=0, keepdims=True)


class SpeakerID:
    def __init__(self, model_path: str = MODEL_PATH, profile_path: str = PROFILE_PATH):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.profile_path = profile_path
        self.profile = np.load(profile_path) if os.path.exists(profile_path) else None

    @property
    def enrolled(self) -> bool:
        return self.profile is not None

    def embed(self, audio: np.ndarray) -> np.ndarray:
        feats = fbank(audio)[None]
        emb = self.session.run(None, {"feats": feats})[0][0]
        return emb / (np.linalg.norm(emb) + 1e-9)

    def similarity(self, audio: np.ndarray) -> float:
        """Similarity to the owner's voice, -1..1 (not enrolled -> 0)."""
        if self.profile is None or len(audio) < 0.3 * config.SAMPLE_RATE:
            return 0.0
        return float(np.dot(self.embed(audio), self.profile))

    def is_owner(self, audio: np.ndarray) -> tuple[bool, float]:
        score = self.similarity(audio)
        return score >= config.SPEAKER_THRESHOLD, score

    def enroll(self, clips: list[np.ndarray]) -> np.ndarray:
        """Save a reference voice from a few spoken sentences. Returns each one's similarity to it (for checking)."""
        embs = np.stack([self.embed(c) for c in clips])
        profile = embs.mean(axis=0)
        profile /= np.linalg.norm(profile) + 1e-9
        os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
        tmp = self.profile_path + ".tmp.npy"
        np.save(tmp, profile)
        os.replace(tmp, self.profile_path)
        self.profile = profile
        return embs @ profile

    def forget(self):
        self.profile = None
        if os.path.exists(self.profile_path):
            os.remove(self.profile_path)
