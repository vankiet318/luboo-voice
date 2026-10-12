"""Voice activity detection with Silero VAD (ONNX, MIT) + an endpointer that finds where an utterance starts / ends.

Silero scores each 512-sample block (32ms @ 16kHz) and returns a 0..1 probability of human speech.
Unlike the old loudness-based approach, keyboard clicks, fans, closing doors... are not counted as speech.
"""
import os

import numpy as np
import onnxruntime as ort

from . import config

BLOCK = 512                    # samples per block @ 16kHz (Silero v5 only accepts exactly this size)
CONTEXT = 64                   # Silero v5 needs the last 64 samples of the previous block prepended to the next
BLOCK_SEC = BLOCK / config.SAMPLE_RATE
MODEL_PATH = os.path.join(config.MODELS_DIR, "silero_vad.onnx")


class SileroVAD:
    def __init__(self, path: str = MODEL_PATH):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.sr = np.array(config.SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(CONTEXT, dtype=np.float32)

    def __call__(self, block: np.ndarray) -> float:
        """block: exactly 512 int16 samples. Returns the speech probability."""
        x = block.astype(np.float32) / 32768.0
        inp = np.concatenate([self.context, x])[None]
        out, self.state = self.session.run(None, {"input": inp, "state": self.state, "sr": self.sr})
        self.context = x[-CONTEXT:]
        return float(out[0, 0])


class Endpointer:
    """Takes (audio block, speech probability) pairs in order and decides where an utterance starts / ends.

    - Start: continuous speech >= VAD_MIN_SPEECH_SEC (ignores short "clicks").
    - End: continuous silence >= SILENCE_SEC after speech has started.
    Shared by the app (live mic) and tools/bench_vad.py (audio files) so the measurements match the app.
    """

    def __init__(self, silence_sec: float | None = None):
        self.silence_sec = config.SILENCE_SEC if silence_sec is None else silence_sec
        self.min_speech = max(1, round(config.VAD_MIN_SPEECH_SEC / BLOCK_SEC))
        self.pre_roll = round(config.VAD_PRE_ROLL_SEC / BLOCK_SEC)
        self.blocks: list[np.ndarray] = []
        self.start = None              # index of the block where speech starts
        self.run = 0                   # number of consecutive voiced blocks
        self.silent = 0                # number of consecutive silent blocks after speech started
        self.last_voiced = None        # index of the last voiced block

    @property
    def started(self) -> bool:
        return self.start is not None

    def feed(self, block: np.ndarray, prob: float) -> bool:
        """Returns True once the utterance has ended."""
        self.blocks.append(block)
        i = len(self.blocks) - 1
        voiced = prob >= config.VAD_THRESHOLD
        if not self.started:
            self.run = self.run + 1 if voiced else 0
            if self.run >= self.min_speech:
                self.start = i - self.run + 1
                self.last_voiced = i
            elif not self.run:
                del self.blocks[:-self.pre_roll or None]   # nobody speaking yet: keep only enough blocks for pre-roll
            return False
        if voiced:
            self.silent, self.last_voiced = 0, i
            return False
        self.silent += 1
        return self.silent * BLOCK_SEC >= self.silence_sec

    def speech_sec(self) -> float:
        return 0.0 if not self.started else (self.last_voiced - self.start + 1) * BLOCK_SEC

    def audio(self) -> np.ndarray:
        """The utterance (plus VAD_PRE_ROLL_SEC before the start) through the end of the trailing silence."""
        return np.concatenate(self.blocks[max(0, self.start - self.pre_roll):])

    def speech_audio(self) -> np.ndarray:
        """Only the voiced part (from the start to the last voiced block) — used for speaker recognition."""
        return np.concatenate(self.blocks[self.start:self.last_voiced + 1])

    def silence_after_speech_sec(self) -> float:
        """How long it has been silent since the last voiced block (to know when the user stopped speaking)."""
        return self.silent * BLOCK_SEC
