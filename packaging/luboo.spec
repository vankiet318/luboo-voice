# PyInstaller — folder (onedir) build: starts fast, fewer antivirus false positives.
# Run via build.ps1, no need to call it directly.
import os
import sys

ROOT = os.path.dirname(SPECPATH)                         # spec lives in packaging/, source is in the repo root
sys.path.insert(0, ROOT)
from luboo import config  # noqa: E402

# Keep the resources/ layout inside the bundle (config.BASE_DIR = _MEIPASS)
datas = [(os.path.join(config.MODELS_DIR, f"{m}.onnx"), "resources/models")
         for m in ("melspectrogram", "embedding_model", "silero_vad", "wespeaker_resnet34")]
datas += [(config.wake_word_path(n), os.path.relpath(os.path.dirname(config.wake_word_path(n)), ROOT))
          for n in config.WAKE_WORDS]
datas += [(config.ICON_PATH, "resources")]

a = Analysis(
    [os.path.join(SPECPATH, "luboo_main.py")],
    pathex=[ROOT],
    datas=datas,
    excludes=[
        # Local Whisper — not included in the public build
        "faster_whisper", "ctranslate2", "tokenizers", "av", "huggingface_hub",
        # Heavy libraries we don't use
        "scipy", "sklearn", "openwakeword", "matplotlib", "pandas", "IPython", "torch",
        "onnxruntime.tools", "onnxruntime.transformers", "onnxruntime.quantization",
        "onnxruntime.training",
        # Unused standard library modules
        "unittest", "pydoc", "pdb", "doctest", "lib2to3", "sqlite3", "xmlrpc",
        "tkinter.test", "idlelib", "turtledemo", "test",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=config.APP_NAME,
    icon=config.ICON_PATH,
    console=False,          # no black console window
    upx=False,              # UPX often triggers antivirus false positives
)
coll = COLLECT(exe, a.binaries, a.datas, name=config.APP_NAME, upx=False)
