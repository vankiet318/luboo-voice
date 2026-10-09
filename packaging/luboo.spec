# PyInstaller — đóng gói dạng thư mục (onedir): mở nhanh, ít bị diệt virus báo nhầm.
# Chạy qua build.ps1, không cần gọi trực tiếp.
import os
import sys

ROOT = os.path.dirname(SPECPATH)                         # spec nằm trong packaging/, mã nguồn ở thư mục gốc
sys.path.insert(0, ROOT)
from luboo import config  # noqa: E402

# Giữ nguyên cấu trúc resources/ bên trong bản đóng gói (config.BASE_DIR = _MEIPASS)
datas = [(os.path.join(config.MODELS_DIR, f"{m}.onnx"), "resources/models") for m in ("melspectrogram", "embedding_model")]
datas += [(config.wake_word_path(n), os.path.relpath(os.path.dirname(config.wake_word_path(n)), ROOT))
          for n in config.WAKE_WORDS]
datas += [(config.ICON_PATH, "resources")]

a = Analysis(
    [os.path.join(SPECPATH, "luboo_main.py")],
    pathex=[ROOT],
    datas=datas,
    excludes=[
        # Whisper chạy trên máy — không kèm trong bản public
        "faster_whisper", "ctranslate2", "tokenizers", "av", "huggingface_hub",
        # Thư viện nặng không dùng tới
        "scipy", "sklearn", "openwakeword", "matplotlib", "pandas", "IPython", "torch",
        "onnxruntime.tools", "onnxruntime.transformers", "onnxruntime.quantization",
        "onnxruntime.training",
        # Thư viện chuẩn không dùng tới
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
    console=False,          # không hiện cửa sổ đen
    upx=False,              # UPX hay làm phần mềm diệt virus báo nhầm
)
coll = COLLECT(exe, a.binaries, a.datas, name=config.APP_NAME, upx=False)
