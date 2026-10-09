"""Cài đặt của người dùng, lưu ở %APPDATA%\\Luboo\\settings.json."""
import json
import os

from . import config

PATH = os.path.join(config.DATA_DIR, "settings.json")

DEFAULTS = {
    "api_key": "",              # Groq API key (trống -> GROQ_API_KEY hoặc key nhúng sẵn của bản test)
    "on_top": False,            # cửa sổ luôn nổi trên các cửa sổ khác
}


def load() -> dict:
    data = dict(DEFAULTS)
    try:
        with open(PATH, encoding="utf-8") as f:
            data.update(json.load(f))
    except (OSError, ValueError):
        pass
    return data


def save(data: dict):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    tmp = PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, PATH)
