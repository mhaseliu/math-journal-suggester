"""Packaged defaults; explicit paths always remain caller-controlled."""
from pathlib import Path
ROOT = Path(__file__).with_name("resources")

def default_path(path):
    value = Path(path)
    if value.is_absolute() or value.exists():
        return value
    packaged = ROOT / value
    return packaged if packaged.is_file() else value
