from __future__ import annotations
import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return name or "file"

def sha1_of_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def get_env(key: str, default: Optional[str]=None) -> Optional[str]:
    v = os.getenv(key, default)
    return v

def dump_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
