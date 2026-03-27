import json
import os
from typing import Any

def load_json_local(path: str) -> Any:
    """
    Load a JSON file from local disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data