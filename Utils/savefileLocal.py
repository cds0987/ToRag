import json
import os
from typing import Union, Dict, Any

def save_json_local(
    data: Union[Dict, Any],
    directory: str,
    filename: str,
    indent: int = 2
):
    """
    Save a dict (or JSON-serializable object) to local disk.
    """
    os.makedirs(directory, exist_ok=True)

    if not filename.endswith(".json"):
        filename += ".json"

    path = os.path.join(directory, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    return path