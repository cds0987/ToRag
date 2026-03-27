import json
from huggingface_hub import hf_hub_download
from typing import Any

def load_json_hf(repo_id: str, filename: str) -> Any:
    """
    Load JSON metadata from HuggingFace Hub dataset repo.
    """
    meta_filename = filename + "meta.json"

    # download file from HF
    path = hf_hub_download(
        repo_id=repo_id,
        filename=meta_filename,
        repo_type="dataset"
    )

    # load json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)