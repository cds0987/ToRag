import json
import os
import tempfile
from huggingface_hub import upload_file

def save_json_hf(metadata: dict, repo_id: str, filename: str):
    """
    Save metadata JSON to HuggingFace Hub dataset repo.
    """
    meta_filename = filename + "meta.json"

    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = os.path.join(tmpdir, meta_filename)

        with open(meta_path, "w") as f:
            json.dump(metadata, f)

        upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=meta_path,
            path_in_repo=meta_filename,
        )

    return f"{repo_id}/{meta_filename}"