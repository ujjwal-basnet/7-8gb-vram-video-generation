"""Pinned user-selected H3 hybrid checkpoint; no ComfyUI runtime required."""
from pathlib import Path
import hashlib
import json

REPO = "smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models"
REVISION = "a36feb17fbd1f20ff4bdd509ccd07e2b7b585a38"
FILENAME = "minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors"
SIZE = 20970379632
SHA256 = "a629cfea8d89a071b140c6e1935dc9a23e72de6badc18975a2bb9e6d1423d76d"
ROOT = Path(__file__).resolve().parent
PATH = ROOT / "models/hybrid" / FILENAME


def download():
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(REPO, FILENAME, revision=REVISION,
                               local_dir=PATH.parent))
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if path.stat().st_size != SIZE or digest != SHA256:
        raise ValueError("Hybrid checkpoint failed size/SHA-256 verification")
    receipt = dict(repo=REPO, revision=REVISION, filename=FILENAME,
                   bytes=SIZE, sha256=digest, status="verified")
    (ROOT / "hybrid-download.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2), flush=True)
    return path


if __name__ == "__main__":
    download()
