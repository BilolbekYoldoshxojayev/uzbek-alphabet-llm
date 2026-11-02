"""Check a downloaded model's weight files against the SHA-256 published on Hugging Face and
re-download only the files that do not match.

    python -X utf8 scripts/repair_model_cache.py unsloth/Mistral-Nemo-Instruct-2407
"""
import hashlib
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from progress import log  # noqa: E402

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


repo = sys.argv[1]
api = HfApi()
sha = api.model_info(repo).sha
files = [f for f in api.list_repo_files(repo, revision=sha) if f.endswith(".safetensors")]
for info in api.get_paths_info(repo, files, revision=sha):
    want = info.lfs.sha256
    path = Path(hf_hub_download(repo, info.path, revision=sha))
    if sha256(path) == want:
        log(f"[repair] {repo} {info.path}: ok")
        continue
    # Without symlinks (Windows, no Developer Mode) the loader reads a *copy* in snapshots/, which
    # two concurrent downloads can corrupt even when the blob is fine. Fix the copy from a good blob.
    blob = path.parents[2] / "blobs" / want
    if not (blob.exists() and sha256(blob) == want):
        log(f"[repair] {repo} {info.path}: BAD ({path.stat().st_size} bytes), re-downloading")
        hf_hub_download(repo, info.path, revision=sha, force_download=True)
    if blob.exists() and not path.is_symlink() and sha256(blob) == want:
        shutil.copyfile(blob, path)
    log(f"[repair] {repo} {info.path}: {'fixed' if sha256(path) == want else 'STILL BAD'}")
log(f"[repair] {repo}: finished")
