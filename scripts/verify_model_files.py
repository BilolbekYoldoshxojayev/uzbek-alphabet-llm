"""Check every model's cached files against the SHA-256 published on Hugging Face. Read-only.

    python -X utf8 scripts/verify_model_files.py
Concurrent downloads silently corrupted one model's cache (same size, wrong bytes), so a model
that loads is not proof that its weights are right. Writes results/logs/model_file_hashes.json.
"""
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, try_to_load_from_cache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from progress import log  # noqa: E402

MODELS = ["unsloth/gemma-3-4b-it", "Qwen/Qwen2.5-7B-Instruct", "unsloth/Llama-3.1-8B-Instruct", "Qwen/Qwen3-8B",
          "unsloth/gemma-3-12b-it", "unsloth/Mistral-Nemo-Instruct-2407"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


api, report, bad = HfApi(), {}, 0
for repo in MODELS:
    rev = api.model_info(repo).sha
    files = [f for f in api.list_repo_files(repo, revision=rev) if f.endswith((".safetensors", ".json", ".model"))]
    report[repo] = {"revision": rev, "files": {}}
    for info in api.get_paths_info(repo, files, revision=rev):
        path = try_to_load_from_cache(repo, info.path, revision=rev)
        if not isinstance(path, str):
            report[repo]["files"][info.path] = "not in cache"
            continue
        if info.lfs:
            ok = sha256(path) == info.lfs.sha256
        else:
            ok = Path(path).stat().st_size == info.size
        report[repo]["files"][info.path] = "ok" if ok else "BAD"
        bad += not ok
        if not ok:
            log(f"[verify] {repo} {info.path}: BAD")
    n = sum(v == "ok" for v in report[repo]["files"].values())
    log(f"[verify] {repo} @ {rev[:7]}: {n} ok, {sum(v == 'BAD' for v in report[repo]['files'].values())} BAD, "
        f"{sum(v == 'not in cache' for v in report[repo]['files'].values())} not in cache")
(ROOT / "results/logs/model_file_hashes.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
log(f"[verify] finished: {bad} bad files")
