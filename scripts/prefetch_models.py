"""Download the queue's remaining models ahead of time, one at a time, in a separate process.

    python -X utf8 scripts/prefetch_models.py
Goes in REVERSE queue order so it works on the models the queue reaches last, and skips any
model the queue has already started (seen in progress.log). Two processes never download the
same model at once, which is what deadlocked the first queue.
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from progress import LOG, log  # noqa: E402
from run_rq2_queue import QUEUE  # noqa: E402

for repo, _ in reversed(QUEUE):
    if f"[rq2] {repo} @" in LOG.read_text(encoding="utf-8"):
        log(f"[prefetch] {repo}: queue already started it, skipping")
        continue
    log(f"[prefetch] {repo}: downloading")
    try:
        p = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
        gb = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 2**30
        log(f"[prefetch] {repo}: done ({gb:.1f} GB)")
    except Exception as e:
        log(f"[prefetch] {repo}: FAILED {type(e).__name__}: {str(e)[:150]}")
log("[prefetch] finished")
