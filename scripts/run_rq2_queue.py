"""Run RQ2 for several models one after another, then analyze each.

    python -X utf8 scripts/run_rq2_queue.py
Each model runs in a fresh process (GPU memory is fully freed between models); a model whose
results are already complete (900 rows in every condition file) is skipped, so the queue can be
restarted after a crash. Everything is written to results/logs/progress.log; each model's full
console output goes to results/logs/<tag>.console.txt. Start it detached so it does not depend
on the terminal that launched it:
    Start-Process -WindowStyle Hidden .venv/Scripts/python.exe "-X utf8 scripts/run_rq2_queue.py"
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conditions import CONDITIONS, EXTRA_CONDITIONS  # noqa: E402

ALL = CONDITIONS + EXTRA_CONDITIONS  # main conditions + letter ablations
from progress import log  # noqa: E402

PY = sys.executable
# (repo, load_4bit). 7B+ do not fit in 12 GB at bf16, so they run 4-bit NF4. Gemma 3 4B is run
# again in 4-bit to measure whether quantization itself changes the gaps between conditions.
QUEUE = [
    ("unsloth/gemma-3-4b-it", True),
    ("Qwen/Qwen2.5-7B-Instruct", True),
    ("unsloth/Llama-3.1-8B-Instruct", True),
    ("Qwen/Qwen3-8B", True),
    ("unsloth/gemma-3-12b-it", True),
    ("unsloth/Mistral-Nemo-Instruct-2407", True),
]
ENV = {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "PYTHONUTF8": "1"}


def complete(tag: str, n: int = 900) -> bool:
    d = ROOT / "results/rq2" / tag
    return all((d / f"{c}.jsonl").exists() and len((d / f"{c}.jsonl").read_text(encoding="utf-8").splitlines()) == n
               for c in ALL)


def main():
    log(f"[queue] start: {len(QUEUE)} models")
    for repo, q4 in QUEUE:
        tag = repo.replace("/", "__") + ("__4bit" if q4 else "")
        if complete(tag):
            log(f"[queue] {tag}: already complete, skipping")
            continue
        cmd = [PY, "-X", "utf8", str(ROOT / "scripts/rq2_eval.py"), "--model", repo,
               "--conditions", ",".join(ALL), "--skip-done"] + (["--load-4bit"] if q4 else [])
        with open(ROOT / f"results/logs/{tag}.console.txt", "w", encoding="utf-8") as out:
            rc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, env=ENV, cwd=ROOT).returncode
        if rc != 0:
            log(f"[queue] {tag}: FAILED (exit {rc}), see results/logs/{tag}.console.txt; continuing")
            continue
        res = subprocess.run([PY, "-X", "utf8", str(ROOT / "scripts/rq2_analyze.py"), str(ROOT / "results/rq2" / tag)],
                             capture_output=True, text=True, encoding="utf-8", env=ENV)
        log(f"[queue] {tag}: analyzed\n{res.stdout.strip()}")
    log("[queue] all models finished")


if __name__ == "__main__":
    main()
