"""Run every remaining experiment in order, logging everything. Finished steps are skipped on restart.

    Start-Process -WindowStyle Hidden .venv/Scripts/python.exe "-X utf8 scripts/run_all_experiments.py"
Phase 1: RQ2 completion (Llama re-run after its corrupted file was repaired; RQ4 conditions on all
         runs; chat-template and Uzbek-frame robustness; a determinism check).
Phase 2: Exp B, LoRA on old vs new text, 3 seeds, order alternated so that any drift in GPU temperature
         does not favor one spelling; each adapted model re-tested on Belebele.
Phase 3: Exp A, from-scratch models, old vs new, separate and shared tokenizers, 3 seeds, order alternated.
Logs: results/logs/progress.log (live), commands.log (every command with start, end, exit code),
manifest_*.json (hashes of code and data, package versions), gpu_monitor.csv, one console file per step.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conditions import CONDITIONS, EXTRA_CONDITIONS  # noqa: E402
from progress import log  # noqa: E402

PY = sys.executable
ENV = {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "PYTHONUTF8": "1"}
LOGS = ROOT / "results/logs"
RUNS = [("unsloth/gemma-3-4b-it", False), ("unsloth/gemma-3-4b-it", True), ("Qwen/Qwen2.5-7B-Instruct", True),
        ("unsloth/Llama-3.1-8B-Instruct", True), ("Qwen/Qwen3-8B", True), ("unsloth/gemma-3-12b-it", True),
        ("unsloth/Mistral-Nemo-Instruct-2407", True)]
SEEDS = [42, 43, 44]
LORA_CHARS, SCRATCH_CHARS = 4_000_000, 200_000_000
ALL = CONDITIONS + EXTRA_CONDITIONS
failed = []


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest():
    files = sorted(list((ROOT / "scripts").glob("*.py")) + list((ROOT / "src").glob("*.py"))
                   + list((ROOT / "data/parallel/belebele_uzn").glob("*.jsonl")) + list((ROOT / "data/train").glob("*/*.txt"))
                   + [ROOT / "converter/YangiAlifbo.Cli/Program.cs"])
    m = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version,
         "sha256": {str(f.relative_to(ROOT)): sha256(f) for f in files},
         "pip_freeze": subprocess.run([PY, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines(),
         "nvidia_smi": subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout}
    (LOGS / f"manifest_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    log(f"[all] manifest written: {len(files)} files hashed")


def step(name, args):
    """Run one script in a fresh process; a failure is recorded and the queue continues."""
    t0 = time.time()
    log(f"[all] START {name}")
    with open(LOGS / f"step_{name}.console.txt", "w", encoding="utf-8") as out:
        rc = subprocess.run([PY, "-X", "utf8"] + args, stdout=out, stderr=subprocess.STDOUT, env=ENV, cwd=ROOT).returncode
    with open(LOGS / "commands.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{name}\texit={rc}\tseconds={time.time() - t0:.0f}\t{' '.join(args)}\n")
    if rc:
        failed.append(name)
    log(f"[all] {'FINISHED' if rc == 0 else f'FAILED (exit {rc}), see step_{name}.console.txt:'} {name} ({(time.time() - t0) / 60:.1f} min)")
    return rc == 0


def tag(repo, q4):
    return repo.replace("/", "__") + ("__4bit" if q4 else "")


def complete(folder, conds, n=900):
    d = ROOT / "results/rq2" / folder
    return all((d / f"{c}.jsonl").exists() and len((d / f"{c}.jsonl").read_text(encoding="utf-8").splitlines()) == n
               for c in conds)


def evaluate(repo, q4, conds, folder, extra=(), name=None):
    if complete(folder, conds):
        log(f"[all] {folder}: already complete, skipping")
    else:
        step(name or f"eval_{folder}", ["scripts/rq2_eval.py", "--model", repo, "--conditions", ",".join(conds), "--skip-done"]
             + (["--load-4bit"] if q4 else []) + list(extra))
    if (ROOT / "results/rq2" / folder).exists():
        step(f"analyze_{folder}", ["scripts/rq2_analyze.py", str(ROOT / "results/rq2" / folder)])


def phase1():
    log("[all] PHASE 1: RQ2 completion")
    for repo, q4 in RUNS:
        evaluate(repo, q4, ALL, tag(repo, q4))
    step("summary", ["scripts/rq2_summary.py"])
    step("determinism", ["scripts/rq2_eval.py", "--model", RUNS[0][0], "--conditions", "latin_okina,new2026", "--limit", "50"])
    for prompt in ("chat", "uz"):
        for repo, q4 in RUNS[1:]:
            evaluate(repo, q4, ["latin_okina", "new2026"], f"{tag(repo, q4)}__{prompt}prompt", ["--prompt", prompt])


def phase2():
    log("[all] PHASE 2: Exp B (LoRA)")
    base = "unsloth/gemma-3-4b-it"
    for i, seed in enumerate(SEEDS):
        for sp in (("old", "new") if i % 2 == 0 else ("new", "old")):
            run = f"{tag(base, True)}__lora_{sp}_s{seed}"
            d = ROOT / "results/train/lora" / run
            if not (d / "metrics.json").exists():
                step(f"lora_{sp}_s{seed}", ["scripts/train_lora.py", "--model", base, "--spelling", sp, "--seed", str(seed),
                                            "--chars", str(LORA_CHARS)])
            if (d / "adapter").exists():
                evaluate(base, True, ["latin_okina", "new2026"], run, ["--adapter", str(d / "adapter")])


def phase3():
    log("[all] PHASE 3: Exp A (from scratch)")
    i = 0
    for seed in SEEDS:
        for mode in ("separate", "shared"):
            for sp in (("old", "new") if i % 2 == 0 else ("new", "old")):
                if not (ROOT / f"results/train/scratch/{mode}_{sp}_s{seed}/metrics.json").exists():
                    step(f"scratch_{mode}_{sp}_s{seed}", ["scripts/train_scratch.py", "--spelling", sp, "--tokenizer", mode,
                                                          "--seed", str(seed), "--chars", str(SCRATCH_CHARS)])
            i += 1


def main():
    log("[all] ========== full experiment run started ==========")
    manifest()
    mon = subprocess.Popen([PY, "-X", "utf8", "scripts/monitor_gpu.py"], cwd=ROOT, env=ENV)
    try:
        phase1()
        phase2()
        phase3()
    finally:
        mon.terminate()
    log(f"[all] ========== ALL EXPERIMENTS FINISHED; failed steps: {failed or 'none'} ==========")


if __name__ == "__main__":
    main()
