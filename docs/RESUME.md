# Run status

The full run finished on 2026-11-02 at 06:12 with no failed steps. It was stopped once by request and killed once by a PC restart (13:37:54 on 2026-11-01, probably a G-Helper GPU mode switch, cause not confirmed); it resumed both times, and partial files were redone from row 0.

Machine: G-Helper Turbo mode with fans at 100%, set during Phase 1 before any training.

To re-run the analysis, which needs no GPU:

```powershell
.venv\Scripts\python.exe -X utf8 scripts\analyze_all.py
```

To re-run an experiment, delete its result and start the runner; finished steps are skipped:

```powershell
Start-Process -WindowStyle Hidden .venv/Scripts/python.exe "-X utf8 scripts/run_all_experiments.py"
```

What is left is in `paper/NEXT_STEPS.md`.
