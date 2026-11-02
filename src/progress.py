"""One shared, human-readable progress log for every long-running script.

Watch it live (PowerShell):  Get-Content results\\logs\\progress.log -Wait -Tail 30
"""
import time
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "results/logs/progress.log"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
