"""Live GPU dashboard for the terminal. Only reads the logs, so it cannot affect a running experiment.

    .venv/Scripts/python.exe scripts/watch_gpu.py
Press Ctrl+C to close it. --once prints one screen and exits.
"""
import argparse
import datetime as dt
import os
import time
from pathlib import Path

LOGS = Path(__file__).resolve().parents[1] / "results/logs"
GREEN, YELLOW, RED, DIM, BOLD, END = "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[1m", "\033[0m"
MEM_TOTAL_MB, POWER_MAX_W = 12282, 125
WINDOW = 360  # readings for the "last hour" line, one every 10 s


def tail(path: Path, n: int, chunk=1 << 16):
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - chunk))
        return f.read().decode("utf-8", "replace").splitlines()[-n:]


def readings():
    rows = []
    for line in tail(LOGS / "gpu_monitor.csv", WINDOW):
        p = line.split(",")
        if len(p) != 8 or p[0] == "time":  # a line can be cut short by a restart
            continue
        try:
            rows.append({"time": p[0], "load": float(p[1]), "mem": float(p[2]), "temp": float(p[3]), "power": float(p[4]),
                         "clock": float(p[5]), "throttle": "Active" in (p[6], p[7])})
        except ValueError:
            continue
    return rows


def bar(value, maximum, color, width=32):
    n = round(min(value / maximum, 1) * width)
    return f"{color}{'█' * n}{END}{DIM}{'░' * (width - n)}{END}"


def level(value, warn, bad):
    return RED if value >= bad else YELLOW if value >= warn else GREEN


def screen():
    rows = readings()
    if not rows:
        return "No GPU readings yet. The monitor starts with the experiment run."
    r = rows[-1]
    age = (dt.datetime.now() - dt.datetime.strptime(r["time"], "%Y-%m-%d %H:%M:%S")).total_seconds()
    temp = level(r["temp"], 80, 87)
    out = [f"{BOLD}GPU MONITOR{END}  {DIM}{r['time']}, Ctrl+C closes this window only{END}", "",
           f"  Temperature {bar(r['temp'], 100, temp)} {temp}{BOLD}{r['temp']:4.0f} °C{END}",
           f"  Load        {bar(r['load'], 100, GREEN)} {r['load']:4.0f} %",
           f"  Memory      {bar(r['mem'], MEM_TOTAL_MB, level(r['mem'], 10500, 11800))} {r['mem'] / 1024:4.1f} / {MEM_TOTAL_MB / 1024:.1f} GB",
           f"  Power       {bar(r['power'], POWER_MAX_W, GREEN)} {r['power']:4.0f} W",
           f"  Clock       {r['clock']:.0f} MHz", ""]
    out.append(f"  Heat slowdown: {RED}{BOLD}ACTIVE{END}" if r["throttle"] else f"  Heat slowdown: {GREEN}none{END}")
    n = sum(x["throttle"] for x in rows)
    out.append(f"  Latest {len(rows)} readings: peak {max(x['temp'] for x in rows):.0f} °C, slowdown in {n}")
    if age > 60:
        out += ["", f"  {RED}{BOLD}No new reading for {age / 60:.0f} min: the run may have stopped.{END}"]
    out += ["", f"{BOLD}Latest progress{END}"] + [f"  {DIM}{x[:100]}{END}" for x in tail(LOGS / "progress.log", 4)]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    os.system("")  # turns on terminal colors on Windows
    if args.once:
        print(screen())
        return
    try:
        while True:
            print("\033[2J\033[H" + screen(), flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
