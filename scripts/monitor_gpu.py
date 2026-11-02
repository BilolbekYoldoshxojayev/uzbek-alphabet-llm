"""Append GPU state to results/logs/gpu_monitor.csv every 10 s until killed.

A laptop GPU can throttle when hot, which would distort training-time comparisons, so temperature,
clock, power and the thermal-throttle flag are recorded for the whole run.
"""
import subprocess
import time
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "results/logs/gpu_monitor.csv"
FIELDS = ("utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.current.graphics,"
          "clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown")
if not LOG.exists():
    LOG.write_text("time," + FIELDS + "\n", encoding="utf-8")
while True:
    out = subprocess.run(["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "," + out.replace(", ", ",") + "\n")
    time.sleep(10)
