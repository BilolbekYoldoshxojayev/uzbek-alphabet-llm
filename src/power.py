"""GPU energy measurement shared by the training scripts."""
import subprocess
import threading
import time


class PowerMeter:
    """Samples GPU power with nvidia-smi every 2 s; energy = mean power × time."""

    def __init__(self):
        self.samples, self._stop = [], threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout
                self.samples.append(float(out.strip().splitlines()[0]))
            except Exception:
                pass
            self._stop.wait(2)

    def __enter__(self):
        self.t0 = time.time()
        self.t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.t.join()
        self.seconds = time.time() - self.t0
        self.mean_w = sum(self.samples) / max(1, len(self.samples))
        self.wh = self.mean_w * self.seconds / 3600
