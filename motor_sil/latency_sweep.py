"""How much command delay can the position controller tolerate? (control period = 10 ms)"""
import sys
from pathlib import Path
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motor_sil.runner import run, Scenario  # noqa: E402

controller = sys.argv[1] if len(sys.argv) > 1 else "python"
delays = list(range(0, 201, 20))
over, ok = [], []
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for d in delays:
    r = run(Scenario(f"delay_{d}", targets=[(0.1, math.pi)], delay_ms=d, duration=6), controller)
    over.append(r["metrics"]["overshoot_pct"]); ok.append(r["passed"])
    if d in (0, 60, 120, 200):
        ax[0].plot(r["t"], r["y"], label=f"{d} ms")
    print(f"delay {d:4d} ms: {'PASS' if r['passed'] else 'FAIL'}  overshoot {over[-1]:5.1f}%  "
          f"settle {r['metrics']['settle_s']:.2f} s  final err {r['metrics']['final_err']:.2f} deg")
last_ok = max(d for d, p in zip(delays, ok) if p)
print(f"\nLargest delay that still passes: {last_ok} ms (= {last_ok // 10} control periods)")
ax[0].axhline(math.pi, c="k", lw=.6)
ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("output angle [rad]"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_title("Step to pi rad with injected delay")
ax[1].plot(delays, over, "o-"); ax[1].axhline(10, ls=":", c="r", label="10% limit")
ax[1].axvline(last_ok, ls="--", c="gray", label=f"last passing: {last_ok} ms")
ax[1].set_xlabel("injected delay [ms]"); ax[1].set_ylabel("overshoot [%]"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); out = ROOT / "results"; out.mkdir(exist_ok=True)
plt.savefig(out / "motor_sil_latency.png", dpi=120)
print("Saved results/motor_sil_latency.png")
