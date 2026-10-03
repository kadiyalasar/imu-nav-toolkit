"""How much command delay can the position controller tolerate?
The PID was tuned assuming zero delay; this measures the real margin."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motor_sil.runner import run, Scenario  # noqa: E402

controller = sys.argv[1] if len(sys.argv) > 1 else "python"
delays = list(range(0, 13))
over, ok = [], []
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for d in delays:
    r = run(Scenario(f"delay_{d}", delay_ms=d, duration=2.0), controller)
    over.append(r["metrics"]["overshoot_pct"]); ok.append(r["passed"])
    if d in (0, 3, 6, 10):
        ax[0].plot(r["t"], np.rad2deg(r["angle"]), label=f"{d} ms")
    print(f"delay {d:3d} ms: {'PASS' if r['passed'] else 'FAIL'}  overshoot {over[-1]:5.1f}%  "
          f"settle {r['metrics']['settle_s']:.3f} s")
last_ok = max(d for d, p in zip(delays, ok) if p)
print(f"\nLargest delay that still passes: {last_ok} ms  (control period = 1 ms)")
ax[0].axhline(90, c="k", lw=.6); ax[0].set_xlim(0, 1.0)
ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("output angle [deg]"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_title("90 deg step with injected delay")
ax[1].plot(delays, over, "o-"); ax[1].axhline(10, ls=":", c="r", label="10% overshoot limit")
ax[1].axvline(last_ok, ls="--", c="gray", label=f"last passing: {last_ok} ms")
ax[1].set_xlabel("injected delay [ms]"); ax[1].set_ylabel("overshoot [%]"); ax[1].legend(); ax[1].grid(alpha=.3)
ax[1].set_title("Latency tolerance")
plt.tight_layout(); out = ROOT / "results"; out.mkdir(exist_ok=True)
plt.savefig(out / "motor_sil_latency.png", dpi=120)
print("Saved results/motor_sil_latency.png")
