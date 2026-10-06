"""Run the SIL scenario suite (position + velocity) against a controller.

    python motor_sil/run_suite.py                   # Python controller
    python motor_sil/run_suite.py --controller cpp  # C++ controller (build it first)

Exit code 0 = all passed, 1 = something failed (that's all CI needs)."""
import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motor_sil.runner import run          # noqa: E402
from motor_sil.scenarios import SUITE     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--controller", default="python", choices=["python", "cpp"])
args = ap.parse_args()

results = [run(s, args.controller) for s in SUITE]
print(f"\nMotor SIL suite -- controller: {args.controller}\n")
print(f"{'scenario':24s} {'result':6s} {'final err':>10s} {'overshoot%':>10s} {'settle s':>9s} "
      f"{'track rms':>9s} {'peak A':>7s} {'%t >0.6A':>8s}")
for r in results:
    m, unit = r["metrics"], ("deg" if r["scenario"].mode == "position" else "RPM")

    def g(k, w, d=2):
        return f"{m[k]:{w}.{d}f}" if k in m else "-".rjust(w)
    print(f"{r['scenario'].name:24s} {'PASS' if r['passed'] else 'FAIL':6s} {g('final_err', 6)} {unit:3s} "
          f"{g('overshoot_pct', 10, 1)} {g('settle_s', 9)} {g('track_rms', 9)} {m['peak_current_A']:7.2f} "
          f"{m['pct_time_over_L293D_cont']:8.0f}")
    for f in r["failures"]:
        print(f"    -> {f}")
rtt = np.concatenate([r["rtt_ms"] for r in results])
print(f"\nround trip sim <-> controller: mean {rtt.mean():.3f} ms (control period 10 ms)")
print("'%t >0.6A' = share of time the current exceeds the L293D's ~0.6 A continuous rating")

out = ROOT / "results"; out.mkdir(exist_ok=True)
fig, ax = plt.subplots(2, 2, figsize=(14, 8))
for r in results:
    s = r["scenario"]
    col = 0 if s.mode == "position" else 1
    scale = 1.0 if s.mode == "position" else 60 / (2 * np.pi)
    line, = ax[0, col].plot(r["t"], r["y"] * scale, lw=1, label=s.name)
    ax[0, col].plot(r["t"], r["ref"] * scale, lw=.7, ls="--", c=line.get_color())
    ax[1, col].plot(r["t"], np.abs(r["current"]), lw=.8)
for col, (title, ylab) in enumerate((("Position control", "output angle [rad]"),
                                     ("Velocity control", "output speed [RPM]"))):
    ax[0, col].set_title(f"{title} (solid = motor, dashed = target)"); ax[0, col].set_ylabel(ylab)
    ax[0, col].legend(fontsize=7); ax[0, col].grid(alpha=.3)
    ax[1, col].axhline(0.6, c="r", ls=":", label="L293D ~0.6 A continuous")
    ax[1, col].axhline(1.2, c="r", ls="--", label="L293D ~1.2 A peak")
    ax[1, col].set_ylabel("|motor current| [A]"); ax[1, col].set_xlabel("time [s]")
    ax[1, col].legend(fontsize=7); ax[1, col].grid(alpha=.3)
plt.tight_layout(); plt.savefig(out / f"motor_sil_{args.controller}.png", dpi=120)
(out / f"motor_sil_metrics_{args.controller}.json").write_text(json.dumps(
    {r["scenario"].name: {**r["metrics"], "passed": r["passed"]} for r in results}, indent=2))
print(f"Saved results/motor_sil_{args.controller}.png")
sys.exit(0 if all(r["passed"] for r in results) else 1)
