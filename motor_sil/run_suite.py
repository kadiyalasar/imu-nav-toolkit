"""Run the motor SIL scenario suite against a controller.

    python motor_sil/run_suite.py                   # Python controller
    python motor_sil/run_suite.py --controller cpp  # C++ controller (build it first)

Exit code 0 = everything passed, 1 = something failed. That exit code is all CI needs.
"""
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
print(f"{'scenario':22s} {'result':6s} {'overshoot%':>10s} {'settle s':>9s} {'ss err deg':>10s} "
      f"{'track rms':>9s} {'peak A':>7s} {'sat %':>6s}")
for r in results:
    m = r["metrics"]
    def g(k, fmt):
        return format(m[k], fmt) if k in m else "-".rjust(int(fmt.split(".")[0]))
    print(f"{r['scenario'].name:22s} {'PASS' if r['passed'] else 'FAIL':6s} {g('overshoot_pct', '10.1f')} "
          f"{g('settle_s', '9.3f')} {g('ss_error_deg', '10.3f')} {g('track_rms_deg', '9.2f')} "
          f"{m['max_current_A']:7.2f} {m['saturated_pct']:6.1f}")
    for f in r["failures"]:
        print(f"    -> {f}")
rtt = np.concatenate([r["rtt_ms"] for r in results])
print(f"\nround-trip time sim<->controller: mean {rtt.mean():.3f} ms, 99th pct {np.percentile(rtt, 99):.3f} ms")

out = ROOT / "results"; out.mkdir(exist_ok=True)
fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
for r in results:
    ax[0].plot(r["t"], np.rad2deg(r["angle"]), label=r["scenario"].name)
    ax[1].plot(r["t"], r["duty"], lw=.8)
    ax[2].plot(r["t"], r["current"], lw=.8)
ax[0].set_ylabel("output angle [deg]"); ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=.3)
ax[1].set_ylabel("PWM duty"); ax[1].grid(alpha=.3)
ax[2].set_ylabel("motor current [A]"); ax[2].set_xlabel("time [s]"); ax[2].grid(alpha=.3)
ax[0].set_title(f"Motor SIL scenario suite ({args.controller} controller)")
plt.tight_layout(); plt.savefig(out / f"motor_sil_{args.controller}.png", dpi=120)
(out / f"motor_sil_metrics_{args.controller}.json").write_text(json.dumps(
    {r["scenario"].name: {**r["metrics"], "passed": r["passed"]} for r in results}, indent=2))
print(f"Saved results/motor_sil_{args.controller}.png")
sys.exit(0 if all(r["passed"] for r in results) else 1)
