"""Step 4: forward velocity from the accelerometer (3 bias strategies) and dead-reckoned path."""
import _setup  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from navlib import deadreckoning as dr, evaluation
from navlib.io import load_csv, RESULTS

FS = evaluation.FS
d = evaluation.build_inputs(load_csv("drive.csv"), load_csv("circles.csv"))
st = d["stationary"]
print(f"stationary detector flagged {st.mean()*100:.1f}% of samples")

vels = {
    "naive integration": dr.velocity_naive(d["accel_x"], FS),
    "constant bias (best-fit line)": dr.velocity_constant_bias(d["accel_x"], FS),
    "ZUPT (re-estimate bias at every stop)": d["v_zupt"],
}
g = ~np.isnan(d["gps_x"])
x0, y0 = d["gps_x"][g][0], d["gps_y"][g][0]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for name, v in vels.items():
    x, y = dr.integrate_position(v, d["yaw_cf"], FS, x0, y0)
    err = np.hypot(x - d["true_x"], y - d["true_y"])
    growth = err - err[0]                      # error added by dead reckoning itself
    within = np.argmax(growth > 2.0) / FS if (growth > 2.0).any() else d["t"][-1]
    print(f"{name:40s} speed RMSE {np.sqrt(np.mean((v - d['true_v'])**2)):7.2f} m/s | "
          f"final position error {err[-1]:8.0f} m | drift passed 2 m after {within:5.1f} s")
    axes[0].plot(d["t"], v, lw=1, label=name)
    if "naive" not in name:
        axes[1].plot(x, y, lw=1, label=name)
axes[0].plot(d["t"], d["true_v"], "k--", lw=1, label="truth")
axes[0].set_ylim(-5, 40); axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("forward speed [m/s]")
axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
axes[1].plot(d["gps_x"][g], d["gps_y"][g], ".", ms=2, c="gray", label="GPS")
axes[1].plot(d["true_x"], d["true_y"], "k--", lw=1, label="truth")
axes[1].set_aspect("equal"); axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
axes[1].set_title("Dead reckoning: NO scale factor, NO rotation fudge")
plt.tight_layout(); plt.savefig(RESULTS / "04_dead_reckoning.png", dpi=120)
print("Saved results/04_dead_reckoning.png")
print("\nNote: road grade leaks gravity into accel_x (g*sin(pitch)); stop-based bias removal "
      "cannot see grade changes between stops. That is the main reason ZUPT alone still drifts.")
