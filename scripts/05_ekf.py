"""Step 5: GPS/IMU/magnetometer EKF, plus a GPS-outage study against Lab 5-style dead reckoning."""
import _setup  # noqa: F401
import json
import numpy as np
import matplotlib.pyplot as plt
from navlib import ekf, heading, evaluation
from navlib.io import load_csv, RESULTS

FS = evaluation.FS
d = evaluation.build_inputs(load_csv("drive.csv"), load_csv("circles.csv"))

X, Pd = ekf.run_ekf(d["t"], d["accel_x"], d["gyro_z"], d["yaw_mag"], d["gps_x"], d["gps_y"],
                    d["stationary"], FS)
g = ~np.isnan(d["gps_x"])
pos_rmse = np.sqrt(np.mean((X[:, 0] - d["true_x"])**2 + (X[:, 1] - d["true_y"])**2))
gps_rmse = np.sqrt(np.mean((d["gps_x"][g] - d["true_x"][g])**2 + (d["gps_y"][g] - d["true_y"][g])**2))
yaw_rmse = np.rad2deg(np.sqrt(np.mean(heading.wrap(X[:, 3] - d["true_yaw"])**2)))
print(f"position RMSE: raw GPS {gps_rmse:.2f} m  ->  EKF {pos_rmse:.2f} m")
print(f"yaw RMSE: EKF {yaw_rmse:.2f} deg")
print(f"gyro bias: estimated {X[-1,5]:.2e}, true {d['true_gyro_bias'][-1]:.2e} rad/s")

r = evaluation.outage_study(d)
print("\n30-second GPS outages (position error at end of each outage, meters):")
for (a, b), eb, ee in zip(r["windows"], r["baseline_err"], r["ekf_err"]):
    print(f"  t = {a/FS:6.0f}-{b/FS:4.0f} s   dead reckoning {eb:7.1f}   EKF {ee:6.1f}")
print(f"mean: dead reckoning {r['baseline_mean_err_m']:.1f} m, EKF {r['ekf_mean_err_m']:.1f} m "
      f"-> {r['reduction_pct']:.0f}% lower (SYNTHETIC DATA)")

metrics = {"ekf_pos_rmse_m": pos_rmse, "gps_pos_rmse_m": gps_rmse, "ekf_yaw_rmse_deg": yaw_rmse,
           "outage_dr_mean_err_m": r["baseline_mean_err_m"],
           "outage_ekf_mean_err_m": r["ekf_mean_err_m"], "outage_reduction_pct": r["reduction_pct"],
           "data": "synthetic"}
(RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
Xo = r["X"]
axes[0].plot(d["true_x"], d["true_y"], "k--", lw=1, label="truth")
axes[0].plot(Xo[:, 0], Xo[:, 1], lw=1, label="EKF (with outages)")
for i, (a, b) in enumerate(r["windows"]):
    axes[0].plot(Xo[a:b, 0], Xo[a:b, 1], "r", lw=2.5, label="GPS outage" if i == 0 else None)
axes[0].set_aspect("equal"); axes[0].legend(); axes[0].grid(alpha=.3)
axes[1].plot(d["t"], d["true_gyro_bias"], "k--", label="true gyro bias")
axes[1].plot(d["t"], X[:, 5], label="EKF estimate")
sd = np.sqrt(Pd[:, 5])
axes[1].fill_between(d["t"], X[:, 5] - 2 * sd, X[:, 5] + 2 * sd, alpha=.2, label="+/- 2 sigma")
axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("rad/s"); axes[1].legend(); axes[1].grid(alpha=.3)
axes[1].set_title("The EKF learns the gyro bias (a complementary filter can't)")
plt.tight_layout(); plt.savefig(RESULTS / "05_ekf.png", dpi=120)
print("Saved results/05_ekf.png and results/metrics.json")
