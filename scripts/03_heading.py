"""Step 3: heading from magnetometer, gyro, complementary filter -- and the sign-convention bug."""
import _setup  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from navlib import heading, evaluation
from navlib.io import load_csv, RESULTS

FS = evaluation.FS
d = evaluation.build_inputs(load_csv("drive.csv"), load_csv("circles.csv"))
yaw_gyro = heading.gyro_yaw(d["gyro_z"], FS, d["yaw_mag"][0])
yaw_gyro_flipped = heading.gyro_yaw(-d["gyro_z"], FS, d["yaw_mag"][0])   # z-down (FRD) data used as z-up

def rmse_deg(a):
    return np.rad2deg(np.sqrt(np.mean(heading.wrap(a - d["true_yaw"])**2)))

print(f"complementary filter alpha=0.99 at {FS:.0f} Hz -> crossover {heading.crossover_hz(0.99, FS):.3f} Hz")
print(f"heading RMSE  magnetometer only : {rmse_deg(d['yaw_mag']):.2f} deg (noisy, but no drift)")
print(f"heading RMSE  gyro only         : {rmse_deg(yaw_gyro):.2f} deg (smooth, but drifts)")
print(f"heading RMSE  complementary     : {rmse_deg(d['yaw_cf']):.2f} deg (best of both)")
drift = np.rad2deg(heading.wrap(yaw_gyro[-1] - d["true_yaw"][-1]))
print(f"gyro-only heading error at the end of the drive: {drift:.1f} deg")

t = d["t"]
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
un = np.unwrap
axes[0].plot(t, np.rad2deg(un(d["yaw_mag"])), lw=.5, label="magnetometer")
axes[0].plot(t, np.rad2deg(yaw_gyro), label="gyro integrated")
axes[0].plot(t, np.rad2deg(un(d["yaw_cf"])), label="complementary filter")
axes[0].plot(t, np.rad2deg(un(d["true_yaw"])), "k--", lw=1, label="truth")
axes[0].set_ylabel("yaw [deg] (unwrapped)"); axes[0].legend(); axes[0].grid(alpha=.3)
axes[0].set_title("Consistent conventions: all estimates turn the same way")

axes[1].plot(t, np.rad2deg(un(d["yaw_mag"])), lw=.5, label="magnetometer (z-up convention)")
axes[1].plot(t, np.rad2deg(yaw_gyro_flipped), label="gyro with z-DOWN sign (bug)")
axes[1].set_title("The Lab 5 symptom: mirror-image yaw curves = frame-convention mismatch")
axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("yaw [deg]"); axes[1].legend(); axes[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(RESULTS / "03_heading.png", dpi=120)
print("Saved results/03_heading.png")
