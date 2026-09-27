"""Step 2: hard/soft-iron magnetometer calibration from driving in circles."""
import _setup  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from navlib import magcal, heading
from navlib.io import load_csv, RESULTS

c = load_csv("circles.csv")
fit = magcal.fit_ellipse(c["mag_x"], c["mag_y"])
hx, hy = magcal.calibrate(c["mag_x"], c["mag_y"], fit)

print(f"hard-iron center : ({fit['center'][0]:.4f}, {fit['center'][1]:.4f}) gauss")
print(f"semi-axes        : major {fit['major']:.4f}, minor {fit['minor']:.4f} gauss "
      f"(ratio {fit['minor']/fit['major']:.3f})")
print(f"tilt             : {fit['tilt']:.4f} rad = {np.rad2deg(fit['tilt']):.1f} deg")
print(f"calibrated radius: mean {np.mean(np.hypot(hx, hy)):.3f}, std {np.std(np.hypot(hx, hy)):.3f} (ideal 1, 0)")

# Why "rotate back" matters: rotate + scale WITHOUT rotating back (the Lab 5 recipe)
th = fit["tilt"]; R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
u = np.vstack([c["mag_x"] - fit["center"][0], c["mag_y"] - fit["center"][1]])
no_back = np.diag([1 / fit["major"], 1 / fit["minor"]]) @ R.T @ u
yaw_ok = heading.mag_yaw(hx, hy)
yaw_nb = heading.mag_yaw(no_back[0], no_back[1])
offset = np.rad2deg(np.mean(heading.wrap(yaw_nb - yaw_ok)))
print(f"\nSkipping the rotate-back step shifts every heading by {offset:+.1f} deg "
      f"(= the tilt angle). Compare the pi/8 = 22.5 deg rotation used in Lab 5.")
if "true_yaw" in c:
    err = np.rad2deg(np.sqrt(np.mean(heading.wrap(yaw_ok - c["true_yaw"])**2)))
    print(f"heading RMSE after calibration: {err:.2f} deg")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(c["mag_x"], c["mag_y"], ".", ms=1, label="raw (ellipse, off-center)")
scale = fit["minor"]                               # draw the circle at a comparable size
ax.plot(hx * scale, hy * scale, ".", ms=1, label="calibrated (scaled for display)")
ax.plot(*fit["center"], "rx", ms=10, label="hard-iron center")
ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(); ax.set_xlabel("mag x [gauss]"); ax.set_ylabel("mag y [gauss]")
plt.tight_layout(); plt.savefig(RESULTS / "02_magcal.png", dpi=120)
print("Saved results/02_magcal.png")
