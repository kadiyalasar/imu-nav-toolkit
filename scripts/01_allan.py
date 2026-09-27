"""Step 1: Allan deviation of a stationary IMU -> white noise, bias instability, rate random walk."""
import _setup  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from navlib import allan, synthetic
from navlib.io import load_csv, RESULTS

FS = 40.0
d = load_csv("stationary.csv")
signals = {
    "gyro_z  [rad/s]": (d["gyro_z"], synthetic.GYRO_MODEL),
    "accel_x [m/s^2]": (d["accel_x"], synthetic.ACCEL_MODEL),
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, (sig, true)) in zip(axes, signals.items()):
    taus, adev = allan.allan_deviation(sig, FS)
    p = allan.noise_parameters(taus, adev)
    ax.loglog(taus, adev, "k", lw=2, label="Allan deviation")
    ax.loglog(taus, p["white_noise_density"] / np.sqrt(taus), "--", label="white noise line (slope -1/2)")
    ax.loglog(taus, p["rate_random_walk"] * np.sqrt(taus / 3), "--", label="rate random walk line (slope +1/2)")
    ax.axhline(p["bias_instability"] * 0.664, ls=":", c="gray", label="bias instability floor")
    ax.plot(1.0, p["white_noise_density"], "o", c="C0")
    ax.plot(3.0, p["rate_random_walk"], "o", c="C1")
    ax.set_title(name); ax.set_xlabel("averaging time tau [s]"); ax.set_ylabel("Allan deviation")
    ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8)
    ax.set_ylim(adev.min() / 3, adev.max() * 3)

    print(f"\n{name}")
    print(f"  white noise density : {p['white_noise_density']:.3e}   (injected {true.white_density:.3e})")
    print(f"  bias instability    : {p['bias_instability']:.3e}   at tau = {p['tau_bias_instability']:.0f} s")
    print(f"  rate random walk    : {p['rate_random_walk']:.3e}   (injected {true.rrw:.3e})")
    if "gyro" in name:
        print(f"  -> ARW = {np.rad2deg(p['white_noise_density'])*60:.3f} deg/sqrt(hr), "
              f"bias instability = {np.rad2deg(p['bias_instability'])*3600:.1f} deg/hr")
print("\n('injected' values only exist for synthetic data -- that's how we know the code works.)")
plt.tight_layout(); plt.savefig(RESULTS / "01_allan.png", dpi=120)
print("Saved results/01_allan.png")
