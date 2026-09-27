"""Metrics shared by the scripts AND the CI tests (one code path, not two)."""
import numpy as np
from . import synthetic, magcal, heading, deadreckoning as dr, ekf

FS = 40.0


def build_inputs(drive=None, circles=None, seed=2, legs=13):
    """Drive log + calibration circles -> everything the filters need.
    Pass your own dicts (e.g. loaded from real CSVs) or leave None to generate synthetic ones."""
    d = dict(drive) if drive is not None else synthetic.drive(fs=FS, seed=seed, legs=legs)
    circ = circles if circles is not None else synthetic.calibration_circles(fs=FS, seed=seed + 100)
    fit = magcal.fit_ellipse(circ["mag_x"], circ["mag_y"])
    hx, hy = magcal.calibrate(d["mag_x"], d["mag_y"], fit)
    d["yaw_mag"] = heading.mag_yaw(hx, hy)
    d["yaw_cf"] = heading.complementary(d["gyro_z"], d["yaw_mag"], FS)
    d["stationary"] = dr.detect_stationary(d["accel_x"], d["accel_y"], d["accel_z"],
                                           d["gyro_z"], FS)
    d["v_zupt"] = dr.velocity_zupt(d["accel_x"], FS, d["stationary"])
    d["fit"] = fit
    return d


def pick_outages(d, duration_s=30.0, count=6, rng_seed=0):
    """Choose GPS-outage windows that start while the car is moving."""
    rng = np.random.default_rng(rng_seed)
    n = len(d["t"]); L = int(duration_s * FS)
    moving_starts = np.flatnonzero(d["true_v"][: n - L] > 3.0)
    starts = np.sort(rng.choice(moving_starts, size=count, replace=False))
    # keep windows from overlapping
    kept = []
    for s in starts:
        if not kept or s - kept[-1] > 2 * L:
            kept.append(s)
    return [(s, s + L) for s in kept]


def outage_study(d, duration_s=30.0, count=6):
    """GPS is removed for each window. Compare position error at the END of each outage:
         baseline = Lab 5 style dead reckoning (ZUPT speed + complementary heading),
                    re-anchored at the last GPS fix before the outage
         ekf      = the EKF, which keeps predicting with its estimated biases
    """
    windows = pick_outages(d, duration_s, count)
    avail = np.ones(len(d["t"]), dtype=bool)
    for a, b in windows:
        avail[a:b] = False
    X, _ = ekf.run_ekf(d["t"], d["accel_x"], d["gyro_z"], d["yaw_mag"], d["gps_x"], d["gps_y"],
                       d["stationary"], FS, gps_available=avail)
    base_err, ekf_err = [], []
    for a, b in windows:
        fixes = np.flatnonzero(~np.isnan(d["gps_x"][:a]))
        k0 = fixes[-1]
        x, y = dr.integrate_position(d["v_zupt"][k0:b], d["yaw_cf"][k0:b], FS,
                                     d["gps_x"][k0], d["gps_y"][k0])
        base_err.append(np.hypot(x[-1] - d["true_x"][b - 1], y[-1] - d["true_y"][b - 1]))
        ekf_err.append(np.hypot(X[b - 1, 0] - d["true_x"][b - 1], X[b - 1, 1] - d["true_y"][b - 1]))
    base_err, ekf_err = np.array(base_err), np.array(ekf_err)
    return {
        "windows": windows, "X": X,
        "baseline_mean_err_m": float(base_err.mean()),
        "ekf_mean_err_m": float(ekf_err.mean()),
        "reduction_pct": float(100 * (1 - ekf_err.mean() / base_err.mean())),
        "baseline_err": base_err, "ekf_err": ekf_err,
    }
