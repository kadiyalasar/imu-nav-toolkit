"""Regression gate for the navigation filter.

Three kinds of checks, which is the general recipe for a simulation release gate:
  1. determinism  -- same seed, same answer, bit for bit
  2. absolute     -- physics-level sanity (fused estimate must beat raw GPS)
  3. regression   -- metrics must not get worse than the stored baseline (+ tolerance)
"""
import json
from pathlib import Path
import numpy as np
import pytest
from navlib import evaluation, ekf

BASELINE = Path(__file__).with_name("baseline_metrics.json")
TOLERANCE = 1.10          # allow 10% worse before failing


@pytest.fixture(scope="module")
def run():
    d = evaluation.build_inputs(seed=2)
    X, _ = ekf.run_ekf(d["t"], d["accel_x"], d["gyro_z"], d["yaw_mag"], d["gps_x"], d["gps_y"],
                       d["stationary"], evaluation.FS)
    return d, X


def metrics(d, X):
    g = ~np.isnan(d["gps_x"])
    out = evaluation.outage_study(d)
    return {
        "ekf_pos_rmse_m": float(np.sqrt(np.mean((X[:, 0] - d["true_x"])**2 + (X[:, 1] - d["true_y"])**2))),
        "gps_pos_rmse_m": float(np.sqrt(np.mean((d["gps_x"][g] - d["true_x"][g])**2
                                                + (d["gps_y"][g] - d["true_y"][g])**2))),
        "outage_ekf_mean_err_m": out["ekf_mean_err_m"],
        "outage_dr_mean_err_m": out["baseline_mean_err_m"],
    }


def test_deterministic(run):
    d, X = run
    X2, _ = ekf.run_ekf(d["t"], d["accel_x"], d["gyro_z"], d["yaw_mag"], d["gps_x"], d["gps_y"],
                        d["stationary"], evaluation.FS)
    assert np.array_equal(X, X2)


def test_fusion_beats_raw_gps(run):
    m = metrics(*run)
    assert m["ekf_pos_rmse_m"] < m["gps_pos_rmse_m"]


def test_gyro_bias_is_learned(run):
    d, X = run
    assert abs(X[-1, 5] - d["true_gyro_bias"][-1]) < 2e-4


def test_ekf_beats_dead_reckoning_in_outages(run):
    m = metrics(*run)
    assert m["outage_ekf_mean_err_m"] < m["outage_dr_mean_err_m"]


def test_no_regression_vs_baseline(run):
    baseline = json.loads(BASELINE.read_text())
    if not baseline:
        pytest.skip("no baseline yet -- run  python scripts/update_baseline.py")
    m = metrics(*run)
    for key in ("ekf_pos_rmse_m", "outage_ekf_mean_err_m"):
        assert m[key] <= baseline[key] * TOLERANCE, \
            f"{key} regressed: {m[key]:.3f} vs baseline {baseline[key]:.3f}"
