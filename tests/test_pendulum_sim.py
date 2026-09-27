"""Simulation release gate example: physics-engine behavior we rely on must not change.
Marked slow -> runs nightly, not on every push."""
import sys
from pathlib import Path
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from pendulum_mujoco import run  # noqa: E402


@pytest.mark.slow
def test_semi_implicit_euler_energy_error_is_bounded():
    # energy error must not GROW: compare a short and a 10x longer run
    _, short = run(0.01, mujoco.mjtIntegrator.mjINT_EULER, seconds=60)
    _, long = run(0.01, mujoco.mjtIntegrator.mjINT_EULER, seconds=600)
    assert np.abs(long).max() < 1.1 * np.abs(short).max()
    assert np.abs(long).max() < 0.15      # joules


@pytest.mark.slow
def test_rk4_is_far_more_accurate():
    _, e_rk4 = run(0.01, mujoco.mjtIntegrator.mjINT_RK4, seconds=60)
    _, e_eul = run(0.01, mujoco.mjtIntegrator.mjINT_EULER, seconds=60)
    assert np.abs(e_rk4).max() < 1e-2 * np.abs(e_eul).max()
