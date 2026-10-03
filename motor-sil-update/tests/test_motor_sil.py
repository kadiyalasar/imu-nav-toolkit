"""Closed-loop software-in-the-loop tests: the controller runs as a separate program
(Python or C++) and controls the simulated motor over a socket."""
import numpy as np
import pytest
from motor_sil.runner import run, Scenario, CPP_EXE
from motor_sil.scenarios import SUITE

CONTROLLERS = ["python"] + (["cpp"] if CPP_EXE.exists() else [])


@pytest.mark.parametrize("controller", CONTROLLERS)
@pytest.mark.parametrize("scenario", SUITE, ids=[s.name for s in SUITE])
def test_scenario_suite(scenario, controller):
    r = run(scenario, controller)
    assert r["passed"], f"{scenario.name} [{controller}]: {r['failures']}"


def test_lockstep_is_deterministic():
    s = Scenario("det", load_time=0.8, load_torque=0.1)
    a, b = run(s), run(s)
    assert np.array_equal(a["angle"], b["angle"]) and np.array_equal(a["duty"], b["duty"])


@pytest.mark.skipif(not CPP_EXE.exists(), reason="C++ controller not built")
def test_back_to_back_python_vs_cpp():
    """Reference (Python) and 'production' (C++) controllers must behave identically."""
    s = Scenario("b2b", sine_amp_deg=30, duration=1.5)
    py, cpp = run(s, "python"), run(s, "cpp")
    assert np.max(np.abs(py["angle"] - cpp["angle"])) < 1e-9


def test_gate_catches_a_broken_controller():
    """Test the test: flipped gains must FAIL, otherwise the gate is worthless."""
    assert not run(Scenario("broken"), gain_scale=-1.0)["passed"]


def test_latency_margin_at_least_4ms():
    """Regression guard: controller changes must not shrink delay tolerance below 4 ms."""
    assert run(Scenario("lat", delay_ms=4, duration=2.0))["passed"]
