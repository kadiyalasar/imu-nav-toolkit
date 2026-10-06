"""Closed-loop SIL tests: the controller runs as a separate program (Python or C++), sees only
encoder counts + potentiometer ADC, and drives PWM + direction pins of the simulated L293D."""
import math
import numpy as np
import pytest
from motor_sil.runner import run, Scenario, CPP_EXE
from motor_sil.scenarios import SUITE, WEAK_BATTERY_HIGH_SPEED

CONTROLLERS = ["python"] + (["cpp"] if CPP_EXE.exists() else [])
RPM = 2 * math.pi / 60


@pytest.mark.parametrize("controller", CONTROLLERS)
@pytest.mark.parametrize("scenario", SUITE, ids=[s.name for s in SUITE])
def test_scenario_suite(scenario, controller):
    r = run(scenario, controller)
    assert r["passed"], f"{scenario.name} [{controller}]: {r['failures']}"


@pytest.mark.xfail(strict=True, reason="known hardware limitation: weak batteries can't reach 20 RPM")
def test_weak_battery_cannot_reach_high_speed():
    assert run(WEAK_BATTERY_HIGH_SPEED)["passed"]


def test_lockstep_is_deterministic():
    s = Scenario("det", targets=[(0.1, 2.0)], load_time=1.0, load_torque=0.05)
    a, b = run(s), run(s)
    assert np.array_equal(a["y"], b["y"]) and np.array_equal(a["duty"], b["duty"])


@pytest.mark.skipif(not CPP_EXE.exists(), reason="C++ controller not built")
@pytest.mark.parametrize("mode,target", [("position", 2.0), ("velocity", 12 * RPM)])
def test_back_to_back_python_vs_cpp(mode, target):
    s = Scenario("b2b", mode=mode, targets=[(0.1, target)], duration=3)
    py, cpp = run(s, "python"), run(s, "cpp")
    assert np.array_equal(py["duty"], cpp["duty"])


def test_gate_catches_a_broken_controller():
    """Test the test: flipped gains must FAIL."""
    assert not run(Scenario("broken", targets=[(0.1, 2.0)]), gain_scale=-1.0)["passed"]


def test_velocity_needs_integral_action():
    """P-only speed control leaves a steady error: holding speed needs a nonzero command,
    and a P controller only produces one when there's an error. PI removes it."""
    s = dict(mode="velocity", targets=[(0.1, 12 * RPM)])
    assert run(Scenario("p", gains=(0.6, 0.0), **s))["metrics"]["final_err"] > 2.0
    assert run(Scenario("pi", **s))["metrics"]["final_err"] < 0.1


def test_anti_windup_reduces_overshoot():
    with_aw = run(Scenario("aw", anti_windup=True))["metrics"]["overshoot_pct"]
    without = run(Scenario("naw", anti_windup=False))["metrics"]["overshoot_pct"]
    assert with_aw < without / 2


def test_deadband_stops_direction_chatter():
    def flips(db):
        d = run(Scenario("db", deadband=db, duration=6))["duty"][-300:]
        return int(np.sum(np.abs(np.diff(np.sign(d))) > 0))
    assert flips(0.02) < flips(0.0)


def test_latency_margin_at_least_20ms():
    assert run(Scenario("lat", delay_ms=20, duration=6))["passed"]
