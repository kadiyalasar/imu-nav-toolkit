"""Verify the simulated hardware against first principles AND against the project's own
Simscape and hardware results, before trusting it to test controllers."""
import numpy as np
import pytest
from motor_sil.motor_model import (MotorPlant, MotorParams, SPEC_MODEL, IDENTIFIED_MODEL,
                                   HARDWARE_TRIALS, SIMSCAPE_RESULTS)


def open_loop(p, pwm, seconds):
    plant = MotorPlant(p)
    n = int(round(seconds / p.control_dt))
    t, w, i = np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(n):
        plant.step(pwm, 1, 0)
        t[k], w[k], i[k] = plant.t, plant.w, plant.i
    return t, w, i, plant


def test_dc_gain_matches_report_value():
    """Report: K = kt / (R b + kt^2) = 52.41 rad/s per volt (motor side)."""
    assert SPEC_MODEL.dc_gain() == pytest.approx(52.41, rel=1e-3)


def test_steady_speed_matches_dc_gain():
    p = SPEC_MODEL
    _, w, _, _ = open_loop(p, 255, 1.0)
    assert w[-1] == pytest.approx(p.dc_gain() * p.V_supply, rel=0.01)


@pytest.mark.parametrize("duty,_hw,_tau", HARDWARE_TRIALS)
def test_reproduces_simscape_steady_speeds(duty, _hw, _tau):
    """Cross-tool check: this Python plant agrees with the project's Simscape model."""
    _, w, _, _ = open_loop(SPEC_MODEL, round(duty * 255), 1.0)
    assert w[-1] / SPEC_MODEL.N == pytest.approx(SIMSCAPE_RESULTS[duty], rel=0.03)


def test_mechanical_time_constant_matches_theory():
    p = SPEC_MODEL.with_changes(control_dt=1e-4, physics_dt=5e-6)
    t, w, _, _ = open_loop(p, 255, 0.2)
    t63 = t[np.argmax(w >= 0.632 * w[-1])]
    assert t63 == pytest.approx(p.mech_time_constant(), rel=0.15)


def test_identified_inertia_matches_hardware_time_constant():
    p = IDENTIFIED_MODEL.with_changes(control_dt=1e-3)
    t, w, _, _ = open_loop(p, round(0.897 * 255), 1.5)
    t63 = t[np.argmax(w >= 0.632 * w[-1])]
    tau_hw = np.mean([tau for *_, tau in HARDWARE_TRIALS])        # 0.135 s
    assert t63 == pytest.approx(tau_hw, rel=0.15)


def test_encoder_has_8256_counts_per_output_rev():
    p = SPEC_MODEL
    plant = MotorPlant(p)
    plant.theta = 0.01                                  # start between edges (avoid rounding traps)
    c0 = plant.encoder_counts()
    plant.theta += 2 * np.pi * p.N
    assert plant.encoder_counts() - c0 == 8256


def test_speed_resolution_at_10ms_matches_hardware_plots():
    """One count per 10 ms sample = 0.0761 rad/s: the step size seen in the Part B plots."""
    p = SPEC_MODEL
    assert (2 * np.pi / p.counts_per_output_rev) / p.control_dt == pytest.approx(0.0762, abs=2e-4)


def test_pwm_is_8_bit():
    assert MotorPlant(SPEC_MODEL).step(128, 1, 0) == pytest.approx(128 / 255)


def test_l293d_direction_logic():
    plant = MotorPlant(SPEC_MODEL)
    assert plant.step(200, 1, 0) > 0          # forward
    assert plant.step(200, 0, 1) < 0          # reverse
    assert plant.step(200, 0, 0) == 0         # brake
    assert plant.step(200, 1, 1) == 0         # brake


def test_brake_slows_the_motor_with_reverse_current():
    plant = MotorPlant(SPEC_MODEL)
    for _ in range(100):
        plant.step(255, 1, 0)
    w0 = plant.w
    plant.step(255, 0, 0)                     # brake: back-EMF drives current backwards
    assert plant.w < w0 and plant.i < 0


def test_result_converges_as_timestep_shrinks():
    _, w1, _, _ = open_loop(SPEC_MODEL.with_changes(physics_dt=5e-5), 128, 0.3)
    _, w2, _, _ = open_loop(SPEC_MODEL.with_changes(physics_dt=2.5e-5), 128, 0.3)
    assert np.max(np.abs(w1 - w2)) / np.max(np.abs(w2)) < 0.01


def test_model_implies_current_above_l293d_rating():
    """A finding, kept as a test: the report's parameters imply ~1 A at Trial 1 speed, above the
    L293D's ~0.6 A continuous rating -> the kt / friction split is probably off."""
    _, _, i, _ = open_loop(SPEC_MODEL, round(0.897 * 255), 1.0)
    assert i[-1] > SPEC_MODEL.i_continuous
