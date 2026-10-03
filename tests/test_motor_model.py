"""Verify the motor SIMULATOR itself against first principles, before trusting it to test
controllers. If the plant model is wrong, every SIL result built on it is wrong too."""
import numpy as np
from motor_sil.motor_model import MotorPlant, MotorParams


def run_open_loop(p, duty, seconds, load=0.0):
    plant = MotorPlant(p)
    n = int(round(seconds / p.control_dt))
    t, i, w = np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(n):
        plant.step(duty, load)
        t[k], i[k], w[k] = plant.t, plant.i, plant.w
    return t, i, w, plant


def test_no_load_speed_matches_theory():
    """Steady state: Kt*i = b*w + tau_c, with i = (V - Ke*w)/R  ->  solve for w."""
    p = MotorParams()
    R = p.R + p.R_switch
    w_theory = (p.Kt * p.V_supply / R - p.tau_coulomb) / (p.b + p.Kt * p.Ke / R)
    _, _, w, _ = run_open_loop(p, 1.0, 1.0)
    assert abs(w[-1] / w_theory - 1) < 0.01


def test_electrical_time_constant_is_L_over_R():
    """Lock the rotor (huge inertia): current rises to 63% of V/R after L/R seconds."""
    p = MotorParams(J_motor=1e3, physics_dt=1e-6, control_dt=1e-5)
    R = p.R + p.R_switch
    tau = p.L / R
    t, i, _, _ = run_open_loop(p, 1.0, 5 * tau)
    t63 = t[np.argmax(i >= 0.632 * p.V_supply / R)]
    assert abs(t63 / tau - 1) < 0.05


def test_stall_current_is_V_over_R():
    p = MotorParams(J_motor=1e3)
    _, i, _, _ = run_open_loop(p, 1.0, 0.02)
    assert abs(i[-1] / (p.V_supply / (p.R + p.R_switch)) - 1) < 0.01


def test_reversing_at_speed_draws_more_than_stall_current():
    """'Plugging': at full speed, back-EMF ADDS to the reversed supply -> current spike."""
    p = MotorParams()
    plant = MotorPlant(p)
    for _ in range(500):
        plant.step(1.0)                       # spin up forward
    peak = 0.0
    for _ in range(20):
        plant.step(-1.0)                      # slam into reverse
        peak = max(peak, abs(plant.i))
    assert peak > 1.5 * p.V_supply / (p.R + p.R_switch)


def test_encoder_counts_one_output_revolution():
    p = MotorParams()
    plant = MotorPlant(p)
    plant.theta = 0.01                        # start between two encoder edges
    c0 = plant.encoder_counts()
    plant.theta += 2 * np.pi * p.N            # one output-shaft revolution
    # (starting exactly ON an edge is a trap: floating-point rounding can land you at 1439.9999)
    assert plant.encoder_counts() - c0 == p.counts_per_output_rev


def test_result_converges_as_timestep_shrinks():
    """Numerical sanity: halving the physics step barely changes the answer."""
    _, _, w1, _ = run_open_loop(MotorParams(physics_dt=5e-5), 0.5, 0.3)
    _, _, w2, _ = run_open_loop(MotorParams(physics_dt=2.5e-5), 0.5, 0.3)
    assert np.max(np.abs(w1 - w2)) / np.max(np.abs(w2)) < 0.01
