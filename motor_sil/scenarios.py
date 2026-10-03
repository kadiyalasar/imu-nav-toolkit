"""Scenario suite: situations the controller must handle, each with pass/fail rules.
Adding a test = adding one line."""
from motor_sil.runner import Scenario

SUITE = [
    Scenario("step_90deg"),
    Scenario("step_360deg", target_deg=360, max_settle_s=0.5),
    Scenario("sine_tracking_1hz", sine_amp_deg=45, duration=2.5, max_track_rms_deg=3.5),
    Scenario("load_disturbance", duration=2.0, load_time=1.0, load_torque=0.15),
    # model uncertainty: the real robot won't match the nominal model exactly
    Scenario("low_battery_9V", params={"V_supply": 9.0}, max_settle_s=0.5),
    Scenario("heavier_load_+50pct", params={"J_load": 1.5e-3}, max_settle_s=0.5),
    Scenario("higher_friction_x3", params={"tau_coulomb": 6e-3}),
    Scenario("latency_3ms", delay_ms=3),
]
