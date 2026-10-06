"""Scenario suite based on the ME 5245 project's position and velocity control tests.
Adding a test = adding one line."""
import math
from motor_sil.runner import Scenario

RPM = 2 * math.pi / 60

SUITE = [
    # ---- position control (the potentiometer sets a target angle 0..2*pi) ----
    Scenario("pos_step_pi", targets=[(0.1, math.pi)]),                         # like Part C Fig 11
    Scenario("pos_step_half_pi", targets=[(0.1, math.pi / 2)]),
    Scenario("pos_pot_sequence", duration=12, targets=[(0.1, 1.0), (3, 3.4), (6, 6.0), (9, 4.8)],
             max_settle_s=2.0),                                                # like Part C Fig 12
    Scenario("pos_sine_tracking", duration=12, sine=(math.pi, math.pi / 2, 0.1), max_track_rms=3.0),
    Scenario("pos_weak_battery_6V", params={"V_supply": 6.0}, max_settle_s=3.0),
    Scenario("pos_friction_model", model="friction"),
    Scenario("pos_latency_30ms", delay_ms=30),
    # ---- velocity control (the potentiometer sets a target speed 0..25 RPM, forward only) ----
    Scenario("vel_step_12rpm", mode="velocity", targets=[(0.1, 12 * RPM)], max_settle_s=1.0),  # Fig 24
    Scenario("vel_steps_6_12_9", mode="velocity", duration=6,
             targets=[(0.1, 6 * RPM), (1.0, 12 * RPM), (4.0, 9 * RPM)], max_settle_s=1.0),     # Fig 18
    Scenario("vel_load_disturbance", mode="velocity", targets=[(0.1, 12 * RPM)], load_time=2.5,
             load_torque=0.15, max_settle_s=10),
    Scenario("vel_friction_model", mode="velocity", model="friction", targets=[(0.1, 12 * RPM)],
             max_settle_s=1.0),
]

# Known limitation, reproduced from the hardware: with weak batteries the motor cannot reach a
# high target speed. Kept OUT of the pass/fail suite; the tests mark it as an expected failure.
WEAK_BATTERY_HIGH_SPEED = Scenario("vel_weak_battery_20rpm", mode="velocity", params={"V_supply": 6.0},
                                   targets=[(0.1, 20 * RPM)], max_settle_s=2.0)
