"""Simulated plant: DC gearmotor + H-bridge driver + quadrature encoder.

These are the same physics as a Simscape DC-motor model, written out by hand:

  Electrical:  L di/dt = V_motor - R i - Ke * w            (back-EMF = Ke * w)
  Mechanical:  J dw/dt = Kt i - b w - tau_coulomb(w) - tau_load / N
  Position:    dtheta/dt = w          (motor shaft)    output shaft = theta / N

H-bridge (average model): V_motor = duty * V_supply - (R_switches) * i, duty in [-1, 1].
Encoder: integer counts on the motor shaft (quadrature), so position is quantized.

PLACEHOLDER PARAMETERS: typical small 12 V gearmotor values. Replace them with the
values from your own project (datasheet or measured) -- everything else stays the same.
"""
from dataclasses import dataclass, replace
import numpy as np


@dataclass
class MotorParams:
    R: float = 2.0            # winding resistance [ohm]
    L: float = 1.0e-3         # winding inductance [H]
    Ke: float = 0.01          # back-EMF constant [V s/rad]  (= Kt in SI units)
    Kt: float = 0.01          # torque constant [N m/A]
    J_motor: float = 2.0e-6   # rotor inertia [kg m^2]
    J_load: float = 1.0e-3    # load inertia at OUTPUT shaft [kg m^2]
    b: float = 1.0e-6         # viscous friction at motor shaft [N m s/rad]
    tau_coulomb: float = 2e-3  # Coulomb friction at motor shaft [N m]
    N: float = 30.0           # gear ratio (motor turns per output turn)
    gear_eff: float = 0.9     # gearbox efficiency (used for load torque)
    V_supply: float = 12.0    # supply voltage [V]
    R_switch: float = 0.2     # H-bridge on-resistance, two switches in the path [ohm]
    counts_per_motor_rev: int = 48   # 12-line encoder x4 quadrature
    physics_dt: float = 5e-5  # 20 kHz physics (electrical time constant L/R = 0.5 ms)
    control_dt: float = 1e-3  # 1 kHz controller

    @property
    def J_total(self):
        """Load inertia seen by the motor shrinks by N^2 (reflected inertia)."""
        return self.J_motor + self.J_load / self.N**2

    @property
    def counts_per_output_rev(self):
        return self.counts_per_motor_rev * self.N

    def with_changes(self, **kw):
        return replace(self, **kw)


class MotorPlant:
    def __init__(self, p: MotorParams = MotorParams()):
        self.p = p
        self.i = 0.0        # current [A]
        self.w = 0.0        # motor speed [rad/s]
        self.theta = 0.0    # motor angle [rad]
        self.t = 0.0
        self.substeps = int(round(p.control_dt / p.physics_dt))

    # ---- what the controller is allowed to see ----
    def encoder_counts(self) -> int:
        return int(np.floor(self.theta / (2 * np.pi) * self.p.counts_per_motor_rev))

    # ---- ground truth (for scoring only) ----
    @property
    def output_angle(self):
        return self.theta / self.p.N

    @property
    def output_speed(self):
        return self.w / self.p.N

    def step(self, duty: float, load_torque: float = 0.0):
        """Hold the PWM duty for one control period. Returns the duty actually applied."""
        p, h = self.p, self.p.physics_dt
        duty = float(np.clip(duty, -1.0, 1.0))
        tau_load_motor = load_torque / (p.N * p.gear_eff)
        for _ in range(self.substeps):
            v = duty * p.V_supply
            # semi-implicit Euler: update current, then speed with the NEW current,
            # then angle with the NEW speed
            di = (v - (p.R + p.R_switch) * self.i - p.Ke * self.w) / p.L
            self.i += di * h
            friction = p.tau_coulomb * np.tanh(self.w / 0.5)   # smooth sign(): avoids chatter at w = 0
            dw = (p.Kt * self.i - p.b * self.w - friction - tau_load_motor) / p.J_total
            self.w += dw * h
            self.theta += self.w * h
            self.t += h
        return duty

    # ---- first-principles numbers, used to verify the model ----
    def theory(self):
        p, R = self.p, self.p.R + self.p.R_switch
        return {
            "electrical_tau_s": p.L / R,
            "mechanical_tau_s": p.J_total * R / (p.Kt * p.Ke),
            "stall_current_A": p.V_supply / R,
            "no_load_speed_rad_s": None,   # depends on friction; see tests
        }
