"""Simulated plant for the ME 5245 mechatronics project hardware:
Pololu 172:1 metal gearmotor with 48 CPR encoder, L293D H-bridge, 5 x AA batteries, Arduino Uno.

Physics (same as the Simscape model: DC motor + separate gearbox, NOT a transfer function)
  Electrical:  L di/dt = V_motor - R i - kb * w
  Mechanical:  J dw/dt = kt i - b w - tau_load / N            (motor shaft)
  Gearbox:     output angle = motor angle / N,  load inertia reflected by 1/N^2

H-bridge (L293D), driven exactly like the Arduino pins in the project:
  PWM 0..255 on the enable pin (pin 9), direction bits IN1/IN2 (pins 6 and 8)
     IN1=1, IN2=0 -> forward    V = +pwm/255 * V_supply
     IN1=0, IN2=1 -> reverse    V = -pwm/255 * V_supply
     IN1 = IN2    -> brake      motor terminals shorted (V = 0), back-EMF current brakes it
  Average model: the 490 Hz Arduino PWM is much faster than the motor, so we use the mean voltage.

Encoder: 48 counts per MOTOR revolution (both edges, both channels) x 172 = 8256 counts per output rev.
"""
from dataclasses import dataclass, replace
import numpy as np

N_GEAR = 172.0


@dataclass
class MotorParams:
    # ---- values from the project report (Part B) ----
    R: float = 5.0                       # ohm (given)
    kt: float = 0.0039                   # N m/A, from spec-sheet stall torque / stall current
    kb: float = 0.0039                   # V s/rad, assumed equal to kt (SI units, as the project instructs)
    b: float = 1.184e-5                  # N m s/rad, fitted from hardware Trial 1 DC gain
    J_motor: float = 0.00976 / N_GEAR**2  # 3.299e-7 kg m^2 (given J at output, referred to motor)
    N: float = N_GEAR
    V_supply: float = 7.5                # 5 x AA, nominal
    counts_per_motor_rev: int = 48
    # ---- assumptions (NOT in the report) ----
    L: float = 1.0e-3                    # H, armature inductance: not on the spec sheet -> assumed
    J_load: float = 0.0                  # extra load inertia at the output shaft [kg m^2]
    tau_coulomb: float = 0.0             # N m at motor shaft: NOT in the Simscape model (hypothesis variant)
    # ---- simulation settings ----
    physics_dt: float = 5e-5             # 20 kHz (electrical time constant L/R = 0.2 ms)
    control_dt: float = 0.01             # 10 ms: matches the 0.0762 rad/s speed steps in the hardware plots
    # ---- driver limits (L293D datasheet, approximate) ----
    i_continuous: float = 0.6            # A per channel
    i_peak: float = 1.2                  # A per channel

    @property
    def J_total(self):
        return self.J_motor + self.J_load / self.N**2

    @property
    def counts_per_output_rev(self):
        return int(self.counts_per_motor_rev * self.N)

    def with_changes(self, **kw):
        return replace(self, **kw)

    # first-principles predictions (used by tests and the identification script)
    def dc_gain(self):
        """Steady-state motor speed per volt: K = kt / (R b + kt kb)."""
        return self.kt / (self.R * self.b + self.kt * self.kb)

    def mech_time_constant(self):
        """tau = J / (b + kt kb / R)   (inductance neglected)."""
        return self.J_total / (self.b + self.kt * self.kb / self.R)


# The two model variants we compare against the hardware
SPEC_MODEL = MotorParams()                                   # what the Simscape model used
IDENTIFIED_MODEL = MotorParams(J_motor=2.01e-6)              # J fitted to the measured tau = 0.135 s
# Hypothesis: add the Coulomb friction the Simscape model left out (reduce viscous b to keep
# roughly the same speed at Trial 1). Used to show WHY the hardware needed PI instead of P.
FRICTION_MODEL = MotorParams(J_motor=2.01e-6, tau_coulomb=1.0e-3, b=0.9e-5)

# Hardware measurements from the Part B report (open loop, 10 s runs)
HARDWARE_TRIALS = [  # (duty, steady output speed rad/s, time constant s)
    (0.897, 2.05, 0.125),
    (0.690, 1.75, 0.1425),
    (0.5327, 1.37, 0.1375),
]
SIMSCAPE_RESULTS = {0.897: 2.0179, 0.690: 1.56, 0.5327: 1.195}   # from the same report


class MotorPlant:
    def __init__(self, p: MotorParams = SPEC_MODEL, theta0=0.0):
        self.p = p
        self.i = 0.0
        self.w = 0.0             # motor-shaft speed [rad/s]
        self.theta = theta0 * p.N  # motor-shaft angle [rad]
        self.t = 0.0
        self.substeps = int(round(p.control_dt / p.physics_dt))

    # ---- sensor the Arduino can read ----
    def encoder_counts(self) -> int:
        return int(np.floor(self.theta / (2 * np.pi) * self.p.counts_per_motor_rev))

    # ---- ground truth (scoring only) ----
    @property
    def output_angle(self):
        return self.theta / self.p.N

    @property
    def output_speed(self):
        return self.w / self.p.N

    def step(self, pwm: int, in1: int, in2: int, load_torque: float = 0.0):
        """Apply the Arduino's pin outputs for one control period (zero-order hold).
        Returns the signed duty actually applied (+ forward, - reverse, 0 brake)."""
        p, h = self.p, self.p.physics_dt
        duty = min(max(int(pwm), 0), 255) / 255.0      # 8-bit PWM, like analogWrite()
        if in1 == 1 and in2 == 0:
            signed = duty
        elif in1 == 0 and in2 == 1:
            signed = -duty
        else:
            signed = 0.0                                # brake: terminals shorted
        v = signed * p.V_supply
        tau_load = load_torque / p.N
        for _ in range(self.substeps):
            di = (v - p.R * self.i - p.kb * self.w) / p.L
            self.i += di * h                            # semi-implicit Euler:
            friction = p.tau_coulomb * np.tanh(self.w / 0.5)   # smooth sign(): no chatter at w = 0
            dw = (p.kt * self.i - p.b * self.w - friction - tau_load) / p.J_total
            self.w += dw * h                            # current first, then speed,
            self.theta += self.w * h                    # then angle (newest values)
            self.t += h
        return signed
