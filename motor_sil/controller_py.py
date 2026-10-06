"""Arduino-style controller as a SEPARATE PROGRAM (the "software" in SIL).

It sees exactly what the Arduino sees, and outputs exactly what the Arduino outputs:
  inputs : encoder count (pins 2,3)  +  potentiometer ADC value 0..1023 (pin A0)
  outputs: PWM 0..255 (pin 9)        +  direction bits IN1, IN2 (pins 6, 8)

Protocol, one exchange per 10 ms control step:
  controller -> sim : HELLO <name>
  sim -> controller : STATE <step> <time> <encoder_counts> <pot_adc>
  controller -> sim : CMD <step> <pwm> <in1> <in2>        (must echo <step>)
  sim -> controller : END

Control structure (same as the project's Simulink hardware models): sign-magnitude drive
  u = PI(error);  PWM = |u| * 255 (clipped);  direction from the sign of u
  position mode : u > +deadband -> forward [1 0],  u < -deadband -> reverse [0 1],  else brake [0 0]
  velocity mode : u > 0 -> forward [1 0],  else brake [0 0]   (never reverses, as in the project)
"""
import argparse
import math
import socket

POT_FULL_SCALE = {"position": 2 * math.pi,          # pot range -> 0..2*pi rad target
                  "velocity": 25.0 * 2 * math.pi / 60}  # pot range -> 0..25 RPM target (rad/s)


class ArduinoController:
    def __init__(self, mode, kp, ki, dt, counts_per_rev, deadband=0.0, anti_windup=False):
        self.mode, self.kp, self.ki, self.dt = mode, kp, ki, dt
        self.rad_per_count = 2 * math.pi / counts_per_rev
        self.deadband, self.anti_windup = deadband, anti_windup
        self.integral = 0.0
        self.prev_counts = None

    def reference(self, adc):
        return adc / 1023.0 * POT_FULL_SCALE[self.mode]

    def update(self, counts, adc):
        ref = self.reference(adc)
        if self.mode == "position":
            measured = counts * self.rad_per_count
        else:   # speed = change in counts over one sample (no filtering, like the Simulink model)
            prev = counts if self.prev_counts is None else self.prev_counts
            measured = (counts - prev) * self.rad_per_count / self.dt
        self.prev_counts = counts
        error = ref - measured
        u = self.kp * error + self.ki * self.integral
        saturated = abs(u) >= 1.0
        if not (self.anti_windup and saturated):
            self.integral += error * self.dt
        pwm = min(255, int(round(abs(u) * 255)))
        if self.mode == "position":
            if u > self.deadband:
                return pwm, 1, 0
            if u < -self.deadband:
                return pwm, 0, 1
            return 0, 0, 0
        return (pwm, 1, 0) if u > 0 else (pwm, 0, 0)


DEFAULT_GAINS = {"position": (8.0, 1.0), "velocity": (0.6, 10.0)}   # tuned in this SIL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--mode", choices=["position", "velocity"], default="position")
    ap.add_argument("--gains", type=float, nargs=2, default=None)      # kp ki
    ap.add_argument("--counts-per-rev", type=float, default=8256.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--deadband", type=float, default=0.0)
    ap.add_argument("--anti-windup", type=int, default=1)
    ap.add_argument("--gain-scale", type=float, default=1.0)
    a = ap.parse_args()
    kp, ki = (g * a.gain_scale for g in (a.gains or DEFAULT_GAINS[a.mode]))
    ctrl = ArduinoController(a.mode, kp, ki, a.dt, a.counts_per_rev, a.deadband, bool(a.anti_windup))

    sock = socket.create_connection(("127.0.0.1", a.port), timeout=10)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    f = sock.makefile("rw", newline="\n")
    f.write("HELLO python\n"); f.flush()
    for line in f:
        parts = line.split()
        if not parts or parts[0] == "END":
            break
        _, step, _t, counts, adc = parts
        pwm, in1, in2 = ctrl.update(int(counts), int(adc))
        f.write(f"CMD {step} {pwm} {in1} {in2}\n"); f.flush()
    sock.close()


if __name__ == "__main__":
    main()
