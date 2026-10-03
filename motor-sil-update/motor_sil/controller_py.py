"""Motor position controller as a SEPARATE PROGRAM (the "software" in SIL).

It only sees what real firmware would see: encoder COUNTS and the target angle.
It must estimate speed itself from the counts.

Protocol (one line each way per control step, 1 kHz):
  controller -> sim : HELLO <name>
  sim -> controller : STATE <step> <time> <encoder_counts> <target_rad>
  controller -> sim : CMD <step> <duty>          duty in [-1, 1], must echo <step>
  sim -> controller : END

Control law: PID on output angle
  * derivative on MEASUREMENT (not error) -> no "kick" when the target jumps
  * speed from count differences, low-pass filtered (raw differences are very jumpy)
  * anti-windup: freeze the integrator while the output is saturated
"""
import argparse
import math
import socket


class PID:
    def __init__(self, kp, ki, kd, dt, counts_per_rev, d_filter_hz=50.0):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.rad_per_count = 2 * math.pi / counts_per_rev
        self.alpha = dt / (dt + 1 / (2 * math.pi * d_filter_hz))   # 1st-order low-pass
        self.integral = 0.0
        self.prev_counts = None
        self.speed = 0.0

    def update(self, counts, target):
        angle = counts * self.rad_per_count
        if self.prev_counts is not None:
            raw_speed = (counts - self.prev_counts) * self.rad_per_count / self.dt
            self.speed += self.alpha * (raw_speed - self.speed)
        self.prev_counts = counts
        error = target - angle
        u = self.kp * error + self.ki * self.integral - self.kd * self.speed
        u_sat = max(-1.0, min(1.0, u))
        if u == u_sat:                       # anti-windup: integrate only when not saturated
            self.integral += error * self.dt
        return u_sat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gains", type=float, nargs=3, default=[12.0, 60.0, 0.2])   # kp ki kd (tuned in simulation)
    ap.add_argument("--counts-per-rev", type=float, default=1440.0)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--gain-scale", type=float, default=1.0)
    a = ap.parse_args()
    kp, ki, kd = (g * a.gain_scale for g in a.gains)
    pid = PID(kp, ki, kd, a.dt, a.counts_per_rev)

    sock = socket.create_connection(("127.0.0.1", a.port), timeout=10)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    f = sock.makefile("rw", newline="\n")
    f.write("HELLO python\n"); f.flush()
    for line in f:
        parts = line.split()
        if not parts or parts[0] == "END":
            break
        _, step, _t, counts, target = parts
        duty = pid.update(int(counts), float(target))
        f.write(f"CMD {step} {duty:.17g}\n"); f.flush()
    sock.close()


if __name__ == "__main__":
    main()
