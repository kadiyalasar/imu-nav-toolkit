"""SIL runner: starts the simulated hardware (motor + L293D + encoder + potentiometer),
launches the controller as a separate process, runs the loop in LOCKSTEP, and scores it.
Lockstep -> simulated time doesn't depend on computer speed -> runs are reproducible."""
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import math
import socket
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motor_sil.motor_model import MotorPlant, SPEC_MODEL, IDENTIFIED_MODEL, FRICTION_MODEL  # noqa: E402
from motor_sil.controller_py import POT_FULL_SCALE  # noqa: E402

HERE = Path(__file__).resolve().parent
CPP_EXE = HERE / "controller_cpp" / ("controller.exe" if sys.platform == "win32" else "controller")
MODELS = {"spec": SPEC_MODEL, "identified": IDENTIFIED_MODEL, "friction": FRICTION_MODEL}
RPM = 2 * math.pi / 60


@dataclass
class Scenario:
    name: str
    mode: str = "position"                 # "position" or "velocity"
    duration: float = 5.0
    targets: list = field(default_factory=lambda: [(0.1, math.pi)])   # (time, target) steps
    sine: tuple = None                     # (offset, amplitude, Hz) instead of steps
    model: str = "identified"
    params: dict = field(default_factory=dict)     # plant changes, e.g. weak battery
    load_time: float = -1.0
    load_torque: float = 0.0               # N m at output shaft
    delay_ms: float = 0.0
    gains: tuple = None                    # override controller gains (kp, ki)
    anti_windup: bool = True
    deadband: float = 0.0
    # pass/fail
    max_final_err: float = 1.5             # position: deg; velocity: RPM (mean over last 1 s)
    max_overshoot_pct: float = 10.0        # position steps only
    max_settle_s: float = 2.5              # position: 2 % band; velocity: 10 % band
    max_track_rms: float = None            # sine tracking (deg)


def target_at(s: Scenario, t):
    if s.sine:
        off, amp, hz = s.sine
        return off + amp * math.sin(2 * math.pi * hz * t)
    value = 0.0
    for t0, v in s.targets:
        if t >= t0:
            value = v
    return value


def to_adc(s, value):
    return int(round(min(max(value / POT_FULL_SCALE[s.mode], 0.0), 1.0) * 1023))


def controller_cmd(kind, port, s: Scenario, counts, dt, gain_scale):
    if kind == "python":
        cmd = [sys.executable, str(HERE / "controller_py.py")]
    elif kind == "cpp":
        if not CPP_EXE.exists():
            raise FileNotFoundError(f"C++ controller not built: {CPP_EXE}")
        cmd = [str(CPP_EXE)]
    else:
        raise ValueError(kind)
    cmd += ["--port", str(port), "--mode", s.mode, "--counts-per-rev", str(counts), "--dt", str(dt),
            "--deadband", str(s.deadband), "--anti-windup", str(int(s.anti_windup)),
            "--gain-scale", str(gain_scale)]
    if s.gains:
        cmd += ["--gains", str(s.gains[0]), str(s.gains[1])]
    return cmd


def run(s: Scenario, controller="python", gain_scale=1.0, timeout_s=10.0):
    p = MODELS[s.model].with_changes(**s.params)
    plant = MotorPlant(p)
    server = socket.socket(); server.bind(("127.0.0.1", 0)); server.listen(1); server.settimeout(timeout_s)
    proc = subprocess.Popen(controller_cmd(controller, server.getsockname()[1], s,
                                           p.counts_per_output_rev, p.control_dt, gain_scale))
    pending = deque([(0, 0, 0)] * int(round(s.delay_ms / 1000 / p.control_dt)))
    log = {k: [] for k in ("t", "ref", "y", "duty", "current", "rtt_ms")}
    try:
        conn, _ = server.accept(); conn.settimeout(timeout_s)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        f = conn.makefile("rw", newline="\n")
        if f.readline().split()[:1] != ["HELLO"]:
            raise RuntimeError("bad handshake")
        for k in range(int(round(s.duration / p.control_dt))):
            adc = to_adc(s, target_at(s, plant.t))
            t0 = time.perf_counter()
            f.write(f"STATE {k} {plant.t:.17g} {plant.encoder_counts()} {adc}\n"); f.flush()
            reply = f.readline().split()
            log["rtt_ms"].append((time.perf_counter() - t0) * 1e3)
            if len(reply) != 5 or reply[0] != "CMD" or int(reply[1]) != k:
                raise RuntimeError(f"protocol error at step {k}: {reply}")
            pending.append(tuple(int(x) for x in reply[2:]))
            load = s.load_torque if 0 <= s.load_time <= plant.t else 0.0
            log["t"].append(plant.t)
            log["ref"].append(adc / 1023 * POT_FULL_SCALE[s.mode])
            log["y"].append(plant.output_angle if s.mode == "position" else plant.output_speed)
            log["current"].append(plant.i)
            log["duty"].append(plant.step(*pending.popleft(), load_torque=load))
        f.write("END\n"); f.flush()
    finally:
        server.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    r = {k: np.array(v) for k, v in log.items()}
    r["metrics"], r["failures"] = score(s, r, p)
    r["passed"] = not r["failures"]
    r["scenario"], r["controller"] = s, controller
    return r


def score(s: Scenario, r, p):
    t, y, ref = r["t"], r["y"], r["ref"]
    to_unit = np.rad2deg if s.mode == "position" else (lambda v: v / RPM)
    m = {"peak_current_A": float(np.abs(r["current"]).max()),
         "pct_time_over_L293D_cont": float(100 * np.mean(np.abs(r["current"]) > p.i_continuous)),
         "rtt_mean_ms": float(r["rtt_ms"].mean())}
    fails = []
    last = t >= t[-1] - 1.0
    m["final_err"] = float(abs(to_unit(np.mean(y[last] - ref[last]))))
    if s.sine:
        late = t > 4.0                          # skip the initial catch-up from 0 to the sine
        m["track_rms"] = float(to_unit(np.sqrt(np.mean((y[late] - ref[late])**2))))
        if s.max_track_rms is not None and m["track_rms"] > s.max_track_rms:
            fails.append(f"tracking RMS {m['track_rms']:.2f} > {s.max_track_rms}")
    else:
        t_step, final = s.targets[-1][0], ref[-1]
        after = t >= t_step
        start = ref[np.argmax(after) - 1] if np.argmax(after) > 0 else 0.0
        jump = final - start
        band = 0.02 if s.mode == "position" else 0.10
        outside = np.flatnonzero(after & (np.abs(y - final) > band * abs(jump)))
        m["settle_s"] = float(t[outside[-1]] - t_step + p.control_dt) if len(outside) else 0.0
        if s.mode == "position" and jump != 0:
            m["overshoot_pct"] = float(max(0.0, (y[after] - final).max() * np.sign(jump) / abs(jump) * 100))
            if m["overshoot_pct"] > s.max_overshoot_pct:
                fails.append(f"overshoot {m['overshoot_pct']:.1f}% > {s.max_overshoot_pct}")
        if m["settle_s"] > s.max_settle_s:
            fails.append(f"settling {m['settle_s']:.2f} s > {s.max_settle_s}")
    unit = "deg" if s.mode == "position" else "RPM"
    if not s.sine and m["final_err"] > s.max_final_err:
        fails.append(f"final error {m['final_err']:.2f} {unit} > {s.max_final_err}")
    return m, fails
