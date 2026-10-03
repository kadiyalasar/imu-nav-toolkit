"""Motor SIL runner: simulator + controller process in lockstep, then scoring.

Lockstep: the simulator waits for each command before advancing, so simulated time is
independent of computer speed -> every run is reproducible (deterministic).
"""
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import socket
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motor_sil.motor_model import MotorPlant, MotorParams  # noqa: E402

HERE = Path(__file__).resolve().parent
CPP_EXE = HERE / "controller_cpp" / ("controller.exe" if sys.platform == "win32" else "controller")


@dataclass
class Scenario:
    name: str
    duration: float = 1.5
    target_deg: float = 90.0           # step target (output shaft)
    step_time: float = 0.05            # when the step command happens
    sine_amp_deg: float = 0.0          # >0: track a sine instead of a step
    sine_hz: float = 1.0
    load_time: float = -1.0            # external load torque on output shaft starts here
    load_torque: float = 0.0           # N m at output shaft
    delay_ms: float = 0.0              # injected command latency
    params: dict = field(default_factory=dict)   # plant parameter changes (model uncertainty)
    # pass/fail
    max_overshoot_pct: float = 10.0
    max_settle_s: float = 0.4          # time (after the step) to stay within 2 % band
    max_ss_error_deg: float = 0.5      # mean error over the last 0.2 s
    max_track_rms_deg: float = 3.0     # sine tracking only
    max_current_A: float = 10.0        # H-bridge driver peak current rating


def target_at(s: Scenario, t):
    if s.sine_amp_deg > 0:
        return np.deg2rad(s.sine_amp_deg) * np.sin(2 * np.pi * s.sine_hz * t)
    return np.deg2rad(s.target_deg) if t >= s.step_time else 0.0


def controller_cmd(kind, port, params: MotorParams, gain_scale):
    if kind == "python":
        cmd = [sys.executable, str(HERE / "controller_py.py")]
    elif kind == "cpp":
        if not CPP_EXE.exists():
            raise FileNotFoundError(f"C++ controller not built: {CPP_EXE}")
        cmd = [str(CPP_EXE)]
    else:
        raise ValueError(kind)
    return cmd + ["--port", str(port), "--counts-per-rev", str(params.counts_per_output_rev),
                  "--dt", str(params.control_dt), "--gain-scale", str(gain_scale)]


def run(s: Scenario, controller="python", gain_scale=1.0, timeout_s=10.0):
    p = MotorParams().with_changes(**s.params)
    plant = MotorPlant(p)
    server = socket.socket(); server.bind(("127.0.0.1", 0)); server.listen(1)
    server.settimeout(timeout_s)
    proc = subprocess.Popen(controller_cmd(controller, server.getsockname()[1], p, gain_scale))
    n = int(round(s.duration / p.control_dt))
    pending = deque([0.0] * int(round(s.delay_ms / 1000 / p.control_dt)))
    log = {k: [] for k in ("t", "target", "angle", "duty", "current", "rtt_ms")}
    try:
        conn, _ = server.accept(); conn.settimeout(timeout_s)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        f = conn.makefile("rw", newline="\n")
        if f.readline().split()[:1] != ["HELLO"]:
            raise RuntimeError("bad handshake")
        for k in range(n):
            tgt = target_at(s, plant.t)
            t0 = time.perf_counter()
            f.write(f"STATE {k} {plant.t:.17g} {plant.encoder_counts()} {tgt:.17g}\n"); f.flush()
            reply = f.readline().split()
            log["rtt_ms"].append((time.perf_counter() - t0) * 1e3)
            if len(reply) != 3 or reply[0] != "CMD" or int(reply[1]) != k:
                raise RuntimeError(f"protocol error at step {k}: {reply}")
            pending.append(float(reply[2]))
            load = s.load_torque if 0 <= s.load_time <= plant.t else 0.0
            log["t"].append(plant.t); log["target"].append(tgt)
            log["angle"].append(plant.output_angle); log["current"].append(plant.i)
            log["duty"].append(plant.step(pending.popleft(), load))
        f.write("END\n"); f.flush()
    finally:
        server.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    out = {k: np.array(v) for k, v in log.items()}
    out["metrics"], out["failures"] = score(s, out)
    out["passed"] = not out["failures"]
    out["scenario"], out["controller"] = s, controller
    return out


def score(s: Scenario, r):
    t, ang, tgt = r["t"], np.rad2deg(r["angle"]), np.rad2deg(r["target"])
    m = {"max_current_A": float(np.abs(r["current"]).max()),
         "saturated_pct": float(100 * np.mean(np.abs(r["duty"]) >= 1.0)),
         "rtt_mean_ms": float(r["rtt_ms"].mean())}
    fails = []
    if s.sine_amp_deg > 0:
        late = t > 0.5                                   # ignore the start-up transient
        m["track_rms_deg"] = float(np.sqrt(np.mean((ang[late] - tgt[late])**2)))
        if m["track_rms_deg"] > s.max_track_rms_deg:
            fails.append(f"tracking RMS {m['track_rms_deg']:.2f} deg > {s.max_track_rms_deg}")
    else:
        A = s.target_deg
        after = t >= s.step_time
        m["overshoot_pct"] = float(max(0.0, (ang[after].max() - A) / abs(A) * 100))
        outside = np.flatnonzero(after & (np.abs(ang - A) > 0.02 * abs(A)))
        m["settle_s"] = float(t[outside[-1]] - s.step_time + 1e-3) if len(outside) else 0.0
        m["ss_error_deg"] = float(np.abs(np.mean(ang[t >= t[-1] - 0.2] - A)))
        if s.load_time < 0 and m["overshoot_pct"] > s.max_overshoot_pct:
            fails.append(f"overshoot {m['overshoot_pct']:.1f}% > {s.max_overshoot_pct}")
        if s.load_time < 0 and m["settle_s"] > s.max_settle_s:
            fails.append(f"settling {m['settle_s']:.3f} s > {s.max_settle_s}")
        if m["ss_error_deg"] > s.max_ss_error_deg:
            fails.append(f"steady-state error {m['ss_error_deg']:.2f} deg > {s.max_ss_error_deg}")
    if m["max_current_A"] > s.max_current_A:
        fails.append(f"current {m['max_current_A']:.2f} A > {s.max_current_A}")
    return m, fails
