"""Reproduce the Part B open-loop experiment in simulation and compare with the hardware.

For each duty cycle: drive the motor forward, measure speed exactly like the Simulink model
did (encoder count difference every 10 ms), then read off the steady speed, the DC gain
K = w_motor / (V * duty) and the time constant (time to 63 %)."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motor_sil.motor_model import (MotorPlant, SPEC_MODEL, IDENTIFIED_MODEL,  # noqa: E402
                                   HARDWARE_TRIALS, SIMSCAPE_RESULTS)


def open_loop(p, duty, seconds=3.0):
    plant = MotorPlant(p)
    pwm = int(round(duty * 255))
    n = int(round(seconds / p.control_dt))
    rad_per_count = 2 * np.pi / p.counts_per_output_rev
    t, w_meas, prev = np.zeros(n), np.zeros(n), plant.encoder_counts()
    for k in range(n):
        plant.step(pwm, 1, 0)
        c = plant.encoder_counts()
        t[k], w_meas[k], prev = plant.t, (c - prev) * rad_per_count / p.control_dt, c
    w_ss = w_meas[-50:].mean()
    tau = t[np.argmax(w_meas >= 0.632 * w_ss)]
    K = w_ss * p.N / (p.V_supply * pwm / 255)
    return t, w_meas, w_ss, tau, K


if __name__ == "__main__":
    print(f"{'duty':>6s} | {'hardware':^22s} | {'Simscape':^9s} | {'SIL spec model':^24s} | {'SIL identified J':^24s}")
    print(f"{'':>6s} | {'speed':>7s} {'K':>6s} {'tau':>6s} | {'speed':>9s} | {'speed':>7s} {'err%':>6s} {'tau':>6s}"
          f"   | {'speed':>7s} {'err%':>6s} {'tau':>6s}")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, (duty, hw_w, hw_tau) in zip(axes, HARDWARE_TRIALS):
        hw_K = hw_w * 172 / (7.5 * duty)
        row = f"{duty*100:5.1f}% | {hw_w:7.3f} {hw_K:6.1f} {hw_tau:6.3f} | {SIMSCAPE_RESULTS[duty]:9.3f} |"
        for name, p in (("spec", SPEC_MODEL), ("identified", IDENTIFIED_MODEL)):
            t, w, w_ss, tau, K = open_loop(p, duty)
            row += f" {w_ss:7.3f} {100*(w_ss-hw_w)/hw_w:+6.1f} {tau:6.3f}   |"
            ax.plot(t, w, lw=1, label=f"SIL {name} model")
        ax.axhline(hw_w, c="k", ls="--", lw=1, label="hardware steady speed")
        ax.axvline(hw_tau, c="gray", ls=":", label=f"hardware tau = {hw_tau} s")
        ax.set_title(f"duty {duty*100:.1f}%"); ax.set_xlim(0, 1.0); ax.set_xlabel("time [s]")
        ax.set_ylabel("output speed [rad/s]"); ax.grid(alpha=.3); ax.legend(fontsize=7)
        print(row)
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    plt.tight_layout(); plt.savefig(out / "motor_identification.png", dpi=120)
    print("\nNotes:")
    print(" * Speed is measured from encoder counts every 10 ms -> steps of 0.0761 rad/s (as in the hardware plots).")
    print(" * Both SIL models use b fitted to Trial 1, so Trial 1 matches; Trials 2-3 come out 10-13% low,")
    print("   the same pattern as the Simscape model -> a single operating point is not enough to fit b.")
    print(" * The spec inertia gives tau ~0.03 s; the hardware's 0.135 s needs ~6x more inertia (identified model).")
    print("Saved results/motor_identification.png")
