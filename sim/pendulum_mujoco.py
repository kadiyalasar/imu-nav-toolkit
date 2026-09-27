"""
Weekend exercise: simulate a simple pendulum in MuJoCo and compare integrators.
Goal: see energy drift with your own eyes, so you can talk about it on Monday.

Setup:   pip install mujoco numpy matplotlib
Run:     python pendulum_mujoco.py
"""
import mujoco
import numpy as np
import matplotlib.pyplot as plt

# A 1 kg point mass on a 0.5 m massless rod, hinged at height 1 m, no friction.
# With no damping, total energy should stay constant forever in the real physics.
XML = """
<mujoco model="pendulum">
  <option timestep="{dt}" gravity="0 0 -9.81">
    <flag energy="enable"/>
  </option>
  <worldbody>
    <body name="pole" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0"/>
      <inertial pos="0 0 -0.5" mass="1" diaginertia="1e-6 1e-6 1e-6"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

INTEGRATORS = {
    "Semi-implicit Euler (MuJoCo default)": mujoco.mjtIntegrator.mjINT_EULER,
    "RK4": mujoco.mjtIntegrator.mjINT_RK4,
    "Implicit-fast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}

def run(dt, integrator, seconds=60.0, start_angle=np.pi / 2):
    model = mujoco.MjModel.from_xml_string(XML.format(dt=dt))
    model.opt.integrator = integrator
    data = mujoco.MjData(model)
    data.qpos[0] = start_angle          # start horizontal, at rest
    mujoco.mj_forward(model, data)      # compute energy for the initial state
    e0 = data.energy[0] + data.energy[1]  # potential + kinetic

    t, drift = [], []
    for _ in range(int(seconds / dt)):
        mujoco.mj_step(model, data)
        e = data.energy[0] + data.energy[1]
        t.append(data.time)
        drift.append(e - e0)
    return np.array(t), np.array(drift)

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, dt in zip(axes, [0.01, 0.001]):
        for name, integ in INTEGRATORS.items():
            t, drift = run(dt, integ)
            ax.plot(t, drift, label=name)
            print(f"dt={dt:<6} {name:<38} final drift = {drift[-1]:+.2e} J, "
                  f"max |drift| = {np.abs(drift).max():.2e} J")
        ax.set_title(f"Energy drift, timestep = {dt} s")
        ax.set_ylabel("E(t) - E(0)  [J]")
        ax.legend()
        ax.grid(True)
    axes[-1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.savefig("energy_drift.png", dpi=120)
    print("Saved plot to energy_drift.png")
