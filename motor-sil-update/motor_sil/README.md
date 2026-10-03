# Motor Software-in-the-Loop (SIL)

Extends my Simscape DC-motor + H-bridge model into a closed-loop SIL setup: the motor
model runs as the simulated plant, and the position controller runs as **separate
software** (Python reference and C++ "production" version) that talks to it over a socket,
the same way firmware would talk to a real motor driver and encoder.

```
 ┌──────────────────────────────┐  STATE step t encoder_counts target  ┌─────────────────────────┐
 │ SIMULATOR  (motor_sil/)      │ ───────────────────────────────────► │ CONTROLLER (separate    │
 │ DC motor: L,R,Ke,Kt,J,b,     │                                      │ process, Python or C++) │
 │ Coulomb friction, gearbox    │ ◄─────────────────────────────────── │ PID on output angle     │
 │ H-bridge (avg model), encoder│          CMD step pwm_duty           │                         │
 └──────────────────────────────┘                                      └─────────────────────────┘
```

## The plant (same physics as the Simscape model)

* Electrical: `L di/dt = V - R i - Ke w`  (R includes H-bridge switch resistance)
* Mechanical: `J dw/dt = Kt i - b w - tau_coulomb - tau_load/N`, with `J = J_motor + J_load/N^2`
* H-bridge, **average model**: `V = duty * V_supply` (duty in [-1, 1]). A switching model would
  need ~microsecond steps to resolve each PWM pulse; the average model is enough for position control.
* Encoder: 48 counts/motor rev (12-line, x4 quadrature) x 30:1 gearbox = 1440 counts/output rev.
* Integration: semi-implicit Euler at 20 kHz. The electrical time constant L/R is about 0.45 ms,
  the mechanical one about 70 ms -- a **stiff** system, so the physics step must resolve the fast part.
* Controller runs at 1 kHz; each command is held between updates (zero-order hold).

Parameters in `motor_model.py` are **placeholders** for a typical 12 V gearmotor.

## The controller

PID on output-shaft angle, written the way firmware would be:
* only sees **encoder counts**, so it must estimate speed itself (count differences, low-pass filtered)
* derivative on **measurement**, not error -> no derivative "kick" when the target jumps
* anti-windup: the integrator freezes while the PWM output is saturated

## Tests

**Plant verification** (`tests/test_motor_model.py`) -- the simulator is checked against first
principles before it is trusted: no-load speed, electrical time constant = L/R, stall current = V/R,
reversal current spike, encoder resolution, and convergence as the timestep shrinks.

**Closed-loop SIL** (`tests/test_motor_sil.py`):
* 8-scenario suite x 2 controllers: steps, sine tracking, load disturbance, 9 V battery,
  +50% load inertia, 3x friction, 3 ms latency -- each with pass/fail criteria
  (overshoot, settling time, steady-state error, tracking RMS, peak current)
* determinism (lockstep -> identical runs), back-to-back Python vs C++ (identical trajectories),
  a deliberately broken controller must fail, and a latency-margin regression guard

## Findings (simulation)

* All scenarios pass for both controllers; Python and C++ produce identical trajectories.
* **Latency tolerance is only about 4 ms** (4 control periods): each extra millisecond adds about
  2% overshoot, and at 6 ms the loop starts to oscillate. Localhost round trip is about 0.03 ms,
  so latency had to be injected -- on real hardware (HIL), USB or bus delay could eat this margin.
* **Reversing at speed draws ~9-10 A, almost 2x the 5.5 A stall current**: back-EMF adds to the
  reversed supply voltage ("plugging"). A real driver needs current limiting or a gentler reversal.

## Run

```
python motor_sil/run_suite.py
python motor_sil/latency_sweep.py
pytest tests/test_motor_model.py tests/test_motor_sil.py -v
```
C++ (optional locally, always built in CI), Windows/MinGW:
`g++ -O2 -std=c++17 motor_sil/controller_cpp/controller.cpp -o motor_sil/controller_cpp/controller.exe -lws2_32`

## Not yet

This is SIL, not HIL: everything runs on one computer, in lockstep, not in real time. HIL would
run the controller on a real microcontroller with the motor model running in real time on the host.
