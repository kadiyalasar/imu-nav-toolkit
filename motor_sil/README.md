# Motor SIL: software-in-the-loop for my ME 5245 mechatronics project

The original project (with Timothy Warner): position and velocity control of a **Pololu 172:1
gearmotor with a 48 CPR encoder**, driven by an **L293D H-bridge** from an **Arduino Uno**,
powered by 5 AA batteries (~7.5 V), with a potentiometer setting the target. We modeled it in
Simscape (MIL), deployed Simulink controllers to the Arduino with the real motor (rapid control
prototyping), and compared the two.

This folder adds the missing **SIL** step: the same plant, simulated in code, and a controller
running as a **separate program** that sees exactly what the Arduino sees and outputs exactly
what the Arduino outputs.

```
 SIMULATED HARDWARE (runner.py + motor_model.py)        CONTROLLER (separate process, Python or C++)
 Pololu motor + 172:1 gearbox + L293D + encoder + pot
        encoder counts (pins 2,3) + pot ADC 0..1023 (A0)  ───────►   PI, sign-magnitude drive
        ◄───────  PWM 0..255 (pin 9) + direction IN1/IN2 (pins 6, 8)
```

| Testing level | What runs | Status |
|---|---|---|
| MIL | Simscape: controller + motor, all models | done in the course |
| **SIL** | real controller code, simulated motor | **this folder** |
| HIL | controller on the Arduino, motor simulated in real time | not done |
| Hardware | controller on the Arduino, real motor | done in the course |

## Plant (`motor_model.py`): parameters from the Part B report

R = 5 Ω, J = 0.00976 kg·m² at the output (3.3e-7 at the motor), kt = kb = 0.0039 N·m/A (spec-sheet
stall torque / stall current), b = 1.184e-5 (fitted to hardware Trial 1), N = 172, 7.5 V,
8256 counts per output revolution, 10 ms control period. **Assumed** (not in the report):
inductance L = 1 mH. Average H-bridge model (490 Hz PWM is much faster than the motor).

Three variants: `spec` (as in Simscape), `identified` (J fitted to the measured 0.135 s time
constant, about 6x the given value), `friction` (adds Coulomb friction, a hypothesis).

## Controller (`controller_py.py`, `controller_cpp/controller.cpp`)

Same structure as the project's Simulink hardware models: PI on the error, `|u|` -> PWM,
sign of `u` -> direction pins. Position mode reverses; velocity mode is forward-or-brake.
Speed is computed exactly like the Simulink model: count difference per 10 ms sample, unfiltered.
Gains were re-tuned here (the project's gains used a PWM conversion gain not recorded in the report).

## Results (simulation)

* **Reproducing the project:** the SIL plant matches our Simscape steady speeds within ~2%, and
  shows the same pattern against hardware: Trial 1 matches (b was fitted to it), Trials 2-3 are
  10-13% low. The given inertia gives a ~0.03 s time constant vs 0.135 s measured. (`identify.py`)
* **Speed resolution:** one count per 10 ms = 0.076 rad/s (0.73 RPM), exactly the steps in our
  hardware plots.
* **Why the hardware needed PI:** P-only speed control leaves ~5 RPM of steady error (holding a
  speed needs a nonzero command). In the spec model, P-only position control is enough, as our
  Simscape result showed.
* **Anti-windup:** without it, a pi-rad step overshoots ~14%; with it, ~4%.
* **Deadband:** zero deadband (as on our hardware) makes the direction flip back and forth at the
  target; a small deadband stops it.
* **Latency:** passes with 20 ms of command delay (2 periods), fails from 40 ms.
* **Driver limits:** the fitted parameters imply ~1 A while running and ~1.5 A peaks, above the
  L293D's ~0.6 A continuous / ~1.2 A peak ratings -> evidence that the kt/friction split is off;
  measuring no-load current would settle it.
* **Known limitation reproduced:** with 6 V batteries the motor can't reach 20 RPM (expected-fail test).

## Honest limits

Placeholder inductance; smooth (tanh) friction creeps instead of truly sticking, so it can't fully
reproduce stiction; lockstep, not real time.

## Run

```
python motor_sil/identify.py          # Part B open-loop comparison: hardware vs Simscape vs SIL
python motor_sil/run_suite.py         # closed-loop scenario suite
python motor_sil/latency_sweep.py
pytest tests/test_motor_model.py tests/test_motor_sil.py -v
```
