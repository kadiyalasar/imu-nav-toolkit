# Weekend Prep: IMU Navigation Toolkit + CI Pipeline + Interview Q&A

Prepared for the Qualcomm manager interview (Monday, Sept 28). Work through it top to bottom.

---

## 0. Read this first: the honesty rules for synthetic data

Everything in this repo runs on **synthetic data**, generated to mimic your Lab 5 setup (the magnetometer ellipse numbers come straight from your report). Synthetic data is a legitimate engineering tool: it's how you prove your analysis code works, because you know the true answer in advance. But the numbers it produces are **not hardware results**.

What you **can** truthfully say on Monday:

> "After re-reading my Lab 5 report, I found a frame-convention bug and some weaknesses in my bias handling, so this weekend I rebuilt the pipeline properly: Allan deviation, ellipse-fit magnetometer calibration, a complementary filter with a correctly derived crossover, ZUPT velocity correction, and a 6-state EKF with bias estimation. I validated each piece against synthetic data with known ground truth, and put it all behind a pytest suite running in GitHub Actions."

What you **must not** say: "My EKF reduced drift by 83%" (as if on a real car), or put any synthetic number on your resume as a result.

**Want real numbers by Monday?** Your phone has an IMU. Free apps such as *phyphox* or *Sensor Logger* can record accelerometer and gyroscope data to CSV. Lay the phone flat on a table, don't touch it, record for 2+ hours tonight (plugged in), then run `01_allan.py` on it (see section 3, "Using real data"). That gives you genuinely real Allan-deviation numbers from real hardware. It's a phone IMU, not a VN-100, and you'd say exactly that.

---

## 1. Schedule

**Saturday afternoon/evening (about 4 hours)**
1. Setup and run everything (30 min) — section 2
2. Start the phone recording if you're doing it (5 min, then leave it)
3. Read sections 3.1–3.4 alongside the plots (2 hours). After each, say the "how to say it" paragraph out loud once.
4. Section 5: go through the resume questions and fill in the brackets with your real details (1 hour). This matters more than any code.

**Sunday (about 5–6 hours)**
1. Section 3.5, the EKF (1.5 hours)
2. Section 4, CI: read it, push the repo to GitHub, watch the tests go green, then break something on purpose and watch it go red (1.5 hours)
3. Run Allan on your phone data if you recorded it (30 min)
4. Section 6, the general questions; practice your 60-second intro out loud (1 hour)
5. Mock interview with Claude (1 hour)

---

## 2. Setup

Install Python 3.10 or newer (python.org). Then, in a terminal, inside the unzipped folder:

```
pip install -r requirements.txt
cd scripts
python 00_generate_data.py
python 01_allan.py
python 02_magcal.py
python 03_heading.py
python 04_dead_reckoning.py
python 05_ekf.py
cd ..
pytest -v
```

Every script prints numbers and saves a plot in `results/`. Open the plot next to the explanation below.

**Repo map**

| Path | What it is |
|---|---|
| `navlib/` | the library: reusable functions (the "real" code) |
| `navlib/synthetic.py` | generates fake-but-realistic sensor data |
| `navlib/allan.py`, `magcal.py`, `heading.py`, `deadreckoning.py`, `ekf.py` | one concept each |
| `navlib/evaluation.py` | metrics shared by scripts AND tests (one code path, not two) |
| `scripts/00–05` | step-by-step runnable analyses that make plots |
| `tests/` | automated checks that run in CI |
| `.github/workflows/ci.yml` | tells GitHub to run the tests automatically |
| `sim/pendulum_mujoco.py` | the MuJoCo integrator exercise from before |

**The conventions (top of `navlib/__init__.py`).** World frame ENU (East, North, Up). Body frame FLU (Forward, Left, Up), which is the ROS standard. Yaw measured from East, counter-clockwise positive. Gyro z positive means turning left. Writing these down *first* and testing them is the direct fix for your Lab 5 bug.

---

## 3. The analysis, step by step

### 3.0 How the synthetic data is made (`navlib/synthetic.py`)

Each gyro/accel reading is `truth + bias(t) + white noise`, where the bias has three parts:

- **turn-on bias**: a constant offset for this power-up
- **Gauss-Markov bias**: wanders but is pulled back toward zero, with a 60 s correlation time. This creates the flat "bias instability" region in the Allan plot.
- **random-walk bias**: drifts without bound, very slowly. This creates the rising right side.

Per-sample white noise sigma = `density × √(sample rate)`. The gyro density (6.1e-5 rad/s/√Hz ≈ 0.0035 °/s/√Hz) is roughly VN-100 class; check the real datasheet before quoting it.

The drive has 13 legs: accelerate, cruise, turn 45° or 90°, cruise, stop at a "traffic light". It also has **road grade** (the road tilting up and down by about 1°). This turns out to be the most important realism detail: tilt makes gravity leak into the forward accelerometer as `g·sin(pitch)`, and 1° of grade = 0.17 m/s² of fake acceleration, *bigger than the sensor bias*. Your Lab 5 report mentions slopes in Q5; this is that effect.

GPS is 1 Hz with 2.5 m noise. The magnetometer is distorted with your exact Lab 5 ellipse: center (0.17, 0.05), axes 0.0889 / 0.0733, tilt 0.4644 rad.

What synthetic data *doesn't* have: real vibration spectra, the IMU mounted off-center or tilted, temperature drift, GPS multipath in Boston's tall-building canyons, magnetic interference from passing cars. Real data is always harder. Say this if asked.

### 3.1 Allan deviation (`01_allan.py` → `results/01_allan.png`)

**What it answers:** "How noisy is this IMU, and in what ways?" You record it perfectly still for hours; anything it reports is error.

**How it works, in plain words.** Pick an averaging time τ, say 10 s. Chop the recording into 10-second chunks and average each chunk. If the sensor were perfect, all chunk averages would be identical. Measure how much *neighboring* chunk averages differ; that's the Allan deviation at τ = 10 s. Repeat for τ from 0.025 s up to 1/10 of the recording, and plot on log-log axes.

**How to read the plot (the black curve):**
- **Left side, slope −½**: averaging longer helps, in exactly the way it helps for pure random noise (averaging N samples shrinks noise by √N). This is **white noise**. Its coefficient is where that line crosses τ = 1 s (blue dot). For a gyro this is called **angle random walk (ARW)**, because integrating white rate-noise makes your angle wander like √t. For an accelerometer it's **velocity random walk**.
- **Bottom, flat**: averaging longer stops helping. The height of this floor (divided by 0.664, a constant from theory) is **bias instability**, the best bias stability you can hope for.
- **Right side, slope +½**: averaging longer makes it *worse*, because the bias itself is drifting. This is **rate random walk**, read where that line crosses τ = 3 s (a convention from the IEEE standard).

**Your output:** white noise recovered as 6.09e-5 vs 6.10e-5 injected (within 0.2%); rate random walk 3.02e-6 vs 3.00e-6. The gyro converts to ARW ≈ 0.21 °/√hr and bias instability ≈ 4.9 °/hr.

**A real insight worth mentioning:** the white-noise estimate is always accurate, but rate random walk bounces around by up to 2× between random seeds. At long τ you only have a handful of independent chunks, so the estimate is statistically weak. That's why the code stops at τ = 1/10 of the record, and why real characterization runs last many hours.

**How this connects to filters and simulators:**
- EKF process noise Q comes from the white-noise density (per-sample variance = density² × rate).
- The bias random-walk states in the EKF get their Q from the rate-random-walk coefficient.
- A simulated IMU is this model run backwards: truth + bias process + white noise with these exact numbers. That is literally the JD line "model sensor behavior from real characterization data."

**How to say it:** "Allan deviation separates IMU errors by timescale. Short averaging windows are dominated by white noise, the slope −½ region, which gives the random-walk coefficient I'd put in the filter's Q. The flat minimum is bias instability. The +½ region is the bias itself random-walking, which is why I model biases as random-walk states. I validated my implementation on synthetic noise with injected parameters: it recovered the white-noise density to within a fraction of a percent, while rate random walk varied up to 2× between seeds, which taught me why characterization recordings need to be hours long."

**Likely follow-ups:**
- *Why not just compute the standard deviation?* It mixes all timescales into one number; you can't tell noise from drift, and a filter needs them separately.
- *Why 1/10 of the record?* Fewer than about 10 independent clusters makes the estimate unreliable.
- *What does bias instability mean physically?* The best you could ever know the bias by averaging, before drift takes over.

**Using real data:** put a CSV named `data/stationary.csv` with columns `t,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z` (SI units: rad/s and m/s²) and set `FS` in the script to your sample rate. Phone apps often log in deg/s or in g; convert first. The "injected" comparison lines won't mean anything for real data; ignore them.

### 3.2 Magnetometer calibration (`02_magcal.py` → `results/02_magcal.png`)

**What it answers:** "Which way is north, when the car's metal is distorting the field?"

**The physics.** As the car turns a full circle, a perfect magnetometer's (x, y) readings trace a circle centered at zero. Two distortions:
- **Hard iron** = permanent magnets and magnetized steel riding along with the sensor. Adds a constant offset, so the circle's center moves. Your data: (0.17, 0.05) gauss.
- **Soft iron** = soft ferromagnetic material (the steel body) that *bends* the Earth's field differently depending on direction. Squashes and tilts the circle into an ellipse. (Correction to your report: it's not laptops and phones "emitting" fields. Current-carrying electronics cause time-varying interference, which neither correction removes.)

**The fix.** Fit an ellipse to the circle-driving data (the code solves a least-squares problem for the general conic equation), then apply the matrix that maps the ellipse back to a unit circle: rotate to the ellipse's axes, scale each axis, **and rotate back**.

**Your output:** center (0.1700, 0.0500), axes 0.0892 / 0.0732 (ratio 0.820, your report had 0.824), tilt 26.8°. Calibrated heading error ≈ 2.1°.

**The finding worth mentioning.** Your report's recipe rotated and scaled but didn't rotate back. The script shows that skipping that step shifts every heading by the tilt angle, **26.8°**. You corrected your Lab 5 trajectory with a **π/8 = 22.5°** rotation. They're close, so the missing rotate-back may explain much of that fudge factor. Say "may": you can't verify without the original data.

**How to say it:** "I found the hard-iron offset as the ellipse center and the soft-iron distortion as the ellipse shape and tilt, then mapped it back to a circle. Revisiting it later, I realized I'd skipped rotating back to the original frame, which leaves a constant heading offset equal to the tilt, about 27° in my data. That's likely part of why I needed a rotation correction on my trajectory."

**Likely follow-ups:**
- *Why drive in circles?* To sweep all headings so the full ellipse is visible.
- *Does the calibration hold forever?* No. If the cargo changes or something gets magnetized, recalibrate; and it's a 2D calibration assuming the car is level.
- *What about declination?* Magnetic north ≠ true north; about 14° West in Boston. Add it after calibration.

### 3.3 Heading: magnetometer + gyro + complementary filter (`03_heading.py` → `results/03_heading.png`)

**The trade-off.** The magnetometer knows *absolute* heading but is noisy (±2° here, worse near metal). The gyro is smooth but you must integrate it, and integrating its bias makes the heading drift forever: 117° off by the end of this drive.

**The complementary filter** trusts the gyro for fast changes and the magnetometer for slow ones:

```
predicted = previous + gyro × dt
new       = predicted + (1 − α) × wrap(mag_yaw − predicted)
```

With α = 0.99 at 40 Hz, the time constant is τ = α·dt/(1−α) ≈ 2.5 s, so the **crossover frequency is 0.064 Hz**: gyro for changes faster than about 16 s, magnetometer for slower. Result: 0.31° error versus 2.2° (magnetometer alone) and 67° (gyro alone).

**Two things fixed from your report:**
1. **Matched crossover.** Your report used separate Butterworth filters at 0.1 Hz (low-pass) and 0.005 Hz (high-pass). For the outputs to add up correctly, the two filters must be *complementary* (sum to one at every frequency), which requires the same crossover. The single-α form guarantees it.
2. **Angle wrapping.** Averaging 359° and 1° naively gives 180°. The code blends the *wrapped difference* instead.

**The sign bug (bottom panel).** Flipping the gyro's sign reproduces your Lab 5 symptom exactly: mirror-image yaw curves. The most likely cause is that the VN-100 natively reports in a z-*down* body frame (x forward, y right, z down), so its positive z-rotation is a *right* turn, while your magnetometer heading assumed counter-clockwise positive. Check which frame your ROS driver published before stating this as fact. `tests/test_conventions.py` is the automated check that would have caught it: "a left turn must increase yaw for every heading source."

**How to say it:** "Looking back at my Lab 5 plots, the magnetometer and gyro headings were mirror images: a frame-convention mismatch, likely the VN-100's z-down frame against my z-up heading math. It's a classic silent bug; the code runs and the plots just look wrong. When I rebuilt the pipeline I wrote the conventions down first and added a unit test that a left turn increases yaw for every source."

### 3.4 Dead reckoning (`04_dead_reckoning.py` → `results/04_dead_reckoning.png`)

Speed comes from integrating forward acceleration; position comes from integrating speed along the heading: `vE = v·cos(yaw)`, `vN = v·sin(yaw)` (with yaw from East). **No scale factor, no rotation fudge.**

Three bias strategies, mirroring your report:

| Method | Speed RMSE | Final position error | Drift passed 2 m after |
|---|---|---|---|
| naive integration | 45 m/s | 1,500 m | 8 s |
| constant bias (best-fit line, your first attempt) | 13 m/s | 2,500 m | 6 s |
| ZUPT: re-estimate bias at every stop (your second method) | 7 m/s | 1,100 m | 35 s |

**ZUPT (zero-velocity update)** is the formal name for your stationary-period trick: when you *know* the car is stopped, reset speed to zero and re-measure the bias. It's standard in pedestrian navigation (foot-mounted IMUs reset every step).

**Why ZUPT alone still fails:** road grade. Between stops the road tilts, gravity leaks into accel_x, and a bias measured at the last light doesn't know about the hill you're on now. This is the same qualitative story as your real Lab 5 (within 2 m for under a minute).

**How to say it:** "Integrating an accelerometer twice turns any bias into quadratically growing position error. Re-estimating the bias at every stop, a zero-velocity update, helped a lot, but road grade makes gravity leak into the forward axis between stops, and nothing in pure dead reckoning can observe that. That's the motivation for fusing GPS in a filter that estimates biases continuously."

### 3.5 The EKF (`05_ekf.py` → `results/05_ekf.png`)

**What a Kalman filter is, from zero.** Keep a best guess of the state *and* how uncertain you are about it (the covariance matrix **P**). Then alternate two steps:

1. **Predict** (every IMU sample, 40 Hz): push the state forward using the motion model and IMU readings. Uncertainty *grows*: P ← F·P·Fᵀ + Q.
2. **Update** (whenever a measurement arrives): compare the measurement to what you predicted you'd measure. The difference is the **innovation**. Nudge the state toward the measurement by an amount set by the **Kalman gain** K, which weighs your uncertainty (P) against the sensor's noise (R). Uncertainty *shrinks*.

"Extended" means the motion model is nonlinear (position depends on cos(yaw)), so we linearize it at each step using the **Jacobian F**: the matrix of how each new state variable changes with each old one.

**This filter's state (6 numbers):** position East, position North, forward speed, yaw, **accelerometer bias**, **gyro bias**.

**Predict:**
```
px  += v·cos(ψ)·dt
py  += v·sin(ψ)·dt
v   += (accel − b_a)·dt
ψ   += (gyro − b_g)·dt
b_a, b_g: unchanged (but allowed to random-walk via Q)
```

**Three measurement types:**
- GPS position (1 Hz), R = (2.5 m)²
- magnetometer yaw (10 Hz), R = (4°)², with the innovation **wrapped** to ±180°
- **ZUPT pseudo-measurement**: when stationary, "measure" v = 0. Your Lab 5 trick, now inside the filter.

**Q vs R, the intuition:** Q says how much you distrust your motion model per step; R says how much you distrust each sensor. Big Q plus small R means "follow the sensors." Here the accel noise in Q (0.3 m/s²) is deliberately larger than the sensor spec, because it has to absorb vibration and unmodeled effects too, and the accel-bias random walk is large because road grade *looks like* a quickly changing bias.

**Joseph form.** The covariance update is written `P = (I−KH)·P·(I−KH)ᵀ + K·R·Kᵀ` instead of the shorter `(I−KH)·P`. They're mathematically equal, but with floating-point rounding the short form can make P lose symmetry or go slightly negative, and the filter diverges. Joseph form stays symmetric and positive. If Bo asks "why Joseph form," that's the answer: numerical robustness.

**Why the bias states matter.** The right-hand plot shows the filter learning the true gyro bias within about 30 s, with its ±2σ uncertainty band shrinking. A complementary filter can't do this. It can only low-pass the magnetometer.

**Your output (synthetic):**
- Position RMSE: raw GPS 3.6 m → EKF 1.8 m. The fused estimate beats GPS because the IMU smooths out GPS noise between fixes.
- Yaw RMSE 0.16°.
- **GPS outage study:** GPS removed for five 30-second windows while driving. Mean position error at the end of the outage: Lab 5-style dead reckoning **195 m**, EKF **32 m** (about 83% lower). Across other random seeds it was 82–91% lower. The one window where the EKF did *no better* (18 m vs 17 m) is a good honesty detail.

**The honest caveat that shows depth:** even the EKF drifts 30+ m in some outages, because road grade changes during the outage and a 2D filter has no way to see pitch. The fix is a full 3D inertial navigation filter that tracks pitch and roll from the gyros and uses all three accelerometer axes, or at least adding a road-grade state. Say this *before* he asks; it's exactly the "honest limits" the JD asks for.

**How to say it:** "I replaced the complementary filter with a 6-state EKF: position, speed, yaw, and accel and gyro biases, with GPS, magnetometer yaw, and zero-velocity pseudo-measurements. The key difference is that it estimates the biases continuously, so when GPS drops out its predictions are already corrected. On synthetic data with ground truth, it cut end-of-outage position error by roughly 80–90% versus my Lab 5 dead reckoning. The remaining error is road grade, which a 2D filter can't observe; the next step would be a full 3D INS."

**Likely follow-ups:**
- *EKF vs particle filter?* EKF assumes roughly Gaussian errors and a locally linear model: cheap, great here. Particle filters handle multi-modal beliefs (for example, "am I on this street or the parallel one?") at much higher compute cost.
- *How did you tune Q and R?* R from sensor specs and Allan numbers; Q started from Allan numbers, then inflated for vibration and grade, checked by whether innovations look like zero-mean noise and whether errors stay inside the ±2σ bands.
- *How would you know the filter is inconsistent?* Errors regularly outside ±2σ, or innovations much bigger than S predicts. The formal check is called NEES/NIS.
- *What's observable?* Gyro bias is observable from magnetometer yaw. Accel bias is only weakly observable from GPS position (through double integration), which is why it's the hard one.

---

## 4. CI pipelines, from zero

### 4.1 The vocabulary

- **Git**: tracks every version of your code. A **commit** is a saved snapshot with a message.
- **GitHub**: hosts git repositories online. **Push** = upload your commits.
- **Pull request (PR)**: "I'd like to merge these changes into the main code; please review." Teams require checks to pass before merging.
- **Test**: code that checks other code. In pytest, any function named `test_…` containing `assert` statements. If an assert is false, the test fails.
- **CI (continuous integration)**: every push or PR automatically triggers a fresh machine (a **runner**) to download the code, install dependencies, and run all the tests. Green check = safe to merge; red X = something broke.
- **CD (continuous deployment/delivery)**: after CI passes, automatically package or release. (Not in this repo; be honest that your experience is CI.)
- **Workflow / job / step**: in GitHub Actions, a workflow file contains jobs, which contain steps (commands).
- **Artifact**: a file the CI run saves for you to download: test reports, plots, metrics.
- **Headless**: running without any screen or graphics window, which is required because CI runners have no display. That's why the scripts use `matplotlib.use("Agg")`.

### 4.2 Walking through `.github/workflows/ci.yml`

```yaml
on:
  push:                  # run on every push
  pull_request:          # and on every PR
  schedule:
    - cron: "0 7 * * *"  # and every night at 07:00 UTC
  workflow_dispatch:     # and when you click "Run workflow"
```

Two jobs:

- **quick-tests** (every push/PR, about a minute): installs requirements, runs `pytest -m "not slow"`, saves the report as an artifact even if tests fail (`if: always()`).
- **full-suite** (nightly or on demand): runs *all* tests including the slow MuJoCo simulation tests, then runs the entire analysis pipeline headless and uploads the plots and `metrics.json`.

This **smoke vs. nightly split** is exactly how real teams handle compute-heavy tests: a developer shouldn't wait an hour for feedback on every commit, but the expensive checks still run every day.

### 4.3 The three kinds of checks (in `tests/`)

This is the core idea to explain to Bo. A simulation release gate needs:

1. **Determinism checks** (`test_deterministic`): same inputs and seed → bit-identical output. Without this, you can't tell whether a failure came from your change or from randomness.
2. **Absolute / physics checks**: things that must always be true regardless of tuning. "Fused position must beat raw GPS." "A left turn must increase yaw" (the Lab 5 bug). "Semi-implicit Euler's energy error must not grow over a 10× longer run" (`test_pendulum_sim.py`, the MuJoCo exercise turned into a test).
3. **Regression checks** (`test_no_regression_vs_baseline`): compare today's metrics to a stored baseline (`tests/baseline_metrics.json`) with a tolerance (10% here). Improvements are fine; getting worse fails. When a change is *supposed* to change the numbers, run `python scripts/update_baseline.py` and explain it in the commit message. Updating the baseline is a deliberate, reviewed decision, never automatic.

**Try the demo:** in `navlib/ekf.py`, change `gps_sigma: float = 2.5` to `25.0` and run `pytest -m "not slow"`. Two tests fail: fusion no longer beats raw GPS, and the regression gate catches the metric getting worse. Change it back and everything's green. That's CI catching a regression before it reaches hardware.

**Stochastic scenarios.** For things like RL policies that vary run to run, instead of one pass/fail you run N seeds and require a pass rate (for example, ≥ 95% success over 20 seeds). The Allan test's loose 35% tolerance on rate random walk is a small example of setting tolerances based on known statistical spread.

### 4.4 Putting it on GitHub (Sunday, about 30 minutes)

1. Make a free GitHub account if you don't have one; click **New repository**, name it `imu-nav-toolkit`, keep it empty.
2. In a terminal inside the folder:
   ```
   git init
   git add .
   git commit -m "IMU navigation toolkit with CI"
   git branch -M main
   git remote add origin https://github.com/<your-username>/imu-nav-toolkit.git
   git push -u origin main
   ```
3. Open the repo on GitHub → **Actions** tab. Watch quick-tests run and turn green.
4. Click **CI → Run workflow** to trigger the full suite, then download the `analysis-results` artifact.
5. Make a branch, break `gps_sigma`, open a pull request, and watch it go red. Screenshot that. It's a concrete thing you've *done*.

Make the repo public only if the README clearly says the data is synthetic (it does).

### 4.5 How this maps to Qualcomm's setup

The JD's "make simulation a release gate: scenario suites, deterministic replay, benchmarks, and metrics that catch regressions before they reach hardware" is this same pattern at huge scale. Scenario suites = many seeded simulated tasks instead of one drive. Benchmarks = tracked metrics over time (success rate, cycle time, sim steps per second). Argo Workflows on Kubernetes = the equivalent of GitHub Actions for GPU-heavy jobs, launching containerized simulation runs across a cluster. The concepts transfer; the scale and tooling don't yet, and you should say so.

**How to say it:** "I hadn't built CI before, so this weekend I set up a small one properly: pytest with three kinds of checks (determinism, physics invariants like frame conventions and integrator energy bounds, and metric regression against a stored baseline with a tolerance), running in GitHub Actions with a fast suite on every push and the slow MuJoCo tests nightly. I know Qualcomm's version runs on Argo and Kubernetes at a different scale, but the gating logic is the same."

---

## 5. Your resume and outreach claims: what Bo is likely to ask

**Important context.** Your LinkedIn message to Bo made several specific claims, and he has read them. Some go further than what you've previously described as your actual project scope. He's a simulation and IMU expert and will ask about exactly these. The strongest move, by far, is to **correct them yourself, calmly, before he catches them**. The JD literally values "honest limits," and a self-correction plus "and here's what I built to close the gap" is a strong answer. Getting caught defending an overclaim ends interviews.

Fill in every `[bracket]` with the truth before Monday.

### Claim 1: "Built a Simscape physics-based simulator (rigid-body dynamics, actuator models, encoder/IMU/camera sensor noise models characterised from real hardware data) used as a SIL release gate before hardware deployment"

*Likely questions:* Walk me through the simulator. What was the release gate? How did the sensor models get into Simscape?

*What's true (as you've described it before):* the DC Motor Control project was a Simscape model correlated against hardware to under 5% error, with no automated test suite. The IMU/camera characterization was separate coursework (EECE 5554 labs, EECE 5550 AprilTag calibration).

*Honest answer:* "Let me be precise, because my message compressed a few projects together. The Simscape work was a DC motor model: I modeled [back-EMF, winding resistance and inductance, rotor inertia, friction — list what you actually modeled] and correlated it against the real motor to within 5% on [what signal: speed step response? current?]. The sensor characterization, camera calibration and IMU/GPS, was separate coursework. 'Release gate' was too strong; there was no automated gate. We [used the model to check controller gains before running on hardware — only if true]."

*Prep:* know what "correlated to under 5%" means exactly. 5% of what, measured how? Which parameters did you tune to get there? (That's a system-identification story.)

### Claim 2: "Built Python regression suites integrated into GitHub CI/CD … <5% sim-to-hardware fidelity gap on state estimation"

*Likely questions:* What did the suite test? What triggered it? Show me a regression it caught.

*Honest answer:* "I want to correct that one. I hadn't actually built a CI pipeline in those projects, and I shouldn't have written it that way. After applying I built one properly: [describe section 4]. And the 5% figure was from the motor model correlation, not state estimation."

This is the hardest sentence to say and the most important one.

### Claim 3: "Deployed complete autonomous stacks (EKF, GTSAM visual-inertial, MPC motion planning, 6-DOF manipulation) from simulation through to real robot hardware — 95% task accuracy"

*What's true (as you've described it):* these were separate projects. MPC was model-in-the-loop simulation only. The UR5 work was feedforward torque via inverse dynamics, not closed-loop control on hardware. GTSAM was with AprilTag observations [check whether IMU was actually in the graph; "visual-inertial" implies it was]. Rebecca ran on real hardware at 27/30 successful picks per color, which is 90%, not 95%.

*Honest answer:* "Those were separate projects, and only some reached hardware. The one fully on real hardware is my pick-and-place system on the ReactorX-200: 27 of 30 successful picks per color, about 90%, with around 6 mm error in the inner workspace and 14 mm at the edges. The MPC work was simulation only, and the UR5 work computed feedforward torques from inverse dynamics."

Then steer to Rebecca, your strongest real story.

### Claim 4: "Stochastic policy evaluation environments with 27-configuration domain-randomisation-analogous DoE"

*Likely questions:* What were the 3 factors and 3 levels (3³ = 27)? What was the metric? What did you learn? How is this different from domain randomization?

*Key distinction:* DoE (design of experiments) *systematically* sweeps a fixed grid of conditions to measure sensitivity. Domain randomization *randomly samples* conditions during training so a policy becomes robust. Related ideas, different purposes. Say "related to" rather than "analogous to."

### Claim 5 (resume): "Allan-variance-analogous noise estimation" and "25% drift reduction" EKF

*Honest answer:* "In the lab I characterized noise from stationary periods, not a full Allan analysis. This weekend I implemented proper overlapping Allan deviation and validated it. [If you recorded your phone: and ran it on real phone IMU data: ARW was X.]" For the 25%: identify exactly where that number came from, what was compared against what, and on what hardware. If you can't, drop it.

### Claim 6 (resume): "<1 mm end-effector accuracy on deployed robot"

*Honest answer:* if it was a simulation or FK-consistency number, say so. Don't defend a hardware accuracy figure you didn't measure.

### Rebecca deep dive (your best story; know it cold)

- **Pipeline:** HSV color detection → AprilTag detection → camera-to-robot registration via a 2D similarity transform (rotation, translation, uniform scale in the table plane, fitted from known point correspondences) → IK (waist angle from atan2(y, x), IKinSpace Newton-Raphson for the rest) → trapezoidal velocity profile you wrote → Cartesian descent interpolation loop you wrote.
- **Be precise about what you didn't write:** the Interbotix `set_ee_pose_components()` linear interpolation (about 150 waypoints) and IKinSpace are library code.
- **Why error grows toward the edge (6 → 14 mm), hypotheses:** (1) lens distortion is strongest at image edges and a 2D similarity transform doesn't model it; (2) small servos sag more under gravity at full reach; (3) small joint errors are amplified by the longer lever arm. *How you'd separate them:* recalibrate with a distortion model and see how much outer error remains; command the arm to known points without the camera to isolate arm error.
- **Sim connection:** "If I built a digital twin of this cell, I'd validate it against exactly this: the same pick targets in sim and real, compare the error maps across the workspace."

### Location

Your message to Bo said "San Diego local," while you've described yourself as Boston-based. Make sure your answer to "where are you now / when could you start on-site" is accurate and consistent with what he read.

---

## 6. Other likely questions

**"Tell me about yourself" (60 seconds).** Throughline: making models match reality and proving it. Collins (FEA held to FAA certification) → Fresenius (hardware V&V and optical sensor validation under ISO 13485) → MS Robotics (estimation, perception, control on real hardware) → simulation is where these meet. End with: "This weekend I rebuilt one of my navigation labs into a tested toolkit with CI, because I wanted to show you how I work, not just tell you."

**Fresenius:** have one specific story, a sensor or subsystem that didn't meet spec, how you found out, what you did. Be precise about what you personally owned.

**Collins:** "What does fatigue analysis have to do with simulation?" Both are about trusting a model enough to make a safety decision, and knowing its margins: FEA mesh convergence ≈ simulation timestep/convergence checks; correlating models to test data ≈ sim-to-real validation.

**C++ fundamentals** (your pendulum is C++; he's a modern C++ person):
- `std::array` vs `std::vector`: fixed size on the stack, known at compile time, vs. dynamic size on the heap. For a fixed 4-element state, `std::array`.
- `unique_ptr` vs `shared_ptr`: single owner, zero overhead, vs. shared ownership with reference counting. Default to unique.
- Why virtual destructors in base classes: deleting through a base pointer must call the derived destructor.
- Pass by `const&` for large objects to avoid copies.
- Be ready to walk through your pendulum code structure: how the integrator, dynamics, and controller are separated.

**"How do you use AI tools?"** A specific story with a failure in it: "Claude helped me write the ellipse-fit and EKF code this weekend, and I verified each piece with a test against known ground truth. At one point [real example: e.g. the Allan code crashed on pure white noise because there was no rising region; the test exposed it and I fixed the edge case]." What he wants to hear is *verification*, not delegation. (That crash really happened while building this repo.)

**"Converge two code paths rather than add a third" (from the JD).** Example in this repo: scripts and tests both call `navlib/evaluation.py`, so the metric CI checks is the same metric the plots show. Duplicated metric code drifts apart and then CI checks something slightly different from what you report.

**"Describe a bug you're proud of finding."** The Lab 5 frame-convention mismatch: symptom (mirror-image yaw), diagnosis (z-down vs. z-up), fix (write conventions first), prevention (unit test).

**"Why simulation, when your background is controls and hardware?"** Every hardware project you've done was bottlenecked by hardware time and by not knowing whether the model could be trusted. Simulation is where you make that scale.

**Expectations:** level (open to his calibration; mid-level range of the posting if asked), relocation (consistent answer!), start date, and a compensation number decided in advance.

**Questions to ask him:** which subsystem needs someone most (physics backends, sensor models, HIL, tooling); what the first six months looks like for someone at the earlier end of the range; which layers are settled vs. still being validated; how sensor-team characterization data flows into the sensor models; how the team is split across sites.

---

## 7. Numbers cheat sheet

| Thing | Number |
|---|---|
| Your Lab 5 hard-iron center | (0.17, 0.05) gauss |
| Your Lab 5 ellipse | 0.0889 / 0.0733 gauss, tilt 0.4644 rad (26.6°), ratio 0.824 |
| Complementary filter | α = 0.99 @ 40 Hz → τ ≈ 2.5 s → crossover ≈ 0.064 Hz |
| Accel bias → position | error = ½·b·t²; 0.01 m/s² → 0.5 m at 10 s, 18 m at 60 s |
| Road grade | 1° → g·sin(1°) ≈ 0.17 m/s² fake acceleration |
| White noise per sample | σ = density × √(sample rate) |
| Allan slopes (deviation plot) | −½ white noise (read at τ=1 s), 0 bias instability (÷0.664), +½ rate random walk (read at τ=3 s) |
| Rebecca | 27/30 per color (90%), ~6 mm inner / ~14 mm outer RMSE |
| Pendulum integrators (MuJoCo, dt=0.01) | Euler energy error ≈ 0.097 J, bounded; RK4 ≈ 2.5e-5 J at 60 s, slowly growing |
