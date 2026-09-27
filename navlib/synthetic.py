"""Synthetic sensor data generator.

WHY THIS EXISTS: to test analysis code against data where the TRUE answer is
known. If your Allan-deviation code is fed noise with a known random-walk
coefficient and gives that number back, you can trust it on real data.

IMPORTANT: numbers produced from this data are NOT hardware results.
"""
from dataclasses import dataclass
import numpy as np

G = 9.81


@dataclass
class ImuErrorModel:
    """Error parameters for one sensor axis (roughly VN-100-class values;
    check the real datasheet before quoting any of them)."""
    white_density: float       # noise density: rad/s/sqrt(Hz) (gyro) or m/s^2/sqrt(Hz) (accel)
    gm_sigma: float            # Gauss-Markov bias std dev -> shows up as "bias instability"
    gm_tau: float              # Gauss-Markov correlation time [s]
    rrw: float                 # rate random walk coefficient: units/s/sqrt(s)
    turn_on_bias: float = 0.0  # constant bias for this power-up


GYRO_MODEL = ImuErrorModel(white_density=6.1e-5, gm_sigma=1.2e-5, gm_tau=60.0, rrw=3.0e-6,
                           turn_on_bias=2.0e-3)
ACCEL_MODEL = ImuErrorModel(white_density=1.4e-3, gm_sigma=2.5e-4, gm_tau=60.0, rrw=6.0e-5,
                            turn_on_bias=0.05)


def imu_errors(n, fs, model: ImuErrorModel, rng):
    """Return bias(t) and white noise sequences of length n.

    white noise   : per-sample sigma = density * sqrt(fs)   (= density / sqrt(dt))
    Gauss-Markov  : b[k+1] = exp(-dt/tau) b[k] + w   -> bias wanders but stays bounded
    random walk   : b[k+1] = b[k] + rrw*sqrt(dt)*w   -> unbounded slow drift
    """
    dt = 1.0 / fs
    white = model.white_density * np.sqrt(fs) * rng.standard_normal(n)

    phi = np.exp(-dt / model.gm_tau)
    drive = model.gm_sigma * np.sqrt(1 - phi**2) * rng.standard_normal(n)
    gm = np.empty(n)
    gm[0] = model.gm_sigma * rng.standard_normal()
    for k in range(1, n):
        gm[k] = phi * gm[k - 1] + drive[k]

    rw = np.cumsum(model.rrw * np.sqrt(dt) * rng.standard_normal(n))
    return model.turn_on_bias + gm + rw, white


def stationary_imu(hours=2.0, fs=40.0, seed=0, axes="xyz"):
    """IMU sitting still on a table: gyro reads only its errors, accel_z reads g + errors."""
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 * fs)
    out = {"t": np.arange(n) / fs}
    for axis in axes:
        b, w = imu_errors(n, fs, GYRO_MODEL, rng)
        out[f"gyro_{axis}"] = b + w
        b, w = imu_errors(n, fs, ACCEL_MODEL, rng)
        out[f"accel_{axis}"] = b + w + (G if axis == "z" else 0.0)
    return out


# ---------- magnetometer distortion (numbers taken from the Lab 5 report) ----------
@dataclass
class MagDistortion:
    center: tuple = (0.17, 0.05)   # hard-iron offset [gauss]
    major: float = 0.0889          # ellipse semi-major axis [gauss]
    minor: float = 0.0733          # ellipse semi-minor axis [gauss]
    tilt: float = 0.4644           # major-axis angle [rad]
    noise: float = 0.003           # per-sample noise [gauss]

    def matrix(self):
        """A = R(tilt) diag(major, minor) R(tilt)^T maps the unit circle onto the ellipse."""
        c, s = np.cos(self.tilt), np.sin(self.tilt)
        R = np.array([[c, -s], [s, c]])
        return R @ np.diag([self.major, self.minor]) @ R.T


def mag_from_yaw(psi, dist: MagDistortion, rng):
    """Horizontal magnetometer reading in the body frame for yaw psi.

    Magnetic north in the world is [0, 1] (declination ignored).
    In the body frame that vector is R(psi)^T [0, 1] = [sin(psi), cos(psi)],
    so an ideal sensor gives psi = atan2(mx, my).
    """
    h = np.vstack([np.sin(psi), np.cos(psi)])
    m = dist.matrix() @ h + np.array(dist.center)[:, None]   # soft iron, then hard iron
    m += dist.noise * rng.standard_normal(m.shape)
    return m[0], m[1]


def calibration_circles(n_circles=5, seconds_per_circle=40.0, fs=40.0, seed=1,
                        dist: MagDistortion = MagDistortion()):
    """Driving in circles (like Ruggles Circle): yaw sweeps through 360 deg repeatedly."""
    rng = np.random.default_rng(seed)
    n = int(n_circles * seconds_per_circle * fs)
    t = np.arange(n) / fs
    psi = 2 * np.pi * t / seconds_per_circle
    mx, my = mag_from_yaw(psi, dist, rng)
    return {"t": t, "mag_x": mx, "mag_y": my, "true_yaw": psi}


# ---------- a ~19-minute city drive with stops, turns and 1 Hz GPS ----------
def _drive_profile(fs, rng, legs):
    v_parts, r_parts = [], []

    def hold(seconds, v, r=0.0):
        n = int(seconds * fs)
        v_parts.append(np.full(n, v)); r_parts.append(np.full(n, r))

    def ramp(seconds, v0, v1):
        n = int(seconds * fs)
        v_parts.append(np.linspace(v0, v1, n)); r_parts.append(np.zeros(n))

    hold(15, 0.0)                                        # start at rest
    for _ in range(legs):
        cruise = rng.uniform(8, 14)                      # m/s (~18-31 mph)
        ramp(8, 0.0, cruise)
        hold(rng.uniform(20, 45), cruise)
        angle = rng.choice([-1, 1]) * np.deg2rad(rng.choice([45, 90]))
        rate = np.deg2rad(12)
        hold(abs(angle) / rate, cruise, np.sign(angle) * rate)   # the turn
        hold(rng.uniform(10, 25), cruise)
        ramp(6, cruise, 0.0)
        hold(rng.uniform(6, 15), 0.0)                    # traffic light: stationary period
    return np.concatenate(v_parts), np.concatenate(r_parts)


def road_grade(n, fs, rng, sigma_deg=1.0, corr_s=40.0):
    """Smoothly varying road pitch. Gravity leaks into accel_x as g*sin(pitch):
    1 deg of grade = 0.17 m/s^2 of fake acceleration -- bigger than the sensor bias."""
    white = rng.standard_normal(n)
    k = np.exp(-np.arange(int(4 * corr_s * fs)) / (corr_s * fs))
    g = np.convolve(white, k, mode="same")
    return np.deg2rad(sigma_deg) * g / g.std()


def drive(fs=40.0, gps_rate=1.0, gps_sigma=2.5, seed=2, legs=13, grade_deg=1.0,
          dist: MagDistortion = MagDistortion(), initial_yaw=np.deg2rad(60)):
    """Simulated car drive: truth + IMU (fs Hz) + GPS (gps_rate Hz, NaN between fixes)."""
    rng = np.random.default_rng(seed)
    v, r = _drive_profile(fs, rng, legs)
    n = len(v); dt = 1.0 / fs
    t = np.arange(n) * dt

    psi = initial_yaw + np.cumsum(r) * dt
    x = np.cumsum(v * np.cos(psi)) * dt
    y = np.cumsum(v * np.sin(psi)) * dt
    a_fwd = np.gradient(v, dt)
    a_lat = v * r                                        # centripetal, to the left when r > 0

    bg, wg = imu_errors(n, fs, GYRO_MODEL, rng)
    bax, wax = imu_errors(n, fs, ACCEL_MODEL, rng)
    bay, way = imu_errors(n, fs, ACCEL_MODEL, rng)
    moving = v > 0.1
    vib = 0.15 * rng.standard_normal(n) * moving         # road/engine vibration while moving

    mx, my = mag_from_yaw(psi, dist, rng)
    pitch = road_grade(n, fs, rng, grade_deg) if grade_deg > 0 else np.zeros(n)

    gps_x = np.full(n, np.nan); gps_y = np.full(n, np.nan)
    idx = np.arange(0, n, int(fs / gps_rate))
    gps_x[idx] = x[idx] + gps_sigma * rng.standard_normal(len(idx))
    gps_y[idx] = y[idx] + gps_sigma * rng.standard_normal(len(idx))

    return {
        "t": t, "true_x": x, "true_y": y, "true_v": v, "true_yaw": psi,
        "accel_x": a_fwd + G * np.sin(pitch) + bax + wax + vib,
        "accel_y": a_lat + bay + way + vib,
        "accel_z": G * np.cos(pitch) + 0.02 * rng.standard_normal(n) + vib,
        "gyro_z": r + bg + wg,
        "mag_x": mx, "mag_y": my,
        "gps_x": gps_x, "gps_y": gps_y,
        "true_gyro_bias": bg, "true_accel_bias": bax, "true_pitch": pitch,
    }
