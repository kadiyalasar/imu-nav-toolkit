"""Loosely coupled GPS / IMU / magnetometer Extended Kalman Filter for a car on flat ground.

State  x = [px, py, v, psi, b_a, b_g]
         position East/North [m], forward speed [m/s], yaw [rad],
         accelerometer bias [m/s^2], gyro bias [rad/s]
Inputs (drive the PREDICT step, 40 Hz): forward accel a_m, yaw rate w_m
Measurements (drive the UPDATE step):
         GPS position   (1 Hz)           z = [px, py]
         magnetometer yaw (10 Hz)        z = psi
         zero-velocity when stationary   z = v = 0     (ZUPT, from Lab 5 idea)

Why bias states matter: a complementary filter can only low-pass the magnetometer;
the EKF *estimates* the gyro and accel biases, so when GPS drops out the IMU
prediction is already corrected and drifts much more slowly.
"""
from dataclasses import dataclass
import numpy as np
from .heading import wrap


@dataclass
class EkfConfig:
    accel_noise: float = 0.3        # m/s^2 per sample -- includes vibration, not just sensor noise
    gyro_noise: float = 0.005       # rad/s per sample
    accel_bias_rw: float = 3e-2     # m/s^2 / sqrt(s): large on purpose -- road grade looks like a fast-changing bias
    gyro_bias_rw: float = 1e-5      # rad/s / sqrt(s)
    gps_sigma: float = 2.5          # m
    mag_sigma: float = np.deg2rad(4.0)
    zupt_sigma: float = 0.02        # m/s
    mag_rate: float = 10.0          # Hz


class NavEKF:
    N = 6

    def __init__(self, x0, P0, cfg: EkfConfig):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.cfg = cfg

    # ---------- predict: push the state forward with the IMU ----------
    def predict(self, a_m, w_m, dt):
        px, py, v, psi, ba, bg = self.x
        c, s = np.cos(psi), np.sin(psi)
        self.x = np.array([px + v * c * dt,
                           py + v * s * dt,
                           max(0.0, v + (a_m - ba) * dt),     # car never reverses
                           wrap(psi + (w_m - bg) * dt),
                           ba, bg])
        F = np.eye(self.N)                    # Jacobian d(new state)/d(old state)
        F[0, 2] = c * dt;  F[0, 3] = -v * s * dt
        F[1, 2] = s * dt;  F[1, 3] = v * c * dt
        F[2, 4] = -dt
        F[3, 5] = -dt
        cfg = self.cfg
        Q = np.diag([0, 0,
                     (cfg.accel_noise * dt) ** 2,
                     (cfg.gyro_noise * dt) ** 2,
                     cfg.accel_bias_rw ** 2 * dt,
                     cfg.gyro_bias_rw ** 2 * dt])
        self.P = F @ self.P @ F.T + Q

    # ---------- update: correct with a measurement ----------
    def update(self, z, h, H, R, angle_rows=()):
        y = np.atleast_1d(z - h)                       # innovation
        for r in angle_rows:                           # yaw residual must be wrapped!
            y[r] = wrap(y[r])
        S = H @ self.P @ H.T + R                       # innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)            # Kalman gain
        self.x = self.x + K @ y
        self.x[3] = wrap(self.x[3])
        I_KH = np.eye(self.N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form: stays symmetric & PSD
        return y, S

    def update_gps(self, gx, gy):
        H = np.zeros((2, self.N)); H[0, 0] = H[1, 1] = 1
        return self.update(np.array([gx, gy]), self.x[:2], H, np.eye(2) * self.cfg.gps_sigma**2)

    def update_mag(self, yaw):
        H = np.zeros((1, self.N)); H[0, 3] = 1
        return self.update(np.array([yaw]), self.x[3:4], H, np.array([[self.cfg.mag_sigma**2]]),
                           angle_rows=(0,))

    def update_zupt(self):
        H = np.zeros((1, self.N)); H[0, 2] = 1
        return self.update(np.array([0.0]), self.x[2:3], H, np.array([[self.cfg.zupt_sigma**2]]))


def run_ekf(t, accel_x, gyro_z, yaw_mag, gps_x, gps_y, stationary, fs,
            cfg: EkfConfig = EkfConfig(), gps_available=None):
    """Run the filter over a whole log. gps_available (bool array) lets you simulate outages."""
    n = len(t); dt = 1.0 / fs
    first_fix = np.flatnonzero(~np.isnan(gps_x))[0]
    x0 = [gps_x[first_fix], gps_y[first_fix], 0.0, yaw_mag[0], 0.0, 0.0]
    P0 = np.diag([cfg.gps_sigma**2, cfg.gps_sigma**2, 0.5**2, np.deg2rad(10)**2, 0.1**2, 0.01**2])
    ekf = NavEKF(x0, P0, cfg)
    mag_every = int(fs / cfg.mag_rate)
    if gps_available is None:
        gps_available = np.ones(n, dtype=bool)

    X = np.zeros((n, 6)); Pdiag = np.zeros((n, 6))
    for k in range(n):
        if k > 0:
            ekf.predict(accel_x[k], gyro_z[k], dt)
        if gps_available[k] and not np.isnan(gps_x[k]):
            ekf.update_gps(gps_x[k], gps_y[k])
        if k % mag_every == 0:
            ekf.update_mag(yaw_mag[k])
        if stationary[k]:
            ekf.update_zupt()
        X[k] = ekf.x; Pdiag[k] = np.diag(ekf.P)
    return X, Pdiag
