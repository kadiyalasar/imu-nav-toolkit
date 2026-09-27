"""Heading (yaw) from magnetometer, gyro, and a complementary filter.
Convention: yaw from East, CCW positive; gyro_z > 0 = left turn (see navlib/__init__.py)."""
import numpy as np


def wrap(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def mag_yaw(hx, hy):
    """Body-frame north vector is [sin(psi), cos(psi)]  ->  psi = atan2(hx, hy)."""
    return np.arctan2(hx, hy)


def gyro_yaw(gyro_z, fs, yaw0=0.0):
    return yaw0 + np.cumsum(gyro_z) / fs


def complementary(gyro_z, yaw_mag, fs, alpha=0.99):
    """psi[k] = psi[k-1] + w dt  +  (1 - alpha) * wrap(psi_mag - that prediction)

    Equivalent to alpha*(psi + w dt) + (1-alpha)*psi_mag, but blending the
    *wrapped difference* avoids the 359 deg vs 1 deg averaging bug.
    Time constant tau = alpha*dt/(1-alpha); crossover f_c = 1/(2*pi*tau).
    """
    dt = 1.0 / fs
    out = np.empty_like(yaw_mag)
    out[0] = yaw_mag[0]
    for k in range(1, len(yaw_mag)):
        pred = out[k - 1] + gyro_z[k] * dt
        out[k] = pred + (1 - alpha) * wrap(yaw_mag[k] - pred)
    return out


def crossover_hz(alpha, fs):
    tau = alpha * (1.0 / fs) / (1 - alpha)
    return 1.0 / (2 * np.pi * tau)
