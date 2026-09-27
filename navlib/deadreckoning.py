"""Forward velocity from the accelerometer, with the Lab 5 bias strategies,
and position by integrating speed along a heading."""
import numpy as np

G = 9.81


def moving_average(x, window):
    return np.convolve(x, np.ones(window) / window, mode="same")


def detect_stationary(ax, ay, az, gz, fs, min_seconds=2.0, window_s=0.5,
                      acc_band=0.08, gyro_thresh=0.01):
    """Stationary = |accel| close to g AND almost no rotation, sustained for min_seconds.
    (Same idea as the Lab 5 detector: norm within 9.65-9.85 m/s^2 for >= 3 s.)
    Vibration from a running engine at a light is why we smooth before thresholding."""
    w = max(1, int(window_s * fs))
    norm = moving_average(np.sqrt(ax**2 + ay**2 + az**2), w)
    acc_var = moving_average((ax - moving_average(ax, w))**2, w)
    candidate = (np.abs(norm - G) < acc_band) & (np.abs(moving_average(gz, w)) < gyro_thresh) \
        & (acc_var < 0.01)
    # keep only runs that last long enough
    out = np.zeros_like(candidate)
    k, n = 0, len(candidate)
    while k < n:
        if candidate[k]:
            j = k
            while j < n and candidate[j]:
                j += 1
            if j - k >= min_seconds * fs:
                out[k:j] = True
            k = j
        else:
            k += 1
    return out


def velocity_naive(ax, fs):
    return np.cumsum(ax) / fs


def velocity_constant_bias(ax, fs):
    """Lab 5 first attempt: fit a straight line to the naively integrated velocity.
    A car that starts and ends near rest has ~zero average acceleration, so the
    slope of that line is taken as ONE constant bias and subtracted everywhere."""
    t = np.arange(len(ax)) / fs
    slope = np.polyfit(t, velocity_naive(ax, fs), 1)[0]
    return np.cumsum(ax - slope) / fs


def velocity_zupt(ax, fs, stationary):
    """Lab 5 method, cleaned up (a.k.a. zero-velocity update, ZUPT):
       * during each stop: re-estimate bias = mean(ax), and force v = 0
       * while moving: integrate (ax - latest bias); clamp v >= 0 (car never reverses)"""
    dt = 1.0 / fs
    v = np.zeros_like(ax)
    bias = 0.0
    k, n = 0, len(ax)
    while k < n:
        if stationary[k]:
            j = k
            while j < n and stationary[j]:
                j += 1
            bias = ax[k:j].mean()
            v[k:j] = 0.0
            k = j
        else:
            v[k] = max(0.0, (v[k - 1] if k else 0.0) + (ax[k] - bias) * dt)
            k += 1
    return v


def integrate_position(v, yaw, fs, x0=0.0, y0=0.0):
    """ENU, yaw from East CCW:  vE = v cos(psi),  vN = v sin(psi)."""
    dt = 1.0 / fs
    return x0 + np.cumsum(v * np.cos(yaw)) * dt, y0 + np.cumsum(v * np.sin(yaw)) * dt
