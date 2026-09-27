"""Overlapping Allan deviation and the three standard IMU noise terms.

Recipe (IEEE Std 952 style):
  1. theta = cumulative integral of the rate signal
  2. for each cluster time tau = m*dt:
        sigma^2(tau) = sum (theta[k+2m] - 2 theta[k+m] + theta[k])^2 / (2 tau^2 (N - 2m))
  3. on a log-log plot of sigma vs tau, read off:
        slope -1/2 region -> white noise (ARW/VRW): value of that line at tau = 1 s
        slope  0  region  -> bias instability:       min sigma / 0.664
        slope +1/2 region -> rate random walk:       value of that line at tau = 3 s
"""
import numpy as np


def allan_deviation(rate, fs, n_taus=100, max_tau_fraction=0.1):
    """Longest tau = max_tau_fraction * record length: beyond ~1/10 of the record there are
    too few independent clusters and the estimate becomes statistically meaningless."""
    rate = np.asarray(rate, dtype=float)
    dt = 1.0 / fs
    n = len(rate)
    theta = np.concatenate([[0.0], np.cumsum(rate) * dt])
    m_max = int(max_tau_fraction * n)
    ms = np.unique(np.logspace(0, np.log10(m_max), n_taus).astype(int))
    taus, adev = [], []
    for m in ms:
        tau = m * dt
        d = theta[2 * m:] - 2 * theta[m:-m] + theta[:-2 * m]
        adev.append(np.sqrt(np.sum(d**2) / (2 * tau**2 * (len(theta) - 2 * m))))
        taus.append(tau)
    return np.array(taus), np.array(adev)


def _line_through(taus, adev, slope, tau_eval):
    """Find the point where the local log-log slope is closest to `slope`,
    draw a line of that slope through it, return its value at tau_eval."""
    logt, loga = np.log10(taus), np.log10(adev)
    local = np.gradient(loga, logt)
    i = np.argmin(np.abs(local - slope))
    intercept = loga[i] - slope * logt[i]
    return 10 ** (intercept + slope * np.log10(tau_eval)), i


def noise_parameters(taus, adev):
    white, i_w = _line_through(taus, adev, -0.5, 1.0)
    i_b = int(np.argmin(adev))
    if len(taus) - i_b >= 3:                  # need a rising region right of the minimum
        rrw, i_r = _line_through(taus[i_b:], adev[i_b:], +0.5, 3.0)
        i_r += i_b
    else:                                     # record too short to see random walk
        rrw, i_r = float("nan"), None
    bias_instability = adev[i_b] / np.sqrt(2 * np.log(2) / np.pi)   # 0.664 factor
    return {
        "white_noise_density": white,        # ARW (gyro) or VRW (accel), units/s/sqrt(Hz)
        "bias_instability": bias_instability,
        "tau_bias_instability": taus[i_b],
        "rate_random_walk": rrw,
        "idx": {"white": i_w, "bias": i_b, "rrw": i_r},
    }
