"""Validate the Allan-deviation code against noise whose parameters we KNOW."""
import numpy as np
from navlib import allan

FS = 100.0


def test_recovers_white_noise_density():
    rng = np.random.default_rng(0)
    density = 5e-4
    x = density * np.sqrt(FS) * rng.standard_normal(int(20 * 60 * FS))   # 20 minutes
    taus, adev = allan.allan_deviation(x, FS)
    p = allan.noise_parameters(taus, adev)
    assert abs(p["white_noise_density"] / density - 1) < 0.05


def test_recovers_rate_random_walk():
    rng = np.random.default_rng(1)
    K = 1e-3
    x = np.cumsum(K * np.sqrt(1 / FS) * rng.standard_normal(int(60 * 60 * FS)))  # 1 hour
    taus, adev = allan.allan_deviation(x, FS)
    p = allan.noise_parameters(taus, adev)
    assert abs(p["rate_random_walk"] / K - 1) < 0.35   # long-tau terms are statistically noisy


def test_white_noise_slope_is_minus_half():
    rng = np.random.default_rng(2)
    taus, adev = allan.allan_deviation(rng.standard_normal(200_000), FS)
    slope = np.polyfit(np.log10(taus[:20]), np.log10(adev[:20]), 1)[0]
    assert abs(slope + 0.5) < 0.05
