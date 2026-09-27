import numpy as np
from navlib import synthetic, magcal, heading


def test_ellipse_fit_recovers_distortion():
    dist = synthetic.MagDistortion()
    c = synthetic.calibration_circles(dist=dist, seed=5)
    fit = magcal.fit_ellipse(c["mag_x"], c["mag_y"])
    assert np.allclose(fit["center"], dist.center, atol=2e-3)
    assert abs(fit["major"] / dist.major - 1) < 0.03
    assert abs(fit["minor"] / dist.minor - 1) < 0.03
    assert abs(fit["tilt"] - dist.tilt) < np.deg2rad(2)


def test_calibrated_heading_accuracy():
    c = synthetic.calibration_circles(seed=6)
    fit = magcal.fit_ellipse(c["mag_x"], c["mag_y"])
    hx, hy = magcal.calibrate(c["mag_x"], c["mag_y"], fit)
    assert abs(np.mean(np.hypot(hx, hy)) - 1) < 0.01
    err = heading.wrap(heading.mag_yaw(hx, hy) - c["true_yaw"])
    assert np.rad2deg(np.sqrt(np.mean(err**2))) < 4.0
