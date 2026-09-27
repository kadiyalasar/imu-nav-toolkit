"""The test that would have caught the Lab 5 bug: every heading source must agree on
which way is a LEFT turn. Frame bugs are silent -- the code runs, the plots just look odd."""
import numpy as np
from navlib import synthetic, heading, deadreckoning as dr

FS = 40.0


def left_turn(seconds=5.0, rate=np.deg2rad(10)):
    n = int(seconds * FS)
    psi = np.deg2rad(30) + rate * np.arange(n) / FS
    return psi, np.full(n, rate)


def test_gyro_and_magnetometer_agree_on_left_turn():
    psi, gyro_z = left_turn()
    dist = synthetic.MagDistortion(center=(0, 0), major=1, minor=1, tilt=0, noise=0)
    mx, my = synthetic.mag_from_yaw(psi, dist, np.random.default_rng(0))
    d_mag = np.unwrap(heading.mag_yaw(mx, my))[-1] - heading.mag_yaw(mx, my)[0]
    d_gyro = heading.gyro_yaw(gyro_z, FS)[-1]
    assert d_mag > 0 and d_gyro > 0, "left turn must INCREASE yaw for every source"
    assert abs(d_mag - d_gyro) < np.deg2rad(1)


def test_yaw_90_deg_moves_north():
    v = np.full(40, 1.0)                       # 1 m/s for 1 s
    x, y = dr.integrate_position(v, np.full(40, np.pi / 2), FS)
    assert abs(x[-1]) < 1e-9 and abs(y[-1] - 1.0) < 1e-9


def test_wrap_range():
    a = heading.wrap(np.array([-4 * np.pi, -np.pi, 0.0, np.pi, 3.5 * np.pi]))
    assert np.all(a >= -np.pi) and np.all(a < np.pi)
