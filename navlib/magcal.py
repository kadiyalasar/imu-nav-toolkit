"""Hard- and soft-iron magnetometer calibration by ellipse fitting.

Model:  m_raw = A h + c      (h = true horizontal field, a unit circle as the vehicle turns)
          c = hard-iron offset (constant magnetized parts of the car)
          A = soft-iron matrix (steel bending the field -> circle becomes a tilted ellipse)
Fix:    h = W^(1/2) (m_raw - c), where W comes from the fitted ellipse.
W^(1/2) rotates to the ellipse axes, rescales each axis, AND rotates back -- skipping
the "rotate back" step leaves every heading offset by the tilt angle.
"""
import numpy as np


def fit_ellipse(x, y):
    """Algebraic conic fit  a x^2 + b xy + c y^2 + d x + e y + f = 0  via SVD."""
    D = np.column_stack([x**2, x * y, y**2, x, y, np.ones_like(x)])
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    a, b, c, d, e, f = Vt[-1]
    center = np.linalg.solve([[2 * a, b], [b, 2 * c]], [-d, -e])
    cx, cy = center
    f0 = a * cx**2 + b * cx * cy + c * cy**2 + d * cx + e * cy + f   # conic value at center
    M = np.array([[a, b / 2], [b / 2, c]])
    W = M / (-f0)                                    # ellipse:  u^T W u = 1
    evals, evecs = np.linalg.eigh(W)                 # ascending: first = major axis
    axes = 1 / np.sqrt(evals)
    tilt = np.arctan2(evecs[1, 0], evecs[0, 0])
    tilt = (tilt + np.pi / 2) % np.pi - np.pi / 2    # an axis direction: keep in [-90, 90) deg
    return {"center": center, "W": W, "major": axes[0], "minor": axes[1], "tilt": tilt}


def correction_matrix(fit):
    """Symmetric square root of W: maps the ellipse back onto the unit circle."""
    evals, evecs = np.linalg.eigh(fit["W"])
    return evecs @ np.diag(np.sqrt(evals)) @ evecs.T


def calibrate(mx, my, fit):
    C = correction_matrix(fit)
    u = np.vstack([mx - fit["center"][0], my - fit["center"][1]])
    h = C @ u
    return h[0], h[1]
