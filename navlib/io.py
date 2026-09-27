"""CSV in/out. Your REAL data just needs the same column names to run through every script."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def save_csv(name, d):
    DATA.mkdir(exist_ok=True)
    keys = list(d.keys())
    arr = np.column_stack([np.asarray(d[k], dtype=float) for k in keys])
    np.savetxt(DATA / name, arr, delimiter=",", header=",".join(keys), comments="", fmt="%.9g")


def load_csv(name):
    path = DATA / name
    if not path.exists():
        raise SystemExit(f"{path} not found -- run  python scripts/00_generate_data.py  first "
                         f"(or put your real CSV there with the same column names).")
    arr = np.genfromtxt(path, delimiter=",", names=True)
    return {k: np.asarray(arr[k]) for k in arr.dtype.names}
