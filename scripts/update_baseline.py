"""Record current metrics as the new regression baseline. Run this ONLY when a change is
supposed to alter the numbers (e.g. you improved the filter) -- and say so in the commit."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from navlib import evaluation, ekf  # noqa: E402
from test_ekf_regression import metrics, BASELINE  # noqa: E402

d = evaluation.build_inputs(seed=2)
X, _ = ekf.run_ekf(d["t"], d["accel_x"], d["gyro_z"], d["yaw_mag"], d["gps_x"], d["gps_y"],
                   d["stationary"], evaluation.FS)
m = metrics(d, X)
BASELINE.write_text(json.dumps(m, indent=2))
print("New baseline:", json.dumps(m, indent=2))
