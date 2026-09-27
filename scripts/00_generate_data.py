"""Step 0: create synthetic CSVs in data/ that mimic the Lab 5 setup.
  stationary.csv : IMU still on a table for 2 hours (for Allan deviation)
  circles.csv    : 5 calibration circles (for magnetometer calibration)
  drive.csv      : ~17 min city drive with stops, turns, road grade, 1 Hz GPS
Truth columns (true_*) exist ONLY because the data is synthetic -- real logs won't have them."""
import _setup  # noqa: F401
from navlib import synthetic
from navlib.io import save_csv

save_csv("stationary.csv", synthetic.stationary_imu(hours=2.0))
save_csv("circles.csv", synthetic.calibration_circles())
save_csv("drive.csv", synthetic.drive())
print("Wrote data/stationary.csv, data/circles.csv, data/drive.csv")
