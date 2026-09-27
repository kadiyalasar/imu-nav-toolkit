import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")          # write PNGs, no window needed (works in CI too)
from navlib.io import RESULTS
RESULTS.mkdir(exist_ok=True)
