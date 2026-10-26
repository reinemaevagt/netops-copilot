"""Configuration pytest : rend le paquet ``src/netops_copilot`` importable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
