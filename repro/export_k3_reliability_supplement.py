from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lc_pipeline.k3.reliability_supplement import ReliabilitySupplementError, export

p = argparse.ArgumentParser()
p.add_argument("--run-root", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument(
    "--mpc-table", type=Path, help="Optional override for the source dump's asteroids.csv"
)
a = p.parse_args()
try:
    print(json.dumps(export(a.run_root, a.output, a.mpc_table), sort_keys=True))
except (OSError, ReliabilitySupplementError) as e:
    p.error(str(e))
