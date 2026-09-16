from __future__ import annotations

import argparse
import json
from pathlib import Path

from lc_pipeline.k3.internal_validation_reporting import export

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--archive", type=Path)
p.add_argument("--archive-sha256")
p.add_argument("--output", type=Path)
a = p.parse_args()
print(
    json.dumps(
        export(root=a.root, archive=a.archive, archive_sha256=a.archive_sha256, output=a.output),
        sort_keys=True,
    )
)
