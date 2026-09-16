"""Small dependency-free helpers used by auditable file contracts.

This module intentionally has no numerical or model imports.  In particular,
the external convex-inversion adapter can use it without importing the neural
data stack.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file, reading it in bounded chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    """Serialize JSON with the stable settings used by the V2 contracts."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
