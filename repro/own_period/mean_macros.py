"""Recover the deleted M1 input list from its surviving scope and paper definitions.

The archived records preserve the complete 183-item M1/M1b scope: 93
original inputs (including seven deliberate exclusions) and 90 additions.
Further printed summaries are audited by median_stats.audit_manuscript.
Neither source is edited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from repro import network_root as network
from repro.own_period import DataPath


def build(published, manuscript):
    saved = json.loads(Path(published).read_text())
    definitions = dict(
        re.findall(r"^\\newcommand\{\\(\w+)\}\{(.*)\}$", Path(manuscript).read_text(), re.M)
    )
    rows = [r for r in saved["entries"] + saved["skipped"]]
    if len({r["mean_macro"] for r in rows}) != len(rows):
        raise ValueError("duplicate original mean-macro scope")
    lines = []
    for row in rows:
        name = row["mean_macro"]
        value = definitions.get(name, row["paper_numeric_text"])
        numeric = re.match(r"[-+]?\d+(?:\.\d+)?", value.replace(r"\ensuremath{-}", "-")).group()
        if numeric != row["paper_numeric_text"]:
            raise ValueError(f"manuscript original mean changed: {name}")
        comment = row["input_source_comment"]
        lines.append(f"{name} = {value}" + (f"    % {comment}" if comment else ""))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published", type=Path, default=DataPath("median-stats.json"))
    parser.add_argument(
        "--manuscript", type=Path, default=DataPath("../../../paper-rev-20261021/sample701.tex")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    network.guard_output(args.output)
    content = build(args.published, args.manuscript)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(
            dict(
                scope="183 original M1/M1b records, including deliberate exclusions; M1b additions audited from the manuscript",
                sources={str(p): network.sha256(p) for p in (args.published, args.manuscript)},
                output_sha256=network.sha256(args.output),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
