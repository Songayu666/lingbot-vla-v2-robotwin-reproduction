#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_stage(stage: Path):
    result = {}
    for path in sorted((stage / "results").glob("*/episodes.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        result[path.parent.name] = (sum(bool(row["success"]) for row in rows), len(rows))
    return result


parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
stages = {name: load_stage(args.run_dir / name) for name in ("chunk50", "chunk25", "chunk10")}
tasks = sorted(set().union(*(stage.keys() for stage in stages.values())))
lines = ["task\tchunk50\tchunk25\tchunk10"]
for task in tasks:
    cells = []
    for name in ("chunk50", "chunk25", "chunk10"):
        success, total = stages[name].get(task, (0, 0))
        cells.append(f"{success}/{total}")
    lines.append("\t".join((task, *cells)))
text = "\n".join(lines) + "\n"
if args.output:
    args.output.write_text(text, encoding="utf-8")
print(text, end="")
