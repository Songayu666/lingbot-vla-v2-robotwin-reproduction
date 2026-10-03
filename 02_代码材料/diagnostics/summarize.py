#!/usr/bin/env python3
import json
from pathlib import Path


root = Path(__file__).resolve().parents[2] / "outputs" / "diagnostics"
print("run\ttask\tsuccess/total\trate")
for run in sorted(root.glob("*_chunk*")):
    for path in sorted(run.glob("results/*/episodes.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        success = sum(bool(row["success"]) for row in rows)
        rate = 100 * success / len(rows) if rows else 0
        print(f"{run.name}\t{path.parent.name}\t{success}/{len(rows)}\t{rate:.1f}%")
