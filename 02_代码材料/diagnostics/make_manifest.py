#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build a fixed episode manifest from a diagnostic run")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    files = sorted((args.source / "results").glob("*/episodes.jsonl"))
    wanted = set(args.tasks or [])
    manifest = {}
    for path in files:
        task = path.parent.name
        if wanted and task not in wanted:
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.limit:
            rows = rows[:args.limit]
        manifest[task] = [{"seed": row["seed"], "instruction": row["instruction"]} for row in rows]

    if wanted - set(manifest):
        raise SystemExit(f"Missing tasks: {sorted(wanted - set(manifest))}")
    if not manifest:
        raise SystemExit("No episode records found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {sum(map(len, manifest.values()))} fixed episodes for {len(manifest)} tasks to {args.output}")


if __name__ == "__main__":
    main()
