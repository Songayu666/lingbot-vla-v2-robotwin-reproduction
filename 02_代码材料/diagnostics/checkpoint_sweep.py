#!/usr/bin/env python3
"""Serial fixed-scene checkpoint screening; preserves per-run logs and videos."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HARD = set("adjust_bottle hanging_mug lift_pot move_playingcard_away open_laptop pick_dual_bottles place_container_plate".split())

def aggregate(directory, manifest):
    rows = []
    for task, expected in manifest.items():
        path = directory / "results" / task / "episodes.jsonl"
        records = [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
        if len(records) != len(expected):
            raise ValueError(f"{task}: expected {len(expected)}, found {len(records)}")
        for actual, reference in zip(records, expected):
            if any(actual[k] != reference[k] for k in ("seed", "instruction")):
                raise ValueError(f"{task}: episode identity mismatch")
        rows.append(dict(task=task, successes=sum(r["success"] is True for r in records), total=len(records)))
    return dict(tasks=rows, successes=sum(r["successes"] for r in rows),
                total=sum(r["total"] for r in rows),
                hard_successes=sum(r["successes"] for r in rows if r["task"] in HARD))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-offset", type=int, default=0, help="Skip screening episodes for validation")
    parser.add_argument("--steps", nargs="+", type=int, default=[5000,10000,8000,9000,7000,6500])
    parser.add_argument("--chunks", nargs="+", type=int, default=[10])
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0: parser.error("episodes must be positive")
    if args.episode_offset < 0: parser.error("episode-offset must be nonnegative")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "sweep.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = ROOT / "outputs/diagnostics/overnight_20260930_200213/chunk50/fixed_manifest.json"
        manifest = {k:v[args.episode_offset:args.episode_offset + args.episodes] for k,v in json.loads(source.read_text()).items()}
        if len(manifest) != 12 or any(len(v) != args.episodes for v in manifest.values()):
            raise ValueError("Need 12 tasks with the requested number of fixed episodes")
        manifest_path = out / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing manifest differs; use a fresh output directory")
        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
        jobs = []
        for step in args.steps:
            group = "competition_expert_only_24h_retry1" if step == 5000 else "competition_grasp_weighted_continue_10k"
            model = ROOT / f"outputs/{group}/checkpoints/global_step_{step}/hf_ckpt"
            index = json.loads((model / "model.safetensors.index.json").read_text())
            for filename in set(index["weight_map"].values()):
                if not (model / filename).is_file(): raise ValueError(f"Missing shard: {model/filename}")
            for chunk in args.chunks:
                if chunk not in (10,25,50): raise ValueError("Unsupported chunk")
                jobs.append((step, chunk, model))
        if args.validate_only:
            print(f"Validated {len(jobs)} jobs, {len(manifest)*args.episodes} episodes each", flush=True)
            return
        summaries = []
        for step,chunk,model in jobs:
            run = out / f"step{step}_chunk{chunk}"
            run.mkdir(exist_ok=True)
            try:
                summary = aggregate(run, manifest)
                print(f"Already complete: {run.name}", flush=True)
            except (ValueError, KeyError, json.JSONDecodeError):
                env = dict(os.environ, MODEL_PATH=str(model), OUT_DIR=str(run),
                           MANIFEST=str(manifest_path), EPISODES=str(args.episodes),
                           TASKS=" ".join(manifest), TEMPORAL_ENSEMBLE="False", VIDEO="True",
                           GUI="0", PORT="9360")
                print(f"Starting {run.name}: {model}", flush=True)
                with (run / "runner.log").open("a") as log:
                    subprocess.run(["bash",str(ROOT/"02_代码材料/diagnostics/run.sh"),str(chunk)],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   check=True)
                summary = aggregate(run, manifest)
            summary.update(step=step,chunk=chunk,temporal_ensemble=False)
            summaries.append(summary)
            (out/"summary.json").write_text(json.dumps(summaries,ensure_ascii=False,indent=2))
            print(f"Finished {run.name}: {summary['successes']}/{summary['total']}", flush=True)
        (out/".complete").touch()
        print("Screening complete. Confirm leading checkpoints on more seeds before training.", flush=True)

if __name__ == "__main__":
    main()
