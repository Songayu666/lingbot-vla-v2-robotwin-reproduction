#!/usr/bin/env python3
"""Build the single-GPU, competition-compliant full Action Expert SFT recipe."""
import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--output-name", default="official_full_expert_balanced50_500")
    args = parser.parse_args()
    if args.max_steps < 1 or args.save_steps < 1 or args.grad_accum < 1:
        raise ValueError("steps and grad-accum must be positive")

    source = ROOT / "02_代码材料/configs/train_robotwin_clean50.yaml"
    cfg = yaml.safe_load(source.read_text())
    cfg["model"]["model_path"] = "./pretrained/lingbot-vla-v2-6b"
    cfg["data"].update({
        "train_path": "assets/training_data/robotwin_clean_50.txt",
        "image_augment": True,
    })
    cfg["train"].update({
        "output_dir": f"./outputs/{args.output_name}",
        # Freeze Qwen3-VL while training every Action Expert layer and all
        # action/state/time projection heads.
        "train_expert_only": True,
        "action_expert_tail_only": False,
        "residual_corrector_enabled": False,
        "residual_corrector_only": False,
        "vlm_lora_enabled": False,
        "vlm_visual_projector_enabled": False,
        "micro_batch_size": 1,
        "global_batch_size": args.grad_accum,
        "gradient_accumulation_steps": args.grad_accum,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "enable_resume": True,
        "enable_gradient_checkpointing": True,
        "enable_activation_offload": True,
        "use_compile": False,
        "use_wandb": False,
    })
    target = ROOT / "02_代码材料/configs/train_full_expert_balanced50_500.yaml"
    target.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(target)
    print(f"output={cfg['train']['output_dir']}")
    print(f"effective_batch={cfg['train']['global_batch_size']}")


if __name__ == "__main__":
    main()
