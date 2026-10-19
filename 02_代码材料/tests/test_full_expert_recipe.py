from pathlib import Path
import unittest
import yaml


ROOT = Path(__file__).resolve().parents[2]


class FullExpertRecipeTest(unittest.TestCase):
    def test_balanced_manifest_has_exactly_50_unique_clean_tasks(self):
        p = ROOT / "assets/training_data/robotwin_clean_50.txt"
        rows = [x.strip() for x in p.read_text().splitlines() if x.strip()]
        self.assertEqual(len(rows), 50)
        self.assertEqual(len(set(rows)), 50)
        self.assertTrue(all("clean_50" in x for x in rows))
        self.assertTrue(all("randomized" not in x for x in rows))

    def test_recipe_starts_from_official_base_and_trains_full_expert(self):
        p = ROOT / "02_代码材料/configs/train_full_expert_balanced50_500.yaml"
        cfg = yaml.safe_load(p.read_text())
        self.assertEqual(cfg["model"]["model_path"], "./pretrained/lingbot-vla-v2-6b")
        self.assertTrue(cfg["train"]["train_expert_only"])
        self.assertFalse(cfg["train"]["action_expert_tail_only"])
        self.assertFalse(cfg["train"]["vlm_lora_enabled"])
        self.assertTrue(cfg["data"]["image_augment"])


if __name__ == "__main__":
    unittest.main()
