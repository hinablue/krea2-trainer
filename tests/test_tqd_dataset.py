import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from krea2_trainer.dataset.bucket import BucketBatchManager
from krea2_trainer.dataset.image_video_dataset import BaseDataset, ItemInfo


class TQDScoreManifestTests(unittest.TestCase):
    def write_manifest(self, directory: Path, records: list[dict]) -> Path:
        path = directory / "scores.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        return path

    def test_loads_valid_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.write_manifest(
                Path(tmp),
                [{"cache_file": "sample_1024x1024_krea2.safetensors", "structure_score": 0.9, "detail_score": 0.4}],
            )

            self.assertEqual(
                BaseDataset._load_tqd_scores(str(manifest)),
                {"sample_1024x1024_krea2.safetensors": (0.9, 0.4)},
            )

    def test_rejects_duplicate_or_out_of_range_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            duplicate = self.write_manifest(
                directory,
                [
                    {"cache_file": "sample_1024x1024_krea2.safetensors", "structure_score": 0.5, "detail_score": 0.5},
                    {"cache_file": "sample_1024x1024_krea2.safetensors", "structure_score": 0.6, "detail_score": 0.4},
                ],
            )
            with self.assertRaisesRegex(ValueError, "Duplicate TQD cache_file"):
                BaseDataset._load_tqd_scores(str(duplicate))

            invalid = self.write_manifest(
                directory,
                [{"cache_file": "other_1024x1024_krea2.safetensors", "structure_score": 1.1, "detail_score": 0.5}],
            )
            with self.assertRaisesRegex(ValueError, "within \\[0, 1\\]"):
                BaseDataset._load_tqd_scores(str(invalid))


class TQDBatchPropagationTests(unittest.TestCase):
    def make_item(self, directory: Path, name: str, structure: float | None, detail: float | None) -> ItemInfo:
        latent_path = directory / f"{name}_1024x1024_krea2.safetensors"
        text_path = directory / f"{name}_krea2_te.safetensors"
        save_file({"latents_1x1x1_float32": torch.zeros(1)}, str(latent_path))
        save_file({"prompt_embed_float32": torch.zeros(1)}, str(text_path))

        item = ItemInfo(name, "", (1024, 1024), (1024, 1024), latent_cache_path=str(latent_path))
        item.text_encoder_output_cache_path = str(text_path)
        item.tqd_structure_score = structure
        item.tqd_detail_score = detail
        return item

    def test_emits_scores_in_item_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            first = self.make_item(directory, "first", 0.8, 0.3)
            second = self.make_item(directory, "second", 0.2, 0.9)
            manager = BucketBatchManager({(1024, 1024): [first, second]}, batch_size=2)

            batch = manager[0]

            torch.testing.assert_close(batch["tqd_structure_score"], torch.tensor([0.8, 0.2]))
            torch.testing.assert_close(batch["tqd_detail_score"], torch.tensor([0.3, 0.9]))

    def test_rejects_mixed_scored_and_unscored_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            scored = self.make_item(directory, "scored", 0.8, 0.3)
            unscored = self.make_item(directory, "unscored", None, None)
            manager = BucketBatchManager({(1024, 1024): [scored, unscored]}, batch_size=2)

            with self.assertRaisesRegex(ValueError, "every item in a batch"):
                manager[0]


if __name__ == "__main__":
    unittest.main()
