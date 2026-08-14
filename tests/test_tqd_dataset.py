import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from krea2_trainer.dataset.bucket import BucketBatchManager
from krea2_trainer.dataset.config_utils import ConfigSanitizer
from krea2_trainer.dataset.image_video_dataset import BaseDataset, ItemInfo


class TQDScoreManifestTests(unittest.TestCase):
    def write_manifest(self, directory: Path, records: list[dict]) -> Path:
        path = directory / "scores.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        return path

    def test_loads_canonical_image_file_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.write_manifest(
                Path(tmp),
                [{"image_file": "sample.png", "structure_score": 0.9, "detail_score": 0.4}],
            )
            self.assertEqual(
                BaseDataset._load_tqd_scores(str(manifest)),
                {"sample": (0.9, 0.4)},
            )

    def test_attaches_canonical_image_file_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.write_manifest(
                Path(tmp),
                [{"image_file": "sample.png", "structure_score": 0.9, "detail_score": 0.4}],
            )
            dataset = BaseDataset((1024, 1024), None, 1, 1, False, False, tqd_score_file=str(manifest))
            cache_file = "/tmp/sample_1024x1024_krea2.safetensors"
            item = ItemInfo("sample", "", (1024, 1024), (1024, 1024), latent_cache_path=cache_file)
            dataset.attach_tqd_scores(item, cache_file)
            self.assertEqual(item.tqd_structure_score, 0.9)
            self.assertEqual(item.tqd_detail_score, 0.4)

    def test_loads_legacy_cache_file_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.write_manifest(
                Path(tmp),
                [{"cache_file": "sample_1024x1024_krea2.safetensors", "structure_score": 0.9, "detail_score": 0.4}],
            )
            self.assertEqual(
                BaseDataset._load_tqd_scores(str(manifest)),
                {"sample": (0.9, 0.4)},
            )

    def test_legacy_cache_normalization_only_removes_final_generated_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.write_manifest(
                Path(tmp),
                [
                    {
                        "cache_file": "portrait_1024x1024_kr2_1024x1536_kr2.safetensors",
                        "structure_score": 0.9,
                        "detail_score": 0.4,
                    }
                ],
            )
            self.assertEqual(
                BaseDataset._load_tqd_scores(str(manifest)),
                {"portrait_1024x1024_kr2": (0.9, 0.4)},
            )

    def test_rejects_both_keys_or_neither_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            both = self.write_manifest(
                directory,
                [{"image_file": "sample.png", "cache_file": "sample_1024x1024_krea2.safetensors", "structure_score": 0.5, "detail_score": 0.5}],
            )
            with self.assertRaisesRegex(ValueError, "exactly one of"):
                BaseDataset._load_tqd_scores(str(both))

            neither = self.write_manifest(
                directory,
                [{"structure_score": 0.5, "detail_score": 0.5}],
            )
            with self.assertRaisesRegex(ValueError, "exactly one of"):
                BaseDataset._load_tqd_scores(str(neither))

    def test_rejects_nested_or_absolute_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            nested = self.write_manifest(
                directory,
                [{"image_file": "nested/sample.png", "structure_score": 0.5, "detail_score": 0.5}],
            )
            with self.assertRaisesRegex(ValueError, "basename"):
                BaseDataset._load_tqd_scores(str(nested))

            absolute = self.write_manifest(
                directory,
                [{"image_file": "/tmp/sample.png", "structure_score": 0.5, "detail_score": 0.5}],
            )
            with self.assertRaisesRegex(ValueError, "basename"):
                BaseDataset._load_tqd_scores(str(absolute))

    def test_rejects_duplicate_stems_or_same_stem_different_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            duplicate = self.write_manifest(
                directory,
                [
                    {"image_file": "sample.png", "structure_score": 0.5, "detail_score": 0.5},
                    {"image_file": "sample.png", "structure_score": 0.6, "detail_score": 0.4},
                ],
            )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                BaseDataset._load_tqd_scores(str(duplicate))

            collision = self.write_manifest(
                directory,
                [
                    {"image_file": "sample.png", "structure_score": 0.5, "detail_score": 0.5},
                    {"image_file": "sample.webp", "structure_score": 0.6, "detail_score": 0.4},
                ],
            )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                BaseDataset._load_tqd_scores(str(collision))

    def test_rejects_malformed_or_out_of_range_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            malformed = self.write_manifest(
                directory,
                [{"image_file": "sample.png", "structure_score": 0.5}],
            )
            with self.assertRaisesRegex(ValueError, "Invalid TQD score record"):
                BaseDataset._load_tqd_scores(str(malformed))

            out_of_range = self.write_manifest(
                directory,
                [{"image_file": "sample.png", "structure_score": 1.1, "detail_score": 0.5}],
            )
            with self.assertRaisesRegex(ValueError, "within \\[0, 1\\]"):
                BaseDataset._load_tqd_scores(str(out_of_range))

    def test_score_file_requires_every_training_cache_even_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.write_manifest(Path(tmp), [])
            dataset = BaseDataset((1024, 1024), None, 1, 1, False, False, tqd_score_file=str(manifest))
            cache_file = "/tmp/sample_1024x1024_krea2.safetensors"
            item = ItemInfo("sample", "", (1024, 1024), (1024, 1024), latent_cache_path=cache_file)

            with self.assertRaisesRegex(ValueError, "Missing TQD score"):
                dataset.attach_tqd_scores(item, cache_file)

    def test_no_score_file_leaves_items_unscored(self):
        dataset = BaseDataset((1024, 1024), None, 1, 1, False, False)
        cache_file = "/tmp/sample.safetensors"
        item = ItemInfo("sample", "", (1024, 1024), (1024, 1024), latent_cache_path=cache_file)

        dataset.attach_tqd_scores(item, cache_file)

        self.assertIsNone(item.tqd_structure_score)
        self.assertIsNone(item.tqd_detail_score)

    def test_dataset_config_accepts_score_file(self):
        config = {
            "general": {},
            "datasets": [
                {
                    "image_directory": "/tmp/images",
                    "cache_directory": "/tmp/cache",
                    "resolution": [1024, 1024],
                    "batch_size": 1,
                    "enable_bucket": True,
                    "tqd_score_file": "/tmp/scores.jsonl",
                }
            ],
        }
        ConfigSanitizer().sanitize_user_config(config)


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
