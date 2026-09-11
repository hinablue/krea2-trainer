"""Preference fixtures are real Krea2 safetensors; no model or dataset downloads."""

import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from krea2_trainer.dataset.bucket import BucketBatchManager
from krea2_trainer.dataset.image_video_dataset import BaseDataset, DatasetGroup, ItemInfo
from krea2_trainer.dataset.preference_dataset import build_preference_dataset_group


class SourceDataset(BaseDataset):
    def __init__(self, items, repeats=1):
        super().__init__(resolution=(32, 32), architecture="kr2", num_repeats=repeats)
        buckets = {}
        for item in items:
            buckets.setdefault(item.bucket_size, []).extend([item] * repeats)
        self.batch_manager = BucketBatchManager(buckets, batch_size=1)
        self.num_train_items = len(items) * repeats

    def __len__(self):
        return len(self.batch_manager)


def identity_collate(batch):
    return batch


class PreferenceDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.items = []

    def item(self, name, value, prompt="a portrait", shape=(2, 1, 4, 4), tokens=3, bucket=None, varlen=True):
        bucket = bucket or (shape[-1] * 8, shape[-2] * 8)
        latent_path = self.directory / f"{name}_{bucket[0]:04d}x{bucket[1]:04d}_kr2.safetensors"
        text_path = self.directory / f"{name}_kr2_te.safetensors"
        f, h, w = shape[-3:]
        save_file({f"latents_{f}x{h}x{w}_float32": torch.full(shape, float(value))}, str(latent_path))
        embed = torch.arange(tokens * 2 * 4, dtype=torch.float32).reshape(tokens, 2, 4)
        prefix = "varlen_" if varlen else ""
        save_file({f"{prefix}krea2_vl_embed_float32": embed}, str(text_path), metadata={"caption1": prompt})
        item = ItemInfo(name, "not trusted", bucket, bucket, latent_cache_path=str(latent_path))
        item.text_encoder_output_cache_path = str(text_path)
        self.items.append(item)
        return item

    def record(self, chosen="winner", rejected="loser", pair_id="pair-1", prompt="a portrait", **extra):
        return dict(pair_id=pair_id, prompt=prompt, chosen=f"{chosen}.png", rejected=f"{rejected}.png", **extra)

    def build(self, records=None, mode="flow_dpo", batch_size=2, seed=41, shared_epoch=None, repeats=1):
        manifest = self.directory / "preferences.jsonl"
        manifest.write_text("\n".join(json.dumps(record) for record in records or [self.record()]), encoding="utf-8")
        source = DatasetGroup([SourceDataset(self.items, repeats)])
        return build_preference_dataset_group(source, str(manifest), mode, batch_size, seed, shared_epoch)

    def test_flow_dpo_returns_normalized_tensors_and_keeps_pair_order(self):
        for name, value in [("winner", 10), ("loser", -10), ("winner2", 20), ("loser2", -20)]:
            self.item(name, value)
        group = self.build([self.record(), self.record("winner2", "loser2", "pair-2")])
        batch = group[0]
        self.assertEqual(group.num_train_items, 2)
        self.assertEqual(len(group), 1)
        self.assertEqual(batch["latents"].shape, (2, 2, 1, 4, 4))
        torch.testing.assert_close(batch["latents"], -batch["rejected_latents"])
        self.assertIsInstance(batch["krea2_vl_embed"], list)
        self.assertEqual(len(batch["krea2_vl_embed"]), 2)
        torch.testing.assert_close(batch["krea2_vl_embed"][0], torch.arange(24, dtype=torch.float32).reshape(3, 2, 4))
        self.assertIsNone(batch["timesteps"])

    def test_rft_deduplicates_winners_and_never_loads_rejected_caches(self):
        winner = self.item("winner", 10)
        rejected = self.item("loser", -10)
        Path(rejected.latent_cache_path).unlink()
        Path(rejected.text_encoder_output_cache_path).unlink()
        from krea2_trainer.dataset import bucket

        with patch.object(bucket, "load_file", wraps=bucket.load_file) as load:
            group = self.build([self.record(), self.record("winner", "absent", "pair-2")], mode="rft", repeats=4)
            batch = group[0]
        self.assertEqual(group.num_train_items, 1)
        self.assertNotIn("rejected_latents", batch)
        self.assertEqual(batch["latents"].flatten()[0].item(), 10)
        self.assertEqual(
            {call.args[0] for call in load.call_args_list}, {winner.latent_cache_path, winner.text_encoder_output_cache_path}
        )

    def test_flow_dpo_rejects_missing_cache(self):
        self.item("winner", 10)
        loser = self.item("loser", -10)
        Path(loser.text_encoder_output_cache_path).unlink()
        with self.assertRaisesRegex(ValueError, "[Mm]issing.*cache"):
            self.build()

    def test_caption_metadata_is_required_and_must_match_on_each_side(self):
        for side in ["winner", "loser"]:
            for metadata in [None, {"caption1": "wrong prompt"}]:
                with self.subTest(side=side, metadata=metadata):
                    self.items = []
                    self.item("winner", 10)
                    self.item("loser", -10)
                    item = next(item for item in self.items if item.item_key == side)
                    save_file(
                        {"varlen_krea2_vl_embed_float32": torch.zeros(3, 2, 4)},
                        item.text_encoder_output_cache_path,
                        metadata=metadata,
                    )
                    with self.assertRaisesRegex(ValueError, "caption1"):
                        self.build()

    def test_conditioning_mismatch_fails(self):
        self.item("winner", 10)
        loser = self.item("loser", -10)
        save_file(
            {"varlen_krea2_vl_embed_float32": torch.ones(3, 2, 4)},
            loser.text_encoder_output_cache_path,
            metadata={"caption1": "a portrait"},
        )
        with self.assertRaisesRegex(ValueError, "conditioning"):
            self.build()

    def test_pair_and_split_validation(self):
        self.item("winner", 10)
        self.item("loser", -10)
        invalid = [
            ([self.record("winner", "winner")], "[Ss]elf"),
            ([self.record(), self.record(pair_id="pair-2")], "[Dd]uplicate.*pair"),
            ([self.record(), self.record("loser", "winner", "pair-2")], "[Rr]eversed|contradict"),
            ([self.record(), self.record("winner", "other", "pair-1")], "pair_id"),
            ([self.record(split="validation")], "train split"),
            ([self.record(), self.record("a", "b", "holdout", split="test")], "across splits"),
        ]
        for records, error in invalid:
            with self.subTest(records=records), self.assertRaisesRegex(ValueError, error):
                self.build(records)

    def test_multiple_shapes_are_bucketed_and_varlen_is_not_stacked(self):
        records = []
        for i, (shape, tokens) in enumerate([((2, 1, 4, 4), 2), ((2, 1, 4, 4), 5), ((2, 1, 4, 8), 3)]):
            self.item(f"w{i}", i + 1, shape=shape, tokens=tokens)
            self.item(f"l{i}", -i - 1, shape=shape, tokens=tokens)
            records.append(self.record(f"w{i}", f"l{i}", f"pair-{i}"))
        group = self.build(records)
        self.assertEqual(len(group), 2)
        batches = [group[i] for i in range(len(group))]
        self.assertEqual(sorted(batch["latents"].shape[0] for batch in batches), [1, 2])
        for batch in batches:
            torch.testing.assert_close(batch["latents"], -batch["rejected_latents"])
        square = next(batch for batch in batches if batch["latents"].shape[-1] == 4)
        self.assertEqual(sorted(t.shape[0] for t in square["krea2_vl_embed"]), [2, 5])

    def test_per_pair_latent_shape_and_bucket_must_match(self):
        self.item("winner", 10)
        self.item("loser", -10, shape=(2, 1, 4, 8))
        with self.assertRaisesRegex(ValueError, "shape|bucket"):
            self.build()

    def test_source_repeats_and_tqd_are_not_carried_or_mutated(self):
        winner = self.item("winner", 10)
        self.item("loser", -10)
        winner.tqd_structure_score = 0.8
        winner.tqd_detail_score = 0.2
        with self.assertLogs("krea2_trainer.dataset.preference_dataset", "INFO") as logs:
            group = self.build(repeats=3)
        batch = group[0]
        self.assertEqual(group.num_train_items, 1)
        self.assertNotIn("tqd_structure_score", batch)
        self.assertEqual(winner.tqd_structure_score, 0.8)
        self.assertTrue(any("repeat" in line.lower() for line in logs.output))
        metadata = group.datasets[0].get_metadata()
        self.assertEqual(metadata["batch_size_per_device"], 2)
        self.assertEqual(metadata["num_repeats"], 1)
        group.set_max_train_steps(12)
        self.assertEqual(group.datasets[0].max_train_steps, 12)

    def test_manifest_rejects_malformed_records_and_noncanonical_names(self):
        self.item("winner", 10)
        self.item("loser", -10)
        for field, value in [
            ("pair_id", ""),
            ("pair_id", 1),
            ("prompt", "   "),
            ("prompt", None),
            ("split", "dev"),
            ("split", None),
            ("chosen", "nested/winner.png"),
            ("chosen", "C:\\winner.png"),
            ("chosen", "../winner.png"),
            ("chosen", "winner_0032x0032_kr2.safetensors"),
            ("chosen", "winner"),
            ("rejected", "https://example.test/loser.png"),
            ("rejected", None),
        ]:
            with self.subTest(field=field, value=value):
                record = self.record()
                record[field] = value
                with self.assertRaises(ValueError):
                    self.build([record])
        for record in [[], {}, {"pair_id": "id", "prompt": "prompt"}]:
            with self.subTest(record=record), self.assertRaises(ValueError):
                self.build([record])
        manifest = self.directory / "broken.jsonl"
        source = DatasetGroup([SourceDataset(self.items)])
        for content in ["{invalid JSON", "", "\n  \n"]:
            manifest.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValueError):
                build_preference_dataset_group(source, str(manifest), "rft", 1, 0)
        with self.assertRaisesRegex(ValueError, "manifest"):
            build_preference_dataset_group(source, str(self.directory / "absent.jsonl"), "rft", 1, 0)

    def test_same_stem_different_extension_is_ambiguous(self):
        self.item("winner", 10)
        self.item("loser", -10)
        second = self.record("winner", "other", "pair-2")
        second["chosen"] = "winner.webp"
        with self.assertRaisesRegex(ValueError, "[Aa]mbiguous.*stem"):
            self.build([self.record(), second], mode="rft")

    def test_source_same_stem_different_cache_is_ambiguous(self):
        self.item("winner", 10)
        self.item("loser", -10)
        # This yields a distinct filename with the same original image stem.
        self.item("winner", 11, shape=(2, 1, 4, 8))
        with self.assertRaisesRegex(ValueError, "[Aa]mbiguous.*stem"):
            self.build(mode="rft")

    def test_holdout_records_do_not_require_any_caches(self):
        self.item("winner", 10)
        self.item("loser", -10)
        records = [
            self.record(),
            self.record("val-w", "val-l", "val", prompt="validation prompt", split="validation"),
            self.record("test-w", "test-l", "test", prompt="test prompt", split="test"),
        ]
        group = self.build(records)
        self.assertEqual(group.num_train_items, 1)
        self.assertEqual(len(group), 1)
        self.assertEqual(group[0]["latents"].flatten()[0].item(), 10)

    def test_rft_requires_selected_winner_and_caption_metadata(self):
        with self.assertRaisesRegex(ValueError, "[Mm]issing.*chosen.*cache"):
            self.build(mode="rft")
        winner = self.item("winner", 10)
        save_file({"varlen_krea2_vl_embed_float32": torch.zeros(3, 2, 4)}, winner.text_encoder_output_cache_path)
        with self.assertRaisesRegex(ValueError, "caption1"):
            self.build(mode="rft")

    def test_missing_latents_and_corrupt_files_fail_up_front(self):
        for side in ["winner", "loser"]:
            for field in ["latent_cache_path", "text_encoder_output_cache_path"]:
                for corrupt in [False, True]:
                    with self.subTest(side=side, field=field, corrupt=corrupt):
                        self.items = []
                        self.item("winner", 10)
                        self.item("loser", -10)
                        target = next(item for item in self.items if item.item_key == side)
                        path = Path(getattr(target, field))
                        if corrupt:
                            path.write_bytes(b"invalid safetensors")
                        else:
                            path.unlink()
                        with self.assertRaisesRegex(ValueError, "cache"):
                            self.build()

    def test_nonfinite_or_integer_cache_tensors_fail_up_front(self):
        for side in ["winner", "loser"]:
            for kind in ["latents", "conditioning"]:
                for value in [float("nan"), float("inf"), -float("inf"), 1]:
                    with self.subTest(side=side, kind=kind, value=value):
                        self.items = []
                        self.item("winner", 10)
                        self.item("loser", -10)
                        target = next(item for item in self.items if item.item_key == side)
                        dtype = torch.int64 if isinstance(value, int) else torch.float32
                        suffix = "int64" if dtype == torch.int64 else "float32"
                        if kind == "latents":
                            save_file(
                                {f"latents_1x4x4_{suffix}": torch.full((2, 1, 4, 4), value, dtype=dtype)}, target.latent_cache_path
                            )
                        else:
                            save_file(
                                {f"varlen_krea2_vl_embed_{suffix}": torch.full((3, 2, 4), value, dtype=dtype)},
                                target.text_encoder_output_cache_path,
                                metadata={"caption1": "a portrait"},
                            )
                        with self.assertRaisesRegex(ValueError, "[Ii]nvalid|finite"):
                            self.build()

    def test_control_and_video_caches_are_rejected(self):
        for variant in ["control", "frames", "metadata", "item"]:
            with self.subTest(variant=variant):
                self.items = []
                winner = self.item("winner", 10)
                self.item("loser", -10)
                latent = torch.zeros(2, 1, 4, 4)
                if variant == "control":
                    save_file(
                        {"latents_1x4x4_float32": latent, "latents_control_0_1x4x4_float32": latent.clone()},
                        winner.latent_cache_path,
                    )
                elif variant == "frames":
                    save_file({"latents_2x4x4_float32": torch.zeros(2, 2, 4, 4)}, winner.latent_cache_path)
                elif variant == "metadata":
                    save_file({"latents_1x4x4_float32": latent}, winner.latent_cache_path, metadata={"frame_count": "1"})
                else:
                    winner.frame_count = 1
                with self.assertRaisesRegex(ValueError, "image-only"):
                    self.build()

    def test_duplicate_logical_keys_and_invalid_latent_shapes_fail(self):
        payloads = [
            {},
            {"latents_1x4x4_float32": torch.zeros(1)},
            {"latents_1x8x4_float32": torch.zeros(2, 1, 4, 4)},
            {"latents_1x4x4_float32": torch.zeros(0, 1, 4, 4)},
            {
                "latents_1x4x4_float32": torch.zeros(2, 1, 4, 4),
                "latents_1x4x4_float16": torch.zeros(2, 1, 4, 4, dtype=torch.float16),
            },
        ]
        for payload in payloads:
            with self.subTest(keys=list(payload)):
                self.items = []
                winner = self.item("winner", 10)
                self.item("loser", -10)
                save_file(payload, winner.latent_cache_path)
                with self.assertRaises(ValueError):
                    self.build()

    def test_conditioning_schema_and_varlen_length_must_match_within_pair(self):
        for tokens, varlen in [(4, True), (3, False)]:
            with self.subTest(tokens=tokens, varlen=varlen):
                self.items = []
                self.item("winner", 10)
                self.item("loser", -10, tokens=tokens, varlen=varlen)
                with self.assertRaisesRegex(ValueError, "conditioning"):
                    self.build()

    def test_dense_conditioning_is_stacked_with_native_loader(self):
        self.item("winner", 10, varlen=False)
        self.item("loser", -10, varlen=False)
        batch = self.build()[0]
        self.assertIsInstance(batch["krea2_vl_embed"], torch.Tensor)
        self.assertEqual(batch["krea2_vl_embed"].shape, (1, 3, 2, 4))

    def test_latent_shapes_split_batches_even_if_source_buckets_match(self):
        self.item("winner", 10)
        self.item("loser", -10)
        self.item("winner2", 20, shape=(2, 1, 4, 8), bucket=(32, 32))
        self.item("loser2", -20, shape=(2, 1, 4, 8), bucket=(32, 32))
        group = self.build([self.record(), self.record("winner2", "loser2", "pair-2")])
        self.assertEqual(len(group), 2)
        self.assertEqual(sorted(group[i]["latents"].shape[-1] for i in range(len(group))), [4, 8])

    def test_identical_latents_in_different_buckets_cannot_form_pair(self):
        self.item("winner", 10)
        self.item("loser", -10, bucket=(64, 32))
        with self.assertRaisesRegex(ValueError, "bucket"):
            self.build()

    def test_repeated_winner_keeps_distinct_flow_dpo_losers(self):
        self.item("winner", 10)
        self.item("loser", -10)
        self.item("loser2", -20)
        group = self.build([self.record(), self.record("winner", "loser2", "pair-2")])
        self.assertEqual(group.num_train_items, 2)
        batch = group[0]
        self.assertEqual(batch["latents"][:, 0, 0, 0, 0].tolist(), [10, 10])
        self.assertEqual(sorted(batch["rejected_latents"][:, 0, 0, 0, 0].tolist()), [-20, -10])

    def test_lazy_loading_reads_fresh_caches_and_does_not_cache_tensors(self):
        winner = self.item("winner", 10)
        loser = self.item("loser", -10)
        group = self.build()
        save_file({"latents_1x4x4_float32": torch.full((2, 1, 4, 4), 99.0)}, winner.latent_cache_path)
        save_file({"latents_1x4x4_float32": torch.full((2, 1, 4, 4), -99.0)}, loser.latent_cache_path)
        batch = group[0]
        self.assertEqual(batch["latents"].flatten()[0].item(), 99)
        self.assertEqual(batch["rejected_latents"].flatten()[0].item(), -99)
        self.assertFalse(any(isinstance(value, torch.Tensor) for value in vars(group.datasets[0]).values()))

    def test_original_stems_survive_dots_and_generated_looking_fragments(self):
        from krea2_trainer.dataset.image_video_dataset import ImageDataset

        name = "portrait.v2_1024x1024_kr2"
        self.item(name, 10)
        self.item("loser", -10)
        source = ImageDataset(
            resolution=(32, 32),
            caption_extension=".txt",
            batch_size=1,
            num_repeats=3,
            enable_bucket=False,
            bucket_no_upscale=False,
            image_directory=str(self.directory),
            cache_directory=str(self.directory),
            architecture="kr2",
        )
        source.prepare_for_training()
        manifest = self.directory / "preferences.jsonl"
        manifest.write_text(json.dumps(self.record(name)), encoding="utf-8")
        group = build_preference_dataset_group(DatasetGroup([source]), str(manifest), "flow_dpo", 2, 41)
        self.assertEqual(group.num_train_items, 1)
        self.assertEqual(group[0]["latents"].flatten()[0].item(), 10)

    def test_epoch_without_shared_state_and_native_collator(self):
        import random
        from krea2_trainer.training.accelerator_setup import collator_class

        self.item("winner", 10)
        self.item("loser", -10)
        group = self.build()
        state = random.getstate()
        group.set_current_epoch(2)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(group.datasets[0].current_epoch, 2)
        shared_epoch = multiprocessing.Value("i", 2)
        group.datasets[0].set_seed(41, shared_epoch)
        loader = torch.utils.data.DataLoader(group, batch_size=1, collate_fn=collator_class(shared_epoch, group))
        self.assertEqual(next(iter(loader))["latents"].shape[0], 1)
        with self.assertRaisesRegex(ValueError, "shared_epoch"):
            group.set_current_epoch(1)

    def test_native_timestep_buckets_are_preserved_and_deterministic(self):
        self.item("winner", 10)
        self.item("loser", -10)
        source = SourceDataset(self.items)
        source.batch_manager.num_timestep_buckets = 4
        manifest = self.directory / "preferences.jsonl"
        manifest.write_text(json.dumps(self.record()), encoding="utf-8")
        group = build_preference_dataset_group(DatasetGroup([source]), str(manifest), "flow_dpo", 2, 41)
        self.assertEqual(len(group[0]["timesteps"]), 1)
        first = group[0]["timesteps"]
        group.set_current_epoch(3)
        self.assertNotEqual(group[0]["timesteps"], first)
        group.set_current_epoch(0)
        self.assertEqual(group[0]["timesteps"], first)
        self.assertTrue(all(0 <= timestep <= 1 for timestep in first))

    def test_invalid_mode_batch_size_and_seed(self):
        for kwargs in [
            {"mode": "dpo"},
            {"batch_size": 0},
            {"batch_size": -1},
            {"batch_size": True},
            {"batch_size": 1.5},
            {"seed": None},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build(**kwargs)

    def test_dataset_pickling_preserves_pair_mapping(self):
        import pickle

        self.item("winner", 10)
        self.item("loser", -10)
        self.item("loser2", -20)
        group = self.build([self.record(), self.record("winner", "loser2", "pair-2")])
        restored = pickle.loads(pickle.dumps(group))
        torch.testing.assert_close(restored[0]["latents"], group[0]["latents"])
        torch.testing.assert_close(restored[0]["rejected_latents"], group[0]["rejected_latents"])

    def test_production_cache_writers_and_bfloat16_tensors(self):
        from krea2_trainer.dataset.cache_io import save_latent_cache_krea2, save_text_encoder_output_cache_krea2

        winner = self.item("winner", 10)
        loser = self.item("loser", -10)
        for item, value in [(winner, 10), (loser, -10)]:
            item.caption = "a portrait"
            Path(item.text_encoder_output_cache_path).unlink()
            save_latent_cache_krea2(item, torch.full((2, 1, 4, 4), value, dtype=torch.bfloat16))
            save_text_encoder_output_cache_krea2(item, torch.ones(3, 2, 4, dtype=torch.bfloat16))
        batch = self.build()[0]
        self.assertEqual(batch["latents"].dtype, torch.bfloat16)
        self.assertEqual(batch["krea2_vl_embed"][0].dtype, torch.bfloat16)
        torch.testing.assert_close(batch["latents"], -batch["rejected_latents"])

    def test_source_repeat_count_does_not_change_preference_order(self):
        records = []
        for i in range(4):
            self.item(f"w{i}", i + 1)
            self.item(f"l{i}", -i - 1)
            records.append(self.record(f"w{i}", f"l{i}", f"pair-{i}"))
        first = self.build(records, repeats=1)
        repeated = self.build(records, repeats=5)
        self.assertEqual(first.num_train_items, repeated.num_train_items)
        for i in range(len(first)):
            torch.testing.assert_close(first[i]["latents"], repeated[i]["latents"])
            torch.testing.assert_close(first[i]["rejected_latents"], repeated[i]["rejected_latents"])

    def test_metadata_reports_actual_preference_bucket_resolution(self):
        self.item("winner", 10)
        self.item("loser", -10)
        metadata = self.build().datasets[0].get_metadata()
        self.assertEqual(metadata["resolution"], (32, 32))
        self.assertEqual(metadata["preference_mode"], "flow_dpo")
        self.assertEqual(metadata["num_train_items"], 1)
        json.dumps(metadata)

    def test_epoch_shuffle_is_reproducible_and_persistent_worker_safe(self):
        records = []
        for i in range(12):
            self.item(f"w{i}", i + 1)
            self.item(f"l{i}", -i - 1)
            records.append(self.record(f"w{i}", f"l{i}", f"pair-{i}"))
        epoch = multiprocessing.Value("i", 1)
        group = self.build(records, shared_epoch=epoch, batch_size=2)
        other = self.build(records, shared_epoch=epoch, batch_size=2)
        loader = torch.utils.data.DataLoader(
            group, batch_size=None, num_workers=2, persistent_workers=True, collate_fn=identity_collate
        )
        try:
            first = [batch["latents"].flatten().tolist() for batch in loader]
            self.assertEqual(first, [other[i]["latents"].flatten().tolist() for i in range(len(other))])
            epoch.value = 3
            group.set_current_epoch(3)
            second_batches = list(loader)
            second = [batch["latents"].flatten().tolist() for batch in second_batches]
            self.assertNotEqual(first, second)
            self.assertEqual(second, [other[i]["latents"].flatten().tolist() for i in range(len(other))])
            for batch in second_batches:
                torch.testing.assert_close(batch["latents"], -batch["rejected_latents"])
            epoch.value = 1
            self.assertEqual(first, [batch["latents"].flatten().tolist() for batch in loader])
        finally:
            if loader._iterator is not None:
                loader._iterator._shutdown_workers()


if __name__ == "__main__":
    unittest.main()
