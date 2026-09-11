"""Validated, disk-backed image preference datasets for RFT and FlowDPO.

Manifest names refer to original image basenames, not generated cache filenames.
Only train records load tensors. RFT loads unique (chosen, prompt) samples; FlowDPO
validates both sides under identical conditioning and keeps the pair atomic while
bucketing/shuffling. Cache tensors are never retained by the dataset between reads.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from krea2_trainer.dataset.architectures import ARCHITECTURE_KREA2, ARCHITECTURE_KREA2_FULL
from krea2_trainer.dataset.bucket import BucketBatchManager
from krea2_trainer.dataset.image_video_dataset import BaseDataset, DatasetGroup, ItemInfo, VideoDataset
from krea2_trainer.dataset.media_utils import IMAGE_EXTENSIONS
from krea2_trainer.utils.model_utils import remove_dtype_suffix

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Pair:
    pair_id: str
    prompt: str
    chosen: str  # original image stem (may itself contain dots)
    rejected: str
    split: str


def _image_stem(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "/" in value
        or "\\" in value
        or ":" in value
        or "\x00" in value
        or Path(value).name != value
    ):
        raise ValueError(f"{field} must be an original image basename, not a path: {value!r}")
    stem, suffix = os.path.splitext(value)
    if not stem or suffix.lower() not in {ext.lower() for ext in IMAGE_EXTENSIONS}:
        raise ValueError(f"{field} must be an original image basename with a supported image extension: {value!r}")
    return stem


def _read_manifest(manifest_path: str) -> list[_Pair]:
    pairs = []
    pair_ids = set()
    image_names = {}
    prompt_splits = {}
    seen_pairs = {}
    try:
        manifest = open(manifest_path, encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Missing or unreadable preference manifest: {manifest_path}") from exc
    with manifest:
        for line_number, line in enumerate(manifest, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be a JSON object")
                for field in ("pair_id", "prompt"):
                    if not isinstance(record.get(field), str) or not record[field].strip():
                        raise ValueError(f"{field} must be a non-empty string")
                pair_id, prompt = record["pair_id"], record["prompt"]
                if pair_id in pair_ids:
                    raise ValueError(f"Duplicate pair_id: {pair_id!r}")
                pair_ids.add(pair_id)
                split = record.get("split", "train")
                if split not in ("train", "validation", "test"):
                    raise ValueError("split must be train, validation, or test")
                if prompt in prompt_splits and prompt_splits[prompt] != split:
                    raise ValueError("The same prompt appears across splits (prompt leakage)")
                prompt_splits[prompt] = split
                stems = []
                for field in ("chosen", "rejected"):
                    name = record.get(field)
                    stem = _image_stem(name, field)
                    if stem in image_names and image_names[stem] != name:
                        raise ValueError(f"Ambiguous image stem with different basename extensions: {stem!r}")
                    image_names[stem] = name
                    stems.append(stem)
                chosen, rejected = stems
                if chosen == rejected:
                    raise ValueError("Self pair: chosen and rejected must be different images")
                unordered = tuple(sorted(stems))
                if unordered in seen_pairs:
                    if seen_pairs[unordered] == (chosen, rejected):
                        raise ValueError("Duplicate image pair")
                    raise ValueError("Reversed image pair contradicts an existing preference")
                seen_pairs[unordered] = (chosen, rejected)
                pairs.append(_Pair(pair_id, prompt, chosen, rejected, split))
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid preference record at {manifest_path}:{line_number}: {exc}") from exc
    train_pairs = [pair for pair in pairs if pair.split == "train"]
    if not train_pairs:
        raise ValueError("Preference manifest must contain at least one record in the train split")
    logger.info(
        "Loaded %d train preference pairs (%d held-out records) from %s",
        len(train_pairs),
        len(pairs) - len(train_pairs),
        manifest_path,
    )
    return train_pairs


def _source_index(source_group: DatasetGroup) -> dict[str, ItemInfo]:
    index = {}
    identities = {}
    repeated = 0
    for dataset in source_group.datasets:
        if isinstance(dataset, VideoDataset):
            raise ValueError("Preference training is image-only; video datasets are not supported")
        if getattr(dataset, "control_directory", None):
            raise ValueError("Preference training is image-only; control datasets are not supported")
        manager = getattr(dataset, "batch_manager", None)
        if manager is None:
            raise ValueError("Source dataset has no prepared cache bucket manager")
        for bucket, items in manager.buckets.items():
            for item in items:
                # prepare_for_training already removed the generated cache suffix.
                # Do not splitext again: 'portrait.v2' is a valid original stem.
                stem = item.item_key
                if not isinstance(stem, str) or not stem or os.path.basename(stem) != stem:
                    raise ValueError(f"Source item_key must be an original image stem: {stem!r}")
                identity = (
                    os.path.realpath(item.latent_cache_path) if item.latent_cache_path else None,
                    os.path.realpath(item.text_encoder_output_cache_path) if item.text_encoder_output_cache_path else None,
                    tuple(bucket),
                )
                if stem in index:
                    if identities[stem] != identity:
                        raise ValueError(f"Ambiguous source image stem across different cache files/buckets: {stem!r}")
                    repeated += 1
                    continue
                index[stem] = item
                identities[stem] = identity
    if repeated:
        logger.info("Ignoring %d source repeat entries; preference manifest defines sample frequency", repeated)
    return index


def _clone_item(item: ItemInfo, prompt: str) -> ItemInfo:
    # Construct a clean descriptor: no cached content, controls, or TQD scores.
    clone = ItemInfo(item.item_key, prompt, item.original_size, item.bucket_size, latent_cache_path=item.latent_cache_path)
    clone.text_encoder_output_cache_path = item.text_encoder_output_cache_path
    return clone


def _normalized_key(key: str) -> tuple[str, bool]:
    varlen = key.startswith("varlen_")
    content = key.replace("varlen_", "") if varlen else key
    if not content.endswith("_mask"):
        content = remove_dtype_suffix(content)
        if content.startswith("latents_"):
            content = content.rsplit("_", 1)[0]
    return content, varlen


def _validate_and_load(item: ItemInfo, prompt: str) -> tuple[dict, tuple]:
    """Validate one selected item, returning only temporary tensors and schema.

    Header inspection catches collisions that the normal loader would otherwise
    merge/overwrite. Actual key normalization and variable-length behavior are
    delegated to BucketBatchManager rather than duplicated in a second loader.
    """
    if item.frame_count is not None or item.control_content is not None or not item.bucket_size or len(item.bucket_size) != 2:
        raise ValueError(f"Preference training is image-only; control/video item: {item.item_key}")
    normalized = {}
    latent_shape = None
    for kind, path in (("latent", item.latent_cache_path), ("text encoder", item.text_encoder_output_cache_path)):
        if not path or not os.path.isfile(path):
            raise ValueError(f"Missing {kind} cache for {item.item_key}: {path}")
        try:
            with safe_open(path, framework="pt", device="cpu") as cache:
                metadata = cache.metadata() or {}
                if "frame_count" in metadata:
                    raise ValueError("image-only preference training does not support video caches")
                if metadata.get("architecture", ARCHITECTURE_KREA2_FULL) != ARCHITECTURE_KREA2_FULL:
                    raise ValueError("image-only Krea2 preference training requires Krea2 caches")
                if kind == "text encoder" and metadata.get("caption1") != prompt:
                    raise ValueError(
                        f"caption1 metadata is required and must exactly equal the manifest prompt for {item.item_key}"
                    )
                keys = list(cache.keys())
                if not keys:
                    raise ValueError(f"Empty {kind} cache")
                for key in keys:
                    logical, varlen = _normalized_key(key)
                    if logical in normalized:
                        raise ValueError(f"Duplicate normalized cache tensor key: {logical}")
                    shape = tuple(cache.get_slice(key).get_shape())
                    if not shape or any(dimension <= 0 for dimension in shape):
                        raise ValueError(f"Invalid empty/scalar tensor shape for {key}: {shape}")
                    if kind == "latent":
                        match = re.fullmatch(r"latents_(\d+)x(\d+)x(\d+)", remove_dtype_suffix(key))
                        if not match or varlen:
                            raise ValueError(
                                f"image-only preference caches cannot contain control/video or extra latent keys: {key}"
                            )
                        encoded_shape = tuple(map(int, match.groups()))
                        if len(shape) != 4 or shape[1] != 1 or encoded_shape != shape[1:]:
                            raise ValueError(f"Invalid image-only latent shape/key (expected C,1,H,W): {key}, {shape}")
                        latent_shape = shape
                    elif logical != "krea2_vl_embed":
                        raise ValueError(f"Unsupported conditioning key for image-only Krea2 preference caches: {key}")
                    elif len(shape) != 3:
                        raise ValueError(f"Invalid Krea2 conditioning shape (expected tokens,layers,hidden): {shape}")
                    normalized[logical] = (varlen, shape)
        except Exception as exc:
            raise ValueError(f"Invalid {kind} cache {path}: {exc}") from exc
    if latent_shape is None or "krea2_vl_embed" not in normalized:
        raise ValueError(f"Missing latents or Krea2 conditioning tensors for {item.item_key}")
    clean_item = _clone_item(item, prompt)
    try:
        batch = BucketBatchManager({tuple(item.bucket_size): [clean_item]}, batch_size=1)[0]
        for key, value in batch.items():
            if key == "timesteps":
                continue
            for tensor in value if isinstance(value, list) else [value]:
                if not tensor.is_floating_point() or not torch.isfinite(tensor.float()).all().item():
                    raise ValueError(f"Invalid or non-finite tensor: {key}")
    except Exception as exc:
        raise ValueError(f"Invalid cache tensors for {item.item_key}: {exc}") from exc
    # Dense conditioning shapes must agree inside a batch; varlen token counts may
    # differ but the selected-layer count and hidden width must still agree.
    schema = tuple((key, varlen, shape[1:] if varlen else shape) for key, (varlen, shape) in sorted(normalized.items()))
    return batch, schema


def _conditioning_equal(chosen: dict, rejected: dict) -> bool:
    for key in chosen.keys() | rejected.keys():
        if key in ("latents", "timesteps"):
            continue
        left, right = chosen.get(key), rejected.get(key)
        if isinstance(left, list) != isinstance(right, list):
            return False
        left = left if isinstance(left, list) else [left]
        right = right if isinstance(right, list) else [right]
        if len(left) != len(right):
            return False
        if any(
            not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or not torch.equal(a, b) for a, b in zip(left, right)
        ):
            return False
    return True


class PreferenceDataset(BaseDataset):
    """One prebatched dataset: batch_size counts pairs (FlowDPO) or samples (RFT)."""

    def __init__(
        self, samples, mode: str, manifest_path: str, batch_size: int, seed: int, shared_epoch=None, num_timestep_buckets=None
    ):
        super().__init__(
            resolution=tuple(samples[0][0].bucket_size),
            batch_size=batch_size,
            num_repeats=1,
            enable_bucket=True,
            architecture=ARCHITECTURE_KREA2,
        )
        self.mode = mode
        self.manifest_path = os.path.abspath(manifest_path)
        self.num_train_items = len(samples)
        self.seed = seed
        self.shared_epoch = shared_epoch
        self._canonical_buckets = {}
        self._rejected_by_chosen = {}
        for chosen, rejected, schema in samples:
            key = (tuple(chosen.bucket_size), schema)
            self._canonical_buckets.setdefault(key, []).append(chosen)
            if rejected is not None:
                self._rejected_by_chosen[chosen] = rejected
        self._canonical_buckets = {key: tuple(items) for key, items in sorted(self._canonical_buckets.items())}
        self.num_timestep_buckets = num_timestep_buckets
        self.current_epoch = int(shared_epoch.value) if shared_epoch is not None else 0
        self.shuffle_buckets()

    def shuffle_buckets(self):
        # Rebuild from immutable order on every epoch. Workers can skip epochs or
        # rewind on resume without depending on their previous shuffle history.
        self.batch_manager = BucketBatchManager(
            {key: list(items) for key, items in self._canonical_buckets.items()}, self.batch_size, self.num_timestep_buckets
        )
        # The shared manager's shuffle uses Python's global RNG. Restore it so
        # preference ordering does not alter the trainer's other random draws.
        state = random.getstate()
        try:
            random.seed(self.seed + self.current_epoch)
            self.batch_manager.shuffle()
        finally:
            random.setstate(state)

    def set_current_epoch(self, epoch):
        if self.shared_epoch is not None and self.shared_epoch.value != epoch:
            raise ValueError("shared_epoch does not match requested preference epoch")
        if self.current_epoch != epoch:
            self.current_epoch = epoch
            self.shuffle_buckets()

    def set_seed(self, seed, shared_epoch=None):
        self.seed = seed
        self.shared_epoch = shared_epoch
        self.current_epoch = int(shared_epoch.value) if shared_epoch is not None else 0
        self.shuffle_buckets()

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata.update(
            preference_mode=self.mode,
            preference_manifest=self.manifest_path,
            num_train_items=self.num_train_items,
            bucket_resolutions=sorted({key[0] for key in self._canonical_buckets}),
        )
        return metadata

    def __len__(self):
        return len(self.batch_manager)

    def __getitem__(self, idx):
        if self.shared_epoch is not None:
            self.set_current_epoch(int(self.shared_epoch.value))
        batch = self.batch_manager[idx]
        if self.mode == "flow_dpo":
            bucket, batch_index = self.batch_manager.bucket_batch_indices[idx]
            start = batch_index * self.batch_size
            chosen_items = self.batch_manager.buckets[bucket][start : start + self.batch_size]
            rejected_items = [self._rejected_by_chosen[item] for item in chosen_items]
            rejected_batch = BucketBatchManager({bucket: rejected_items}, self.batch_size)[0]
            batch["rejected_latents"] = rejected_batch["latents"]
        return batch


def build_preference_dataset_group(
    source_group: DatasetGroup, manifest_path: str, mode: str, batch_size: int, seed: int, shared_epoch=None
) -> DatasetGroup:
    """Validate the manifest/selected caches and wrap their descriptors for training.

    Held-out rows are validated for manifest contradictions/leakage but not loaded.
    In RFT neither rejected item lookup nor rejected cache reads are required.
    Validation retains at most a chosen/rejected pair of tensors at a time.
    """
    if mode not in ("rft", "flow_dpo"):
        raise ValueError("Preference mode must be rft or flow_dpo")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Preference batch_size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Preference seed must be an integer")
    pairs = _read_manifest(manifest_path)
    index = _source_index(source_group)
    samples = []
    selected_winners = set()
    timestep_buckets = {dataset.batch_manager.num_timestep_buckets for dataset in source_group.datasets}
    if len(timestep_buckets) != 1:
        raise ValueError("Source datasets must use the same num_timestep_buckets for preference training")
    for pair in pairs:
        winner_key = (pair.chosen, pair.prompt)
        if mode == "rft" and winner_key in selected_winners:
            continue
        if pair.chosen not in index:
            raise ValueError(f"Missing chosen cache item for pair {pair.pair_id}: {pair.chosen}")
        chosen = index[pair.chosen]
        chosen_batch, schema = _validate_and_load(chosen, pair.prompt)
        rejected = None
        if mode == "flow_dpo":
            if pair.rejected not in index:
                raise ValueError(f"Missing rejected cache item for pair {pair.pair_id}: {pair.rejected}")
            rejected = index[pair.rejected]
            rejected_batch, rejected_schema = _validate_and_load(rejected, pair.prompt)
            if chosen.bucket_size != rejected.bucket_size or chosen_batch["latents"].shape != rejected_batch["latents"].shape:
                raise ValueError(f"Chosen/rejected latent shape and bucket must match for pair {pair.pair_id}")
            if schema != rejected_schema or not _conditioning_equal(chosen_batch, rejected_batch):
                raise ValueError(f"Chosen/rejected conditioning tensors must be identical for pair {pair.pair_id}")
            del rejected_batch
        samples.append((_clone_item(chosen, pair.prompt), _clone_item(rejected, pair.prompt) if rejected else None, schema))
        selected_winners.add(winner_key)
        del chosen_batch
    dataset = PreferenceDataset(samples, mode, manifest_path, batch_size, seed, shared_epoch, timestep_buckets.pop())
    logger.info(
        "Prepared %s preference dataset: %d %s in %d batches",
        mode,
        dataset.num_train_items,
        "pairs" if mode == "flow_dpo" else "samples",
        len(dataset),
    )
    return DatasetGroup([dataset])
