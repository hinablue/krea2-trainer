# Krea2 Structure–Detail TQD Implementation Plan

> **For Hermes:** Execute this plan in small, independently verified commits. Preserve unrelated working-tree changes.

**Goal:** Add an opt-in, per-image Structure–Detail TQD mode for Krea2 LoRA training.

**Architecture:** Krea2 is an image Flow Matching trainer, so this adapts TQD's motion/visual split to structure/detail quality. A dataset-local JSONL manifest supplies normalized per-cache structure and detail scores. Scores travel through `ItemInfo` and `BucketBatchManager`; the new sampler maps a sample-specific Beta CDF draw back into Krea2's native resolution-aware logit-normal schedule. A separate optional deterministic quality weighting approximates TQD sample retention without changing epoch cardinality or distributed batch alignment.

**Tech Stack:** Python, PyTorch, argparse, TOML dataset config, safetensors cache loader.

---

### Task 1: Add score-manifest plumbing

**Objective:** Accept `tqd_score_file` in dataset TOML, validate a JSONL manifest keyed by cache filename, and emit `[B]` structure/detail tensors with each training batch.

**Files:**
- Modify: `src/krea2_trainer/dataset/config_utils.py`
- Modify: `src/krea2_trainer/dataset/image_video_dataset.py`
- Modify: `src/krea2_trainer/dataset/bucket.py`
- Create: `tests/test_tqd_dataset.py`

**Verification:** Test full score coverage, duplicate/out-of-range errors, and score tensor batch order. Run `python -m compileall src` and targeted unittest.

**Commit:** `feat: add TQD score manifest dataset plumbing`

### Task 2: Add Krea2-native conditional sampler

**Objective:** Add `--timestep_sampling tqd_krea2_shift`, preserving native Krea2 shifted-logit-normal behavior when structure and detail scores are equal.

**Files:**
- Modify: `src/krea2_trainer/training/parser_common.py`
- Modify: `src/krea2_trainer/training/trainer_base.py`
- Create: `tests/test_tqd_sampler.py`

**Verification:** Parser accepts the mode; equal scores statistically match the baseline path; high-structure scores yield higher mean timestep than high-detail scores; invalid/missing score tensors fail clearly.

**Commit:** `feat: add Krea2 structure-detail TQD sampler`

### Task 3: Add optional deterministic sample-quality weighting

**Objective:** Implement opt-in per-sample weighting proportional to `max(structure, detail)`, normalized per batch to retain stable loss scale without changing loader cardinality.

**Files:**
- Modify: `src/krea2_trainer/training/parser_common.py`
- Modify: `src/krea2_trainer/training/trainer_base.py`
- Extend: `tests/test_tqd_sampler.py`

**Verification:** Weighting is off by default, normalized weights have mean one, and a lower-quality sample contributes a lower loss when enabled.

**Commit:** `feat: add optional TQD quality weighting`

### Task 4: Integration review

**Objective:** Verify tests, parser help, source compilation, clean staged diff for only TQD files, and obtain independent spec/quality review.

**Verification:** `python -m unittest discover -s tests -v`, `python -m compileall src`, training entrypoint `--help`, and `git diff`/`git status` inspection.
