"""Opt-in bounded microstep profiling; never enabled by default.

Wall times include CPU work and the final CUDA synchronization. CUDA values
are *stream elapsed times*, not kernel-only time. Named phases may be nested:
never add them to estimate total time. DataLoader time includes Accelerate
placement/prefetch work, not just filesystem IO. No tensors or prompts are saved.
"""

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time

import torch


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise ValueError("must be nonnegative")
    return result


class TrainingStepProfiler:
    def __init__(self, steps, warmup, device, output, *, metadata=None):
        if type(steps) is not int or type(warmup) is not int or steps <= 0 or warmup < 0:
            raise ValueError("profile steps must be positive and warmup must be nonnegative integers")
        self.steps, self.warmup = steps, warmup
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("phase profiling supports CPU or CUDA only")
        self.output = Path(output)
        self.metadata = dict(metadata or {})
        self.seen = 0
        self.rows = []
        self.current = None
        self._pending = None
        self._events = []

    @property
    def eligible(self):
        return self.warmup <= self.seen < self.warmup + self.steps

    def _start(self):
        if self.device.type == "cuda":
            # A clean boundary isolates this sample from preceding warmup work.
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        event = self._event()
        return start, event

    def _event(self):
        if self.device.type != "cuda":
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(self.device))
        return event

    def iter_batches(self, loader):
        iterator = iter(loader)
        while True:
            start, event = self._start() if self.eligible else (None, None)
            wait_start = time.perf_counter() if self.eligible else None
            try:
                batch = next(iterator)
            except StopIteration:
                self._pending = None
                return
            if self.eligible:
                self._pending = (start, event, (time.perf_counter() - wait_start) * 1000)
            yield batch

    def begin_step(self, optimizer_step, units=1):
        if self.current is not None:
            raise RuntimeError("profile step already active")
        if not self.eligible:
            return
        start, event, wait = self._pending if self._pending is not None else (*self._start(), None)
        self._pending = None
        self._start_time, self._start_event = start, event
        self.current = {
            "microstep": self.seen,
            "optimizer_step_before": int(optimizer_step),
            "units": int(units),
            "loader_and_placement_wall_ms": wait,
            "phases": {},
        }
        self._events = []

    @contextmanager
    def phase(self, name):
        if self.current is None:
            yield
            return
        start = time.perf_counter()
        event = self._event()
        try:
            yield
        finally:
            end_event = self._event()
            self.current["phases"].setdefault(name, []).append({"wall_ms": (time.perf_counter() - start) * 1000})
            if event is not None:
                self._events.append((event, end_event, self.current["phases"][name][-1]))

    def end_step(self, *, optimizer_updated, auxiliary_work=False):
        if self.current is not None:
            end_event = self._event()
            if end_event is not None:
                torch.cuda.synchronize(self.device)
                for start, end, entry in self._events:
                    entry["cuda_stream_ms"] = start.elapsed_time(end)
                self.current["cuda_stream_ms"] = self._start_event.elapsed_time(end_event)
                self.current["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
                self.current["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device)
            else:
                self.current.update(cuda_stream_ms=None, peak_allocated_bytes=None, peak_reserved_bytes=None)
            self.current["wall_ms"] = (time.perf_counter() - self._start_time) * 1000
            self.current["optimizer_updated"] = bool(optimizer_updated)
            self.current["sampling_or_save"] = bool(auxiliary_work)
            self.rows.append(self.current)
            self.current = None
            self._events = []
        self.seen += 1
        if len(self.rows) == self.steps and self.seen == self.warmup + self.steps:
            self.finish()

    def finish(self):
        if self.current is not None:
            raise RuntimeError("cannot finalize an incomplete profile step")
        times = sorted(row["wall_ms"] for row in self.rows)
        summary = None
        if times:
            summary = {
                "median_wall_ms": statistics.median(times),
                "p95_wall_ms": times[math.ceil(0.95 * len(times)) - 1],
                "units_per_second": sum(row["units"] for row in self.rows) / (sum(times) / 1000),
            }
        payload = {
            "schema_version": 1,
            "requested_microsteps": self.steps,
            "warmup_microsteps": self.warmup,
            "collected_microsteps": len(self.rows),
            "complete": len(self.rows) == self.steps,
            "device": str(self.device),
            "metadata": self.metadata,
            "measurement_notes": [
                "Opt-in synchronized diagnostic; compare throughput separately with profiling disabled.",
                "CUDA durations are current-stream elapsed time, not kernel-only execution time.",
                "Nested phases are inclusive and must not be summed.",
                "Loader wall time includes Accelerate device placement; epoch setup and iterator creation are excluded.",
                "Sampling/save within a measured step is included and flagged; final save is outside this window.",
                "p95 is nearest-rank; units are per-device images or preference pairs, not optimizer updates.",
            ],
            "summary": summary,
            "steps": self.rows,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.output.parent, delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, indent=2, allow_nan=False)
                handle.write("\n")
            os.replace(temporary, self.output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return payload


def validate_profile_args(args):
    steps = getattr(args, "profile_steps", 0)
    warmup = getattr(args, "profile_warmup_steps", 5)
    if type(steps) is not int or type(warmup) is not int or steps < 0 or warmup < 0:
        raise ValueError("profile_steps/profile_warmup_steps must be nonnegative integers")
    return steps, warmup


def create_step_profiler(args, accelerator):
    steps, warmup = validate_profile_args(args)
    if not steps:
        return None
    path = Path(getattr(args, "profile_output", None) or Path(args.output_dir) / "training-profile.json")
    rank = accelerator.process_index
    if accelerator.num_processes > 1:
        path = path.with_name(f"{path.stem}.rank{rank}{path.suffix}")
    mode = getattr(args, "post_training", "none")
    return TrainingStepProfiler(steps, warmup, accelerator.device, path, metadata={
        "mode": mode,
        "unit": "pairs" if mode in ("flow_dpo", "flow_cpo") else "images",
        "rank": rank,
        "world_size": accelerator.num_processes,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "mixed_precision": args.mixed_precision,
        "gradient_checkpointing": args.gradient_checkpointing,
        "effective_max_train_steps": args.max_train_steps,
    })
