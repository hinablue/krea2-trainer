"""Fresh-only performance recipe and bounded benchmark CLI contracts."""

import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

from tests import test_train_from_env as launcher_fixtures

ROOT = launcher_fixtures.ROOT


class FreshPerformanceTests(unittest.TestCase):
    def launch(self, directory, performance=None, overrides=None, mode="tqd"):
        env, log = launcher_fixtures.TrainingLauncherTests().base_env(directory)
        for key in ("TRAIN_PERFORMANCE", "ENABLE_COMPILE", "ENABLE_FP8", "COMPILE_MODE", "COMPILE_DYNAMIC"):
            env.pop(key, None)
        env.update(TRAIN_MODE=mode, CACHE_MODE="none")
        if performance is not None:
            env["TRAIN_PERFORMANCE"] = performance
        env.update(overrides or {})
        result = subprocess.run(["bash", str(ROOT / "scripts/train_from_env.sh")], env=env,
                                capture_output=True, text=True)
        argv = shlex.split(log.read_text()) if log.exists() else []
        return result, argv

    def test_balanced_preserves_existing_default(self):
        with tempfile.TemporaryDirectory() as temp:
            result, argv = self.launch(Path(temp))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("--compile", argv)
            self.assertIn("performance=balanced", result.stdout)

    def test_throughput_compiles_only_without_changing_recipe(self):
        for mode, sampler in (("standard", "krea2_shift"), ("tqd", "tqd_krea2_shift")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                result, argv = self.launch(Path(temp), "throughput", mode=mode)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--compile", argv)
                self.assertEqual(argv[argv.index("--compile_mode") + 1], "max-autotune-no-cudagraphs")
                self.assertEqual(argv[argv.index("--compile_dynamic") + 1], "auto")
                self.assertIn("--gradient_checkpointing", argv)
                self.assertNotIn("--fp8_base", argv)
                for key, value in (("--mixed_precision", "bf16"), ("--network_dim", "32"),
                                   ("--network_alpha", "16"), ("--optimizer_type", "Adopt_adv"),
                                   ("--learning_rate", "5e-5"), ("--timestep_sampling", sampler)):
                    self.assertEqual(argv[argv.index(key) + 1], value)
                self.assertNotIn("--post_training", argv)
                self.assertIn("performance=throughput compile=1", result.stdout)

    def test_explicit_compile_override_wins(self):
        with tempfile.TemporaryDirectory() as temp:
            result, argv = self.launch(Path(temp), "throughput", {"ENABLE_COMPILE": "0"})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("--compile", argv)
            self.assertIn("performance=throughput compile=0", result.stdout)

    def test_bad_profile_fails_before_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            result, argv = self.launch(Path(temp), "unknown")
            self.assertEqual(result.returncode, 2)
            self.assertEqual(argv, [])
            self.assertIn("TRAIN_PERFORMANCE", result.stderr)


class FreshBenchmarkCLITests(unittest.TestCase):
    @staticmethod
    def memory_helper():
        import importlib.util
        spec = importlib.util.spec_from_file_location("fresh_benchmark_under_test", ROOT / "scripts/benchmark_fresh_training.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.profile_window_memory

    def test_profile_memory_uses_window_max_not_last_step(self):
        report = {"complete": True, "collected_microsteps": 2, "steps": [
            {"peak_allocated_bytes": 100, "peak_reserved_bytes": 120},
            {"peak_allocated_bytes": 50, "peak_reserved_bytes": 60},
        ]}
        self.assertEqual(self.memory_helper()(report, 2),
                         {"peak_allocated_bytes": 100, "peak_reserved_bytes": 120})

    def test_profile_memory_rejects_incomplete_or_missing_measurements(self):
        helper = self.memory_helper()
        for report in (
            {"complete": False, "collected_microsteps": 0, "steps": []},
            {"complete": True, "collected_microsteps": 2, "steps": []},
            {"complete": True, "collected_microsteps": 1, "steps": [
                {"peak_allocated_bytes": None, "peak_reserved_bytes": None}]},
        ):
            with self.subTest(report=report), self.assertRaises(ValueError):
                helper(report, report["collected_microsteps"])

    def test_help_is_fresh_only(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/benchmark_fresh_training.py"), "--help"],
                                capture_output=True, text=True, timeout=40)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--training-mode {standard,tqd}", result.stdout)
        self.assertIn("--variant {eager,compiled}", result.stdout)
        self.assertNotIn("--post_training", result.stdout)

    def test_unbounded_request_rejected_before_output_or_model_access(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "must-not-exist"
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/benchmark_fresh_training.py"),
                "--dit", str(Path(temp) / "missing.safetensors"), "--output", str(output), "--steps", "1000",
            ], capture_output=True, text=True, timeout=40)
            self.assertEqual(result.returncode, 2)
            self.assertIn("2..24", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
