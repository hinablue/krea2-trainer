import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TrainingLauncherTests(unittest.TestCase):
    def make_fixture(self, directory: Path):
        log = directory / "commands.log"
        fake = directory / "fake-command"
        fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$COMMAND_LOG"\n', encoding="utf-8")
        fake.chmod(0o755)
        files = {}
        for name in ("dataset.toml", "dit.safetensors", "vae.safetensors", "te.safetensors"):
            path = directory / name
            path.write_text("ready", encoding="utf-8")
            files[name] = path
        return log, fake, files

    def base_env(self, directory: Path):
        log, fake, files = self.make_fixture(directory)
        env = os.environ.copy()
        env.pop("WANDB_API_KEY", None)
        env.pop("HF_TOKEN", None)
        env.update(
            PROJECT_DIR=str(ROOT),
            DATASET_CONFIG=str(files["dataset.toml"]),
            RAW_DIT=str(files["dit.safetensors"]),
            VAE=str(files["vae.safetensors"]),
            TEXT_ENCODER=str(files["te.safetensors"]),
            OUTPUT_DIR=str(directory / "output"),
            LOGGING_DIR=str(directory / "logs"),
            MODEL_DIR=str(directory / "models"),
            PYTHON_BIN=str(fake),
            ACCELERATE_BIN=str(fake),
            COMMAND_LOG=str(log),
            ENABLE_COMPILE="0",
        )
        return env, log

    def test_all_cache_runs_in_fixed_order_then_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, log = self.base_env(Path(tmp))
            env.update(CACHE_MODE="all", CACHE_SKIP_EXISTING="1", WANDB_API_KEY="super-secret-marker")
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/train_from_env.sh"), "--learning_rate", "0.1"],
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )
            lines = log.read_text(encoding="utf-8").splitlines()
            self.assertNotIn("super-secret-marker", result.stdout + result.stderr + "\n".join(lines))
            self.assertIn("cache_latents", lines[0])
            self.assertIn("--skip_existing", lines[0])
            self.assertIn("cache_text_encoder", lines[1])
            self.assertIn("train_lora", lines[2])
            self.assertIn("--learning_rate 0.1", lines[2])

    def test_force_cache_omits_skip_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, log = self.base_env(Path(tmp))
            env.update(CACHE_MODE="latents", CACHE_SKIP_EXISTING="0", FORCE_REBUILD_CACHE="1")
            subprocess.run(["bash", str(ROOT / "scripts/train_from_env.sh")], env=env, check=True)
            self.assertNotIn("--skip_existing", log.read_text(encoding="utf-8").splitlines()[0])

    def test_incompatible_cache_flags_fail_before_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, log = self.base_env(Path(tmp))
            env.update(CACHE_MODE="all", CACHE_SKIP_EXISTING="1", FORCE_REBUILD_CACHE="1")
            result = subprocess.run(["bash", str(ROOT / "scripts/train_from_env.sh")], env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mutually exclusive", result.stderr)
            self.assertFalse(log.exists())

    def test_tqd_mode_selects_tqd_sampler(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, log = self.base_env(Path(tmp))
            env.update(CACHE_MODE="none", TRAIN_MODE="tqd")
            subprocess.run(["bash", str(ROOT / "scripts/train_from_env.sh")], env=env, check=True)
            self.assertIn("--timestep_sampling tqd_krea2_shift", log.read_text(encoding="utf-8"))

    def test_dotenv_does_not_execute_shell_or_override_process_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            env_file = directory / "train.env"
            marker = directory / "owned"
            env_file.write_text(f"OUTPUT_NAME=from-file\nEVIL=$(touch {marker})\nCACHE_MODE=none\n", encoding="utf-8")
            env, log = self.base_env(directory)
            env.update(ENV_FILE=str(env_file), OUTPUT_NAME="from-process", MODEL_FETCH="never", SKIP_CUDA_CHECK="1", CACHE_MODE="none")
            subprocess.run(["bash", str(ROOT / "runpod/train.sh")], env=env, check=True)
            self.assertFalse(marker.exists())
            self.assertIn("--output_name from-process", log.read_text(encoding="utf-8"))


class RunpodWrapperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.bin_dir = self.tmp_path / "bin"
        self.bin_dir.mkdir()

        self.log_file = self.tmp_path / "calls.log"
        self.log_file.touch()

        for cmd in ["python", "accelerate"]:
            mock = self.bin_dir / cmd
            mock.write_text(f'#!/usr/bin/env bash\necho "{cmd} $@" >> "{self.log_file}"\n')
            mock.chmod(0o755)

        self.env = os.environ.copy()
        self.env.pop("WANDB_API_KEY", None)
        self.env.pop("HF_TOKEN", None)
        self.env["PATH"] = f"{self.bin_dir}:{self.env.get('PATH', '')}"

        # Avoid CUDA check failing in mock environment
        self.env["SKIP_CUDA_CHECK"] = "1"
        self.env["OUTPUT_DIR"] = str(self.tmp_path / "output")
        self.env["LOGGING_DIR"] = str(self.tmp_path / "logs")
        self.env["MODEL_DIR"] = str(self.tmp_path / "models")

        self.script = Path(__file__).parent.parent / "runpod" / "train.sh"

    def tearDown(self):
        self.tmp.cleanup()

    def test_runpod_default_behavior(self):
        scripts_dir = self.tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "train_from_env.sh").symlink_to(Path(__file__).parent.parent / "scripts" / "train_from_env.sh")

        self.env["PROJECT_DIR"] = str(self.tmp_path)
        self.env["DATASET_CONFIG"] = str(self.tmp_path / "dataset.toml")

        for f in [self.env["DATASET_CONFIG"]]:
            Path(f).touch()

        models_dir = self.tmp_path / "models"
        models_dir.mkdir()

        # Mock download_models to create the files
        mock_py = self.bin_dir / "python"
        mock_py.write_text(f'''#!/usr/bin/env bash
if [[ "$*" == *download_models* ]]; then
    mkdir -p "{models_dir}/diffusion_models" "{models_dir}/text_encoders" "{models_dir}/vae"
    touch "{models_dir}/diffusion_models/krea2_raw_bf16.safetensors"
    touch "{models_dir}/text_encoders/qwen3vl_4b_bf16.safetensors"
    touch "{models_dir}/vae/qwen_image_vae.safetensors"
fi
echo "python $@" >> "{self.log_file}"
''')

        res = subprocess.run([str(self.script)], env=self.env, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, msg=res.stderr)

        calls = self.log_file.read_text().strip().split('\n')
        self.assertTrue(any("download_models" in call for call in calls))
        self.assertTrue(any("cache_latents" in call for call in calls))
        self.assertTrue(any("cache_text_encoder" in call for call in calls))
        self.assertTrue(any("accelerate launch" in call for call in calls))

    def test_runpod_rejects_incompatible_cache_flags(self):
        self.env["CACHE_SKIP_EXISTING"] = "1"
        self.env["FORCE_REBUILD_CACHE"] = "1"
        res = subprocess.run([str(self.script)], env=self.env, capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Incompatible cache flags", res.stderr)

    def test_runpod_rejects_missing_dataset(self):
        self.env["DATASET_CONFIG"] = str(self.tmp_path / "missing.toml")
        res = subprocess.run([str(self.script)], env=self.env, capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Dataset TOML not found", res.stderr)

if __name__ == "__main__":
    unittest.main()
