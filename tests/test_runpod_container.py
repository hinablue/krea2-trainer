import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RunpodContainerDefinitionTests(unittest.TestCase):
    def test_verified_cuda_13_pytorch_291_base_and_locked_sync(self):
        dockerfile = (ROOT / "runpod/Dockerfile").read_text(encoding="utf-8")
        self.assertIn("FROM runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404", dockerfile)
        self.assertIn("uv sync --frozen --extra cu130", dockerfile)
        self.assertIn('ENTRYPOINT ["/opt/krea2-trainer/runpod/entrypoint.sh"]', dockerfile)

    def test_container_only_copies_allowlisted_project_content(self):
        dockerfile = (ROOT / "runpod/Dockerfile").read_text(encoding="utf-8")
        self.assertNotIn("COPY . ", dockerfile)
        for forbidden in ("models/", "datasets/", "output/", ".env"):
            self.assertNotIn(f"COPY {forbidden}", dockerfile)

    def test_entrypoint_does_not_start_training_by_default(self):
        entrypoint = (ROOT / "runpod/entrypoint.sh").read_text(encoding="utf-8")
        self.assertIn('AUTO_START_TRAINING:-0', entrypoint)
        self.assertIn("No GPU training was started", entrypoint)
        self.assertIn("exec sleep infinity", entrypoint)

    def test_shell_files_parse(self):
        subprocess.run(
            [
                "bash",
                "-n",
                str(ROOT / "runpod/entrypoint.sh"),
                str(ROOT / "runpod/train.sh"),
                str(ROOT / "scripts/train_from_env.sh"),
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
