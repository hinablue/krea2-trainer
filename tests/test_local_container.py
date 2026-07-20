import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LocalContainerDefinitionTests(unittest.TestCase):
    def test_local_image_is_multi_arch_and_reuses_explicit_launcher(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3", dockerfile)
        self.assertIn("uv sync --frozen --extra cu130", dockerfile)
        self.assertIn('ENTRYPOINT ["/opt/krea2-trainer/runpod/entrypoint.sh"]', dockerfile)
        self.assertNotIn("COPY . ", dockerfile)

    def test_compose_uses_gpu_workspace_and_no_auto_training(self):
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("gpus: all", compose)
        self.assertIn("${KREA2_WORKSPACE:-./workspace}:/workspace", compose)
        self.assertIn('AUTO_START_TRAINING: "0"', compose)
        self.assertIn('command: ["sleep", "infinity"]', compose)

    def test_compose_parses(self):
        env = {
            "PATH": __import__("os").environ["PATH"],
            "HF_TOKEN": "",
            "WANDB_API_KEY": "",
        }
        subprocess.run(
            ["docker", "compose", "-f", str(ROOT / "docker-compose.yml"), "config", "--quiet"],
            check=True,
            cwd=ROOT,
            env=env,
        )


if __name__ == "__main__":
    unittest.main()
