import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from krea2_trainer.scripts.download_models import ASSETS, provision_models


class ModelProvisioningTests(unittest.TestCase):
    def fake_downloader(self, **kwargs):
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"complete")
        return str(target)

    @patch("krea2_trainer.scripts.download_models.shutil.disk_usage")
    def test_downloads_default_layout_and_forwards_revision(self, disk_usage):
        disk_usage.return_value = Mock(free=100 * 1024**3)
        calls = []

        def download(**kwargs):
            calls.append(kwargs)
            return self.fake_downloader(**kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            paths = provision_models(Path(tmp), revision="commit-sha", token="secret", downloader=download)
            self.assertEqual({path.relative_to(tmp).as_posix() for path in paths}, set(ASSETS))
            self.assertEqual({call["filename"] for call in calls}, set(ASSETS))
            self.assertTrue(all(call["revision"] == "commit-sha" for call in calls))
            self.assertTrue(all(call["token"] == "secret" for call in calls))
            self.assertTrue(all(call["force_download"] is False for call in calls))

    @patch("krea2_trainer.scripts.download_models.shutil.disk_usage")
    def test_if_missing_reuses_complete_files(self, disk_usage):
        disk_usage.return_value = Mock(free=100 * 1024**3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ASSETS:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"ready")
            downloader = Mock()
            provision_models(root, downloader=downloader)
            downloader.assert_not_called()

    @patch("krea2_trainer.scripts.download_models.shutil.disk_usage")
    def test_if_missing_only_fetches_missing_assets(self, disk_usage):
        disk_usage.return_value = Mock(free=100 * 1024**3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / next(iter(ASSETS))
            existing.parent.mkdir(parents=True, exist_ok=True)
            existing.write_bytes(b"ready")
            calls = []

            def download(**kwargs):
                calls.append(kwargs["filename"])
                return self.fake_downloader(**kwargs)

            provision_models(root, downloader=download)
            self.assertNotIn(existing.relative_to(root).as_posix(), calls)
            self.assertEqual(len(calls), len(ASSETS) - 1)

    @patch("krea2_trainer.scripts.download_models.shutil.disk_usage")
    def test_force_redownloads_every_asset(self, disk_usage):
        disk_usage.return_value = Mock(free=100 * 1024**3)
        with tempfile.TemporaryDirectory() as tmp:
            downloader = Mock(side_effect=self.fake_downloader)
            provision_models(Path(tmp), mode="force", downloader=downloader)
            self.assertEqual(downloader.call_count, len(ASSETS))
            self.assertTrue(all(call.kwargs["force_download"] for call in downloader.call_args_list))

    def test_never_rejects_missing_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "MODEL_FETCH=never"):
                provision_models(Path(tmp), mode="never")

    @patch("krea2_trainer.scripts.download_models.shutil.disk_usage")
    def test_low_disk_fails_before_download(self, disk_usage):
        disk_usage.return_value = Mock(free=1)
        with tempfile.TemporaryDirectory() as tmp:
            downloader = Mock()
            with self.assertRaisesRegex(OSError, "Insufficient disk space"):
                provision_models(Path(tmp), downloader=downloader)
            downloader.assert_not_called()

    @patch("krea2_trainer.scripts.download_models.shutil.disk_usage")
    def test_download_failure_propagates(self, disk_usage):
        disk_usage.return_value = Mock(free=100 * 1024**3)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "network failed"):
                provision_models(Path(tmp), downloader=Mock(side_effect=RuntimeError("network failed")))


if __name__ == "__main__":
    unittest.main()
