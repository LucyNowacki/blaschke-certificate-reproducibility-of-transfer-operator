"""Regression tests for the non-circular deployment manifest boundary."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import refresh_deployment_manifest as refresh


class DeploymentManifestTests(unittest.TestCase):
    def test_generated_release_metadata_is_excluded_from_global_manifest(self) -> None:
        candidates = (
            "MANIFEST.sha256",
            "release/reproducibility_manifest.json",
            "release/source-only-replay-inventory.json",
            "kept.txt",
        )
        with tempfile.TemporaryDirectory(prefix="blaschke-manifest-boundary-") as raw:
            root = Path(raw)
            for relative in candidates:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(relative + "\n", encoding="utf-8")
            completed = mock.Mock(
                stdout=("\0".join(candidates) + "\0").encode("utf-8")
            )
            with mock.patch.object(refresh.subprocess, "run", return_value=completed):
                selected = refresh.authoritative_paths(root)
        self.assertEqual(selected, (Path("kept.txt"),))


if __name__ == "__main__":
    unittest.main()
