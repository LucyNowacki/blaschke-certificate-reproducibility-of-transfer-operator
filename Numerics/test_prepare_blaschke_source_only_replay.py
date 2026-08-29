"""Reduced tests for fail-closed source-only replay preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import prepare_blaschke_source_only_replay as preparation
from run_blaschke_clean_room_replay import run_replay


HERE = Path(__file__).resolve().parent
POLICY = HERE / "blaschke_source_only_replay_policy.json"


class SourceOnlyReplayPreparationTests(unittest.TestCase):
    def _bundle(self, parent: Path) -> Path:
        root = parent / preparation.BUNDLE_ROOT_NAME
        numerics = root / "Numerics"
        output = numerics / "outputs/blaschke_deformation_certifier/data"
        output.mkdir(parents=True)
        files = {
            root / ".final_deployment_generated": "marker\n",
            root / "REPLAY.md": "replay\n",
            numerics / "blaschke_deformation_certifier_template.ipynb": "{}\n",
            numerics / "blaschke_deformation_certifier.ipynb": "generated source\n",
            numerics / "blaschke_deformation_certifier_thesis_math.ipynb": "executed\n",
            numerics / "helper.py": "value = 1\n",
            output / "generated.csv": "x\n1\n",
        }
        for path, text in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        shutil.copyfile(POLICY, numerics / POLICY.name)

        staged_paths = sorted(
            path for path in root.rglob("*") if path.is_file()
        )
        records = [
            {
                "archive_path": path.relative_to(root).as_posix(),
                "sha256": preparation.sha256_file(path),
            }
            for path in staged_paths
        ]
        inventory = preparation.make_archive_inventory(
            staging_root=root,
            file_records=records,
        )
        inventory_path = root / preparation.INVENTORY_NAME
        inventory_path.write_text(
            json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
        )
        records.append({
            "archive_path": preparation.INVENTORY_NAME,
            "sha256": preparation.sha256_file(inventory_path),
        })
        (root / preparation.INTERNAL_MANIFEST_NAME).write_text(
            json.dumps({
                "bundle_format": "blaschke-deformation-certifier-reproducibility-v3",
                "files": records,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        return root

    def test_prepare_removes_only_generated_and_preserves_immutable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="source-only-prep-test-") as temp:
            root = self._bundle(Path(temp))
            template = root / "Numerics/blaschke_deformation_certifier_template.ipynb"
            helper = root / "Numerics/helper.py"
            hashes = {
                template: preparation.sha256_file(template),
                helper: preparation.sha256_file(helper),
            }
            result = run_replay(
                bundle_root=root,
                kernel_name="unused",
                assembly_workers=1,
                surface_workers=1,
                prepare_only=True,
                published_archive=None,
            )
            self.assertEqual(result["status"], "prepared")
            self.assertFalse((root / "Numerics/outputs").exists())
            self.assertFalse(
                (root / "Numerics/blaschke_deformation_certifier.ipynb").exists()
            )
            self.assertFalse(
                (root / "Numerics/blaschke_deformation_certifier_thesis_math.ipynb").exists()
            )
            for path, digest in hashes.items():
                self.assertTrue(path.is_file())
                self.assertEqual(preparation.sha256_file(path), digest)
            checked = preparation.check_prepared_bundle(root)
            self.assertEqual(checked["status"], "source-only replay root prepared")

    def test_inventory_drift_is_rejected_before_deletion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="source-only-drift-test-") as temp:
            root = self._bundle(Path(temp))
            generated = root / (
                "Numerics/outputs/blaschke_deformation_certifier/data/generated.csv"
            )
            (root / "unexpected.txt").write_text("drift\n", encoding="utf-8")
            with self.assertRaises(preparation.SourceOnlyReplayError):
                preparation.prepare_bundle(root)
            self.assertTrue(generated.is_file())

    def test_traversal_inventory_path_is_rejected_before_deletion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="source-only-traversal-test-") as temp:
            parent = Path(temp)
            root = self._bundle(parent)
            outside = parent / "outside.txt"
            outside.write_text("preserve\n", encoding="utf-8")
            inventory_path = root / preparation.INVENTORY_NAME
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            inventory["files"][0]["path"] = "../outside.txt"
            inventory_path.write_text(
                json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaises(preparation.SourceOnlyReplayError):
                preparation.prepare_bundle(root)
            self.assertEqual(outside.read_text(encoding="utf-8"), "preserve\n")

    def test_broad_or_renamed_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="source-only-root-test-") as temp:
            parent = Path(temp)
            root = self._bundle(parent)
            with self.assertRaises(preparation.SourceOnlyReplayError):
                preparation.prepare_bundle(parent)
            renamed = parent / "renamed-bundle"
            root.rename(renamed)
            with self.assertRaises(preparation.SourceOnlyReplayError):
                preparation.prepare_bundle(renamed)


if __name__ == "__main__":
    unittest.main()
