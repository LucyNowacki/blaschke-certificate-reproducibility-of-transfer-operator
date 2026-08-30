"""Fail-closed tests for the locked source-notebook template."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock


NUMERICS_DIR = Path(__file__).resolve().parent
DEPLOYMENT_ROOT = NUMERICS_DIR.parent
MANIFEST = DEPLOYMENT_ROOT / "MANIFEST.sha256"
if str(NUMERICS_DIR) not in sys.path:
    sys.path.insert(0, str(NUMERICS_DIR))

import build_blaschke_deformation_certifier as builder


class LockedTemplateTests(unittest.TestCase):
    def test_checked_in_template_matches_builder_lock(self) -> None:
        self.assertEqual(
            builder._sha256(builder.TEMPLATE),
            builder.LOCKED_TEMPLATE_SHA256,
        )

    def test_manifest_authenticates_checked_in_template_lock(self) -> None:
        records = {
            relative: digest
            for digest, relative in (
                line.split("  ", 1)
                for line in MANIFEST.read_text(encoding="utf-8").splitlines()
            )
        }
        relative = builder.TEMPLATE.relative_to(DEPLOYMENT_ROOT).as_posix()
        self.assertIn(relative, records)
        self.assertEqual(records[relative], builder.LOCKED_TEMPLATE_SHA256)
        self.assertEqual(records[relative], builder._sha256(builder.TEMPLATE))

    def test_ordinary_build_rejects_template_drift(self) -> None:
        with tempfile.TemporaryDirectory(prefix="blaschke-template-drift-") as root:
            template = Path(root) / "template.ipynb"
            target = Path(root) / "source.ipynb"
            shutil.copyfile(builder.TEMPLATE, template)
            template.write_bytes(template.read_bytes() + b"\n")
            with mock.patch.object(builder, "TEMPLATE", template), mock.patch.object(
                builder, "TARGET", target
            ), self.assertRaisesRegex(RuntimeError, "template digest changed"):
                builder.build()
            self.assertFalse(target.exists())

    def test_locked_build_writes_an_output_free_source_notebook(self) -> None:
        with tempfile.TemporaryDirectory(prefix="blaschke-template-build-") as root:
            target = Path(root) / "source.ipynb"
            with mock.patch.object(builder, "TARGET", target):
                builder.build()
            notebook = json.loads(target.read_text(encoding="utf-8"))
            code_cells = [
                cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
            ]
            self.assertTrue(code_cells)
            self.assertTrue(all(cell.get("execution_count") is None for cell in code_cells))
            self.assertTrue(all(cell.get("outputs") == [] for cell in code_cells))


if __name__ == "__main__":
    unittest.main()
