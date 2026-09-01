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
    def test_historical_phase4_has_explicit_theorem_only_skip_contract(self) -> None:
        notebook = json.loads(builder.TEMPLATE.read_text(encoding="utf-8"))
        cell35 = builder._source(
            builder._unique_cell(notebook["cells"], "# Cell 35\n")
        )
        for marker in (
            'os.environ.get("BLASCHKE_SKIP_HISTORICAL_PHASE4", "0")',
            "elif HISTORICAL_PHASE4_DIAGNOSTIC_SKIPPED:",
            '"execution_status": "diagnostic_skipped"',
            '"diagnostic_skipped": True',
            '"theorem_authoritative": False',
            "historical_phase4_result = rebuild_historical_phase4(",
        ):
            self.assertIn(marker, cell35)

    def test_hardy_starting_audit_has_explicit_theorem_only_skip_contract(self) -> None:
        notebook = json.loads(builder.TEMPLATE.read_text(encoding="utf-8"))
        hardy = builder._source(
            builder._unique_cell(notebook["cells"], "# Cell 102A\n")
        )
        summary = builder._source(
            builder._unique_cell(notebook["cells"], "# Cell 104\n")
        )
        for marker in (
            'os.environ.get("BLASCHKE_SKIP_HARDY_STARTING_AUDIT", "0")',
            "STARTING_HARDY_MATRIX_AUDIT = None",
            "if not HARDY_MATRIX_SKIP_STARTING_AUDIT:",
        ):
            self.assertIn(marker, hardy)
        for marker in (
            '"hardy_starting_audit_skipped": bool(HARDY_MATRIX_SKIP_STARTING_AUDIT)',
            '"hardy_starting_audit_authoritative": False',
            "None if HARDY_MATRIX_SKIP_STARTING_AUDIT else int(HARDY_MATRIX_STARTING_BITS)",
            "None if HARDY_MATRIX_SKIP_STARTING_AUDIT else str(STARTING_HARDY_MATRIX_AUDIT",
        ):
            self.assertIn(marker, summary)

    def test_builder_rejects_short_theorem_config_without_policing_prose(self) -> None:
        short_config = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": "Historical presentation-only radius: 2.47367.",
                },
                {
                    "cell_type": "code",
                    "source": 'r_target: str = "2.473669807791324"\n',
                },
            ]
        }
        with self.assertRaisesRegex(AssertionError, "canonical selected Hardy radius"):
            builder._validate_exact_radius_comparators(short_config)

    def test_builder_rejects_tolerant_inline_radius_selection(self) -> None:
        tolerant = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": (
                        'r_target: str = "2.473669807791324109321273260"\n'
                        "def _rp_close_decimal(left, right, tolerance='1e-12'):\n"
                        "    return abs(left - right) <= tolerance\n"
                        "_rp_tail_candidates = []\n"
                    ),
                }
            ]
        }
        with self.assertRaisesRegex(AssertionError, "tolerance/float comparator"):
            builder._validate_exact_radius_comparators(tolerant)

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
