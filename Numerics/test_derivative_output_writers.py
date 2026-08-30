"""Focused non-heavy checks for the deterministic derivative-output writers."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import tempfile
import unittest

import nbformat
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPLACEMENTS_PATH = HERE / "plotting_cell_replacements.json"
CURATED_NOTEBOOK_PATH = HERE / "blaschke_deformation_certifier_thesis_math.ipynb"


def _replacement_source(cell_id: str) -> str:
    payload = json.loads(REPLACEMENTS_PATH.read_text(encoding="utf-8"))
    record = payload[cell_id]
    if isinstance(record, str):
        return record
    return "\n".join(record["source_lines"]).rstrip() + "\n"


class DerivativeOutputWriterTests(unittest.TestCase):
    maxDiff = None

    def test_atomic_csv_and_parquet_writer_preserves_dataframe(self) -> None:
        source = _replacement_source("0e01d379")
        tree = ast.parse(source)
        writer_names = {
            "_atomic_dataframe_write",
            "save_csv_dataframe",
            "save_dataframe",
        }
        writer_module = ast.Module(
            body=[
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in writer_names
            ],
            type_ignores=[],
        )
        namespace = {
            "Path": Path,
            "np": np,
            "os": os,
            "pd": pd,
            "tempfile": tempfile,
        }
        exec(compile(writer_module, "cell-0e01d379-writers", "exec"), namespace)
        frame = pd.DataFrame(
            {
                "N": pd.Series([12, 20], dtype="int64"),
                "value": pd.Series([1.25, 2.5], dtype="float64"),
                "label": pd.Series(["alpha^1", "mu^1"], dtype="object"),
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            namespace["DATA_DIR"] = data_dir
            namespace["save_dataframe"](frame, "paired")
            self.assertEqual(
                (data_dir / "paired.csv").read_bytes(),
                frame.to_csv(index=False).encode("utf-8"),
            )
            parquet_frame = pd.read_parquet(data_dir / "paired.parquet")
            pd.testing.assert_frame_equal(parquet_frame, frame, check_dtype=False)
            self.assertEqual(str(parquet_frame["N"].dtype), "int64")
            self.assertEqual(str(parquet_frame["value"].dtype), "float64")
            self.assertEqual(
                [path.name for path in data_dir.iterdir() if path.name.startswith(".")],
                [],
            )
            invalid_path = data_dir / "invalid.bin"
            with self.assertRaisesRegex(ValueError, "Unsupported dataframe format"):
                namespace["_atomic_dataframe_write"](
                    frame,
                    invalid_path,
                    file_format="invalid",
                )
            self.assertFalse(invalid_path.exists())
            self.assertEqual(
                [path.name for path in data_dir.iterdir() if path.name.startswith(".")],
                [],
            )

    def test_exact_seventeen_path_contract_is_present(self) -> None:
        helper_source = _replacement_source("0e01d379")
        phase1_source = _replacement_source("cc338a81")
        phase3_source = "\n".join(
            _replacement_source(cell_id)
            for cell_id in ("7d75c6ff", "df668eaa", "5cb2c38a", "3410c676")
        )
        self.assertIn("f'{stem}.parquet'", helper_source)

        paired_phase1_stems = {
            "phase1_N12_M12_to_M24_visible_cloud_displacements",
            "phase1_N20_M20_to_M30_cloud_displacements",
            "phase1_eigencloud_leading_interval_comparison_points",
        }
        for stem in paired_phase1_stems:
            self.assertIn(stem, phase1_source)

        raw_parquet_stems = {
            "raw_spectrum_square_N100_24_nontrivial_spectral_points",
            "raw_spectrum_square_N_sweep",
            "raw_spectrum_square_N_sweep_eigenvalues",
            "raw_spectrum_square_N_sweep_extended_targets",
        }
        source_notebook = nbformat.read(
            HERE / "blaschke_deformation_certifier.ipynb",
            as_version=4,
        )
        raw_source = next(
            "".join(cell.source)
            for cell in source_notebook.cells
            if cell.get("id") == "a5385979"
        )
        for stem in raw_parquet_stems:
            if stem.startswith("raw_spectrum_square_N100"):
                self.assertIn(
                    "raw_24_spectral_points_display_df",
                    raw_source,
                )
                self.assertIn("24_nontrivial_spectral_points", raw_source)
            else:
                self.assertIn(stem, raw_source)

        phase3_csv_stems = {
            "branch_image_phase3_square_raw_cluster_data",
            "branch_image_phase3_certificate_comparison",
            "branch_image_phase3_rate_diagnostics",
            "branch_image_phase3_raw_cluster_curves",
            "branch_image_phase3_raw_cluster_fitted_bases",
            "branch_image_phase3_fitted_base_table_for_rate_panel",
            "branch_image_phase3_components_vs_geometric_rate",
        }
        for stem in phase3_csv_stems:
            self.assertIn(stem, phase3_source)

        expected_count = (
            2 * len(paired_phase1_stems)
            + len(raw_parquet_stems)
            + len(phase3_csv_stems)
        )
        self.assertEqual(expected_count, 17)

    def test_affected_cells_are_source_regenerated_and_schema_locked(self) -> None:
        affected = {
            cell_id: _replacement_source(cell_id)
            for cell_id in (
                "cc338a81",
                "7d75c6ff",
                "df668eaa",
                "5cb2c38a",
                "3410c676",
            )
        }
        for cell_id, source in affected.items():
            with self.subTest(cell_id=cell_id):
                ast.parse(source)
                self.assertNotIn("pd.read_csv", source)
                self.assertNotIn(".exists()", source)

        self.assertIn(
            "columns=('N', 'M', 'source', 'eig_re', 'eig_im', 'eig_abs')",
            affected["cc338a81"],
        )
        self.assertIn(
            "columns=('N', 'M_base', 'M_comparison', 'z_base_re', 'z_base_im', 'z_comparison_re', 'z_comparison_im', 'displacement_abs')",
            affected["cc338a81"],
        )
        self.assertIn(
            "[['source', 'short_source', 'N', 'M', 'rho', 'r', 'r_tau', 'q_star', 'B_out', 'B_in', 'collocation', 'epsilon', 'status', 'qstar_power', 'epsilon_over_qstar_power', 'effective_base', 'row_symbol']]",
            affected["5cb2c38a"],
        )
        self.assertIn("'row_role'", affected["3410c676"])
        self.assertIn("'theorem_certified'", affected["3410c676"])

    def test_official_builder_output_contains_writer_sources(self) -> None:
        notebook = nbformat.read(CURATED_NOTEBOOK_PATH, as_version=4)
        cell_index = {cell.get("id"): "".join(cell.source) for cell in notebook.cells}
        required_markers = {
            "0e01d379": "def _atomic_dataframe_write",
            "cc338a81": "phase1_eigencloud_comparison_df",
            "7d75c6ff": "branch_image_phase3_square_raw_cluster_data",
            "df668eaa": "branch_image_phase3_raw_cluster_fitted_bases",
            "5cb2c38a": "branch_image_phase3_rate_diagnostics",
            "3410c676": "branch_image_phase3_components_vs_geometric_rate",
        }
        for cell_id, marker in required_markers.items():
            with self.subTest(cell_id=cell_id):
                self.assertIn(marker, cell_index[cell_id])


if __name__ == "__main__":
    unittest.main()
