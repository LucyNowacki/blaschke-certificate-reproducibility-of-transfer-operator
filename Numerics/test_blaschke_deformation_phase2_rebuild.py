"""Focused tests for the clean-room Phase 2 producer chain."""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import blaschke_deformation_phase2_geometry as geometry
import blaschke_deformation_historical_comparisons as historical
import blaschke_deformation_phase2_matrix as matrix
import blaschke_deformation_phase2_pipeline as pipeline
import blaschke_deformation_phase2_transport as transport
import build_blaschke_deformation_thesis_math_notebook as counterpart_builder
import integrate_phase2_clean_room_notebook as integration
import prepare_blaschke_deformation_thesis_appendix as preparation


_DISPLAY_NUMBER_RE = re.compile(
    r"^#(?:\d+[A-Z]*N|\s*Cell\s+\d+[A-Z]*)\s*$"
)


def _semantic_code_source(cell: dict[str, object]) -> str:
    """Remove presentation-only numbering and provenance from code."""

    source = preparation._strip_existing_provenance_header(
        integration._source(cell)
    )
    lines = source.splitlines()
    magic = None
    if lines and lines[0].lstrip().startswith("%%"):
        magic = lines.pop(0).rstrip()
    lines = [
        line.rstrip()
        for line in lines
        if not _DISPLAY_NUMBER_RE.fullmatch(line.strip())
    ]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    body = "\n".join(lines) + "\n"
    return (magic + "\n" if magic is not None else "") + body


class Phase2ProducerTests(unittest.TestCase):
    def test_pipeline_output_members_are_portable_and_fail_closed(self) -> None:
        self.assertEqual(
            pipeline._canonical_phase2_output_member("data", "geometry.csv"),
            "Numerics/outputs/blaschke_deformation_certifier/data/geometry.csv",
        )
        for area, filename in (
            ("data", "/tmp/replay/data/geometry.csv"),
            ("data", "../geometry.csv"),
            ("data", "nested/geometry.csv"),
            ("scratch", "geometry.csv"),
        ):
            with self.subTest(area=area, filename=filename), self.assertRaises(
                ValueError
            ):
                pipeline._canonical_phase2_output_member(area, filename)

    def test_selected_radius_and_exact_q_gap_contract_reject_short_decimal(self) -> None:
        r_tau = "2.293091911822557449340820312"
        contract = geometry.exact_q_gap_contract(
            r_tau_upper=r_tau,
            hardy_radius=geometry.CANONICAL_SELECTED_HARDY_RADIUS_TEXT,
            q_gap_target=geometry.CANONICAL_SELECTED_Q_GAP_TARGET_TEXT,
        )
        self.assertTrue(contract["q_gap_derived_le_target"])
        self.assertEqual(contract["q_gap_target_text"], "0.927")
        self.assertLessEqual(
            Decimal(contract["q_gap_derived_decimal_text"]), Decimal("0.927")
        )
        with self.assertRaisesRegex(ValueError, "canonical selected Hardy radius"):
            geometry.require_canonical_selected_hardy_radius(
                "2.473669807791324"
            )
        with self.assertRaisesRegex(ArithmeticError, "exceeds"):
            geometry.exact_q_gap_contract(
                r_tau_upper=r_tau,
                hardy_radius="2.473669807791324",
                q_gap_target="0.927",
                require_canonical_radius=False,
            )

    def test_production_geometry_grid_has_36_configuration_candidates(self) -> None:
        self.assertEqual(len(geometry.RHO_CANDIDATES), 6)
        self.assertEqual(len(geometry.Q_GAP_CANDIDATES), 6)
        self.assertEqual(
            len(geometry.RHO_CANDIDATES) * len(geometry.Q_GAP_CANDIDATES),
            36,
        )

    def test_matrix_rejects_short_radius_transport_before_theorem_use(self) -> None:
        selected = {
            "configuration_digest": "1" * 64,
            "r_candidate": geometry.CANONICAL_SELECTED_HARDY_RADIUS_TEXT,
            "rho": "2.725",
            "r_tau_interval_u": "2.293091911822557449340820312",
        }
        short_transport = {
            "N": 600,
            "geometry_configuration_digest": selected["configuration_digest"],
            "r": "2.473669807791324",
            "rho": selected["rho"],
            "r_tau": selected["r_tau_interval_u"],
            "transport_certified": True,
        }
        with tempfile.TemporaryDirectory(prefix="phase2-matrix-radius-negative-") as root:
            with self.assertRaisesRegex(RuntimeError, "different exact Hardy radius"):
                matrix.certify_matrix_row(
                    matrix.Phase2MatrixConfig(N=600, M=610),
                    selected_geometry=selected,
                    transport_record=short_transport,
                    data_dir=Path(root) / "data",
                    report_dir=Path(root) / "reports",
                )

    def test_small_complete_boundary_chain(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase2-rebuild-test-") as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            report_dir = root / "reports"
            geometry_result = geometry.certify_geometry_scan(
                geometry.Phase2GeometryConfig(
                    N=8,
                    M=10,
                    cells=4096,
                    precision_bits=128,
                    workers=4,
                ),
                data_dir=data_dir,
                report_dir=report_dir,
                process_workers=4,
            )
            self.assertEqual(len(geometry_result.rows), 36)
            self.assertTrue(
                all(row["status"] == "ok" for row in geometry_result.rows)
            )
            geometry_report = json.loads(
                geometry_result.report_path.read_text(encoding="utf-8")
            )
            self.assertTrue(
                geometry_report["complete_boundary_cover"]["uses_arb_pi"]
            )
            self.assertTrue(
                geometry_report["complete_boundary_cover"][
                    "covers_zero_to_two_pi"
                ]
            )

            selected = geometry_result.selected_geometry
            transport_result = transport.certify_transport(
                transport.Phase2TransportConfig(
                    N=8,
                    r=str(selected["r_candidate"]),
                    rho=str(selected["rho"]),
                    r_tau=str(selected["r_tau_interval_u"]),
                    geometry_configuration_digest=str(
                        selected["configuration_digest"]
                    ),
                    precision_bits=128,
                    flint_threads=2,
                ),
                data_dir=data_dir,
                report_dir=report_dir,
            )
            transport_record = transport_result.record
            self.assertTrue(transport_record["transport_certified"])
            self.assertLess(
                Decimal(transport_record["residual_delta_cert"]), Decimal(1)
            )
            self.assertGreater(
                Decimal(transport_record["lambda_min_cert"]), Decimal(0)
            )

            matrix_result = matrix.certify_matrix_row(
                matrix.Phase2MatrixConfig(N=8, M=10, precision_bits=128),
                selected_geometry=selected,
                transport_record=transport_record,
                data_dir=data_dir,
                report_dir=report_dir,
            )
            matrix_record = matrix_result.record
            self.assertLessEqual(
                Decimal(matrix_record["B_mat_star_arb_u"]),
                Decimal(matrix_record["B_mat_unstar_arb_u"]),
            )
            self.assertFalse(matrix_record["finite_M_certified"])
            self.assertIn(
                "blaschke_deformation_phase2_finite_m.py",
                matrix_record["D_M_route"],
            )


    def test_pipeline_source_identity_is_location_independent(self) -> None:
        record = pipeline._producer_source_record(Path(pipeline.__file__))
        self.assertEqual(
            record["path"],
            "Numerics/blaschke_deformation_phase2_pipeline.py",
        )
        self.assertEqual(len(record["sha256"]), 64)
        self.assertFalse(Path(record["path"]).is_absolute())

    def test_small_historical_comparisons_are_seed_free(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="phase2-historical-comparison-test-"
        ) as temporary:
            root = Path(temporary)
            result = historical.rebuild_historical_comparisons(
                historical.HistoricalComparisonConfig(
                    N=8,
                    M=10,
                    baseline_cells=4096,
                    branch_image_cells=4096,
                    geometry_precision_bits=128,
                    transport_precision_bits=128,
                    zwx13_maximum_extra=80,
                    workers=4,
                ),
                data_dir=root / "data",
                report_dir=root / "reports",
                process_workers=4,
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["legacy_seed_dependency"])
            self.assertTrue(report["complete_boundary_cover"]["uses_arb_pi"])
            self.assertEqual(result.effect_record["cells"], 4096)
            self.assertGreater(
                Decimal(result.effect_record["new_branch_image_B_in_u"]),
                Decimal(0),
            )
            self.assertLess(
                Decimal(result.effect_record["new_branch_image_B_in_u"]),
                Decimal(result.effect_record["old_B_in"]),
            )


class Phase2NotebookIntegrationTests(unittest.TestCase):
    def test_diagnostic_audit_cell_uses_explicit_generated_inputs(self) -> None:
        built, _ = counterpart_builder.build_curated()
        source = next(
            integration._source(cell)
            for cell in built["cells"]
            if "# Cell 100: restore the symmetric-benchmark" in integration._source(cell)
        )
        for forbidden in (
            "transfer_lab_exact_or_reference_targets",
            "transfer_lab_active_geometry_record",
            "finite_envelope_df.copy()",
            "phase4_moat_export.copy()",
            "_phase4_restored_data_dir()",
        ):
            self.assertNotIn(forbidden, source)
        for filename in (
            "raw_spectrum_square_N_sweep_extended_targets.csv",
            "transfer_lab_blaschke_mu_0p3_active_geometry.csv",
            "transfer_lab_blaschke_mu_0p3_generic_sampled_schur_envelope.csv",
            "branch_image_wide_candidate_first15_contour_moats_N600_M610_J128.csv",
        ):
            self.assertIn(filename, source)

    def test_curated_builder_matches_canonical_executable_cells(self) -> None:
        built, _ = counterpart_builder.build_curated()
        counterpart_builder.validate_curated_counterpart(built)
        canonical = json.loads(
            (HERE / "blaschke_deformation_certifier_thesis_math.ipynb")
            .read_text(encoding="utf-8")
        )

        built_ids = tuple(str(cell.get("id")) for cell in built["cells"])
        canonical_ids = tuple(
            str(cell.get("id")) for cell in canonical["cells"]
        )
        self.assertEqual(built_ids, canonical_ids)

        built_code = {
            str(cell.get("id")): _semantic_code_source(cell)
            for cell in built["cells"]
            if cell.get("cell_type") == "code"
        }
        canonical_code = {
            str(cell.get("id")): _semantic_code_source(cell)
            for cell in canonical["cells"]
            if cell.get("cell_type") == "code"
        }
        self.assertEqual(built_code, canonical_code)

    def test_canonical_notebook_preserves_numbering_and_inline_sources(self) -> None:
        path = HERE / "blaschke_deformation_certifier_thesis_math.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        counterpart_builder.validate_inline_helper_sync(
            notebook,
            helper_ordinals=(6, 7, 8, 9, 10),
            allow_notebook_provenance=True,
        )
        sources = {
            str(cell.get("id")): integration._source(cell)
            for cell in notebook["cells"]
        }
        expected_labels = (
            "35AM",
            "#35AN",
            "35BM",
            "#35BN",
            "35CM",
            "#35CN",
            "35DM",
            "#35DN",
            "35EM",
            "#35EN",
        )
        all_labels = []
        for source in sources.values():
            all_labels.extend(
                re.findall(r"(?m)^#?\d+[A-Z]*[MN]$", source)
            )
        for label in expected_labels:
            self.assertEqual(all_labels.count(label), 1)

        cell19 = sources["6451c7fe"]
        self.assertEqual(
            cell19.count("from blaschke_deformation_phase2_pipeline import"),
            1,
        )
        self.assertEqual(cell19.count("rebuild_phase2_inputs("), 1)
        self.assertEqual(cell19.count("rebuild_historical_comparisons("), 1)
        self.assertEqual(
            integration._replace_cell19_seed_block(cell19), cell19
        )

        cell4a = sources["code-bffea704"]
        for filename in integration.REBUILT_SEEDS:
            executable_occurrences = tuple(
                line
                for line in cell4a.splitlines()
                if filename in line and not line.lstrip().startswith("#")
            )
            self.assertEqual(executable_occurrences, ())


if __name__ == "__main__":
    unittest.main()
