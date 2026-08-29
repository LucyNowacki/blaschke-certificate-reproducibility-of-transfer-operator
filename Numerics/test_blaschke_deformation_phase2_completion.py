"""Focused tests for the standalone Phase 2 completion helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from blaschke_deformation_phase2_finite_m import (
    Phase2FiniteMConfig,
    certify_finite_m_completion,
)
from blaschke_deformation_phase2_resolved_response import (
    Phase2ResolvedResponseConfig,
    certify_resolved_response_completion,
)
from blaschke_deformation_phase2_final_aggregation import (
    FINAL_CERTIFICATION_GATES,
    INPUT_CERTIFICATION_GATES,
    Phase2FinalAggregationConfig,
    RESPONSE_CERTIFICATION_GATES,
    certify_final_phase2_aggregation,
)


HERE = Path(__file__).resolve().parent
CANONICAL_OUTPUT = HERE / "outputs" / "blaschke_deformation_certifier"
UPSTREAM_NAMES = (
    "branch_image_balanced_candidate_single_space_row_N600_M610.csv",
    "branch_image_balanced_candidate_transport_cert_N600.csv",
    "branch_image_radius_reoptimisation_balanced_highcell_scan.csv",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Phase2CompletionHelperTests(unittest.TestCase):
    def test_three_stages_use_only_the_explicit_output_directory(self) -> None:
        canonical_data = CANONICAL_OUTPUT / "data"
        watched = tuple(canonical_data / name for name in UPSTREAM_NAMES)
        before = {path: _sha256(path) for path in watched}

        with tempfile.TemporaryDirectory(prefix="phase2-completion-test-") as root:
            output_dir = Path(root)
            data_dir = output_dir / "data"
            data_dir.mkdir(parents=True)
            for name in UPSTREAM_NAMES:
                shutil.copy2(canonical_data / name, data_dir / name)

            finite = certify_finite_m_completion(
                Phase2FiniteMConfig.production_n600_m610(),
                output_dir=output_dir,
            )
            resolved = certify_resolved_response_completion(
                Phase2ResolvedResponseConfig(cells=1024, prefix_terms=4),
                output_dir=output_dir,
            )
            final = certify_final_phase2_aggregation(
                Phase2FinalAggregationConfig(cells=1024, prefix_terms=4),
                output_dir=output_dir,
            )

            produced_paths = (
                finite.certificate_csv_path,
                finite.safe_row_csv_path,
                finite.report_json_path,
                resolved.principal_csv_path,
                resolved.profile_csv_path,
                resolved.candidate_csv_path,
                resolved.report_json_path,
                final.input_certificate_csv_path,
                final.input_profile_csv_path,
                final.phase2_summary_csv_path,
            )
            for path in produced_paths:
                self.assertTrue(path.is_file(), msg=str(path))
                self.assertTrue(path.is_relative_to(output_dir), msg=str(path))

            self.assertTrue(finite.certificate["finite_M_prefactor_certified"])
            self.assertTrue(resolved.response_summary["response_prefactor_certified"])
            self.assertTrue(final.final_certificate["certified"])
            self.assertTrue(final.refreshed_candidate["total_certified"])
            self.assertTrue(final.refreshed_candidate["final_phase2_certified"])
            for gate in (*FINAL_CERTIFICATION_GATES, "input_tail_certified"):
                self.assertTrue(final.refreshed_candidate[gate], msg=gate)
            self.assertEqual(len(final.input_certificate["profile"]), 1024)
            self.assertEqual(
                final.final_certificate["B_in_selection"],
                "coherent_branchwise_intersection",
            )

            # An inherited True aggregate must be replaced by the result of
            # the freshly computed input certificate.  Flip one fresh gate
            # while preserving the upstream candidate's previous True value.
            failed_input = {
                **final.input_certificate,
                "summary": dict(final.input_certificate["summary"]),
            }
            failed_input["summary"]["geometric_remainder_certified"] = False

            def failed_certificate(*_args, **_kwargs):
                return failed_input

            with mock.patch(
                "blaschke_deformation_phase2_final_aggregation.certify_input_tail_rows",
                new=failed_certificate,
            ):
                failed = certify_final_phase2_aggregation(
                    Phase2FinalAggregationConfig(cells=1024, prefix_terms=4),
                    output_dir=output_dir,
                )
            self.assertFalse(failed.final_certificate["certified"])
            self.assertFalse(
                failed.refreshed_candidate["input_geometric_remainder_certified"]
            )
            self.assertFalse(failed.refreshed_candidate["input_tail_certified"])
            self.assertFalse(failed.refreshed_candidate["final_phase2_certified"])
            self.assertFalse(failed.refreshed_candidate["total_certified"])
            self.assertTrue(
                all(
                    failed.refreshed_candidate[gate]
                    for gate in INPUT_CERTIFICATION_GATES
                    if gate != "input_geometric_remainder_certified"
                )
            )
            self.assertTrue(
                all(
                    failed.refreshed_candidate[gate]
                    for gate in RESPONSE_CERTIFICATION_GATES
                )
            )

        after = {path: _sha256(path) for path in watched}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
