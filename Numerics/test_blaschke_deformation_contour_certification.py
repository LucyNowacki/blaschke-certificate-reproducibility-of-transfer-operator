"""Structural tests for Phase 4 count and moat provenance."""

from __future__ import annotations

import csv
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from flint import arb

import blaschke_deformation_contour_certification as certificate


class CountAndMoatProvenanceTests(unittest.TestCase):
    @staticmethod
    def _write_epsilon_row(path: Path, row: dict[str, object]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(row))
            writer.writeheader()
            writer.writerow(row)

    def _contour(
        self,
        *,
        expected_multiplicity: int = 2,
        centre: Fraction = Fraction(1, 1),
        radius: Fraction = Fraction(1, 4),
        laurent_sample_count: int | None = None,
    ) -> certificate.TargetContour:
        return certificate.TargetContour(
            rank=1,
            name="toy",
            family="toy",
            power=1,
            expected_multiplicity=expected_multiplicity,
            centre=centre,
            nearest_target_separation=Fraction(2, 1),
            radius_fraction=Fraction(1, 4),
            radius=radius,
            laurent_sample_count=laurent_sample_count,
        )

    @staticmethod
    def _schur_report(eta_schur: str = "0.01") -> dict[str, object]:
        return {
            "Q_condition_upper": 1.0,
            "Q_condition_upper_text": "1",
            "eta_schur_upper": float(eta_schur),
            "eta_schur_upper_text": eta_schur,
        }

    @staticmethod
    def _diagonal_matrix() -> np.ndarray:
        return np.diag(np.asarray([1.0, 1.0, 3.0], dtype=np.complex128))

    def test_repeated_diagonal_count_survives_nonnormality(self) -> None:
        diagonal = self._diagonal_matrix()
        nonnormal = diagonal.copy()
        nonnormal[0, 1] = 8.0
        contour = self._contour()

        diagonal_geometry = certificate._certified_schur_diagonal_geometry(
            diagonal, contour
        )
        nonnormal_geometry = certificate._certified_schur_diagonal_geometry(
            nonnormal, contour
        )
        diagonal_moat = certificate._uniform_triangular_inverse_bound(
            diagonal, np.abs(diagonal), contour, diagonal_geometry
        )
        nonnormal_moat = certificate._uniform_triangular_inverse_bound(
            nonnormal, np.abs(nonnormal), contour, nonnormal_geometry
        )

        self.assertEqual(
            diagonal_geometry["schur_diagonal_algebraic_count"], 2
        )
        self.assertEqual(
            nonnormal_geometry["schur_diagonal_algebraic_count"], 2
        )
        self.assertEqual(diagonal_geometry["inside_flags"], (True, True, False))
        self.assertEqual(nonnormal_geometry["inside_flags"], (True, True, False))
        self.assertAlmostEqual(
            float(diagonal_geometry["minimum_diagonal_boundary_distance"]),
            0.25,
        )
        self.assertGreater(
            float(nonnormal_moat["triangular_inverse_bound"]),
            float(diagonal_moat["triangular_inverse_bound"]),
        )

    def test_epsilon_loading_requires_every_fresh_phase2_gate(self) -> None:
        config = certificate.ContourCertificateConfig()
        row: dict[str, object] = {
            "N": config.N,
            "M": config.M,
            "rho": config.rho,
            "r": config.r,
            "phase2_aggregation_status": (
                "authoritative standalone final-aggregation refresh"
            ),
            "new_epsilon_response_prefactor_candidate_text": "0.01",
            **{
                gate: True
                for gate in certificate.REQUIRED_EPSILON_CERTIFICATION_GATES
            },
        }
        with tempfile.TemporaryDirectory(prefix="epsilon-gate-test-") as root:
            path = Path(root) / "epsilon.csv"
            self._write_epsilon_row(path, row)
            epsilon, text = certificate._load_epsilon(path, config)
            self.assertEqual(text, "0.01")
            self.assertGreater(float(epsilon), 0.0)

            failed = dict(row)
            failed["input_geometric_remainder_certified"] = False
            self._write_epsilon_row(path, failed)
            with self.assertRaisesRegex(
                ArithmeticError, "input_geometric_remainder_certified"
            ):
                certificate._load_epsilon(path, config)

            missing = dict(row)
            del missing["input_exact_prefix_certified"]
            self._write_epsilon_row(path, missing)
            with self.assertRaisesRegex(ArithmeticError, "input_exact_prefix_certified"):
                certificate._load_epsilon(path, config)

    def test_wrong_expected_multiplicity_does_not_invalidate_count(self) -> None:
        triangular = self._diagonal_matrix()
        row = certificate._schur_contour_attempt(
            self._contour(expected_multiplicity=1),
            triangular,
            np.abs(triangular),
            self._schur_report(),
            arb("0.01"),
            arb("0.001"),
        )

        self.assertEqual(row["schur_diagonal_algebraic_count"], 2)
        self.assertTrue(row["schur_diagonal_membership_certified"])
        self.assertTrue(row["A_N_circ_count_transport_certified"])
        self.assertTrue(row["mathematical_finite_count_transport_certified"])
        self.assertTrue(row["finite_count_certified"])
        self.assertFalse(row["finite_count_matches_expected"])
        self.assertTrue(row["finite_to_exact_rank_transfer_certified"])
        self.assertFalse(row["theorem_certified"])

    def test_failed_schur_homotopy_blocks_count_transport(self) -> None:
        triangular = self._diagonal_matrix()
        row = certificate._schur_contour_attempt(
            self._contour(),
            triangular,
            np.abs(triangular),
            self._schur_report("0.30"),
            arb("0.01"),
            arb("0.001"),
        )

        self.assertEqual(row["schur_diagonal_algebraic_count"], 2)
        self.assertTrue(row["schur_diagonal_membership_certified"])
        self.assertTrue(row["finite_count_matches_expected"])
        self.assertFalse(row["A_N_circ_count_transport_certified"])
        self.assertFalse(row["mathematical_finite_count_transport_certified"])
        self.assertFalse(row["finite_count_certified"])
        self.assertFalse(row["theorem_certified"])

    def test_failed_laurent_residual_preserves_count_provenance(self) -> None:
        triangular = self._diagonal_matrix()
        contour = self._contour(laurent_sample_count=4)
        proposal = (
            np.asarray([0], dtype=int),
            np.zeros((1, 3, 3), dtype=np.complex128),
            np.asarray([0.0]),
            "zero-coefficient-proposal",
            0.0,
        )
        with mock.patch.object(
            certificate, "_laurent_coefficients", return_value=proposal
        ):
            row, _ = certificate._laurent_contour_certificate(
                contour,
                triangular,
                certificate._numpy_to_acb_exact(triangular),
                self._schur_report(),
                arb("0.01"),
                arb("0.001"),
            )

        self.assertGreaterEqual(row["exact_dyadic_residual_sum_upper"], 1.0)
        self.assertLessEqual(row["triangular_laurent_moat_lower"], 0.0)
        self.assertEqual(
            row["count_method"], certificate.COUNT_METHOD_SCHUR_DIAGONAL
        )
        self.assertEqual(row["moat_method"], certificate.MOAT_METHOD_LAURENT)
        self.assertEqual(
            row["certificate_route"], certificate.CERTIFICATE_ROUTE_LAURENT
        )
        self.assertFalse(row["laurent_used_for_count"])
        self.assertTrue(row["laurent_used_for_moat"])
        self.assertEqual(row["schur_diagonal_algebraic_count"], 2)
        self.assertTrue(row["schur_diagonal_membership_certified"])
        self.assertFalse(row["finite_count_certified"])
        self.assertFalse(row["theorem_certified"])

    def test_laurent_proposal_table_has_recorded_reference_digests(self) -> None:
        self.assertEqual(
            set(certificate._LAURENT_PROPOSALS),
            set(certificate._LAURENT_REFERENCE_DIGESTS),
        )
        self.assertTrue(
            all(
                len(digest) == 64
                for digest in certificate._LAURENT_REFERENCE_DIGESTS.values()
            )
        )

    def test_laurent_candidate_generation_is_repeatable(self) -> None:
        contour = self._contour(
            expected_multiplicity=1,
            centre=Fraction(2, 1),
            radius=Fraction(1, 4),
            laurent_sample_count=4,
        )
        triangular = np.diag(
            np.asarray([0.5, 0.75], dtype=np.complex128)
        )

        first = certificate._laurent_coefficients(triangular, contour)
        second = certificate._laurent_coefficients(triangular, contour)

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[3], second[3])
        self.assertEqual(first[1].shape, (4, 2, 2))
        self.assertTrue(first[1].flags.c_contiguous)

    def test_checkpoint_requires_all_seven_reconstruction_records(self) -> None:
        witnesses = []
        certificate_rows = []
        for rank, name in enumerate(certificate._LAURENT_PROPOSALS, 18):
            sample_count = certificate._LAURENT_PROPOSALS[name][1]
            digest = certificate._LAURENT_REFERENCE_DIGESTS[name]
            certificate_rows.append(
                {
                    "name": name,
                    "moat_method": certificate.MOAT_METHOD_LAURENT,
                    "coefficient_sha256": digest,
                }
            )
            witnesses.append(
                {
                    "rank": str(rank),
                    "name": name,
                    "laurent_sample_count": str(sample_count),
                    "coefficient_matrix_count": str(sample_count),
                    "coefficient_matrix_rows": "600",
                    "coefficient_matrix_columns": "600",
                    "coefficient_sha256": digest,
                    "reference_coefficient_sha256": digest,
                    "digest_matches_recorded_reference": "True",
                    "digest_used_in_theorem_gate": "False",
                    "generated_in_recorded_run": "True",
                    "candidate_coefficients_validated_exact_dyadic": "True",
                    "exact_dyadic_residual_sum_upper": "0.5",
                    "theorem_certified": "True",
                }
            )

        self.assertTrue(
            certificate._laurent_witness_records_are_reusable(
                witnesses,
                certificate_rows,
            )
        )
        witnesses[0]["coefficient_sha256"] = "0" * 64
        self.assertFalse(
            certificate._laurent_witness_records_are_reusable(
                witnesses,
                certificate_rows,
            )
        )

    def test_zero_complement_requires_strict_exclusion(self) -> None:
        touching = self._contour(
            expected_multiplicity=1,
            centre=Fraction(1, 4),
            radius=Fraction(1, 4),
        )
        with self.assertRaises(AssertionError):
            certificate._validate_target_contour_plan([touching])

        separated = self._contour(
            expected_multiplicity=1,
            centre=Fraction(1, 4),
            radius=Fraction(1, 8),
        )
        self.assertEqual(
            certificate._validate_target_contour_plan([separated]),
            [separated],
        )

    def test_logical_source_hashes_ignore_temporary_parent_paths(self) -> None:
        with tempfile.TemporaryDirectory() as left_root, tempfile.TemporaryDirectory() as right_root:
            left = Path(left_root) / "helper.py"
            right = Path(right_root) / "helper.py"
            left.write_text("value = 1\n", encoding="utf-8")
            right.write_text("value = 1\n", encoding="utf-8")
            left_hashes = certificate.logical_source_hashes((left,))
            right_hashes = certificate.logical_source_hashes((right,))

        self.assertEqual(left_hashes, right_hashes)
        self.assertEqual(tuple(left_hashes), ("helper.py",))

    def test_small_gain_reaggregation_preserves_finite_geometry(self) -> None:
        triangular = self._diagonal_matrix()
        original = certificate._schur_contour_attempt(
            self._contour(),
            triangular,
            np.abs(triangular),
            self._schur_report(),
            arb("0.01"),
            arb("0.001"),
        )
        original_moat = original["lifted_finite_section_moat_lower"]

        refreshed = certificate._reaggregate_small_gain_rows(
            [original], arb("0.002")
        )[0]

        self.assertEqual(
            refreshed["lifted_finite_section_moat_lower"], original_moat
        )
        self.assertEqual(refreshed["count_method"], original["count_method"])
        self.assertGreaterEqual(refreshed["epsilon_upper"], 0.002)
        self.assertAlmostEqual(refreshed["epsilon_upper"], 0.002)
        self.assertTrue(refreshed["certified_small_gain_pass"])
        self.assertTrue(refreshed["finite_to_exact_rank_transfer_certified"])
        self.assertTrue(refreshed["theorem_certified"])


if __name__ == "__main__":
    unittest.main()
