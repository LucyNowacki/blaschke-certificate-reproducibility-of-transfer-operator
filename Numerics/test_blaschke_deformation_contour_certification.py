"""Structural tests for Phase 4 count and moat provenance."""

from __future__ import annotations

import csv
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import flint
import numpy as np
from flint import arb

import blaschke_deformation_contour_certification as certificate


# Exact source-resident projection retained from the authenticated 5c0 rows.
_AUTHENTICATED_5C0_EPSILON_TEXT = (
    "0.000000000000000000033264433839017426342179265983234893950031511766904056153485116"
)
_AUTHENTICATED_5C0_EPSILON_UPPER = 3.326443383901744e-20
_AUTHENTICATED_5C0_LEGACY_SMALL_GAIN_ROWS = (
    (1, "0.028096548411368008", "1.1839331063725426e-18"),
    (2, "0.00422740732731099", "7.868755306382225e-18"),
    (3, "9.508007756212867e-05", "3.4985703306017286e-16"),
    (4, "7.29619394357647e-05", "4.559148796791962e-16"),
    (5, "0.00019239308416855842", "1.7289828261120858e-16"),
    (6, "1.435174463617861e-05", "2.317797221333131e-15"),
    (7, "1.70993414705515e-06", "1.9453634455049327e-14"),
    (8, "1.4050419477033225e-06", "2.367504677948683e-14"),
    (9, "1.2436620910454112e-06", "2.6747163943105855e-14"),
    (10, "4.231148058175618e-08", "7.861798590276789e-13"),
    (11, "1.5143383182271744e-08", "2.1966315874486952e-12"),
    (12, "9.876620157830163e-09", "3.36799768619688e-12"),
    (13, "1.8764024728966627e-09", "1.772777126416066e-11"),
    (14, "1.1661969436238611e-11", "2.8523856129866845e-09"),
    (15, "6.810055442291137e-12", "4.884605436901723e-09"),
    (16, "1.2016682997299054e-11", "2.7681876809510726e-09"),
    (17, "3.888874814934859e-13", "8.553742514741928e-08"),
    (18, "2.1423514271401273e-13", "1.5527066856357403e-07"),
    (19, "1.6551087788942995e-13", "2.009803480182122e-07"),
    (20, "3.36000049215256e-11", "9.90012766864413e-10"),
    (21, "4.285735844943455e-12", "7.761662184164863e-09"),
    (22, "1.8096240519593842e-13", "1.8381958287413417e-07"),
    (23, "1.6361202218220478e-13", "2.0331289470875723e-07"),
    (24, "3.7774649210294747e-13", "8.806020581112869e-08"),
)


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

    def test_small_gain_precision_roles_match_legacy_and_explicit_256(self) -> None:
        stored_rows = [
            {
                "rank": str(rank),
                "lifted_finite_section_moat_lower": moat_text,
                "finite_count_certified": "True",
                "finite_count_matches_expected": "True",
                "complete_circle_covered": "True",
                "zero_outside_enclosed_region": "True",
                "sampled_values_used_in_theorem_gate": "False",
            }
            for rank, moat_text, _ in _AUTHENTICATED_5C0_LEGACY_SMALL_GAIN_ROWS
        ]

        previous_precision = int(flint.ctx.prec)
        with certificate._fixed_flint_precision(256):
            epsilon = arb(_AUTHENTICATED_5C0_EPSILON_TEXT).upper()
            expected_epsilon = certificate._upper_float(epsilon.upper())
            expected_products = tuple(
                certificate._upper_float(
                    (
                        epsilon.upper()
                        / arb(row["lifted_finite_section_moat_lower"]).lower()
                    ).upper()
                )
                for row in stored_rows
            )
        self.assertEqual(int(flint.ctx.prec), previous_precision)

        refreshed = certificate._reaggregate_small_gain_rows(
            stored_rows,
            epsilon,
            epsilon_text=_AUTHENTICATED_5C0_EPSILON_TEXT,
            precision_bits=256,
        )

        self.assertEqual(int(flint.ctx.prec), previous_precision)
        self.assertEqual(len(refreshed), 24)
        self.assertTrue(
            any(
                row["theorem_small_gain_product_upper"]
                != row["legacy_5c0_compatibility_small_gain_product_upper"]
                for row in refreshed
            )
        )
        for fixture, row, expected_product in zip(
            _AUTHENTICATED_5C0_LEGACY_SMALL_GAIN_ROWS,
            refreshed,
            expected_products,
            strict=True,
        ):
            rank, moat_text, legacy_product_text = fixture
            with self.subTest(rank=rank):
                self.assertEqual(int(row["rank"]), rank)
                self.assertEqual(row["theorem_precision_bits"], 256)
                self.assertEqual(row["theorem_epsilon_upper"], expected_epsilon)
                self.assertEqual(
                    row["theorem_small_gain_product_upper"], expected_product
                )
                self.assertEqual(
                    row["theorem_rounding_role"],
                    "configured-precision interval upper bounds used by theorem gates",
                )
                self.assertEqual(row["epsilon_upper"], expected_epsilon)
                self.assertEqual(
                    row["certified_small_gain_product_upper"], expected_product
                )
                self.assertTrue(row["certified_small_gain_pass"])
                self.assertTrue(row["finite_to_exact_rank_transfer_certified"])
                self.assertTrue(row["theorem_certified"])
                self.assertEqual(row["status"], "theorem_certified")

                self.assertEqual(
                    row["legacy_5c0_compatibility_precision_bits"], 53
                )
                self.assertEqual(
                    row["legacy_5c0_compatibility_epsilon_upper"],
                    _AUTHENTICATED_5C0_EPSILON_UPPER,
                )
                self.assertEqual(
                    row["legacy_5c0_compatibility_small_gain_product_upper"],
                    float(legacy_product_text),
                )
                self.assertEqual(
                    row["legacy_5c0_compatibility_serialized_moat_lower"],
                    float(moat_text),
                )
                self.assertEqual(
                    row["legacy_5c0_compatibility_rounding_role"],
                    "fixed-53-bit upward epsilon over downward serialized moat; "
                    "diagnostic compatibility only",
                )
                self.assertTrue(
                    row["legacy_5c0_compatibility_is_conservative_upper"]
                )
                self.assertFalse(row["legacy_5c0_compatibility_theorem_gate"])

        roles = certificate._precision_role_report(
            refreshed,
            theorem_precision_bits=256,
        )
        self.assertEqual(
            roles,
            {
                "theorem_projection": {
                    "precision_bits": 256,
                    "row_count": 24,
                    "rounding_role": (
                        "configured-precision interval upper bounds used by theorem gates"
                    ),
                    "canonical_fields_are_theorem_values": True,
                    "theorem_gate": True,
                },
                "legacy_5c0_compatibility_projection": {
                    "precision_bits": 53,
                    "row_count": 24,
                    "rounding_role": (
                        "fixed-53-bit upward epsilon over downward serialized moat; "
                        "diagnostic compatibility only"
                    ),
                    "is_conservative_upper": True,
                    "theorem_gate": False,
                },
            },
        )
        self.assertEqual(int(flint.ctx.prec), previous_precision)

    def test_fixed_flint_precision_restores_after_failure(self) -> None:
        previous_precision = int(flint.ctx.prec)
        with self.assertRaisesRegex(RuntimeError, "probe failure"):
            with certificate._fixed_flint_precision(197):
                self.assertEqual(int(flint.ctx.prec), 197)
                raise RuntimeError("probe failure")
        self.assertEqual(int(flint.ctx.prec), previous_precision)


if __name__ == "__main__":
    unittest.main()
