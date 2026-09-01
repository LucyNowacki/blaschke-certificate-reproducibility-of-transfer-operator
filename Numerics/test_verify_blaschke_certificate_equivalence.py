from __future__ import annotations

import csv
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


NUMERICS = Path(__file__).resolve().parent
RELEASE_ROOT = NUMERICS.parent
sys.path.insert(0, str(NUMERICS))

import verify_blaschke_certificate_equivalence as verifier
import run_blaschke_clean_room_replay as clean_replay
import run_blaschke_certificate_reproduction as reproduction
import stage_blaschke_certificate_replay as stager


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _policy() -> dict[str, object]:
    return json.loads(
        (RELEASE_ROOT / "release/blaschke-certificate-semantic-policy.json").read_text(
            encoding="utf-8"
        )
    )


def _artifact(root: Path, policy: dict[str, object], key: str) -> Path:
    return root / policy["artifact_paths"][key]


def _true_false(value: bool) -> str:
    return "True" if value else "False"


class TargetFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.policy = _policy()
        self.plan_rows: list[dict[str, object]] = []
        self.certificate_rows: list[dict[str, object]] = []
        self.mode_rows: list[dict[str, object]] = []
        self.reconstruction_rows: list[dict[str, object]] = []
        self.q_condition = Fraction(2)
        for geometry in verifier.expected_target_geometry():
            rank = int(geometry["rank"])
            name = str(geometry["name"])
            multiplicity = int(geometry["expected_multiplicity"])
            route = (
                "Schur-count/Schur-moat"
                if rank <= 17
                else "Schur-count/Laurent-moat"
            )
            centre = str(geometry["centre"])
            radius = str(geometry["radius"])
            self.plan_rows.append(
                {
                    "rank": rank,
                    "name": name,
                    "family": geometry["family"],
                    "power": geometry["power"],
                    "expected_multiplicity": multiplicity,
                    "centre_exact": centre,
                    "nearest_target_separation_exact": str(
                        geometry["nearest_separation"]
                    ),
                    "radius_fraction_exact": str(geometry["radius_fraction"]),
                    "radius_exact": radius,
                    "distance_to_zero_exact": str(
                        geometry["centre"] - geometry["radius"]
                    ),
                    "planned_certificate_route": route,
                }
            )
            row: dict[str, object] = {
                "rank": rank,
                "name": name,
                "family": geometry["family"],
                "power": geometry["power"],
                "expected_multiplicity": multiplicity,
                "centre_exact": centre,
                "radius_fraction_exact": str(geometry["radius_fraction"]),
                "radius_exact": radius,
                "certificate_route": route,
                "distance_to_zero_lower": "0.0003",
                "status": "theorem_certified",
                "count_method": "certified Schur-diagonal algebraic count",
                "count_reference_matrix": "exact-binary upper-triangular Schur matrix T",
                "schur_diagonal_algebraic_count": multiplicity,
                "sampled_values_used_in_theorem_gate": "False",
                "laurent_used_for_count": "False",
                "laurent_used_for_moat": _true_false(rank > 17),
                "eta_schur_upper": "0.01",
                "eta_A_upper": "0.01",
                "epsilon_upper": "0.000001",
                "A_N_circ_moat_lower": "0.2",
                "mathematical_finite_matrix_moat_lower": "0.19",
                "lifted_finite_section_moat_lower": "0.18",
                # Deliberately below the primitive reaggregation: regression
                # for independently outward-rounded display aggregates.
                "certified_small_gain_product_upper": "0",
                "Q_condition_upper": "2",
                "triangular_inverse_bound_upper": "2" if rank <= 17 else "",
                "exact_dyadic_residual_sum_upper": "" if rank <= 17 else "0.1",
                "coefficient_frobenius_sum_upper": "" if rank <= 17 else "2",
                "candidate_coefficients_validated_exact_dyadic": (
                    "" if rank <= 17 else "True"
                ),
                "candidate_generated_in_recorded_run": "" if rank <= 17 else "True",
                "coefficient_digest_used_in_theorem_gate": "" if rank <= 17 else "False",
                "coefficient_sha256": "" if rank <= 17 else f"{rank:064x}",
                "minimum_laurent_mode": "" if rank <= 17 else "0",
                "maximum_laurent_mode": "" if rank <= 17 else "0",
                "laurent_sample_count": "" if rank <= 17 else "1",
                "moat_method": (
                    "uniform complete-circle triangular Schur resolvent"
                    if rank <= 17
                    else "complete-circle Laurent approximate inverse in Schur coordinates"
                ),
            }
            for gate in verifier.TRUE_ROW_GATES:
                row[gate] = "True"
            self.certificate_rows.append(row)
            if rank > 17:
                digest = str(row["coefficient_sha256"])
                self.mode_rows.extend(
                    [
                        {
                            "name": name,
                            "mode": "0",
                            "coefficient_present": "True",
                            "coefficient_frobenius_upper": "1",
                            "residual_frobenius_upper": "0.01",
                            "coefficient_sha256": digest,
                            "precision_bits": "256",
                        },
                        {
                            "name": name,
                            "mode": "1",
                            "coefficient_present": "False",
                            "coefficient_frobenius_upper": "0",
                            "residual_frobenius_upper": "0.09",
                            "coefficient_sha256": digest,
                            "precision_bits": "256",
                        },
                    ]
                )
                self.reconstruction_rows.append(
                    {
                        "name": name,
                        "laurent_sample_count": "1",
                        "minimum_laurent_mode": "0",
                        "maximum_laurent_mode": "0",
                        "coefficient_sha256": digest,
                        "generated_in_recorded_run": "True",
                        "candidate_coefficients_validated_exact_dyadic": "True",
                        "theorem_certified": "True",
                        "digest_used_in_theorem_gate": "False",
                    }
                )
        self.report = {
            "eta_schur_upper_text": "0.01",
            "eta_A_upper_text": "0.01",
        }
        self.write()

    def write(self) -> None:
        _write_csv(_artifact(self.root, self.policy, "contour_plan"), self.plan_rows)
        _write_csv(
            _artifact(self.root, self.policy, "spectral_certificate"),
            self.certificate_rows,
        )
        _write_csv(
            _artifact(self.root, self.policy, "laurent_mode_bounds"), self.mode_rows
        )
        _write_csv(
            _artifact(self.root, self.policy, "laurent_reconstruction"),
            self.reconstruction_rows,
        )

    def validate(self, epsilon: str = "0.000001") -> dict[str, object]:
        self.write()
        return verifier._validate_targets_and_moats(
            self.root,
            self.policy,
            self.report,
            epsilon_precise=Fraction(Decimal(epsilon)),
            schur_primitives={
                "eta_schur_row_upper": Fraction(Decimal("0.01")),
                "Q_condition_row_upper": self.q_condition,
            },
        )

    def close(self) -> None:
        self.temporary.cleanup()


class ExactArithmeticTests(unittest.TestCase):
    def test_phase2_exact_rss_and_triangle(self) -> None:
        result = verifier.validate_phase2_inequalities(
            b_out_value="3",
            b_in_value="4",
            collocation_value="1",
            epsilon_value="6",
            triangle_value="8",
        )
        self.assertEqual(result["epsilon"], Fraction(6))

    def test_phase2_rss_failure(self) -> None:
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "RSS-square"):
            verifier.validate_phase2_inequalities(
                b_out_value="3",
                b_in_value="4",
                collocation_value="1",
                epsilon_value="5.9999999999999999999999999999",
                triangle_value="8",
            )

    def test_phase2_triangle_failure(self) -> None:
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "triangle"):
            verifier.validate_phase2_inequalities(
                b_out_value="3",
                b_in_value="4",
                collocation_value="1",
                epsilon_value="6",
                triangle_value="7.9999999999999999999999999999",
            )

    def test_radius_and_exact_q_contract(self) -> None:
        policy = _policy()
        canonical = verifier._validate_radius_contract(
            policy, {"a": policy["canonical_hardy_radius"], "b": Decimal(policy["canonical_hardy_radius"])}
        )
        r_tau = Decimal("2.293091911822557449340820312")
        ratio = Fraction(r_tau) / Fraction(canonical)
        row = {
            "q_gap_target_text": "0.927",
            "q_gap_derived_exact_numerator": str(ratio.numerator),
            "q_gap_derived_exact_denominator": str(ratio.denominator),
            "q_gap_derived_le_target": "True",
            "q_gap_derived_decimal_text": "0.927",
        }
        self.assertEqual(
            verifier._validate_q_contract(
                row,
                radius=canonical,
                r_tau_value=str(r_tau),
                q_target=Decimal("0.927"),
            ),
            ratio,
        )

    def test_short_radius_fails_q_gap(self) -> None:
        policy = _policy()
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "not in"):
            verifier._validate_q_contract(
                {},
                radius=Decimal("2.473669807791324"),
                r_tau_value="2.293091911822557449340820312",
                q_target=Decimal(policy["q_gap_target"]),
            )

    def test_negative_r_tau_fails_q_gap(self) -> None:
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "not in"):
            verifier._validate_q_contract(
                {},
                radius=Decimal(verifier.CANONICAL_HARDY_RADIUS),
                r_tau_value="-1",
                q_target=Decimal(verifier.CANONICAL_Q_GAP_TARGET),
            )

    def test_radius_mismatch_fails(self) -> None:
        policy = _policy()
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "radius contract"):
            verifier._validate_radius_contract(
                policy,
                {"canonical": policy["canonical_hardy_radius"], "short": "2.473669807791324"},
            )

    def test_spectrally_bound_phase2_candidate_is_reaggregated(self) -> None:
        candidate = {
            "B_out_response_prefactor_cert": "3",
            "B_in_selected_cert_text": "4",
            "B_in_selected_cert_u": repr(math.nextafter(4.0, math.inf)),
            "collocation_selected_cert_u": repr(math.nextafter(1.0, math.inf)),
            "new_epsilon_response_prefactor_candidate": repr(
                math.nextafter(6.0, math.inf)
            ),
            "new_epsilon_response_prefactor_candidate_text": "6",
            "epsilon_triangle_check_text": "8",
        }
        final = {
            "B_out": "3",
            "B_in": "4",
            "collocation": "1",
            "epsilon_upper_text": "6",
            "epsilon_triangle_upper_text": "8",
        }
        self.assertEqual(
            verifier.validate_phase2_candidate_binding(candidate, final)["epsilon"],
            Fraction(6),
        )
        candidate["B_out_response_prefactor_cert"] = "3.0000000000000000001"
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "B_out binding"):
            verifier.validate_phase2_candidate_binding(candidate, final)

        candidate["B_out_response_prefactor_cert"] = "3"
        candidate["B_in_selected_cert_text"] = "4.0000000000000001"
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "exactly equal"):
            verifier.validate_phase2_candidate_binding(candidate, final)

    def test_phase2_candidate_nextafter_serializations_are_exact(self) -> None:
        candidate = {
            "B_out_response_prefactor_cert": "3",
            "B_in_selected_cert_text": "4",
            "B_in_selected_cert_u": repr(math.nextafter(4.0, math.inf)),
            "collocation_selected_cert_u": repr(math.nextafter(1.0, math.inf)),
            "new_epsilon_response_prefactor_candidate": repr(
                math.nextafter(6.0, math.inf)
            ),
            "new_epsilon_response_prefactor_candidate_text": "6",
            "epsilon_triangle_check_text": "8",
        }
        final = {
            "B_out": "3",
            "B_in": "4",
            "collocation": "1",
            "epsilon_upper_text": "6",
            "epsilon_triangle_upper_text": "8",
        }
        for field in (
            "B_in_selected_cert_u",
            "collocation_selected_cert_u",
            "new_epsilon_response_prefactor_candidate",
        ):
            broken = dict(candidate)
            broken[field] = "9"
            with self.subTest(field=field), self.assertRaisesRegex(
                verifier.CertificateEquivalenceError, "nextafter serialization"
            ):
                verifier.validate_phase2_candidate_binding(broken, final)

        broken = dict(candidate)
        broken["new_epsilon_response_prefactor_candidate_text"] = "6.1"
        with self.assertRaisesRegex(
            verifier.CertificateEquivalenceError, "epsilon displays disagree"
        ):
            verifier.validate_phase2_candidate_binding(broken, final)

    def test_real_input_tail_schema_without_q_fields_passes(self) -> None:
        policy = _policy()
        report = {
            "map_label": "blaschke_mu_0p3",
            "N": 600,
            "rho": "2.725",
            "r": verifier.CANONICAL_HARDY_RADIUS,
            "mu": "0.3",
            "cells": 65536,
            "precision_bits": 192,
            "prefix_terms": 24,
            "J": 623,
            "branchwise_profile_cert_u": "3.4e-20",
            "combined_row_cert_u": "3.3e-20",
            "maximum_branch_radius_u": "2.29",
            "boundary_cover_certified": True,
            "exact_prefix_certified": True,
            "geometric_remainder_certified": True,
            "branchwise_profile_certified": True,
            "combined_row_certified": True,
        }
        csv_row = {key: str(value) for key, value in report.items()}
        result = verifier.validate_input_tail_configuration(policy, report, csv_row)
        self.assertEqual(result["N"], 600)
        self.assertEqual(result["r"], verifier.CANONICAL_HARDY_RADIUS)
        self.assertNotIn("r_tau", report)
        self.assertNotIn("q_gap", csv_row)

        final_row = {
            "B_in_selection": "coherent_branchwise_intersection",
            "B_in": "3.3000000000000000000001e-20",
            "B_in_branchwise": "3.4e-20",
            "B_in_combined_row": "3.3e-20",
        }
        candidate = {
            "B_in_selection": "coherent_branchwise_intersection",
            "B_in_selected_cert_text": "3.3000000000000000000001e-20",
        }
        refresh = {
            "B_in_selection": "coherent_branchwise_intersection",
            "B_in_selected_cert_text": "3.3000000000000000000001e-20",
        }
        bounds = verifier.validate_input_tail_bounds(
            report, csv_row, candidate, refresh, final_row
        )
        self.assertEqual(bounds["combined"], Fraction(Decimal("3.3e-20")))

    def test_input_tail_provenance_binds_sources_and_canonical_path(self) -> None:
        report = {
            "producer_helper": "blaschke_deformation_phase2_final_aggregation.py",
            "producer_schema": "phase2-final-aggregation-v3",
            "module_sha256": verifier._sha256_file(
                RELEASE_ROOT
                / verifier.SOURCE_PATHS["blaschke_deformation_certification.py"]
            ),
            "source_sha256": verifier._sha256_file(
                RELEASE_ROOT
                / verifier.SOURCE_PATHS[
                    "blaschke_deformation_phase2_final_aggregation.py"
                ]
            ),
            "profile_path": (
                "Numerics/outputs/blaschke_deformation_certifier/data/"
                "blaschke_deformation_input_tail_boundary_profile_N600.csv"
            ),
        }
        result = verifier.validate_input_tail_provenance(RELEASE_ROOT, report)
        self.assertEqual(result["profile_path"], report["profile_path"])

        mutations = {
            "producer_helper": "wrong.py",
            "producer_schema": "phase2-final-aggregation-v2",
            "module_sha256": "0" * 64,
            "source_sha256": "0" * 64,
            "profile_path": str(
                RELEASE_ROOT
                / "Numerics/outputs/blaschke_deformation_certifier/data/"
                "blaschke_deformation_input_tail_boundary_profile_N600.csv"
            ),
        }
        for field, value in mutations.items():
            broken = dict(report)
            broken[field] = value
            with self.subTest(field=field), self.assertRaises(
                verifier.CertificateEquivalenceError
            ):
                verifier.validate_input_tail_provenance(RELEASE_ROOT, broken)

    def test_input_tail_primitive_and_selection_drift_fails(self) -> None:
        report = {
            "branchwise_profile_cert_u": "3.4e-20",
            "combined_row_cert_u": "3.3e-20",
            "maximum_branch_radius_u": "2.29",
            "boundary_cover_certified": True,
            "exact_prefix_certified": True,
            "geometric_remainder_certified": True,
            "branchwise_profile_certified": True,
            "combined_row_certified": True,
        }
        csv_row = {key: str(value) for key, value in report.items()}
        candidate = {
            "B_in_selection": "coherent_branchwise_intersection",
            "B_in_selected_cert_text": "3.3000000000000000000001e-20",
        }
        refresh = dict(candidate)
        final_row = {
            "B_in_selection": "coherent_branchwise_intersection",
            "B_in": "3.3000000000000000000001e-20",
            "B_in_branchwise": "3.4e-20",
            "B_in_combined_row": "3.3e-20",
        }

        cases: tuple[str, dict[str, object], str] = (
            (
                "csv primitive",
                {"csv_row": {**csv_row, "combined_row_cert_u": "3.2e-20"}},
                "primitives disagree",
            ),
            (
                "combined enclosure",
                {"report": {**report, "combined_row_cert_u": "3.5e-20"},
                 "csv_row": {**csv_row, "combined_row_cert_u": "3.5e-20"}},
                "not enclosed",
            ),
            (
                "local radius",
                {"report": {**report, "maximum_branch_radius_u": "0"},
                 "csv_row": {**csv_row, "maximum_branch_radius_u": "0"}},
                "outside",
            ),
            (
                "final primitive",
                {"final_row": {**final_row, "B_in": "3.2e-20"}},
                "one-sidedly bind",
            ),
            (
                "selection text",
                {"candidate": {**candidate, "B_in_selected_cert_text": "3.3000000000000001e-20"}},
                "not exactly bound",
            ),
            (
                "selection route",
                {"refresh": {**refresh, "B_in_selection": "other"}},
                "selection route",
            ),
            (
                "CSV theorem gate",
                {"csv_row": {**csv_row, "combined_row_certified": "False"}},
                "combined_row_certified",
            ),
        )
        base = {
            "report": report,
            "csv_row": csv_row,
            "candidate": candidate,
            "refresh": refresh,
            "final_row": final_row,
        }
        for label, changes, message in cases:
            values = {**base, **changes}
            with self.subTest(label=label), self.assertRaisesRegex(
                verifier.CertificateEquivalenceError, message
            ):
                verifier.validate_input_tail_bounds(
                    values["report"],
                    values["csv_row"],
                    values["candidate"],
                    values["refresh"],
                    values["final_row"],
                )

    def test_missing_input_tail_configuration_fails(self) -> None:
        policy = _policy()
        report = {
            "map_label": "blaschke_mu_0p3",
            "N": 600,
            "rho": "2.725",
            "r": verifier.CANONICAL_HARDY_RADIUS,
            "mu": "0.3",
            "cells": 65536,
            "precision_bits": 192,
            "prefix_terms": 24,
            "J": 623,
        }
        csv_row = {key: str(value) for key, value in report.items()}
        for field in (
            "map_label",
            "N",
            "rho",
            "r",
            "mu",
            "cells",
            "precision_bits",
            "prefix_terms",
            "J",
        ):
            broken = dict(report)
            broken.pop(field)
            with self.subTest(field=field), self.assertRaises(
                verifier.CertificateEquivalenceError
            ):
                verifier.validate_input_tail_configuration(policy, broken, csv_row)

    def test_wrong_or_nonidentical_input_tail_configuration_fails(self) -> None:
        policy = _policy()
        report = {
            "map_label": "blaschke_mu_0p3",
            "N": 600,
            "rho": "2.725",
            "r": verifier.CANONICAL_HARDY_RADIUS,
            "mu": "0.3",
            "cells": 65536,
            "precision_bits": 192,
            "prefix_terms": 24,
            "J": 623,
        }
        wrong_values = {
            "map_label": "wrong_map",
            "N": "601",
            "rho": "2.726",
            "r": "2.473669807791324",
            "mu": "0.31",
            "cells": "65535",
            "precision_bits": "191",
            "prefix_terms": "23",
            "J": "622",
        }
        for field, wrong in wrong_values.items():
            wrong_report = dict(report)
            wrong_report[field] = wrong
            csv_row = {key: str(value) for key, value in wrong_report.items()}
            with self.subTest(field=field), self.assertRaisesRegex(
                verifier.CertificateEquivalenceError,
                "configuration drifted",
            ):
                verifier.validate_input_tail_configuration(
                    policy, wrong_report, csv_row
                )
        nonidentical_csv = {key: str(value) for key, value in report.items()}
        nonidentical_csv["rho"] = "2.726"
        with self.assertRaisesRegex(
            verifier.CertificateEquivalenceError, "configuration identity"
        ):
            verifier.validate_input_tail_configuration(
                policy, report, nonidentical_csv
            )

    def test_report_hash_maps_require_canonical_relative_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = Path("Numerics/outputs/example.csv")
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_text("evidence\n", encoding="utf-8")
            digest = verifier._sha256_file(path)
            self.assertEqual(
                verifier._validate_named_output_hashes(
                    root,
                    {relative.as_posix(): digest},
                    label="fixture manifest",
                ),
                1,
            )
            with self.assertRaisesRegex(
                verifier.CertificateEquivalenceError, "Unsafe"
            ):
                verifier._validate_named_output_hashes(
                    root,
                    {str(path): digest},
                    label="fixture manifest",
                )


class TargetSemanticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = TargetFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_positive_reaggregation_ignores_lower_display_aggregate(self) -> None:
        result = self.fixture.validate()
        self.assertEqual(result["target_count"], 24)
        self.assertEqual(result["total_multiplicity"], 30)
        self.assertEqual(
            result["route_counts"],
            {"Schur-count/Schur-moat": 17, "Schur-count/Laurent-moat": 7},
        )

    def test_bad_schur_count_fails(self) -> None:
        self.fixture.certificate_rows[0]["schur_diagonal_algebraic_count"] = 2
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "Schur count"):
            self.fixture.validate()

    def test_bad_route_fails(self) -> None:
        self.fixture.certificate_rows[0]["certificate_route"] = "Schur-count/Laurent-moat"
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "route|target set"):
            self.fixture.validate()

    def test_bad_row_gate_fails(self) -> None:
        self.fixture.certificate_rows[0]["theorem_certified"] = "False"
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "theorem_certified"):
            self.fixture.validate()

    def test_non_upper_triangular_marker_fails(self) -> None:
        self.fixture.certificate_rows[0]["count_reference_matrix"] = "generic dense matrix"
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "Schur count marker"):
            self.fixture.validate()

    def test_nonpositive_laurent_moat_fails(self) -> None:
        row = self.fixture.certificate_rows[17]
        row["exact_dyadic_residual_sum_upper"] = "0.99"
        row["coefficient_frobenius_sum_upper"] = "100"
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "nonpositive"):
            self.fixture.validate()

    def test_laurent_q_condition_is_in_denominator(self) -> None:
        for row in self.fixture.certificate_rows:
            row["Q_condition_upper"] = "100"
        self.fixture.q_condition = Fraction(100)
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "nonpositive"):
            self.fixture.validate()

    def test_gain_at_least_one_fails(self) -> None:
        for row in self.fixture.certificate_rows:
            row["epsilon_upper"] = "1"
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "not below one"):
            self.fixture.validate(epsilon="1")

    def test_sampled_theorem_use_fails(self) -> None:
        self.fixture.certificate_rows[0]["sampled_values_used_in_theorem_gate"] = "True"
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "sampled_values"):
            self.fixture.validate()

    def test_digest_only_theorem_use_fails(self) -> None:
        self.fixture.certificate_rows[17]["coefficient_digest_used_in_theorem_gate"] = "True"
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "digest"):
            self.fixture.validate()

    def test_modewise_laurent_residual_is_bound_into_moat(self) -> None:
        name = str(self.fixture.certificate_rows[17]["name"])
        for mode in self.fixture.mode_rows:
            if mode["name"] == name:
                mode["residual_frobenius_upper"] = "0.49"
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "nonpositive"):
            self.fixture.validate()

    def test_exact_target_centre_drift_fails(self) -> None:
        self.fixture.plan_rows[0]["centre_exact"] = "1/2"
        self.fixture.certificate_rows[0]["centre_exact"] = "1/2"
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "contour geometry"):
            self.fixture.validate()


class SchurStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.policy = _policy()
        self.plan_rows: list[dict[str, object]] = []
        diagonal: list[complex] = []
        for geometry in verifier.expected_target_geometry():
            rank = int(geometry["rank"])
            name = str(geometry["name"])
            multiplicity = int(geometry["expected_multiplicity"])
            self.plan_rows.append(
                {
                    "rank": rank,
                    "name": name,
                    "expected_multiplicity": multiplicity,
                    "centre_exact": str(geometry["centre"]),
                    "radius_exact": str(geometry["radius"]),
                }
            )
            diagonal.extend([complex(float(geometry["centre"]), 0)] * multiplicity)
        diagonal.extend([complex(-1000, 0)] * (600 - len(diagonal)))
        self.matrix = np.diag(np.asarray(diagonal, dtype=np.complex128))
        self.schur_report = {
            "certificate_schema": self.policy["certificate_schemas"]["validated_schur"],
            "map_label": self.policy["map_label"],
            "N": 600,
            "M": 610,
            "rho": "2.725",
            "r": verifier.CANONICAL_HARDY_RADIUS,
            "precision_bits": 256,
            "similarity_identity": "A_N_circ Q = Q T + R",
            "status": "interval-certified exact-binary Schur similarity",
            "schur_similarity_certified": True,
            "eta_schur_upper": 0.01,
            "eta_schur_upper_text": "0.01",
            "Q_condition_upper": 1.0,
            "Q_condition_upper_text": "1",
            "midpoint_sha256": "a" * 64,
            "exact_dyadic_schur_below_diagonal_entry_count": 179700,
            "exact_dyadic_schur_below_diagonal_zero_count": 179700,
            "exact_dyadic_schur_below_diagonal_all_zero": True,
            "exact_dyadic_schur_upper_triangular_certified": True,
        }
        self.spectral_report = {
            "eta_schur_upper_text": "0.01",
            "exact_dyadic_schur_below_diagonal_entry_count": 179700,
            "exact_dyadic_schur_below_diagonal_zero_count": 179700,
            "exact_dyadic_schur_below_diagonal_all_zero": True,
            "exact_dyadic_schur_upper_triangular_certified": True,
        }
        self._write()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self) -> None:
        _write_csv(_artifact(self.root, self.policy, "contour_plan"), self.plan_rows)
        path = _artifact(self.root, self.policy, "validated_schur_npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            T=self.matrix,
            Q=np.eye(600, dtype=np.complex128),
            absolute_T_upper=np.abs(self.matrix),
            midpoint_sha256=np.asarray("a" * 64),
            eta_schur_upper=np.asarray(0.01),
            Q_condition_upper=np.asarray(1.0),
        )

    def test_recomputes_all_exact_schur_counts(self) -> None:
        result = verifier._validate_exact_dyadic_schur(
            self.root, self.policy, self.schur_report, self.spectral_report
        )
        self.assertEqual(result["exact_diagonal_counts"], self.policy["expected_multiplicities"])

    def test_nonzero_lower_triangle_fails(self) -> None:
        self.matrix[1, 0] = 1
        self._write()
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "below the diagonal"):
            verifier._validate_exact_dyadic_schur(
                self.root, self.policy, self.schur_report, self.spectral_report
            )

    def test_incorrect_exact_diagonal_count_fails(self) -> None:
        self.matrix[0, 0] = -1000
        self._write()
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "Exact Schur-diagonal count"):
            verifier._validate_exact_dyadic_schur(
                self.root, self.policy, self.schur_report, self.spectral_report
            )

    def test_false_upper_triangular_gate_fails(self) -> None:
        self.schur_report["exact_dyadic_schur_upper_triangular_certified"] = False
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "upper_triangular"):
            verifier._validate_exact_dyadic_schur(
                self.root, self.policy, self.schur_report, self.spectral_report
            )

    def test_npz_q_scalar_mismatch_fails(self) -> None:
        self.schur_report["Q_condition_upper"] = 2.0
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "NPZ/report"):
            verifier._validate_exact_dyadic_schur(
                self.root, self.policy, self.schur_report, self.spectral_report
            )

    def test_npz_eta_scalar_mismatch_fails(self) -> None:
        self.schur_report["eta_schur_upper"] = 0.02
        self.schur_report["eta_schur_upper_text"] = "0.02"
        self.spectral_report["eta_schur_upper_text"] = "0.02"
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "NPZ/report"):
            verifier._validate_exact_dyadic_schur(
                self.root, self.policy, self.schur_report, self.spectral_report
            )


class GlobalGateTests(unittest.TestCase):
    def test_global_report_gate_failure(self) -> None:
        policy = _policy()
        report: dict[str, object] = {
            "certificate_schema": policy["certificate_schemas"]["spectral_contours"],
            "map_label": policy["map_label"],
            "N": 600,
            "M": 610,
            "rho": "2.725",
            "alpha_numerator": 13,
            "alpha_denominator": 20,
            "alpha_power_count": 18,
            "mu_numerator": 3,
            "mu_denominator": 10,
            "mu_power_count": 6,
            "default_radius_numerator": 1,
            "default_radius_denominator": 5,
            "contour_precision_bits": 256,
            "target_count": 24,
            "total_expected_algebraic_multiplicity": 30,
            "total_certified_algebraic_multiplicity": 30,
            "schur_count_target_count": 24,
            "schur_triangular_moat_target_count": 17,
            "laurent_moat_target_count": 7,
            "r": policy["canonical_hardy_radius"],
            "epsilon_upper_text": "0.001",
            "eta_A_upper_text": "0.01",
            "status": "theorem-certified twenty-four-target Riesz-rank package",
        }
        for gate in policy["true_report_gates"]:
            report[gate] = True
        for gate in policy["false_report_gates"]:
            report[gate] = False
        verifier._validate_global_report(
            policy,
            report,
            radius=Decimal(policy["canonical_hardy_radius"]),
            epsilon=Fraction(Decimal("0.001")),
            eta_a=Decimal("0.01"),
        )
        report["all_small_gain_tests_pass"] = False
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "all_small_gain"):
            verifier._validate_global_report(
                policy,
                report,
                radius=Decimal(policy["canonical_hardy_radius"]),
                epsilon=Fraction(Decimal("0.001")),
                eta_a=Decimal("0.01"),
            )

    def test_global_rho_drift_fails(self) -> None:
        policy = _policy()
        report: dict[str, object] = {
            "certificate_schema": policy["certificate_schemas"]["spectral_contours"],
            "map_label": policy["map_label"],
            "N": 600,
            "M": 610,
            "rho": "2.726",
            "alpha_numerator": 13,
            "alpha_denominator": 20,
            "alpha_power_count": 18,
            "mu_numerator": 3,
            "mu_denominator": 10,
            "mu_power_count": 6,
            "default_radius_numerator": 1,
            "default_radius_denominator": 5,
            "contour_precision_bits": 256,
            "target_count": 24,
            "total_expected_algebraic_multiplicity": 30,
            "total_certified_algebraic_multiplicity": 30,
            "schur_count_target_count": 24,
            "schur_triangular_moat_target_count": 17,
            "laurent_moat_target_count": 7,
            "r": policy["canonical_hardy_radius"],
            "epsilon_upper_text": "0.001",
            "eta_A_upper_text": "0.01",
            "status": "theorem-certified twenty-four-target Riesz-rank package",
        }
        report.update({gate: True for gate in policy["true_report_gates"]})
        report.update({gate: False for gate in policy["false_report_gates"]})
        with self.assertRaisesRegex(
            verifier.CertificateEquivalenceError, "metadata drifted"
        ):
            verifier._validate_global_report(
                policy,
                report,
                radius=Decimal(policy["canonical_hardy_radius"]),
                epsilon=Fraction(Decimal("0.001")),
                eta_a=Decimal("0.01"),
            )


class PolicyAndHardyBindingTests(unittest.TestCase):
    def test_policy_weakening_is_rejected(self) -> None:
        original = _policy()
        mutations = (
            ("q gap", lambda value: value.__setitem__("q_gap_target", "0.928")),
            ("radius", lambda value: value.__setitem__("canonical_hardy_radius", "2.473669807791324")),
            ("target", lambda value: value["target_order"].__setitem__(0, "alpha^0")),
            ("multiplicity", lambda value: value["expected_multiplicities"].__setitem__(0, 2)),
            ("gate", lambda value: value["true_report_gates"].pop()),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                policy = json.loads(json.dumps(original))
                mutate(policy)
                path = root / verifier.POLICY_RELATIVE
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(policy), encoding="utf-8")
                with self.assertRaises(verifier.CertificateEquivalenceError):
                    verifier._load_policy(root)

    def test_hardy_summary_cannot_underreport_eta(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = _policy()
            gates = (
                "gauss_nodes_disjoint",
                "gauss_weight_sum_contains_two",
                "branch_values_certified",
                "branch_weights_positive",
                "branch_ordering_certified",
                "legendre_recurrence_certified",
                "connection_inverse_certified",
                "all_matrix_products_certified",
                "reference_is_exact_dyadic",
                "binary64_midpoint_is_diagnostic_only",
                "mathematical_block_enclosed",
                "matrix_enclosure_certified",
            )
            report: dict[str, object] = {
                "schema": policy["certificate_schemas"]["hardy_matrix"],
                "map_label": policy["map_label"],
                "N": 600,
                "M": 610,
                "rho": "2.725",
                "r": verifier.CANONICAL_HARDY_RADIUS,
                "precision_bits": 2048,
                "entry_count": 360000,
                "eta_A_upper_text": "0.01",
                "payload_sha256": "a" * 64,
                "midpoint_sha256": "b" * 64,
                "spectral_use_ready": True,
                "precision_budget_pass": True,
                "precision_budget_enforced": True,
            }
            report.update({gate: True for gate in gates})
            summary = {
                "schema": policy["certificate_schemas"]["hardy_matrix"],
                "map_label": policy["map_label"],
                "N": "600",
                "M": "610",
                "rho": "2.725",
                "r": verifier.CANONICAL_HARDY_RADIUS,
                "precision_bits": "2048",
                "entry_count": "360000",
                "eta_A_upper_text": "0.01",
                "payload_sha256": "a" * 64,
                "midpoint_sha256": "b" * 64,
                "spectral_use_ready": "True",
                "precision_budget_pass": "True",
            }
            summary.update({gate: "True" for gate in gates})
            path = _artifact(root, policy, "hardy_2048_summary")
            _write_csv(path, [summary])
            midpoint_path = _artifact(root, policy, "hardy_2048_midpoint")
            midpoint_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                midpoint_path,
                A_N_circ_binary64_diagnostic=np.zeros(
                    (600, 600), dtype=np.float64
                ),
                N=np.asarray(600),
                M=np.asarray(610),
                rho=np.asarray("2.725"),
                r=np.asarray(verifier.CANONICAL_HARDY_RADIUS),
                precision_bits=np.asarray(2048),
                midpoint_sha256=np.asarray("b" * 64),
                authoritative_exact_dyadic_payload=np.asarray(
                    str(_artifact(root, policy, "hardy_2048_payload"))
                ),
            )
            self.assertNotEqual(
                verifier._sha256_file(midpoint_path), report["midpoint_sha256"]
            )
            verifier._validate_hardy_report(
                root, policy, report, bits=2048, expected_ready=True
            )
            summary["eta_A_upper_text"] = "0.009"
            _write_csv(path, [summary])
            with self.assertRaisesRegex(
                verifier.CertificateEquivalenceError, "theorem primitive"
            ):
                verifier._validate_hardy_report(
                    root, policy, report, bits=2048, expected_ready=True
                )

            summary["eta_A_upper_text"] = "0.01"
            _write_csv(path, [summary])
            np.savez_compressed(
                midpoint_path,
                A_N_circ_binary64_diagnostic=np.zeros(
                    (600, 600), dtype=np.float64
                ),
                N=np.asarray(600),
                M=np.asarray(610),
                rho=np.asarray("2.725"),
                r=np.asarray(verifier.CANONICAL_HARDY_RADIUS),
                precision_bits=np.asarray(2048),
                midpoint_sha256=np.asarray("c" * 64),
                authoritative_exact_dyadic_payload=np.asarray(
                    str(_artifact(root, policy, "hardy_2048_payload"))
                ),
            )
            with self.assertRaisesRegex(
                verifier.CertificateEquivalenceError, "midpoint identity"
            ):
                verifier._validate_hardy_report(
                    root, policy, report, bits=2048, expected_ready=True
                )

    def test_1024_diagnostic_is_not_in_theorem_closure(self) -> None:
        paths = _policy()["theorem_required_generated_paths"]
        self.assertEqual(len(paths), 27)
        self.assertFalse(any("bits1024" in path for path in paths))
        self.assertTrue(any("bits2048" in path for path in paths))


class RunnerSafetyTests(unittest.TestCase):
    def test_ephemeral_kernel_rejects_injected_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            python = Path(sys.executable).resolve()
            environment = reproduction._environment()
            path = reproduction._provision_ephemeral_kernel(
                workspace=workspace,
                python=python,
                kernel_name="blaschke-certificate-replay",
                environment=environment,
            )
            reproduction.validate_ephemeral_kernel_spec(path, python)
            spec = json.loads(path.read_text(encoding="utf-8"))
            spec["env"]["PYTHONPATH"] = "/tmp/injected"
            path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaisesRegex(reproduction.ReproductionError, "argv/env"):
                reproduction.validate_ephemeral_kernel_spec(path, python)

    def test_workspace_inside_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            inside = source / "scratch"
            outside = Path(temporary) / "outside"
            inside.mkdir(parents=True)
            outside.mkdir()
            with self.assertRaisesRegex(stager.StageError, "separate"):
                stager._require_separate_workspace(source.resolve(), inside.resolve())
            stager._require_separate_workspace(source.resolve(), outside.resolve())

    def test_workspace_must_be_empty_and_have_no_symlink_component(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = root / "empty"
            empty.mkdir()
            self.assertEqual(stager._empty_workspace(empty), empty.resolve())
            (empty / "ephemeral-jupyter").mkdir()
            with self.assertRaisesRegex(stager.StageError, "empty and fresh"):
                stager._empty_workspace(empty)

            real = root / "real"
            real.mkdir()
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(stager.StageError, "Symlink forbidden"):
                stager._empty_workspace(linked)

    def test_authenticated_copy_detects_mutation_during_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "staged" / "member"
            original = b"authenticated bytes"
            source.write_bytes(original)
            original_copy = stager.shutil.copyfile

            def mutate_then_copy(source_path: object, destination_path: object) -> object:
                Path(source_path).write_bytes(b"mutated during copy")
                return original_copy(source_path, destination_path)

            with mock.patch.object(
                stager.shutil, "copyfile", side_effect=mutate_then_copy
            ):
                with self.assertRaisesRegex(stager.StageError, "authenticated release row"):
                    stager._copy_authenticated_file(
                        source,
                        destination,
                        expected_size=len(original),
                        expected_sha256=hashlib.sha256(original).hexdigest(),
                        label="test member",
                    )

    def test_preplanted_ephemeral_kernel_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            target = workspace / "target"
            target.mkdir()
            (workspace / "ephemeral-jupyter").symlink_to(
                target, target_is_directory=True
            )
            with self.assertRaisesRegex(
                reproduction.ReproductionError, "pre-existing ephemeral"
            ):
                reproduction._provision_ephemeral_kernel(
                    workspace=workspace,
                    python=Path(sys.executable).resolve(),
                    kernel_name="blaschke-certificate-replay",
                    environment=reproduction._environment(),
                )

    def test_safe_loader_ignores_poisoned_ambient_import_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            poison = Path(temporary)
            marker = poison / "poison-imported"
            (poison / "prepare_blaschke_source_only_replay.py").write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n"
                "raise RuntimeError('poison imported')\n",
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment.update(
                {
                    "PYTHONPATH": str(poison),
                    "PYTHONSAFEPATH": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(NUMERICS / "run_blaschke_clean_room_replay.py"),
                    "--help",
                ],
                cwd=poison,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(marker.exists())

    def test_child_import_preflight_uses_only_authenticated_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            poison = Path(temporary)
            (poison / "Numerics").mkdir()
            (poison / "Numerics" / "__init__.py").write_text(
                "raise RuntimeError('ambient package imported')\n", encoding="utf-8"
            )
            environment = dict(os.environ)
            environment.update(
                {
                    "PYTHONPATH": os.pathsep.join(
                        (str(RELEASE_ROOT), str(NUMERICS))
                    ),
                    "PYTHONSAFEPATH": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "BLASCHKE_AUTHENTICATED_REPLAY_ROOT": str(RELEASE_ROOT),
                }
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    clean_replay.PROJECT_IMPORT_PREFLIGHT + "\nprint('PASS')",
                ],
                cwd=poison,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.strip(), "PASS")

    def test_theorem_only_closure_and_final_export_contract(self) -> None:
        paths = clean_replay.THEOREM_REQUIRED_GENERATED_PATHS
        self.assertEqual(len(paths), 27)
        self.assertEqual(len(set(paths)), 27)
        self.assertTrue(any("coherent_packet" in path for path in paths))
        self.assertTrue(any("phase2_certified_single_space_row" in path for path in paths))
        self.assertFalse(any("bits1024" in path for path in paths))
        self.assertFalse(any("historical" in path for path in paths))
        self.assertFalse(any(path.endswith(".ipynb") for path in paths))
        row = clean_replay.final_certificate_compatibility_row(
            {
                "kappa_hat": 3,
                "tail_components_interval": True,
                "certified": True,
                "q_gap_target_text": "0.927",
            }
        )
        self.assertEqual(row["C_tr"], 3)
        self.assertIs(row["tail_mismatch_certified"], True)
        self.assertIs(row["branch_data_certified"], True)
        self.assertEqual(row["q_gap_target_text"], "0.927")

    def test_final_export_preserves_exact_mpmath_decimal_text(self) -> None:
        class ExactMpmathScalar:
            __module__ = "mpmath.ctx_mp_python"

            def __init__(self, text: str) -> None:
                self.text = text

            def __str__(self) -> str:
                return self.text

            def __float__(self) -> float:
                return float(self.text)

        exact = "0.000000000000000000033264421486718413000000000000000000000000000963"
        row = clean_replay.final_certificate_compatibility_row(
            {
                "kappa_hat": 3,
                "tail_components_interval": True,
                "certified": True,
                "B_in": ExactMpmathScalar(exact),
            }
        )
        self.assertIsInstance(row["B_in"], str)
        self.assertEqual(row["B_in"], exact)
        self.assertNotEqual(row["B_in"], repr(float(exact)))

    def test_caller_inventory_identity_is_not_self_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            digest = "a" * 64
            result = stager._validate_external_identity(
                root,
                inventory_sha256=digest,
                expected_inventory_sha256=digest,
                expected_git_commit=None,
            )
            self.assertEqual(result["value"], digest)
            with self.assertRaisesRegex(stager.StageError, "caller-supplied"):
                stager._validate_external_identity(
                    root,
                    inventory_sha256=digest,
                    expected_inventory_sha256="b" * 64,
                    expected_git_commit=None,
                )

    def test_wrapper_rejects_quick_to_full_escalation(self) -> None:
        completed = subprocess.run(
            ["bash", str(RELEASE_ROOT / "reproduce-certificate.sh"), "quick", "--mode", "full-replay"],
            cwd=RELEASE_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("protected wrapper option", completed.stderr)

    def test_verifier_cli_has_no_integrity_bypass(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-I", "-B", str(NUMERICS / "verify_blaschke_certificate_equivalence.py"), "--help"],
            cwd=RELEASE_ROOT,
            text=True,
            check=True,
            stdout=subprocess.PIPE,
        )
        self.assertNotIn("skip-release-integrity", completed.stdout)


class TheoremOnlyEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.release_inventory = "a" * 64
        self.staged_inventory = "b" * 64
        self.nonce = "c" * 64
        self.records: list[dict[str, object]] = []
        for relative in verifier.EXPECTED_THEOREM_REQUIRED_PATHS:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode("utf-8"))
            stat_result = path.stat()
            self.records.append(
                {
                    "path": relative,
                    "bytes": stat_result.st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "mtime_ns": stat_result.st_mtime_ns,
                }
            )
        self.preparation = {"inventory_sha256": self.staged_inventory}
        preparation_path = self.root / "source-only-replay-preparation.json"
        preparation_path.write_text(json.dumps(self.preparation), encoding="utf-8")
        self.forced = {
            "BLASCHKE_FORCE_HARDY_MATRIX": "1",
            "BLASCHKE_FORCE_CONTOURS": "1",
            "BLASCHKE_SKIP_HARDY_STARTING_AUDIT": "1",
            "BLASCHKE_SKIP_HISTORICAL_PHASE4": "1",
            "MPMATH_PF_ASSEMBLY_WORKERS": "24",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONPATH": os.pathsep.join(
                (str(self.root), str(self.root / "Numerics"))
            ),
            "BLASCHKE_AUTHENTICATED_REPLAY_ROOT": str(self.root),
        }
        python = str(Path(sys.executable).resolve())
        driver = " ".join(
            (
                "rebuild_phase2_inputs",
                "certify_finite_m_completion",
                "certify_resolved_response_completion",
                "certify_final_phase2_aggregation",
                "build_or_load_hardy_matrix_certificate",
            )
        )
        commands = [
            [python, "-B", "-c", "blas preflight"],
            [python, "-B", "Numerics/build_blaschke_deformation_certifier.py"],
            [python, "-B", "Numerics/build_blaschke_deformation_thesis_math_notebook.py"],
            [python, "-u", "-B", "-c", driver],
            [python, "-u", "-B", "Numerics/blaschke_deformation_contour_certification.py", "--root", ".", "--force"],
        ]
        self.evidence = {
            "receipt_schema": "blaschke-theorem-only-compute-v1",
            "status": "theorem-only source reconstruction complete",
            "stage": "theorem-only-compute",
            "normalization_run": False,
            "provenance_refresh_run": False,
            "published_comparison_run": False,
            "kernel_used_for_theorem_compute": False,
            "historical_phase4_diagnostics_skipped": True,
            "historical_phase4_diagnostics_authoritative": False,
            "hardy_1024_diagnostic_skipped": True,
            "hardy_1024_diagnostic_authoritative": False,
            "forced_rebuild_environment": self.forced,
            "python_executable": python,
            "commands": [{"command": command, "returncode": 0} for command in commands],
            "preparation": self.preparation,
            "run_started_ns": 0,
            "run_finished_ns": 2**63 - 1,
            "generated_closure": {
                "status": "declared generated output closure complete",
                "theorem_only": True,
                "generated_path_count": 27,
                "generated_paths": list(verifier.EXPECTED_THEOREM_REQUIRED_PATHS),
                "generated_files": self.records,
                "missing": [],
            },
        }
        self._write_evidence_and_attestation()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_evidence_and_attestation(self) -> None:
        evidence_path = self.root / verifier.COMPUTE_EVIDENCE_RELATIVE
        evidence_path.write_text(json.dumps(self.evidence), encoding="utf-8")
        preparation_path = self.root / "source-only-replay-preparation.json"
        attestation = {
            "receipt_schema": "blaschke-certificate-current-run-attestation-v1",
            "status": "current theorem-only run hash-bound",
            "run_nonce": self.nonce,
            "outer_started_ns": 0,
            "outer_finished_ns": 2**63 - 1,
            "release_inventory_sha256": self.release_inventory,
            "staged_inventory_sha256": self.staged_inventory,
            "compute_evidence_path": verifier.COMPUTE_EVIDENCE_RELATIVE.as_posix(),
            "compute_evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            "preparation_receipt_path": "source-only-replay-preparation.json",
            "preparation_receipt_sha256": hashlib.sha256(preparation_path.read_bytes()).hexdigest(),
            "theorem_generated_path_count": 27,
            "theorem_generated_files": self.records,
            "forced_rebuild_environment": self.forced,
            "interpreter_sha256": hashlib.sha256(
                Path(sys.executable).resolve().read_bytes()
            ).hexdigest(),
            "historical_diagnostics_skipped": True,
            "historical_diagnostics_authoritative": False,
            "hardy_1024_diagnostic_skipped": True,
            "hardy_1024_diagnostic_authoritative": False,
            "clean_replay_command": [
                str(Path(sys.executable).resolve()),
                "-B",
                "Numerics/run_blaschke_clean_room_replay.py",
                "--expected-inventory-sha256",
                self.staged_inventory,
                "--theorem-only-compute",
            ],
        }
        (self.root / verifier.RUN_ATTESTATION_RELATIVE).write_text(
            json.dumps(attestation), encoding="utf-8"
        )

    def test_theorem_only_current_run_binding_passes(self) -> None:
        result = verifier._validate_compute_evidence(
            self.root,
            expected_run_nonce=self.nonce,
            expected_release_inventory_sha256=self.release_inventory,
        )
        self.assertEqual(result["theorem_generated_path_count"], 27)

    def test_stale_or_forged_nonce_fails(self) -> None:
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "attestation"):
            verifier._validate_compute_evidence(
                self.root,
                expected_run_nonce="d" * 64,
                expected_release_inventory_sha256=self.release_inventory,
            )

    def test_missing_current_theorem_output_fails(self) -> None:
        (self.root / verifier.EXPECTED_THEOREM_REQUIRED_PATHS[0]).unlink()
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "current theorem output"):
            verifier._validate_compute_evidence(
                self.root,
                expected_run_nonce=self.nonce,
                expected_release_inventory_sha256=self.release_inventory,
            )

    def test_old_compute_schema_fails(self) -> None:
        self.evidence["receipt_schema"] = "blaschke-clean-room-compute-only-v1"
        self._write_evidence_and_attestation()
        with self.assertRaisesRegex(verifier.CertificateEquivalenceError, "not complete"):
            verifier._validate_compute_evidence(
                self.root,
                expected_run_nonce=self.nonce,
                expected_release_inventory_sha256=self.release_inventory,
            )

    def test_poisoned_compute_import_roots_fail(self) -> None:
        self.forced["PYTHONPATH"] = "/tmp/poisoned"
        self._write_evidence_and_attestation()
        with self.assertRaisesRegex(
            verifier.CertificateEquivalenceError, "authenticated staged import roots"
        ):
            verifier._validate_compute_evidence(
                self.root,
                expected_run_nonce=self.nonce,
                expected_release_inventory_sha256=self.release_inventory,
            )

    def test_interpreter_hash_mismatch_fails(self) -> None:
        attestation_path = self.root / verifier.RUN_ATTESTATION_RELATIVE
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
        attestation["interpreter_sha256"] = "0" * 64
        attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
        with self.assertRaisesRegex(
            verifier.CertificateEquivalenceError, "interpreter executable hash"
        ):
            verifier._validate_compute_evidence(
                self.root,
                expected_run_nonce=self.nonce,
                expected_release_inventory_sha256=self.release_inventory,
            )


if __name__ == "__main__":
    unittest.main()
