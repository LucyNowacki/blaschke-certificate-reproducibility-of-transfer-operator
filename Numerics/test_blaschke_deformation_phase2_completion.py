"""Focused tests for the standalone Phase 2 completion helpers."""

from __future__ import annotations

import csv
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
    _canonical_resolved_response_input_member,
    _require_transport_geometry_binding,
    certify_resolved_response_completion,
)
from blaschke_deformation_phase2_final_aggregation import (
    FINAL_CERTIFICATION_GATES,
    INPUT_CERTIFICATION_GATES,
    Phase2FinalAggregationConfig,
    RESPONSE_CERTIFICATION_GATES,
    _bind_refreshed_candidate_exact_exports,
    _canonical_input_tail_profile_member,
    _require_canonical_input_tail_profile_member,
    _validate_response_artifact_geometry_contract,
    certify_final_phase2_aggregation,
)
from blaschke_deformation_phase2_geometry import (
    CANONICAL_SELECTED_HARDY_RADIUS_TEXT,
    exact_q_gap_contract,
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
    def test_resolved_response_input_members_are_portable_and_fail_closed(self) -> None:
        self.assertEqual(
            _canonical_resolved_response_input_member("generated", "safe.csv"),
            "Numerics/outputs/blaschke_deformation_certifier/data/safe.csv",
        )
        self.assertEqual(
            _canonical_resolved_response_input_member(
                "certification_source",
                "blaschke_deformation_certification.py",
            ),
            "Numerics/blaschke_deformation_certification.py",
        )
        for kind, filename in (
            ("generated", "/tmp/replay/data/safe.csv"),
            ("generated", "../safe.csv"),
            ("generated", "nested/safe.csv"),
            ("certification_source", "wrong_source.py"),
            ("unknown", "safe.csv"),
        ):
            with self.subTest(kind=kind, filename=filename), self.assertRaises(
                ValueError
            ):
                _canonical_resolved_response_input_member(kind, filename)

    def test_refreshed_candidate_rereads_the_exact_final_b_out_text(self) -> None:
        short = "2.8540338522337406e-23"
        exact = (
            "2.8540338522337406000000000000000000000000000000000000000000001621e-23"
        )
        candidate = {
            "B_out_response_prefactor_cert": short,
            "C_resp_coherent_packet_cert_u": "0.000000000000000000000006",
            "response_prefactor_certified": True,
        }
        final_certificate = {"B_out": exact}
        refreshed = _bind_refreshed_candidate_exact_exports(
            candidate,
            final_certificate,
        )
        self.assertEqual(refreshed["B_out_response_prefactor_cert"], exact)
        self.assertEqual(
            refreshed["C_resp_coherent_packet_cert_u"],
            candidate["C_resp_coherent_packet_cert_u"],
        )
        self.assertTrue(refreshed["response_prefactor_certified"])

        with tempfile.TemporaryDirectory(prefix="exact-b-out-export-") as root:
            path = Path(root) / "candidate.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(refreshed))
                writer.writeheader()
                writer.writerow(refreshed)
            with path.open("r", encoding="utf-8", newline="") as handle:
                reread = next(csv.DictReader(handle))
        self.assertEqual(reread["B_out_response_prefactor_cert"], exact)
        self.assertEqual(
            reread["B_out_response_prefactor_cert"],
            str(final_certificate["B_out"]),
        )
        self.assertNotEqual(reread["B_out_response_prefactor_cert"], short)
        self.assertEqual(
            reread["C_resp_coherent_packet_cert_u"],
            candidate["C_resp_coherent_packet_cert_u"],
        )
        self.assertEqual(reread["response_prefactor_certified"], "True")

    def test_input_tail_profile_report_member_is_portable_and_exact(self) -> None:
        expected = (
            "Numerics/outputs/blaschke_deformation_certifier/data/"
            "blaschke_deformation_input_tail_boundary_profile_N600.csv"
        )
        member = _canonical_input_tail_profile_member(600)
        self.assertEqual(member, expected)
        self.assertFalse(Path(member).is_absolute())
        self.assertEqual(
            _require_canonical_input_tail_profile_member(member, N=600),
            expected,
        )

        with self.assertRaisesRegex(ValueError, "canonical project-relative"):
            _require_canonical_input_tail_profile_member(
                "/tmp/blaschke-certificate-replay.NONSOURCE/"
                "data/blaschke_deformation_input_tail_boundary_profile_N600.csv",
                N=600,
            )

    def test_response_artifacts_and_transport_are_exactly_same_radius_bound(self) -> None:
        r_tau = "2.293091911822557449340820312"
        q_contract = exact_q_gap_contract(
            r_tau_upper=r_tau,
            hardy_radius=CANONICAL_SELECTED_HARDY_RADIUS_TEXT,
            q_gap_target="0.927",
        )
        geometry_digest = "1" * 64
        transport_digest = "2" * 64
        safe = {
            "N": "600",
            "M": "610",
            "rho": "2.725",
            "r": CANONICAL_SELECTED_HARDY_RADIUS_TEXT,
            "r_tau_interval_u": r_tau,
            "q_gap": "0.927",
            "lambda_min_cert": "0.5",
            "geometry_configuration_digest": geometry_digest,
            "transport_configuration_digest": transport_digest,
            **q_contract,
        }
        candidate = {
            **safe,
            "N": "600",
            "M": "610",
        }
        response_summary = {
            "map_label": "blaschke_mu_0p3",
            "N": "600",
            "rho": "2.725",
            "r": CANONICAL_SELECTED_HARDY_RADIUS_TEXT,
            "r_tau_interval_u": r_tau,
            "q_gap": "0.927",
            **q_contract,
        }
        coherent_summary = {
            "map_label": "blaschke_mu_0p3",
            "N": "600",
            "rho": "2.725",
            "r": CANONICAL_SELECTED_HARDY_RADIUS_TEXT,
            "r_tau": r_tau,
            "q_gap_target": "0.927",
            **q_contract,
        }
        config = Phase2FinalAggregationConfig()
        self.assertEqual(
            _validate_response_artifact_geometry_contract(
                config=config,
                safe_row=safe,
                candidate=candidate,
                response_summary=response_summary,
                coherent_summary=coherent_summary,
            ),
            q_contract,
        )
        for label, artifact in (
            ("candidate", candidate),
            ("response", response_summary),
            ("coherent", coherent_summary),
        ):
            corrupted = dict(artifact)
            corrupted["r"] = "2.473669807791324"
            kwargs = {
                "config": config,
                "safe_row": safe,
                "candidate": candidate,
                "response_summary": response_summary,
                "coherent_summary": coherent_summary,
            }
            kwargs[
                {
                    "candidate": "candidate",
                    "response": "response_summary",
                    "coherent": "coherent_summary",
                }[label]
            ] = corrupted
            with self.subTest(label=label), self.assertRaisesRegex(
                (RuntimeError, ValueError), "Hardy radius|exact decimal r"
            ):
                _validate_response_artifact_geometry_contract(**kwargs)

        with tempfile.TemporaryDirectory(prefix="transport-binding-") as root:
            output_dir = Path(root)
            data_dir = output_dir / "data"
            data_dir.mkdir()
            witness_path = (
                data_dir
                / "branch_image_balanced_candidate_transport_inverse_witness_N600.npz"
            )
            witness_path.write_bytes(b"exact transport witness fixture")
            transport = {
                "N": "600",
                "rho": "2.725",
                "r": CANONICAL_SELECTED_HARDY_RADIUS_TEXT,
                "r_tau": r_tau,
                "lambda_min_cert": "0.5",
                "geometry_configuration_digest": geometry_digest,
                "configuration_digest": transport_digest,
                "inverse_witness_sha256": _sha256(witness_path),
                "inverse_witness_path": str(witness_path),
            }
            tail = {
                "r_tau_interval_u": r_tau,
                "configuration_digest": geometry_digest,
            }
            _require_transport_geometry_binding(
                transport=transport,
                balanced=safe,
                tail=tail,
                config=Phase2ResolvedResponseConfig(),
                output_dir=output_dir,
            )
            short_transport = dict(transport)
            short_transport["r"] = "2.473669807791324"
            with self.assertRaisesRegex(RuntimeError, "exact decimal r"):
                _require_transport_geometry_binding(
                    transport=short_transport,
                    balanced=safe,
                    tail=tail,
                    config=Phase2ResolvedResponseConfig(),
                    output_dir=output_dir,
                )
            witness_path.write_bytes(b"tampered transport witness fixture")
            with self.assertRaisesRegex(RuntimeError, "payload digest"):
                _require_transport_geometry_binding(
                    transport=transport,
                    balanced=safe,
                    tail=tail,
                    config=Phase2ResolvedResponseConfig(),
                    output_dir=output_dir,
                )

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
            witness_name = (
                "branch_image_balanced_candidate_transport_inverse_witness_N600.npz"
            )
            witness_path = data_dir / witness_name
            shutil.copy2(canonical_data / witness_name, witness_path)
            transport_path = (
                data_dir / "branch_image_balanced_candidate_transport_cert_N600.csv"
            )
            with transport_path.open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                transport_rows = list(csv.DictReader(handle))
            self.assertEqual(len(transport_rows), 1)
            transport_rows[0]["inverse_witness_path"] = str(witness_path)
            transport_rows[0]["inverse_witness_sha256"] = _sha256(witness_path)
            with transport_path.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(transport_rows[0]))
                writer.writeheader()
                writer.writerows(transport_rows)

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
            self.assertEqual(
                final.final_certificate["r"],
                "2.473669807791324109321273260",
            )
            self.assertTrue(final.final_certificate["q_gap_derived_le_target"])
            self.assertEqual(final.final_certificate["q_gap_target_text"], "0.927")
            self.assertEqual(
                final.refreshed_candidate["q_gap_derived_decimal_text"],
                final.final_certificate["q_gap_derived_decimal_text"],
            )
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
