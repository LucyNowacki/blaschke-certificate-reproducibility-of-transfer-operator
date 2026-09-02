"""Focused tests for the seed-free Blaschke diagnostic-audit generators."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import blaschke_deformation_diagnostic_audits as audits


DATA_DIR = HERE / "outputs" / "blaschke_deformation_certifier" / "data"
TARGETS_PATH = DATA_DIR / "raw_spectrum_square_N_sweep_extended_targets.csv"
GEOMETRY_PATH = DATA_DIR / "transfer_lab_blaschke_mu_0p3_active_geometry.csv"
SCHUR_PATH = (
    DATA_DIR / "transfer_lab_blaschke_mu_0p3_generic_sampled_schur_envelope.csv"
)
MOAT_PATH = (
    DATA_DIR / "branch_image_wide_candidate_first15_contour_moats_N600_M610_J128.csv"
)
RETAINED_UNIVERSAL_PATH = DATA_DIR / audits.UNIVERSAL_AUDIT_FILENAME
RETAINED_FIRST14_PATH = DATA_DIR / audits.FIRST14_AUDIT_FILENAME


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def _source_frames() -> dict[str, pd.DataFrame]:
    return {
        "targets": _read(TARGETS_PATH),
        "geometry": _read(GEOMETRY_PATH),
        "schur_rows": _read(SCHUR_PATH),
        "moat_rows": _read(MOAT_PATH),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DiagnosticAuditPureGeneratorTests(unittest.TestCase):
    def test_geometry_lock_uses_full_canonical_radius(self) -> None:
        radius = 2.473669807791324
        r_tau = 2.2926584027370747
        rho = 2.725
        geometry = pd.DataFrame(
            [
                {
                    "map_label": audits.MAP_LABEL,
                    "rho": rho,
                    "r_tau": r_tau,
                    "r": radius,
                    "Phi": 4.763076532415687,
                    "Phi_star": 4.7255168399801315,
                    "q_out": radius / rho,
                    "q_gap": r_tau / radius,
                    "q_star": r_tau / radius,
                }
            ]
        )

        normalised = audits._normalise_geometry(geometry)
        self.assertEqual(normalised["r"], radius)

        rounded = geometry.copy(deep=True)
        rounded.loc[0, "r"] = 2.47367
        rounded.loc[0, "q_out"] = 2.47367 / rho
        rounded.loc[0, "q_gap"] = r_tau / 2.47367
        rounded.loc[0, "q_star"] = r_tau / 2.47367
        with self.assertRaisesRegex(
            audits.DiagnosticAuditValidationError,
            "geometry violates.*lock for r",
        ):
            audits._normalise_geometry(rounded)

    def test_pure_generators_preserve_schema_bytes_and_inputs(self) -> None:
        frames = _source_frames()
        originals = {name: frame.copy(deep=True) for name, frame in frames.items()}

        universal, first14 = audits.generate_diagnostic_audit_frames(
            map_label=audits.MAP_LABEL,
            **frames,
        )

        for name, original in originals.items():
            pd.testing.assert_frame_equal(frames[name], original)
        self.assertEqual(tuple(universal.columns), audits.UNIVERSAL_AUDIT_COLUMNS)
        self.assertEqual(tuple(first14.columns), audits.FIRST14_AUDIT_COLUMNS)
        self.assertEqual(
            first14.to_csv(index=False, lineterminator="\n").encode("utf-8"),
            RETAINED_FIRST14_PATH.read_bytes(),
        )

        status_by_item = dict(zip(universal["audit_item"], universal["status"]))
        self.assertEqual(
            status_by_item,
            {
                "target source and contour role": "theorem_certified",
                "Bernstein branch geometry": "sampled_not_interval_certified",
                "single-space Schur envelope": ("sampled_not_theorem_certified"),
                "finite packet counts": "sampled_pass",
                "finite-section contour moats": ("sampled_not_interval_certified"),
                "sampled small-gain test": "sampled_pass",
                "overall spectral certification": ("diagnostic_not_theorem_certified"),
            },
        )
        retained = _read(RETAINED_UNIVERSAL_PATH)
        self.assertEqual(
            universal["audit_item"].tolist(), retained["audit_item"].tolist()
        )
        self.assertEqual(
            universal["evidence"].iloc[:6].tolist(),
            retained["evidence"].iloc[:6].tolist(),
        )
        self.assertEqual(
            universal.iloc[-1]["evidence"],
            "Missing theorem gates: certified branch geometry; "
            "certified X-to-X perturbation bound; validated contour-moat "
            "lower bound; certified finite-section packet count",
        )
        self.assertFalse(first14["contour_interval_certified"].any())
        self.assertFalse(first14["finite_count_certified"].any())
        self.assertFalse(first14["certified_small_gain_pass"].any())
        self.assertEqual(
            set(first14["riesz_rank_status"]),
            {"sampled_pass_not_theorem_certified"},
        )

    def test_map_lock_fails_closed(self) -> None:
        with self.assertRaisesRegex(audits.DiagnosticAuditValidationError, "map lock"):
            audits.generate_diagnostic_audit_frames(
                map_label="asymmetric_blaschke_degree2",
                **_source_frames(),
            )

    def test_missing_schema_column_fails_closed(self) -> None:
        frames = _source_frames()
        frames["schur_rows"] = frames["schur_rows"].drop(
            columns=["epsilon_schur_diagnostic"]
        )
        with self.assertRaisesRegex(
            audits.DiagnosticAuditValidationError,
            "missing required columns.*epsilon_schur_diagnostic",
        ):
            audits.generate_diagnostic_audit_frames(
                map_label=audits.MAP_LABEL,
                **frames,
            )

    def test_target_alignment_fails_closed(self) -> None:
        frames = _source_frames()
        frames["moat_rows"].loc[0, "target"] = "alpha^99"
        with self.assertRaisesRegex(
            audits.DiagnosticAuditValidationError, "not aligned"
        ):
            audits.generate_diagnostic_audit_frames(
                map_label=audits.MAP_LABEL,
                **frames,
            )

    def test_sampled_claims_cannot_be_upgraded(self) -> None:
        frames = _source_frames()
        frames["moat_rows"].loc[0, "contour_interval_certified"] = True
        frames["moat_rows"].loc[0, "validation_status"] = "interval_certified"
        with self.assertRaisesRegex(
            audits.DiagnosticAuditValidationError,
            "cannot set contour_interval_certified",
        ):
            audits.generate_diagnostic_audit_frames(
                map_label=audits.MAP_LABEL,
                **frames,
            )

        frames = _source_frames()
        frames["schur_rows"]["status"] = "theorem_certified"
        with self.assertRaisesRegex(
            audits.DiagnosticAuditValidationError,
            "sampled diagnostics",
        ):
            audits.generate_diagnostic_audit_frames(
                map_label=audits.MAP_LABEL,
                **frames,
            )


class DiagnosticAuditWriterTests(unittest.TestCase):
    def test_inline_temp_location_resolves_canonical_sources(self) -> None:
        with tempfile.TemporaryDirectory(prefix="diagnostic-audits-inline-") as temporary:
            temporary_source = Path(temporary) / audits.__file__.split("/")[-1]
            temporary_source.write_text("temporary inline copy\n", encoding="utf-8")
            with mock.patch.object(audits, "__file__", str(temporary_source)):
                self.assertEqual(audits._deployment_root(), HERE.parent)
                notebook_record = audits._notebook_source_record()
            self.assertEqual(
                notebook_record["path"],
                "Numerics/blaschke_deformation_certifier.ipynb",
            )
            self.assertEqual(
                set(notebook_record["cells"]), {"Cell 100", "Cell 102"}
            )

    def test_atomic_writes_and_seed_free_report(self) -> None:
        with tempfile.TemporaryDirectory(prefix="diagnostic-audits-test-") as temporary:
            root = Path(temporary)
            real_replace = os.replace
            with mock.patch.object(
                audits.os, "replace", side_effect=real_replace
            ) as replace:
                result = audits.rebuild_diagnostic_audits(
                    map_label=audits.MAP_LABEL,
                    targets=TARGETS_PATH,
                    geometry=GEOMETRY_PATH,
                    schur_rows=SCHUR_PATH,
                    moat_rows=MOAT_PATH,
                    data_dir=root / "data",
                    report_path=root / "reports" / audits.REPORT_FILENAME,
                )

            self.assertEqual(replace.call_count, 3)
            destinations = [Path(call.args[1]).name for call in replace.call_args_list]
            self.assertEqual(
                destinations,
                [
                    audits.UNIVERSAL_AUDIT_FILENAME,
                    audits.FIRST14_AUDIT_FILENAME,
                    audits.REPORT_FILENAME,
                ],
            )
            self.assertEqual(list(root.rglob("*.tmp")), [])
            self.assertEqual(
                result.first14_csv_path.read_bytes(),
                RETAINED_FIRST14_PATH.read_bytes(),
            )

            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertIs(report["legacy_seed_dependency"], False)
            self.assertIs(report["diagnostic_only"], True)
            self.assertIs(report["checks"]["map_lock"], True)
            self.assertIs(report["checks"]["schema_alignment"], True)
            self.assertIs(report["checks"]["target_alignment"], True)
            self.assertIs(report["checks"]["sampled_claims_remain_diagnostic"], True)
            self.assertEqual(
                report["source_extraction"]["notebook"]["cells"]["Cell 100"]["cell_id"],
                "a2b9f84258b4",
            )
            self.assertEqual(
                report["source_extraction"]["notebook"]["cells"]["Cell 102"]["cell_id"],
                "fd3908b317fe",
            )
            for filename, path in (
                (audits.UNIVERSAL_AUDIT_FILENAME, result.universal_csv_path),
                (audits.FIRST14_AUDIT_FILENAME, result.first14_csv_path),
            ):
                self.assertEqual(report["outputs"][filename]["sha256"], _sha256(path))

    def test_retained_audit_cannot_be_an_input_seed(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="diagnostic-audits-seed-test-"
        ) as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                audits.DiagnosticAuditValidationError,
                "cannot use retained audit output",
            ):
                audits.rebuild_diagnostic_audits(
                    map_label=audits.MAP_LABEL,
                    targets=RETAINED_UNIVERSAL_PATH,
                    geometry=GEOMETRY_PATH,
                    schur_rows=SCHUR_PATH,
                    moat_rows=MOAT_PATH,
                    data_dir=root / "data",
                    report_path=root / "reports" / audits.REPORT_FILENAME,
                )
            self.assertFalse((root / "data").exists())
            self.assertFalse((root / "reports").exists())


if __name__ == "__main__":
    unittest.main()
