"""Verify a Blaschke reproducibility archive without extracting its members."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import tarfile
from typing import Mapping
import zipfile
import zlib


BUNDLE_ROOT = "blaschke_deformation_certifier_reproducibility"
ARCHIVE_NAME = f"{BUNDLE_ROOT}.tar.gz"
CHECKSUM_NAME = f"{ARCHIVE_NAME}.sha256"
EXTERNAL_MANIFEST_NAME = f"{BUNDLE_ROOT}_manifest.json"
INTERNAL_MANIFEST_NAME = "reproducibility_manifest.json"
SOURCE_MANIFEST_NAME = "MANIFEST.sha256"
REPLAY_NAME = "REPLAY.md"
CONDA_LOCK_NAME = "conda-explicit-lock.txt"
PIP_LOCK_NAME = "pip-requirements-lock.txt"
SELECTED_VERSIONS_NAME = "selected-package-versions.json"
EFFECTIVE_PLAN_NAME = "effective-reproducibility-plan.json"
SOURCE_ONLY_INVENTORY_NAME = "source-only-replay-inventory.json"
PIP_LOCK_PACKAGES = ("mpmath", "pip", "python-flint", "threadpoolctl")

OUTPUT_RELATIVE = PurePosixPath("Numerics/outputs/blaschke_deformation_certifier")
NOTEBOOK_RELATIVE = PurePosixPath(
    "Numerics/blaschke_deformation_certifier_thesis_math.ipynb"
)
PLAN_RELATIVE = (
    OUTPUT_RELATIVE / "reports/blaschke_deformation_reproducibility_plan.json"
)
SPECTRAL_REPORT_RELATIVE = OUTPUT_RELATIVE / (
    "reports/blaschke_deformation_24_target_N600_M610_spectral_certificate.json"
)
HARDY_MATRIX_INPUT_RELATIVES = {
    "matrix_payload": OUTPUT_RELATIVE
    / "data/blaschke_deformation_balanced_hardy_reference_N600_M610_bits2048.pkl.gz",
    "matrix_midpoint": OUTPUT_RELATIVE
    / (
        "data/blaschke_deformation_balanced_hardy_reference_N600_M610_"
        "bits2048_diagnostic_midpoint.npz"
    ),
    "matrix_report": OUTPUT_RELATIVE
    / (
        "reports/blaschke_deformation_balanced_hardy_reference_N600_M610_"
        "bits2048_certificate.json"
    ),
}
EPSILON_CERTIFICATE_RELATIVE = OUTPUT_RELATIVE / (
    "data/branch_image_balanced_response_prefactor_candidate_row_N600_M610.csv"
)
THEOREM_INPUT_RELATIVES = {
    **HARDY_MATRIX_INPUT_RELATIVES,
    "epsilon_certificate": EPSILON_CERTIFICATE_RELATIVE,
}
TEMPLATE_NOTEBOOK_RELATIVE = PurePosixPath(
    "Numerics/blaschke_deformation_certifier_template.ipynb"
)
SOURCE_NOTEBOOK_RELATIVE = PurePosixPath(
    "Numerics/blaschke_deformation_certifier.ipynb"
)
BUILDER_RELATIVE = PurePosixPath(
    "Numerics/build_blaschke_deformation_thesis_math_notebook.py"
)
CONTOUR_CERTIFICATE_RELATIVE = OUTPUT_RELATIVE / (
    "data/blaschke_deformation_24_target_N600_M610_spectral_certificate.csv"
)
PHASE2_CERTIFIED_ROW_RELATIVE = OUTPUT_RELATIVE / (
    "data/branch_image_phase2_certified_single_space_row_N600_M610.csv"
)
HISTORICAL_REPORT_RELATIVE = OUTPUT_RELATIVE / (
    "reports/historical_phase2_comparison_rebuild.json"
)
HISTORICAL_BASELINE_RELATIVE = OUTPUT_RELATIVE / (
    "data/final_blaschke_N600_schur_certificate.csv"
)
HISTORICAL_EFFECT_RELATIVE = OUTPUT_RELATIVE / (
    "data/branch_image_input_tail_interval_effect_N600.csv"
)
HISTORICAL_PLOT_RELATIVES = tuple(
    OUTPUT_RELATIVE / f"figures/{name}"
    for name in (
        "branch_image_phase2_tail_improvement_ratios.png",
        "branch_image_phase2_certificate_components.png",
        "branch_image_phase2_certificate_array_1x2.png",
    )
)
HISTORICAL_PHASE4_REPORT_RELATIVE = OUTPUT_RELATIVE / (
    "reports/blaschke_deformation_historical_phase4_rebuild_N600_M610.json"
)
HISTORICAL_PHASE4_DATA_RELATIVES = {
    "scaled_block": OUTPUT_RELATIVE
    / "data/branch_image_wide_candidate_B_scaled_mp_N600_M610_r2p225974769705636.npz",
    "hardy_matrix": OUTPUT_RELATIVE
    / "data/branch_image_wide_candidate_A_X_from_mp_scaled_N600_M610_r2p225974769705636.npz",
    "eigenvalues": OUTPUT_RELATIVE
    / "data/phase4_hp_hardy_gauge_eigenvalues_N600_M610.csv",
    "packets": OUTPUT_RELATIVE
    / "data/branch_image_wide_candidate_first15_sampled_spectral_packets_N600_M610.csv",
    "moats": OUTPUT_RELATIVE
    / "data/branch_image_wide_candidate_first15_contour_moats_N600_M610_J128.csv",
    "profile_summary": OUTPUT_RELATIVE
    / "data/branch_image_wide_candidate_first15_contour_profile_summary_N600_M610_J128.csv",
    "profiles": OUTPUT_RELATIVE
    / "data/branch_image_wide_candidate_first15_contour_profiles_N600_M610_J128.csv",
    "plot_table": OUTPUT_RELATIVE
    / "data/branch_image_wide_candidate_first15_contour_moat_plot_table_N600_M610_J128.csv",
    "minima": OUTPUT_RELATIVE
    / "data/branch_image_wide_candidate_first15_contour_profile_minima_table_N600_M610_J128.csv",
    "fragile": OUTPUT_RELATIVE
    / "data/branch_image_wide_candidate_first15_fragile_robustness_N600_M610.csv",
    "final_packets": OUTPUT_RELATIVE
    / "data/final_blaschke_branch-image_first15_sampled_spectral_packets_N600_M610.csv",
    "final_moats": OUTPUT_RELATIVE
    / "data/final_blaschke_branch-image_first15_contour_moats_N600_M610.csv",
    "final_fragile": OUTPUT_RELATIVE
    / "data/final_blaschke_branch-image_first15_fragile_robustness_N600_M610.csv",
    "compatibility_summary": OUTPUT_RELATIVE
    / "data/final_blaschke_branch-image_first15_phase4_summary.json",
    "local_surface": OUTPUT_RELATIVE
    / "data/branch_image_wide_candidate_mu2_hardy_moat_surface_N600_M610_grid25.npz",
    "local_surface_csv": OUTPUT_RELATIVE
    / "data/branch_image_wide_candidate_mu2_hardy_moat_surface_N600_M610.csv",
    "global_surface": OUTPUT_RELATIVE
    / "data/branch_image_wide_candidate_first15_global_hardy_moat_surface_N600_M610_grid45.npz",
    "global_surface_csv": OUTPUT_RELATIVE
    / "data/branch_image_wide_candidate_first15_global_hardy_moat_surface_N600_M610.csv",
    "zoom_surface": OUTPUT_RELATIVE
    / "data/branch_image_wide_candidate_first15_zoom_hardy_moat_surface_N600_M610_grid161.npz",
    "deep_zoom_surface": OUTPUT_RELATIVE
    / "data/branch_image_wide_candidate_first15_deep_zoom_hardy_moat_surface_N600_M610_grid201.npz",
}
HISTORICAL_PHASE4_CSV_ROW_COUNTS = {
    "eigenvalues": 600,
    "packets": 15,
    "moats": 15,
    "profile_summary": 15,
    "profiles": 15 * 128,
    "plot_table": 15,
    "minima": 15,
    "fragile": 4,
    "final_packets": 15,
    "final_moats": 15,
    "final_fragile": 4,
    "local_surface_csv": 25 * 25,
    "global_surface_csv": 45 * 45,
}
HISTORICAL_PHASE4_NPZ_MEMBER_NAMES = {
    "scaled_block": frozenset(
        {
            "B_scaled.npy",
            "B_scaled_sha256.npy",
            "producer_schema.npy",
            "configuration_digest.npy",
            "source_digest.npy",
            "cache_key.npy",
            "diagnostic_status.npy",
        }
    ),
    "hardy_matrix": frozenset(
        {
            "A_X.npy",
            "B_scaled.npy",
            "T.npy",
            "eigenvalues.npy",
            "B_scaled_sha256.npy",
            "A_X_sha256.npy",
            "T_sha256.npy",
            "eigenvalues_sha256.npy",
            "producer_schema.npy",
            "configuration_digest.npy",
            "source_digest.npy",
            "cache_key.npy",
            "diagnostic_status.npy",
        }
    ),
}
_HISTORICAL_PHASE4_SURFACE_NPZ_MEMBERS = frozenset(
    {
        "x.npy",
        "y.npy",
        "s_min.npy",
        "row_block_dir.npy",
        "matrix_path.npy",
        "A_X_sha256.npy",
        "surface_digest.npy",
        "surface_tag.npy",
        "s_min_sha256.npy",
        "producer_schema.npy",
        "configuration_digest.npy",
        "source_digest.npy",
        "cache_key.npy",
        "diagnostic_status.npy",
    }
)
for _surface_key in (
    "local_surface",
    "global_surface",
    "zoom_surface",
    "deep_zoom_surface",
):
    HISTORICAL_PHASE4_NPZ_MEMBER_NAMES[_surface_key] = (
        _HISTORICAL_PHASE4_SURFACE_NPZ_MEMBERS
    )
HISTORICAL_PHASE4_VOLATILE_NPZ_MEMBERS = {
    key: frozenset({"row_block_dir.npy", "matrix_path.npy"})
    for key in (
        "local_surface",
        "global_surface",
        "zoom_surface",
        "deep_zoom_surface",
    )
}
HISTORICAL_PHASE4_PLOT_RELATIVES = tuple(
    OUTPUT_RELATIVE / f"figures/{name}"
    for name in (
        "branch_image_wide_candidate_first15_sampled_validation_1x2.png",
        "branch_image_wide_candidate_first15_small_gain_horizontal.png",
        "branch_image_wide_candidate_first15_contour_moat_heatmap.png",
        "branch_image_wide_candidate_first15_contour_moat_profiles_lines.png",
        "branch_image_wide_candidate_first15_contour_profile_minima_table.png",
        "branch_image_wide_candidate_first15_moat_minima_profile.png",
        "branch_image_wide_candidate_sampled_hardy_moat_profile_alpha11_N600_M610.png",
        "branch_image_wide_candidate_first15_fragile_robustness.png",
        "branch_image_wide_candidate_mu2_hardy_moat_surface_3d_2d_N600_M610.png",
        "branch_image_wide_candidate_first15_global_hardy_moat_surface_3d_2d_N600_M610.png",
        "branch_image_wide_candidate_first15_global_hardy_moat_zoom_symlog_N600_M610.png",
        "branch_image_wide_candidate_first15_interactive_dashboard_snapshot_N600_M610.png",
    )
)
DIAGNOSTIC_AUDIT_REPORT_RELATIVE = OUTPUT_RELATIVE / (
    "reports/blaschke_deformation_diagnostic_audits_rebuild.json"
)
DIAGNOSTIC_AUDIT_DATA_RELATIVES = {
    "universal_audit": OUTPUT_RELATIVE
    / "data/transfer_lab_blaschke_mu_0p3_universal_certification_audit.csv",
    "first14_audit": OUTPUT_RELATIVE
    / "data/transfer_lab_blaschke_mu_0p3_first14_packet_audit.csv",
}
DIAGNOSTIC_AUDIT_PLOT_RELATIVES = tuple(
    OUTPUT_RELATIVE / f"figures/{name}"
    for name in (
        "transfer_lab_blaschke_mu_0p3_Cell_100_universal_certification_ladder.png",
        "transfer_lab_blaschke_mu_0p3_Cell_102_first14_packet_audit.png",
    )
)

EXPECTED_MAP_LABEL = "blaschke_mu_0p3"
EXPECTED_TARGET_COUNT = 24
EXPECTED_MULTIPLICITY = 30
EXPECTED_SCHUR_MOAT_COUNT = 17
EXPECTED_LAURENT_MOAT_COUNT = 7
EXPECTED_NOTEBOOK_CELLS = 140
EXPECTED_NOTEBOOK_CODE_CELLS = 68
EXPECTED_REPORT_STATUSES = frozenset({
    "theorem-certified twenty-four-target Riesz-rank package",
    (
        "theorem-certified twenty-four-target Riesz-rank package; "
        "finite moats reused and small-gain products reaggregated"
    ),
})
EXPECTED_HISTORICAL_SCHEMA = "phase2-historical-comparisons-v1"
EXPECTED_HISTORICAL_CONFIGURATION = {
    "N": 600,
    "M": 610,
    "mu": "0.3",
    "rho": "1.5999",
    "q_gap": "0.967",
    "baseline_cells": 32768,
    "branch_image_cells": 65536,
    "geometry_precision_bits": 192,
    "transport_precision_bits": 384,
    "zwx13_maximum_extra": 800,
    "workers": 24,
}
EXPECTED_HISTORICAL_PHASE4_SCHEMA = "historical-wide-phase4-source-rebuild-v2"
EXPECTED_HISTORICAL_PHASE4_STATUS = "diagnostic_only_not_theorem_gate"
EXPECTED_HISTORICAL_PHASE4_CONFIGURATION = {
    "mode": "production",
    "N": 600,
    "M": 610,
    "mu": "0.3",
    "rho": "2.45",
    "r": "2.225974769705636",
    "epsilon_x": "1.1641169413997097e-17",
    "dps": 200,
    "target_count": 15,
    "maximum_target_power": 28,
    "contour_samples": 128,
    "fragile_samples": [128, 256],
    "local_surface_grid": 25,
    "global_surface_grid": 45,
    "zoom_surface_grid": 161,
    "deep_zoom_surface_grid": 201,
    "surface_row_block_size": 2,
    "surface_log_floor": "1e-18",
    "assembly_row_block_size": 32,
}
EXPECTED_DIAGNOSTIC_AUDIT_SCHEMA = "blaschke-deformation-diagnostic-audits-v1"
EXPECTED_DIAGNOSTIC_AUDIT_STATUS = (
    "retained sampled comparison audits; not inputs to theorem gates"
)
EXPECTED_DIAGNOSTIC_TARGET_NAMES = (
    "alpha^1",
    "alpha^2",
    "mu^1",
    "alpha^3",
    "alpha^4",
    "alpha^5",
    "mu^2",
    "alpha^6",
    "alpha^7",
    "alpha^8",
    "mu^3",
    "alpha^9",
    "alpha^10",
    "alpha^11",
)
EXPECTED_HISTORICAL_PHASE4_TARGET_NAMES = (
    *EXPECTED_DIAGNOSTIC_TARGET_NAMES,
    "mu^4",
)
EXPECTED_DIAGNOSTIC_SCHUR_N = (30, 40, 50, 60, 80, 100)
EXPECTED_UNIVERSAL_AUDIT_FIELDS = (
    "map_label",
    "audit_item",
    "status",
    "evidence",
)
EXPECTED_FIRST14_AUDIT_FIELDS = (
    "name",
    "family_packet",
    "power",
    "multiplicity_packet",
    "packet_centre",
    "target_mode",
    "rank",
    "target",
    "centre",
    "multiplicity_moat",
    "radius",
    "finite_eigenvalue_count",
    "cluster_radius",
    "nearest_outside_distance",
    "radius_status",
    "s_gamma_H",
    "theta_min",
    "d_Gamma0",
    "m_gamma_X",
    "epsilon_X",
    "epsilon_m_gamma",
    "pass_sampled_small_gain",
    "selected_count",
    "selected_eigenvalues",
    "max_numeric_error",
    "pass_sampled_packet",
    "family_moat",
    "count_ok",
    "sampled_validation_pass",
    "contour_interval_certified",
    "validation_status",
    "small_gain_diagnostic",
    "target_identity_status",
    "finite_section_status",
    "contour_role",
    "finite_count_certified",
    "finite_count_matches",
    "sampled_small_gain_pass",
    "small_gain_pass",
    "certified_small_gain_pass",
    "riesz_rank_status",
)
CONTOUR_TIMING_FIELDS = frozenset({"inverse_sampling_seconds", "elapsed_seconds"})
NON_STABLE_THEOREM_REPORT_FIELDS = frozenset(
    {
        "artifacts",
        "binary64_candidate_runtime",
        "elapsed_seconds",
        "geometry_input_hashes",
        "input_hashes",
    }
)
HISTORICAL_STABLE_FIELDS = (
    "map_label",
    "producer_schema",
    "configuration",
    "diagnostic_status",
    "historical_design_inputs",
    "complete_boundary_cover",
    "aggregation",
    "current_theorem_aggregation",
    "legacy_seed_dependency",
    "zwx13_terms_used",
    "baseline_record",
    "effect_record",
    "baseline_geometry_configuration_digest",
    "branch_geometry_configuration_digest",
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
NPY_SIGNATURE = b"\x93NUMPY"
MAX_NPZ_UNCOMPRESSED_BYTES = 512 * 1024 * 1024

TRUE_THEOREM_GATES = (
    "all_24_targets_theorem_certified",
    "all_schur_diagonal_memberships_certified",
    "all_finite_count_transports_certified",
    "all_finite_counts_certified",
    "all_finite_counts_match_expected",
    "all_finite_counts_schur_derived",
    "all_finite_to_exact_rank_transfers_certified",
    "all_small_gain_tests_pass",
    "all_complete_circles_covered",
    "all_complete_circle_moats_positive",
    "zero_excluded_from_every_contour",
)
FALSE_THEOREM_GATES = (
    "sampled_values_used_in_any_theorem_gate",
    "laurent_digests_used_in_any_theorem_gate",
)
PLAN_TRUE_GATES = (
    "contour_all_finite_counts_schur_derived",
    "contour_all_schur_diagonal_memberships_certified",
    "contour_all_finite_count_transports_certified",
    "contour_all_finite_counts_match_expected",
    "contour_all_finite_to_exact_rank_transfers_certified",
)
GENERATED_FILES = frozenset(
    {
        REPLAY_NAME,
        CONDA_LOCK_NAME,
        SELECTED_VERSIONS_NAME,
        EFFECTIVE_PLAN_NAME,
        SOURCE_ONLY_INVENTORY_NAME,
    }
)
MANIFEST_LINE_RE = re.compile(r"([0-9a-f]{64})  (.+)")
CHECKSUM_RE = re.compile(r"([0-9a-f]{64})  (.+)\n")


class VerificationError(RuntimeError):
    """Raised when an archive verification invariant does not hold."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"Invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"{label} must be a JSON object.")
    return value


def _exact_pip_pins(payload: bytes) -> dict[str, str]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise VerificationError(f"{PIP_LOCK_NAME} is not UTF-8.") from exc
    pins: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
        if match is None:
            raise VerificationError(
                f"A pip requirement is not exactly pinned: {line!r}."
            )
        name = match.group(1).lower().replace("_", "-")
        if name in pins:
            raise VerificationError(f"Duplicate pip requirement for {name}.")
        pins[name] = match.group(2)
    if set(pins) != set(PIP_LOCK_PACKAGES):
        raise VerificationError(
            "The pip requirements do not contain the required exact package set."
        )
    return dict(sorted(pins.items()))


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        return _json_object(path.read_bytes(), label=label)
    except OSError as exc:
        raise VerificationError(f"Cannot read {label} {path}: {exc}") from exc


def _safe_relative_path(value: str, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise VerificationError(f"{label} must be a non-empty string.")
    if "\\" in value or any(character in value for character in "\x00\n\r"):
        raise VerificationError(f"Unsafe {label}: {value!r}.")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise VerificationError(f"Unsafe {label}: {value!r}.")
    relative = PurePosixPath(value)
    if relative.is_absolute() or relative.as_posix() != value:
        raise VerificationError(f"Unsafe {label}: {value!r}.")
    return relative


def _expect_exact_int(mapping: Mapping[str, object], key: str, expected: int) -> None:
    observed = mapping.get(key)
    if type(observed) is not int or observed != expected:
        raise VerificationError(f"Expected {key}={expected}, observed {observed!r}.")


def _expect_bool(mapping: Mapping[str, object], key: str, expected: bool) -> None:
    observed = mapping.get(key)
    if type(observed) is not bool or observed is not expected:
        raise VerificationError(f"Expected {key}={expected}, observed {observed!r}.")


def _validate_plan_and_report(
    plan: Mapping[str, object], report: Mapping[str, object]
) -> dict[str, object]:
    if set(plan) != {"precision_settings", "upstream_artifact_names"}:
        raise VerificationError("Unexpected effective-plan top-level schema.")
    precision = plan.get("precision_settings")
    artifact_names = plan.get("upstream_artifact_names")
    if not isinstance(precision, dict) or not isinstance(artifact_names, list):
        raise VerificationError("Invalid effective-plan field types.")
    if not artifact_names:
        raise VerificationError("The effective plan has no upstream artifacts.")

    if precision.get("map_label") != EXPECTED_MAP_LABEL:
        raise VerificationError("Unexpected plan map_label.")
    if report.get("map_label") != EXPECTED_MAP_LABEL:
        raise VerificationError("Unexpected report map_label.")
    _expect_exact_int(precision, "N", 600)
    _expect_exact_int(precision, "M", 610)
    _expect_exact_int(report, "N", 600)
    _expect_exact_int(report, "M", 610)
    for key in ("rho", "r"):
        if precision.get(key) != report.get(key):
            raise VerificationError(f"Plan/report mismatch for {key}.")

    schema = precision.get("contour_certificate_schema")
    if not isinstance(schema, str) or not schema:
        raise VerificationError("Missing effective contour certificate schema.")
    if report.get("certificate_schema") != schema:
        raise VerificationError("Plan/report certificate schema mismatch.")

    _expect_exact_int(precision, "contour_target_count", EXPECTED_TARGET_COUNT)
    _expect_exact_int(precision, "contour_total_multiplicity", EXPECTED_MULTIPLICITY)
    _expect_exact_int(
        precision, "contour_schur_count_target_count", EXPECTED_TARGET_COUNT
    )
    _expect_exact_int(
        precision,
        "contour_schur_triangular_moat_target_count",
        EXPECTED_SCHUR_MOAT_COUNT,
    )
    _expect_exact_int(
        precision,
        "contour_laurent_moat_target_count",
        EXPECTED_LAURENT_MOAT_COUNT,
    )
    for key in PLAN_TRUE_GATES:
        _expect_bool(precision, key, True)

    _expect_exact_int(report, "target_count", EXPECTED_TARGET_COUNT)
    _expect_exact_int(report, "schur_count_target_count", EXPECTED_TARGET_COUNT)
    _expect_exact_int(
        report, "schur_triangular_moat_target_count", EXPECTED_SCHUR_MOAT_COUNT
    )
    _expect_exact_int(report, "laurent_moat_target_count", EXPECTED_LAURENT_MOAT_COUNT)
    _expect_exact_int(
        report,
        "total_expected_algebraic_multiplicity",
        EXPECTED_MULTIPLICITY,
    )
    _expect_exact_int(
        report,
        "total_certified_algebraic_multiplicity",
        EXPECTED_MULTIPLICITY,
    )
    if report.get("status") not in EXPECTED_REPORT_STATUSES:
        raise VerificationError("Unexpected theorem-facing report status.")
    for key in TRUE_THEOREM_GATES:
        _expect_bool(report, key, True)
    for key in FALSE_THEOREM_GATES:
        _expect_bool(report, key, False)

    for plan_key, report_key in (
        ("contour_arb_bits", "contour_precision_bits"),
        ("contour_eta_schur_upper", "eta_schur_upper_text"),
        ("hardy_matrix_eta_A_upper", "eta_A_upper_text"),
        ("hardy_matrix_midpoint_sha256", "matrix_midpoint_sha256"),
    ):
        if precision.get(plan_key) != report.get(report_key):
            raise VerificationError(
                f"Plan/report mismatch for {plan_key}/{report_key}."
            )
    return {
        "certificate_schema": schema,
        "target_count": EXPECTED_TARGET_COUNT,
        "total_algebraic_multiplicity": EXPECTED_MULTIPLICITY,
    }


def _refresh_plan(
    source_plan: Mapping[str, object], report: Mapping[str, object]
) -> dict[str, object]:
    if set(source_plan) != {"precision_settings", "upstream_artifact_names"}:
        raise VerificationError("Unexpected source-plan top-level schema.")
    source_precision = source_plan.get("precision_settings")
    source_artifacts = source_plan.get("upstream_artifact_names")
    if not isinstance(source_precision, dict) or not isinstance(source_artifacts, list):
        raise VerificationError("Invalid source-plan field types.")
    precision = dict(source_precision)
    precision.update(
        {
            "map_label": report.get("map_label"),
            "N": report.get("N"),
            "M": report.get("M"),
            "rho": report.get("rho"),
            "r": report.get("r"),
            "hardy_matrix_eta_A_upper": report.get("eta_A_upper_text"),
            "hardy_matrix_midpoint_sha256": report.get("matrix_midpoint_sha256"),
            "contour_certificate_schema": report.get("certificate_schema"),
            "contour_schur_count_target_count": report.get("schur_count_target_count"),
            "contour_schur_triangular_moat_target_count": report.get(
                "schur_triangular_moat_target_count"
            ),
            "contour_laurent_moat_target_count": report.get(
                "laurent_moat_target_count"
            ),
            "contour_all_finite_counts_schur_derived": report.get(
                "all_finite_counts_schur_derived"
            ),
            "contour_all_schur_diagonal_memberships_certified": report.get(
                "all_schur_diagonal_memberships_certified"
            ),
            "contour_all_finite_count_transports_certified": report.get(
                "all_finite_count_transports_certified"
            ),
            "contour_all_finite_counts_match_expected": report.get(
                "all_finite_counts_match_expected"
            ),
            "contour_all_finite_to_exact_rank_transfers_certified": report.get(
                "all_finite_to_exact_rank_transfers_certified"
            ),
            "contour_arb_bits": report.get("contour_precision_bits"),
            "contour_eta_schur_upper": report.get("eta_schur_upper_text"),
            "contour_target_count": report.get("target_count"),
            "contour_total_multiplicity": report.get(
                "total_certified_algebraic_multiplicity"
            ),
        }
    )
    effective = {
        "precision_settings": precision,
        "upstream_artifact_names": list(source_artifacts),
    }
    _validate_plan_and_report(effective, report)
    return effective


def _validate_notebook(notebook: Mapping[str, object]) -> dict[str, object]:
    cells = notebook.get("cells")
    if not isinstance(cells, list) or len(cells) != EXPECTED_NOTEBOOK_CELLS:
        raise VerificationError(
            f"Executed notebook does not have {EXPECTED_NOTEBOOK_CELLS} cells."
        )
    code_cells = [
        cell
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "code"
    ]
    if len(code_cells) != EXPECTED_NOTEBOOK_CODE_CELLS:
        raise VerificationError(
            "Executed notebook does not have exactly "
            f"{EXPECTED_NOTEBOOK_CODE_CELLS} code cells."
        )
    counts: list[int] = []
    errors = 0
    for cell in code_cells:
        count = cell.get("execution_count")
        if type(count) is not int:
            raise VerificationError("The archive contains an unexecuted code cell.")
        counts.append(count)
        outputs = cell.get("outputs")
        if not isinstance(outputs, list):
            raise VerificationError("A code cell has a non-array outputs field.")
        errors += sum(
            isinstance(output, dict) and output.get("output_type") == "error"
            for output in outputs
        )
    if counts != list(range(1, EXPECTED_NOTEBOOK_CODE_CELLS + 1)):
        raise VerificationError("Notebook execution counts are not contiguous 1-63.")
    if errors:
        raise VerificationError("The executed notebook contains error outputs.")
    return {
        "cell_count": len(cells),
        "code_cell_count": len(code_cells),
        "executed_code_cell_count": len(code_cells),
        "stored_error_output_count": errors,
    }


def _parse_source_manifest(payload: bytes) -> dict[str, str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationError("The source manifest is not UTF-8.") from exc
    if not text or not text.endswith("\n"):
        raise VerificationError("The source manifest is not newline-terminated.")
    entries: dict[str, str] = {}
    paths: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = MANIFEST_LINE_RE.fullmatch(line)
        if match is None:
            raise VerificationError(
                f"Malformed source manifest line {line_number}: {line!r}."
            )
        digest, path_text = match.groups()
        relative = _safe_relative_path(path_text, label="source manifest path")
        label = relative.as_posix()
        if label == SOURCE_MANIFEST_NAME or label in entries:
            raise VerificationError(f"Invalid source manifest path: {label}.")
        entries[label] = digest
        paths.append(label)
    if paths != sorted(paths):
        raise VerificationError("Source manifest paths are not sorted.")
    return entries


def _validate_gzip_header(archive_path: Path) -> None:
    try:
        with archive_path.open("rb") as stream:
            header = stream.read(10)
    except OSError as exc:
        raise VerificationError(f"Cannot read archive header: {exc}") from exc
    if len(header) != 10 or header[:2] != b"\x1f\x8b":
        raise VerificationError("The archive is not a gzip stream.")
    if int.from_bytes(header[4:8], "little") != 0:
        raise VerificationError("The gzip header mtime is not deterministic zero.")
    if header[3] & 0x08:
        raise VerificationError("The gzip header contains a source filename.")


def _read_safe_archive(
    archive_path: Path, *, commit_epoch: int
) -> tuple[dict[str, bytes], int]:
    _validate_gzip_header(archive_path)
    try:
        archive = tarfile.open(archive_path, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise VerificationError(f"Cannot open archive: {exc}") from exc
    with archive:
        members = archive.getmembers()
        names: list[str] = []
        files: dict[str, bytes] = {}
        directories: set[str] = set()
        for member in members:
            name = member.name
            if name in names:
                raise VerificationError(f"Duplicate archive member: {name}.")
            names.append(name)
            if name == BUNDLE_ROOT:
                relative = None
            else:
                prefix = f"{BUNDLE_ROOT}/"
                if not name.startswith(prefix):
                    raise VerificationError(
                        f"Archive member escapes bundle root: {name}."
                    )
                relative = _safe_relative_path(
                    name[len(prefix) :], label="archive member path"
                )
            if member.uid != 0 or member.gid != 0:
                raise VerificationError(f"Non-deterministic owner metadata: {name}.")
            if member.uname or member.gname:
                raise VerificationError(f"Non-empty owner names: {name}.")
            if member.mtime != commit_epoch:
                raise VerificationError(f"Non-deterministic mtime: {name}.")
            if member.isdir():
                if member.mode & 0o777 != 0o755:
                    raise VerificationError(f"Unexpected directory mode: {name}.")
                directories.add("" if relative is None else relative.as_posix())
                continue
            if not member.isreg() or relative is None:
                raise VerificationError(f"Unsafe archive member type: {name}.")
            if member.mode & 0o777 not in {0o644, 0o755}:
                raise VerificationError(f"Unexpected regular-file mode: {name}.")
            stream = archive.extractfile(member)
            if stream is None:
                raise VerificationError(f"Cannot read archive member: {name}.")
            payload = stream.read()
            if len(payload) != member.size:
                raise VerificationError(f"Truncated archive member: {name}.")
            files[relative.as_posix()] = payload

    expected_directories = {""}
    for file_name in files:
        parent = PurePosixPath(file_name).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if directories != expected_directories:
        raise VerificationError("Archive directory closure is incomplete or excessive.")
    expected_order = [BUNDLE_ROOT] + [
        f"{BUNDLE_ROOT}/{name}"
        for name in sorted(set(files) | (expected_directories - {""}))
    ]
    if names != expected_order:
        raise VerificationError("Archive members are not in deterministic order.")
    return files, len(members)


def _regular_repo_file(root: Path, relative: PurePosixPath) -> Path:
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise VerificationError(
            f"Replay path is not a regular file: {relative.as_posix()}."
        )
    try:
        candidate.resolve(strict=True).relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise VerificationError(
            f"Replay path escapes replay root: {relative.as_posix()}."
        ) from exc
    return candidate


def _compare_replay_root(
    root: Path, source_manifest_payload: bytes, source_entries: Mapping[str, str]
) -> None:
    replay_root = root.resolve()
    if not replay_root.is_dir():
        raise VerificationError(f"Replay root is not a directory: {replay_root}.")
    observed_manifest = _regular_repo_file(
        replay_root, PurePosixPath(SOURCE_MANIFEST_NAME)
    ).read_bytes()
    if observed_manifest != source_manifest_payload:
        raise VerificationError("Replay-root MANIFEST.sha256 differs from the archive.")
    for path_text, expected_hash in source_entries.items():
        relative = _safe_relative_path(path_text, label="replay comparison path")
        observed_hash = _sha256_file(_regular_repo_file(replay_root, relative))
        if observed_hash != expected_hash:
            raise VerificationError(
                f"Replay-root hash mismatch for {relative.as_posix()}."
            )


def _archive_payload(
    files: Mapping[str, bytes], relative: PurePosixPath, *, label: str
) -> bytes:
    payload = files.get(relative.as_posix())
    if payload is None:
        raise VerificationError(
            f"Archive baseline lacks {label}: {relative.as_posix()}."
        )
    return payload


def _read_replay_bytes(
    replay_root: Path, relative: PurePosixPath, *, label: str
) -> bytes:
    path = _regular_repo_file(replay_root, relative)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise VerificationError(
            f"Cannot read replay {label} {relative.as_posix()}: {exc}"
        ) from exc


def _read_replay_json(
    replay_root: Path, relative: PurePosixPath, *, label: str
) -> dict[str, object]:
    return _json_object(
        _read_replay_bytes(replay_root, relative, label=label),
        label=f"replay {label}",
    )


def _validate_replay_inline_helper_sync(
    replay_root: Path, notebook: Mapping[str, object]
) -> None:
    builder_path = _regular_repo_file(replay_root, BUILDER_RELATIVE)
    module_name = "_blaschke_replay_builder_" + _sha256_file(builder_path)[:16]
    spec = importlib.util.spec_from_file_location(module_name, builder_path)
    if spec is None or spec.loader is None:
        raise VerificationError(
            f"Cannot load replay inline-helper validator from {builder_path}."
        )
    builder = importlib.util.module_from_spec(spec)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    missing_module = object()
    previous_module = sys.modules.get(module_name, missing_module)
    try:
        sys.dont_write_bytecode = True
        sys.modules[module_name] = builder
        spec.loader.exec_module(builder)
        helper_ordinals = getattr(builder, "CURATED_INLINE_HELPER_ORDINALS")
        validator = getattr(builder, "validate_inline_helper_sync")
        if not callable(validator):
            raise TypeError("validate_inline_helper_sync is not callable")
        validator(
            notebook,
            helper_ordinals=helper_ordinals,
            allow_notebook_provenance=True,
        )
    except Exception as exc:
        raise VerificationError(
            f"Replay inline-helper validation failed: {exc}"
        ) from exc
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        if previous_module is missing_module:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module


def _csv_table(
    payload: bytes, *, label: str
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationError(f"{label} is not UTF-8.") from exc
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        raw_fields = reader.fieldnames
        if raw_fields is None:
            raise VerificationError(f"{label} has no CSV header.")
        fields = tuple(raw_fields)
        if (
            not fields
            or any(not field for field in fields)
            or len(set(fields)) != len(fields)
        ):
            raise VerificationError(f"{label} has an invalid CSV header.")
        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=1):
            if None in row or any(value is None for value in row.values()):
                raise VerificationError(
                    f"{label} row {row_number} does not match its header."
                )
            rows.append(dict(row))
    except csv.Error as exc:
        raise VerificationError(f"Invalid {label}: {exc}") from exc
    return fields, rows


def _compare_contour_certificate_csv(
    archive_payload: bytes, replay_payload: bytes
) -> int:
    archive_fields, archive_rows = _csv_table(
        archive_payload, label="archive contour certificate CSV"
    )
    replay_fields, replay_rows = _csv_table(
        replay_payload, label="replay contour certificate CSV"
    )
    if archive_fields != replay_fields:
        raise VerificationError("Replay contour certificate CSV schema differs.")
    missing_timing = CONTOUR_TIMING_FIELDS - set(archive_fields)
    if missing_timing:
        raise VerificationError(
            "Contour certificate CSV lacks timing fields: "
            + ", ".join(sorted(missing_timing))
            + "."
        )
    if (
        len(archive_rows) != EXPECTED_TARGET_COUNT
        or len(replay_rows) != EXPECTED_TARGET_COUNT
    ):
        raise VerificationError(
            "Archive and replay contour certificate CSVs must each contain "
            f"{EXPECTED_TARGET_COUNT} rows."
        )
    stable_fields = tuple(
        field for field in archive_fields if field not in CONTOUR_TIMING_FIELDS
    )
    archive_stable = [
        tuple(row[field] for field in stable_fields) for row in archive_rows
    ]
    replay_stable = [
        tuple(row[field] for field in stable_fields) for row in replay_rows
    ]
    if archive_stable != replay_stable:
        raise VerificationError(
            "Replay contour certificate CSV stable fields differ from the archive."
        )
    return len(replay_rows)


def _compare_exact_single_row_csv(
    archive_payload: bytes,
    replay_payload: bytes,
    *,
    label: str,
) -> tuple[tuple[str, ...], dict[str, str]]:
    if replay_payload != archive_payload:
        raise VerificationError(f"Replay {label} differs byte-for-byte.")
    fields, rows = _csv_table(archive_payload, label=f"archive {label}")
    if len(rows) != 1:
        raise VerificationError(f"{label} must contain exactly one data row.")
    return fields, rows[0]


def _csv_scalar_text(value: object, *, label: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, (str, int, float, bool)):
        raise VerificationError(f"{label} contains a non-scalar CSV value.")
    return str(value)


def _validate_report_record_matches_csv(
    record: Mapping[str, object],
    fields: tuple[str, ...],
    row: Mapping[str, str],
    *,
    label: str,
) -> None:
    if set(record) != set(fields):
        raise VerificationError(f"{label} record/CSV schema mismatch.")
    for field in fields:
        expected = _csv_scalar_text(record[field], label=f"{label}.{field}")
        if row.get(field) != expected:
            raise VerificationError(f"{label} record/CSV value mismatch for {field}.")


def _validate_historical_report(
    report: Mapping[str, object],
    *,
    baseline_csv_payload: bytes,
    effect_csv_payload: bytes,
    label: str,
) -> dict[str, object]:
    if report.get("map_label") != EXPECTED_MAP_LABEL:
        raise VerificationError(f"Unexpected {label} map_label.")
    if report.get("producer_schema") != EXPECTED_HISTORICAL_SCHEMA:
        raise VerificationError(f"Unexpected {label} producer schema.")
    if report.get("configuration") != EXPECTED_HISTORICAL_CONFIGURATION:
        raise VerificationError(f"Unexpected {label} configuration.")
    _expect_bool(report, "legacy_seed_dependency", False)
    _expect_exact_int(report, "zwx13_terms_used", 801)

    expected_design = {
        "rho": EXPECTED_HISTORICAL_CONFIGURATION["rho"],
        "q_gap": EXPECTED_HISTORICAL_CONFIGURATION["q_gap"],
        "baseline_cells": EXPECTED_HISTORICAL_CONFIGURATION["baseline_cells"],
        "branch_image_cells": EXPECTED_HISTORICAL_CONFIGURATION["branch_image_cells"],
    }
    if report.get("historical_design_inputs") != expected_design:
        raise VerificationError(f"Unexpected {label} historical design inputs.")

    cover = report.get("complete_boundary_cover")
    expected_cover_keys = {
        "uses_arb_pi",
        "covers_zero_to_two_pi",
        "baseline_cells",
        "branch_image_cells",
        "branch_image_maximum_cell",
    }
    if not isinstance(cover, dict) or set(cover) != expected_cover_keys:
        raise VerificationError(f"Unexpected {label} boundary-cover schema.")
    _expect_bool(cover, "uses_arb_pi", True)
    _expect_bool(cover, "covers_zero_to_two_pi", True)
    _expect_exact_int(
        cover,
        "baseline_cells",
        int(EXPECTED_HISTORICAL_CONFIGURATION["baseline_cells"]),
    )
    branch_cells = int(EXPECTED_HISTORICAL_CONFIGURATION["branch_image_cells"])
    _expect_exact_int(cover, "branch_image_cells", branch_cells)
    maximum_cell = cover.get("branch_image_maximum_cell")
    if type(maximum_cell) is not int or not 0 <= maximum_cell < branch_cells:
        raise VerificationError(f"Invalid {label} branch-image maximum cell.")

    baseline_record = report.get("baseline_record")
    effect_record = report.get("effect_record")
    if not isinstance(baseline_record, dict) or not isinstance(effect_record, dict):
        raise VerificationError(f"Missing {label} baseline/effect records.")
    _expect_exact_int(baseline_record, "N", 600)
    _expect_exact_int(baseline_record, "M", 610)
    for key in (
        "branch_data_certified",
        "output_tail_certified",
        "input_tail_certified",
        "transport_certified",
        "schur_matrix_certified",
        "total_certified",
    ):
        _expect_bool(baseline_record, key, True)
    _expect_exact_int(effect_record, "N", 600)
    _expect_exact_int(effect_record, "M", 610)
    _expect_exact_int(effect_record, "cells", branch_cells)

    baseline_fields, baseline_rows = _csv_table(
        baseline_csv_payload, label=f"{label} baseline CSV"
    )
    effect_fields, effect_rows = _csv_table(
        effect_csv_payload, label=f"{label} effect CSV"
    )
    if len(baseline_rows) != 1 or len(effect_rows) != 1:
        raise VerificationError(
            f"{label} historical CSVs must each contain exactly one data row."
        )
    _validate_report_record_matches_csv(
        baseline_record,
        baseline_fields,
        baseline_rows[0],
        label=f"{label} baseline",
    )
    _validate_report_record_matches_csv(
        effect_record,
        effect_fields,
        effect_rows[0],
        label=f"{label} effect",
    )

    outputs = report.get("outputs")
    expected_output_hashes = {
        HISTORICAL_BASELINE_RELATIVE.name: _sha256_bytes(baseline_csv_payload),
        HISTORICAL_EFFECT_RELATIVE.name: _sha256_bytes(effect_csv_payload),
    }
    if outputs != expected_output_hashes:
        raise VerificationError(f"{label} historical output hashes differ.")

    stable: dict[str, object] = {}
    for key in HISTORICAL_STABLE_FIELDS:
        if key not in report:
            raise VerificationError(f"{label} lacks stable field {key}.")
        stable[key] = report[key]
    return stable


def _sha256_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise VerificationError(f"{label} is not a lowercase SHA-256 digest.")
    return value


def _positive_int(value: object, *, label: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise VerificationError(f"{label} must be an integer of at least {minimum}.")
    return value


def _npz_member_payloads(
    payload: bytes,
    *,
    expected_members: frozenset[str],
    label: str,
) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise VerificationError(f"{label} has duplicate NPZ members.")
            if set(names) != expected_members:
                raise VerificationError(f"{label} has an unexpected NPZ member schema.")
            if any(info.is_dir() or info.flag_bits & 0x1 for info in infos):
                raise VerificationError(f"{label} has an invalid NPZ member.")
            total_size = sum(info.file_size for info in infos)
            if total_size > MAX_NPZ_UNCOMPRESSED_BYTES:
                raise VerificationError(f"{label} exceeds the NPZ size limit.")
            members = {info.filename: archive.read(info) for info in infos}
    except VerificationError:
        raise
    except (
        NotImplementedError,
        OSError,
        RuntimeError,
        ValueError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as exc:
        raise VerificationError(f"Invalid {label}: {exc}") from exc
    for name, member_payload in members.items():
        if not member_payload.startswith(NPY_SIGNATURE):
            raise VerificationError(f"{label} member {name} is not an NPY array.")
    return members


def _compare_historical_phase4_npz(
    archive_payload: bytes,
    replay_payload: bytes,
    *,
    artifact_key: str,
) -> int:
    expected_members = HISTORICAL_PHASE4_NPZ_MEMBER_NAMES[artifact_key]
    archive_members = _npz_member_payloads(
        archive_payload,
        expected_members=expected_members,
        label=f"archive historical Phase 4 {artifact_key} NPZ",
    )
    replay_members = _npz_member_payloads(
        replay_payload,
        expected_members=expected_members,
        label=f"replay historical Phase 4 {artifact_key} NPZ",
    )
    volatile_members = HISTORICAL_PHASE4_VOLATILE_NPZ_MEMBERS.get(
        artifact_key, frozenset()
    )
    stable_members = sorted(expected_members - volatile_members)
    for name in stable_members:
        if replay_members[name] != archive_members[name]:
            raise VerificationError(
                "Replay historical Phase 4 NPZ stable member differs: "
                f"{artifact_key}/{name}."
            )
    return len(stable_members)


def _validate_historical_phase4_compatibility_summary(
    payload: bytes,
    *,
    report: Mapping[str, object],
    label: str,
) -> None:
    summary = _json_object(payload, label=label)
    expected_fields = {
        "N",
        "M",
        "rho",
        "r",
        "epsilon_X",
        "finite_M_prefactor_certified",
        "sampled_validation_passes",
        "target_packets",
        "algebraic_count",
        "worst_epsilon_m_gamma",
        "contour_interval_certified",
        "moat_status",
        "diagnostic_status",
        "theorem_gate_eligible",
        "legacy_seed_dependency",
        "configuration_digest",
        "source_digest",
        "cache_key",
    }
    if set(summary) != expected_fields:
        raise VerificationError(f"Unexpected {label} schema.")
    _expect_exact_int(summary, "N", 600)
    _expect_exact_int(summary, "M", 610)
    if summary.get("rho") != 2.45 or summary.get("r") != 2.225974769705636:
        raise VerificationError(f"Unexpected {label} historical radii.")
    if summary.get("epsilon_X") != 1.1641169413997097e-17:
        raise VerificationError(f"Unexpected {label} epsilon_X.")
    _expect_exact_int(summary, "target_packets", 15)
    _expect_exact_int(summary, "algebraic_count", 19)
    _expect_exact_int(summary, "sampled_validation_passes", 15)
    worst_product = summary.get("worst_epsilon_m_gamma")
    if (
        isinstance(worst_product, bool)
        or not isinstance(worst_product, (int, float))
        or not math.isfinite(float(worst_product))
        or not 0.45 <= float(worst_product) <= 0.55
    ):
        raise VerificationError(f"Invalid {label} worst sampled product.")
    for key in (
        "finite_M_prefactor_certified",
        "contour_interval_certified",
        "theorem_gate_eligible",
        "legacy_seed_dependency",
    ):
        _expect_bool(summary, key, False)
    if summary.get("moat_status") != "sampled finite-section moat diagnostic only":
        raise VerificationError(f"Unexpected {label} moat status.")
    if summary.get("diagnostic_status") != EXPECTED_HISTORICAL_PHASE4_STATUS:
        raise VerificationError(f"Unexpected {label} diagnostic status.")
    for key in ("configuration_digest", "source_digest", "cache_key"):
        if summary.get(key) != report.get(key):
            raise VerificationError(f"{label} differs from its report for {key}.")


def _validate_historical_phase4_report(
    report: Mapping[str, object],
    *,
    artifact_payloads: Mapping[str, bytes],
    label: str,
) -> tuple[dict[str, object], dict[str, int]]:
    expected_fields = {
        "map_label",
        "producer_schema",
        "diagnostic_status",
        "diagnostic_description",
        "record_role",
        "authoritative_for_current_thesis",
        "theorem_gate_eligible",
        "contour_interval_certified",
        "legacy_seed_dependency",
        "retained_csv_inputs",
        "retained_npz_inputs",
        "configuration",
        "configuration_digest",
        "source_digest",
        "cache_key",
        "producer_sources",
        "runtime_versions",
        "process_execution",
        "cache_validation",
        "historical_regression",
        "target_plan",
        "target_plan_digest",
        "sampled_diagnostics",
        "surfaces",
        "matrix_hashes",
        "outputs",
    }
    if set(report) != expected_fields:
        raise VerificationError(f"Unexpected {label} schema.")
    if report.get("map_label") != EXPECTED_MAP_LABEL:
        raise VerificationError(f"Unexpected {label} map_label.")
    if report.get("producer_schema") != EXPECTED_HISTORICAL_PHASE4_SCHEMA:
        raise VerificationError(f"Unexpected {label} producer schema.")
    if report.get("diagnostic_status") != EXPECTED_HISTORICAL_PHASE4_STATUS:
        raise VerificationError(f"Unexpected {label} diagnostic status.")
    if report.get("diagnostic_description") != (
        "retained historical wide-radius sampled finite-section diagnostic; "
        "not interval-certified and not eligible for a theorem gate"
    ):
        raise VerificationError(f"Unexpected {label} diagnostic description.")
    if report.get("record_role") != "historical_exploratory_diagnostic":
        raise VerificationError(f"Unexpected {label} record role.")
    for key in (
        "authoritative_for_current_thesis",
        "theorem_gate_eligible",
        "contour_interval_certified",
        "legacy_seed_dependency",
    ):
        _expect_bool(report, key, False)
    if (
        report.get("retained_csv_inputs") != []
        or report.get("retained_npz_inputs") != []
    ):
        raise VerificationError(f"{label} declares retained seed inputs.")
    if report.get("configuration") != EXPECTED_HISTORICAL_PHASE4_CONFIGURATION:
        raise VerificationError(f"Unexpected {label} production configuration.")
    for key in ("configuration_digest", "source_digest", "cache_key"):
        _sha256_text(report.get(key), label=f"{label}.{key}")

    source_names = {
        "historical_phase4_producer": "blaschke_deformation_historical_phase4.py",
        "mpmath_row_worker": "mpmath_pf_raw.py",
        "hardy_surface_worker": "hardy_moat_surface_worker.py",
    }
    producer_sources = report.get("producer_sources")
    if not isinstance(producer_sources, dict) or set(producer_sources) != set(
        source_names
    ):
        raise VerificationError(f"Unexpected {label} producer sources.")
    stable_sources: dict[str, object] = {}
    for name, filename in source_names.items():
        record = producer_sources[name]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise VerificationError(f"Invalid {label} source record {name}.")
        path = record.get("path")
        if not isinstance(path, str) or PurePosixPath(path).name != filename:
            raise VerificationError(f"Unexpected {label} source path for {name}.")
        stable_sources[name] = {
            "filename": filename,
            "sha256": _sha256_text(
                record.get("sha256"), label=f"{label}.{name}.sha256"
            ),
        }

    runtime_versions = report.get("runtime_versions")
    runtime_fields = {
        "python",
        "python_implementation",
        "numpy",
        "scipy",
        "mpmath",
        "byteorder",
    }
    if (
        not isinstance(runtime_versions, dict)
        or set(runtime_versions) != runtime_fields
    ):
        raise VerificationError(f"Unexpected {label} runtime-version schema.")
    if any(
        not isinstance(value, str) or not value for value in runtime_versions.values()
    ):
        raise VerificationError(f"Invalid {label} runtime version.")
    if runtime_versions.get("byteorder") not in {"little", "big"}:
        raise VerificationError(f"Invalid {label} byte order.")

    process = report.get("process_execution")
    process_fields = {
        "row_block_assembly",
        "assembly_workers",
        "assembly_row_block_size",
        "surface_row_blocks",
        "surface_workers",
        "surface_row_block_size",
    }
    if not isinstance(process, dict) or set(process) != process_fields:
        raise VerificationError(f"Unexpected {label} process schema.")
    _expect_bool(process, "row_block_assembly", True)
    _expect_bool(process, "surface_row_blocks", True)
    _positive_int(
        process.get("assembly_workers"), label=f"{label}.assembly_workers", minimum=2
    )
    _positive_int(
        process.get("surface_workers"), label=f"{label}.surface_workers", minimum=2
    )
    _positive_int(
        process.get("assembly_row_block_size"),
        label=f"{label}.assembly_row_block_size",
    )
    _expect_exact_int(process, "surface_row_block_size", 2)

    cache = report.get("cache_validation")
    cache_fields = {
        "keyed_by_configuration_digest",
        "keyed_by_source_digest",
        "keyed_by_runtime_versions",
        "surface_keyed_by_hardy_matrix_hash",
        "cache_hits",
        "contour_cache_path",
    }
    if not isinstance(cache, dict) or set(cache) != cache_fields:
        raise VerificationError(f"Unexpected {label} cache-validation schema.")
    cache_guards = (
        "keyed_by_configuration_digest",
        "keyed_by_source_digest",
        "keyed_by_runtime_versions",
        "surface_keyed_by_hardy_matrix_hash",
    )
    for key in cache_guards:
        _expect_bool(cache, key, True)
    expected_cache_hits = {
        "scaled_block",
        "hardy_matrix",
        "eigenvalues",
        "contours",
        "surface_local",
        "surface_global",
        "surface_zoom",
        "surface_deep_zoom",
    }
    cache_hits = cache.get("cache_hits")
    if not isinstance(cache_hits, dict) or set(cache_hits) != expected_cache_hits:
        raise VerificationError(f"Unexpected {label} cache-hit schema.")
    if any(type(value) is not bool for value in cache_hits.values()):
        raise VerificationError(f"Invalid {label} cache-hit value.")
    contour_cache_path = cache.get("contour_cache_path")
    if not isinstance(contour_cache_path, str) or not contour_cache_path:
        raise VerificationError(f"Invalid {label} contour cache path.")

    historical_regression = report.get("historical_regression")
    regression_fields = {
        "contract",
        "pure_scaled_binary64_boundary_after_mpmath_dps",
        "hardy_similarity",
        "contour_radius_rule",
        "scaled_matrix_maximum_modulus",
        "hardy_matrix_maximum_modulus",
        "sampled_validation_passes",
        "worst_finite_epsilon_m_gamma",
        "theorem_gate_eligible",
    }
    if (
        not isinstance(historical_regression, dict)
        or set(historical_regression) != regression_fields
    ):
        raise VerificationError(f"Unexpected {label} historical regression schema.")
    if historical_regression.get("contract") != (
        "source-rederived historical sampled comparison"
    ):
        raise VerificationError(f"Unexpected {label} historical regression contract.")
    _expect_exact_int(
        historical_regression,
        "pure_scaled_binary64_boundary_after_mpmath_dps",
        200,
    )
    if historical_regression.get("hardy_similarity") != (
        "binary64 T B_scaled T_inverse historical path"
    ):
        raise VerificationError(f"Unexpected {label} Hardy similarity contract.")
    if historical_regression.get("contour_radius_rule") != (
        "finite-eigenvalue cluster-to-nearest-outside mid-gap"
    ):
        raise VerificationError(f"Unexpected {label} contour-radius contract.")
    _expect_bool(historical_regression, "theorem_gate_eligible", False)
    _expect_exact_int(historical_regression, "sampled_validation_passes", 15)
    scaled_maximum = historical_regression.get("scaled_matrix_maximum_modulus")
    hardy_maximum = historical_regression.get("hardy_matrix_maximum_modulus")
    regression_product = historical_regression.get(
        "worst_finite_epsilon_m_gamma"
    )
    if not isinstance(scaled_maximum, (int, float)) or isinstance(
        scaled_maximum, bool
    ) or not 1.9e9 <= float(scaled_maximum) <= 2.1e9:
        raise VerificationError(f"Invalid {label} pure-scaled regression scale.")
    if not isinstance(hardy_maximum, (int, float)) or isinstance(
        hardy_maximum, bool
    ) or not 1.7e9 <= float(hardy_maximum) <= 1.9e9:
        raise VerificationError(f"Invalid {label} Hardy regression scale.")
    if not isinstance(regression_product, (int, float)) or isinstance(
        regression_product, bool
    ) or not 0.45 <= float(regression_product) <= 0.55:
        raise VerificationError(f"Invalid {label} historical sampled product.")

    target_plan = report.get("target_plan")
    target_fields = {
        "rank",
        "name",
        "family",
        "power",
        "centre",
        "expected_multiplicity",
        "nearest_target_separation",
        "radius",
    }
    if not isinstance(target_plan, list) or len(target_plan) != 15:
        raise VerificationError(f"Unexpected {label} target plan cardinality.")
    if any(
        not isinstance(record, dict) or set(record) != target_fields
        for record in target_plan
    ):
        raise VerificationError(f"Unexpected {label} target plan schema.")
    if (
        tuple(record["name"] for record in target_plan)
        != EXPECTED_HISTORICAL_PHASE4_TARGET_NAMES
    ):
        raise VerificationError(f"Unexpected {label} target plan ordering.")
    if tuple(record["rank"] for record in target_plan) != tuple(range(1, 16)):
        raise VerificationError(f"Unexpected {label} target ranks.")
    if any(
        type(record["expected_multiplicity"]) is not int
        or record["expected_multiplicity"] < 1
        for record in target_plan
    ):
        raise VerificationError(f"Invalid {label} target multiplicity.")
    if sum(record["expected_multiplicity"] for record in target_plan) != 19:
        raise VerificationError(f"Unexpected {label} target multiplicity.")
    _sha256_text(report.get("target_plan_digest"), label=f"{label}.target_plan_digest")

    sampled = report.get("sampled_diagnostics")
    sampled_fields = {
        "target_count",
        "algebraic_multiplicity",
        "contour_samples",
        "profile_rows",
        "fragile_rows",
        "sampled_validation_passes",
        "worst_finite_epsilon_m_gamma",
    }
    if not isinstance(sampled, dict) or set(sampled) != sampled_fields:
        raise VerificationError(f"Unexpected {label} sampled-diagnostic schema.")
    _expect_exact_int(sampled, "target_count", 15)
    _expect_exact_int(sampled, "algebraic_multiplicity", 19)
    _expect_exact_int(sampled, "contour_samples", 128)
    _expect_exact_int(sampled, "profile_rows", 15 * 128)
    _expect_exact_int(sampled, "fragile_rows", 4)
    _expect_exact_int(sampled, "sampled_validation_passes", 15)
    worst_product = sampled.get("worst_finite_epsilon_m_gamma")
    if (
        isinstance(worst_product, bool)
        or not isinstance(worst_product, (int, float))
        or not math.isfinite(float(worst_product))
        or not 0.45 <= float(worst_product) <= 0.55
    ):
        raise VerificationError(f"Invalid {label} worst sampled product.")

    surfaces = report.get("surfaces")
    expected_surface_grids = {
        "local": 25,
        "global": 45,
        "zoom": 161,
        "deep_zoom": 201,
    }
    if not isinstance(surfaces, dict) or set(surfaces) != set(expected_surface_grids):
        raise VerificationError(f"Unexpected {label} surface schema.")
    stable_surfaces: dict[str, object] = {}
    for name, grid in expected_surface_grids.items():
        record = surfaces[name]
        if not isinstance(record, dict) or set(record) != {
            "xlim",
            "ylim",
            "grid",
            "minimum_singular_value",
            "cache_hit",
        }:
            raise VerificationError(f"Unexpected {label} surface record {name}.")
        _expect_exact_int(record, "grid", grid)
        if type(record.get("cache_hit")) is not bool:
            raise VerificationError(f"Invalid {label} {name} cache-hit value.")
        for axis in ("xlim", "ylim"):
            limits = record.get(axis)
            if (
                not isinstance(limits, list)
                or len(limits) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in limits
                )
                or float(limits[0]) >= float(limits[1])
            ):
                raise VerificationError(f"Invalid {label} {name} {axis}.")
        minimum = record.get("minimum_singular_value")
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, (int, float))
            or not math.isfinite(float(minimum))
            or float(minimum) < 0.0
        ):
            raise VerificationError(f"Invalid {label} {name} minimum.")
        stable_surfaces[name] = {
            key: value for key, value in record.items() if key != "cache_hit"
        }

    matrix_hashes = report.get("matrix_hashes")
    expected_matrix_hashes = {
        "B_scaled_sha256",
        "A_X_sha256",
        "T_sha256",
        "eigenvalues_sha256",
    }
    if (
        not isinstance(matrix_hashes, dict)
        or set(matrix_hashes) != expected_matrix_hashes
    ):
        raise VerificationError(f"Unexpected {label} matrix-hash schema.")
    for key, value in matrix_hashes.items():
        _sha256_text(value, label=f"{label}.matrix_hashes.{key}")

    if set(artifact_payloads) != set(HISTORICAL_PHASE4_DATA_RELATIVES):
        raise VerificationError(f"Incomplete {label} artifact payload set.")
    outputs = report.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(
        HISTORICAL_PHASE4_DATA_RELATIVES
    ):
        raise VerificationError(f"Unexpected {label} output schema.")
    stable_outputs: dict[str, object] = {}
    for key, relative in HISTORICAL_PHASE4_DATA_RELATIVES.items():
        record = outputs[key]
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "diagnostic_status",
        }:
            raise VerificationError(f"Unexpected {label} output record {key}.")
        if record.get("path") != relative.name:
            raise VerificationError(f"Unexpected {label} output path for {key}.")
        if record.get("diagnostic_status") != EXPECTED_HISTORICAL_PHASE4_STATUS:
            raise VerificationError(f"Unexpected {label} output status for {key}.")
        observed_hash = _sha256_bytes(artifact_payloads[key])
        if record.get("sha256") != observed_hash:
            raise VerificationError(f"{label} output hash differs for {key}.")
        stable_outputs[key] = {
            "path": relative.name,
            "diagnostic_status": EXPECTED_HISTORICAL_PHASE4_STATUS,
        }

    csv_field_counts: dict[str, int] = {}
    metadata_fields = {
        "diagnostic_status",
        "configuration_digest",
        "source_digest",
        "cache_key",
    }
    for key, expected_rows in HISTORICAL_PHASE4_CSV_ROW_COUNTS.items():
        fields, rows = _csv_table(artifact_payloads[key], label=f"{label} {key} CSV")
        if len(rows) != expected_rows:
            raise VerificationError(
                f"{label} {key} CSV must contain {expected_rows} rows."
            )
        if not metadata_fields.issubset(fields):
            raise VerificationError(f"{label} {key} CSV lacks producer metadata.")
        for row in rows:
            if row["diagnostic_status"] != EXPECTED_HISTORICAL_PHASE4_STATUS:
                raise VerificationError(f"Unexpected {label} {key} row status.")
            for digest_key in ("configuration_digest", "source_digest", "cache_key"):
                if row[digest_key] != report[digest_key]:
                    raise VerificationError(
                        f"{label} {key} row differs for {digest_key}."
                    )
        csv_field_counts[key] = len(fields)
    for compatibility_key, source_key in (
        ("final_packets", "packets"),
        ("final_moats", "moats"),
        ("final_fragile", "fragile"),
    ):
        if artifact_payloads[compatibility_key] != artifact_payloads[source_key]:
            raise VerificationError(
                f"{label} compatibility CSV differs from {source_key}."
            )
    _validate_historical_phase4_compatibility_summary(
        artifact_payloads["compatibility_summary"],
        report=report,
        label=f"{label} compatibility summary",
    )

    stable = {
        key: report[key]
        for key in (
            "map_label",
            "producer_schema",
            "diagnostic_status",
            "diagnostic_description",
            "record_role",
            "authoritative_for_current_thesis",
            "theorem_gate_eligible",
            "contour_interval_certified",
            "legacy_seed_dependency",
            "retained_csv_inputs",
            "retained_npz_inputs",
            "configuration",
            "configuration_digest",
            "source_digest",
            "cache_key",
            "runtime_versions",
            "target_plan",
            "target_plan_digest",
            "sampled_diagnostics",
            "matrix_hashes",
        )
    }
    stable["producer_sources"] = stable_sources
    stable["process_execution"] = {
        "row_block_assembly": True,
        "surface_row_blocks": True,
        "surface_row_block_size": process["surface_row_block_size"],
    }
    stable["cache_validation"] = {key: cache[key] for key in cache_guards}
    stable["surfaces"] = stable_surfaces
    stable["outputs"] = stable_outputs
    return stable, csv_field_counts


def _without_path_fields(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_path_fields(item)
            for key, item in value.items()
            if key != "path"
        }
    if isinstance(value, list):
        return [_without_path_fields(item) for item in value]
    return value


def _validate_diagnostic_audit_report(
    report: Mapping[str, object],
    *,
    csv_payloads: Mapping[str, bytes],
    label: str,
) -> tuple[dict[str, object], dict[str, int]]:
    expected_fields = {
        "producer_schema",
        "map_label",
        "diagnostic_only",
        "diagnostic_status",
        "legacy_seed_dependency",
        "source_extraction",
        "inputs",
        "checks",
        "outputs",
    }
    if set(report) != expected_fields:
        raise VerificationError(f"Unexpected {label} schema.")
    if report.get("producer_schema") != EXPECTED_DIAGNOSTIC_AUDIT_SCHEMA:
        raise VerificationError(f"Unexpected {label} producer schema.")
    if report.get("map_label") != EXPECTED_MAP_LABEL:
        raise VerificationError(f"Unexpected {label} map_label.")
    if report.get("diagnostic_status") != EXPECTED_DIAGNOSTIC_AUDIT_STATUS:
        raise VerificationError(f"Unexpected {label} diagnostic status.")
    _expect_bool(report, "diagnostic_only", True)
    _expect_bool(report, "legacy_seed_dependency", False)

    source_extraction = report.get("source_extraction")
    if not isinstance(source_extraction, dict) or set(source_extraction) != {
        "notebook",
        "producer",
    }:
        raise VerificationError(f"Unexpected {label} source-extraction schema.")
    notebook_source = source_extraction["notebook"]
    producer_source = source_extraction["producer"]
    if not isinstance(notebook_source, dict) or set(notebook_source) != {
        "path",
        "sha256",
        "cells",
    }:
        raise VerificationError(f"Unexpected {label} notebook-source schema.")
    if not isinstance(producer_source, dict) or set(producer_source) != {
        "path",
        "sha256",
    }:
        raise VerificationError(f"Unexpected {label} producer-source schema.")
    for source, filename, source_label in (
        (
            notebook_source,
            "blaschke_deformation_certifier.ipynb",
            "notebook",
        ),
        (
            producer_source,
            "blaschke_deformation_diagnostic_audits.py",
            "producer",
        ),
    ):
        path = source.get("path")
        if not isinstance(path, str) or PurePosixPath(path).name != filename:
            raise VerificationError(f"Unexpected {label} {source_label} path.")
        _sha256_text(source.get("sha256"), label=f"{label}.{source_label}.sha256")
    cells = notebook_source.get("cells")
    if not isinstance(cells, dict) or set(cells) != {"Cell 100", "Cell 102"}:
        raise VerificationError(f"Unexpected {label} notebook-cell schema.")
    for cell_label, record in cells.items():
        if not isinstance(record, dict) or set(record) != {
            "cell_id",
            "source_sha256",
        }:
            raise VerificationError(f"Unexpected {label} {cell_label} record.")
        if not isinstance(record.get("cell_id"), str) or not record["cell_id"]:
            raise VerificationError(f"Invalid {label} {cell_label} ID.")
        _sha256_text(
            record.get("source_sha256"),
            label=f"{label}.{cell_label}.source_sha256",
        )

    inputs = report.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "targets",
        "geometry",
        "schur_rows",
        "moat_rows",
    }:
        raise VerificationError(f"Unexpected {label} input schema.")
    for name, record in inputs.items():
        if not isinstance(record, dict):
            raise VerificationError(f"Invalid {label} input record {name}.")
        kind = record.get("kind")
        expected_record_fields = {"kind", "rows", "columns", "sha256"}
        if kind == "explicit_generated_path":
            expected_record_fields.add("path")
        elif kind != "explicit_generated_dataframe":
            raise VerificationError(f"Unexpected {label} input kind for {name}.")
        if set(record) != expected_record_fields:
            raise VerificationError(f"Unexpected {label} input record {name}.")
        if type(record.get("rows")) is not int or record["rows"] < 1:
            raise VerificationError(f"Invalid {label} input row count for {name}.")
        columns = record.get("columns")
        if (
            not isinstance(columns, list)
            or not columns
            or any(not isinstance(column, str) or not column for column in columns)
            or len(columns) != len(set(columns))
        ):
            raise VerificationError(f"Invalid {label} input columns for {name}.")
        _sha256_text(record.get("sha256"), label=f"{label}.{name}.sha256")
        if "path" in record and (
            not isinstance(record["path"], str) or not record["path"]
        ):
            raise VerificationError(f"Invalid {label} input path for {name}.")

    checks = report.get("checks")
    expected_checks = {
        "map_lock": True,
        "schema_alignment": True,
        "target_alignment": True,
        "sampled_claims_remain_diagnostic": True,
        "target_count": 14,
        "target_names": list(EXPECTED_DIAGNOSTIC_TARGET_NAMES),
        "sampled_schur_N": list(EXPECTED_DIAGNOSTIC_SCHUR_N),
    }
    if checks != expected_checks:
        raise VerificationError(f"Unexpected {label} checks.")

    if set(csv_payloads) != set(DIAGNOSTIC_AUDIT_DATA_RELATIVES):
        raise VerificationError(f"Incomplete {label} CSV payload set.")
    outputs = report.get("outputs")
    expected_output_by_key = {
        "universal_audit": (
            EXPECTED_UNIVERSAL_AUDIT_FIELDS,
            7,
        ),
        "first14_audit": (
            EXPECTED_FIRST14_AUDIT_FIELDS,
            14,
        ),
    }
    expected_output_names = {
        DIAGNOSTIC_AUDIT_DATA_RELATIVES[key].name for key in expected_output_by_key
    }
    if not isinstance(outputs, dict) or set(outputs) != expected_output_names:
        raise VerificationError(f"Unexpected {label} output schema.")

    row_counts: dict[str, int] = {}
    for key, (expected_csv_fields, expected_rows) in expected_output_by_key.items():
        relative = DIAGNOSTIC_AUDIT_DATA_RELATIVES[key]
        payload = csv_payloads[key]
        fields, rows = _csv_table(payload, label=f"{label} {key} CSV")
        if fields != expected_csv_fields:
            raise VerificationError(f"Unexpected {label} {key} CSV schema.")
        if len(rows) != expected_rows:
            raise VerificationError(
                f"{label} {key} CSV must contain {expected_rows} rows."
            )
        output = outputs[relative.name]
        if not isinstance(output, dict) or set(output) != {
            "path",
            "rows",
            "columns",
            "sha256",
            "diagnostic_only",
        }:
            raise VerificationError(f"Unexpected {label} output {relative.name}.")
        path = output.get("path")
        if not isinstance(path, str) or PurePosixPath(path).name != relative.name:
            raise VerificationError(f"Unexpected {label} output path {relative.name}.")
        _expect_exact_int(output, "rows", expected_rows)
        if output.get("columns") != list(expected_csv_fields):
            raise VerificationError(
                f"Unexpected {label} output columns {relative.name}."
            )
        if output.get("sha256") != _sha256_bytes(payload):
            raise VerificationError(f"{label} output hash differs for {relative.name}.")
        _expect_bool(output, "diagnostic_only", True)
        row_counts[key] = len(rows)

        if key == "universal_audit":
            expected_items = (
                "target source and contour role",
                "Bernstein branch geometry",
                "single-space Schur envelope",
                "finite packet counts",
                "finite-section contour moats",
                "sampled small-gain test",
                "overall spectral certification",
            )
            if tuple(row["audit_item"] for row in rows) != expected_items:
                raise VerificationError(f"Unexpected {label} universal audit rows.")
            if any(row["map_label"] != EXPECTED_MAP_LABEL for row in rows):
                raise VerificationError(f"Unexpected {label} universal map label.")
            theorem_items = {
                row["audit_item"]
                for row in rows
                if row["status"] == "theorem_certified"
            }
            if not theorem_items.issubset({"target source and contour role"}):
                raise VerificationError(f"{label} promotes sampled universal evidence.")
            if rows[-1]["status"] != "diagnostic_not_theorem_certified":
                raise VerificationError(f"{label} overall audit is not diagnostic.")
        else:
            if tuple(row["name"] for row in rows) != EXPECTED_DIAGNOSTIC_TARGET_NAMES:
                raise VerificationError(f"Unexpected {label} first14 target ordering.")
            for row in rows:
                if (
                    row["contour_interval_certified"] != "False"
                    or row["finite_count_certified"] != "False"
                    or row["certified_small_gain_pass"] != "False"
                    or "theorem_certified_equal" in row["riesz_rank_status"]
                ):
                    raise VerificationError(
                        f"{label} promotes sampled packet evidence."
                    )

    stable = {
        key: report[key]
        for key in (
            "producer_schema",
            "map_label",
            "diagnostic_only",
            "diagnostic_status",
            "legacy_seed_dependency",
            "checks",
        )
    }
    stable["source_extraction"] = _without_path_fields(source_extraction)
    stable["inputs"] = _without_path_fields(inputs)
    stable["outputs"] = _without_path_fields(outputs)
    return stable, row_counts


def _validate_geometry_input_hashes(
    report: Mapping[str, object], payloads: Mapping[str, bytes]
) -> None:
    """Validate run-local container hashes without treating them as semantic data."""

    recorded = report.get("geometry_input_hashes")
    if not isinstance(recorded, dict) or set(recorded) != set(
        HARDY_MATRIX_INPUT_RELATIVES
    ):
        raise VerificationError("Unexpected theorem-report geometry input hashes.")
    if set(payloads) != set(THEOREM_INPUT_RELATIVES):
        raise VerificationError("Incomplete theorem input payload set.")
    for key in HARDY_MATRIX_INPUT_RELATIVES:
        payload = payloads[key]
        if recorded.get(key) != _sha256_bytes(payload):
            raise VerificationError(
                f"Theorem-report geometry input hash mismatch for {key}."
            )
    all_recorded = report.get("input_hashes")
    if not isinstance(all_recorded, dict) or set(all_recorded) != set(
        THEOREM_INPUT_RELATIVES
    ):
        raise VerificationError("Unexpected theorem-report input hashes.")
    for key, payload in payloads.items():
        if all_recorded.get(key) != _sha256_bytes(payload):
            raise VerificationError(
                f"Theorem-report input hash mismatch for {key}."
            )


def _compare_stable_theorem_report(
    archive_report: Mapping[str, object], replay_report: Mapping[str, object]
) -> int:
    if set(archive_report) != set(replay_report):
        raise VerificationError(
            "Replay theorem report schema differs from the archive."
        )
    stable_fields = sorted(set(archive_report) - NON_STABLE_THEOREM_REPORT_FIELDS)
    for key in stable_fields:
        if replay_report[key] != archive_report[key]:
            raise VerificationError(
                f"Replay theorem report stable field differs: {key}."
            )
    return len(stable_fields)


def _png_critical_chunk_visual_digest(payload: bytes, *, label: str) -> str:
    if not payload.startswith(PNG_SIGNATURE):
        raise VerificationError(f"{label} is not a PNG file.")
    offset = len(PNG_SIGNATURE)
    critical_digest = hashlib.sha256()
    critical_digest.update(b"png-critical-chunks-v1\x00")
    chunk_index = 0
    seen_ihdr = False
    seen_plte = False
    seen_idat = False
    idat_ended = False
    seen_iend = False
    while offset < len(payload):
        if len(payload) - offset < 12:
            raise VerificationError(f"{label} has a truncated PNG chunk.")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        chunk_end = data_end + 4
        if chunk_end > len(payload):
            raise VerificationError(f"{label} has a truncated PNG chunk payload.")
        if not all(
            65 <= character <= 90 or 97 <= character <= 122 for character in chunk_type
        ):
            raise VerificationError(f"{label} has an invalid PNG chunk type.")
        chunk_data = payload[data_start:data_end]
        recorded_crc = struct.unpack(">I", payload[data_end:chunk_end])[0]
        observed_crc = zlib.crc32(chunk_type)
        observed_crc = zlib.crc32(chunk_data, observed_crc) & 0xFFFFFFFF
        if recorded_crc != observed_crc:
            raise VerificationError(
                f"{label} has a PNG CRC mismatch in {chunk_type!r}."
            )

        is_critical = not bool(chunk_type[0] & 0x20)
        if is_critical:
            if chunk_type not in {b"IHDR", b"PLTE", b"IDAT", b"IEND"}:
                raise VerificationError(
                    f"{label} has an unknown critical PNG chunk {chunk_type!r}."
                )
            critical_digest.update(struct.pack(">I", length))
            critical_digest.update(chunk_type)
            critical_digest.update(chunk_data)

        if chunk_type == b"IHDR":
            if chunk_index != 0 or seen_ihdr or length != 13:
                raise VerificationError(f"{label} has an invalid PNG IHDR chunk.")
            width, height, bit_depth, colour_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", chunk_data)
            )
            allowed_depths = {
                0: {1, 2, 4, 8, 16},
                2: {8, 16},
                3: {1, 2, 4, 8},
                4: {8, 16},
                6: {8, 16},
            }
            if (
                width == 0
                or height == 0
                or colour_type not in allowed_depths
                or bit_depth not in allowed_depths[colour_type]
                or compression != 0
                or filtering != 0
                or interlace not in {0, 1}
            ):
                raise VerificationError(f"{label} has invalid PNG image metadata.")
            seen_ihdr = True
        elif not seen_ihdr:
            raise VerificationError(f"{label} does not begin with PNG IHDR.")
        elif chunk_type == b"PLTE":
            if seen_plte or seen_idat or length == 0 or length % 3:
                raise VerificationError(f"{label} has an invalid PNG PLTE chunk.")
            seen_plte = True
        elif chunk_type == b"IDAT":
            if idat_ended:
                raise VerificationError(f"{label} has non-contiguous PNG IDAT chunks.")
            seen_idat = True
        else:
            if seen_idat and chunk_type != b"IEND":
                idat_ended = True
            if chunk_type == b"IEND":
                if seen_iend or length != 0 or not seen_idat:
                    raise VerificationError(f"{label} has an invalid PNG IEND chunk.")
                seen_iend = True
                offset = chunk_end
                if offset != len(payload):
                    raise VerificationError(f"{label} has data after PNG IEND.")
                break
        offset = chunk_end
        chunk_index += 1
    if not (seen_ihdr and seen_idat and seen_iend):
        raise VerificationError(f"{label} has an incomplete critical PNG closure.")
    return critical_digest.hexdigest()


def _compare_plot_group(
    replay_root: Path,
    *,
    files: Mapping[str, bytes],
    relatives: tuple[PurePosixPath, ...],
    label: str,
) -> dict[str, str]:
    digests: dict[str, str] = {}
    for relative in relatives:
        archive_digest = _png_critical_chunk_visual_digest(
            _archive_payload(files, relative, label=f"{label} plot"),
            label=f"archive {label} plot {relative.name}",
        )
        replay_digest = _png_critical_chunk_visual_digest(
            _read_replay_bytes(replay_root, relative, label=f"{label} plot"),
            label=f"replay {label} plot {relative.name}",
        )
        if replay_digest != archive_digest:
            raise VerificationError(
                f"Replay {label} plot visual digest differs: {relative.name}."
            )
        digests[relative.name] = replay_digest
    return digests


def _compare_executed_replay_root(
    root: Path,
    *,
    files: Mapping[str, bytes],
    source_entries: Mapping[str, str],
    archive_effective_plan: Mapping[str, object],
    archive_report: Mapping[str, object],
) -> dict[str, object]:
    replay_root = Path(root).resolve()
    if not replay_root.is_dir():
        raise VerificationError(
            f"Executed replay root is not a directory: {replay_root}."
        )

    output_prefix = f"{OUTPUT_RELATIVE.as_posix()}/"
    immutable_count = 0
    for path_text in source_entries:
        if path_text == NOTEBOOK_RELATIVE.as_posix() or path_text.startswith(
            output_prefix
        ):
            continue
        relative = _safe_relative_path(
            path_text, label="executed replay immutable path"
        )
        archive_bytes = _archive_payload(
            files, relative, label="immutable source-manifest file"
        )
        replay_bytes = _read_replay_bytes(
            replay_root, relative, label="immutable source-manifest file"
        )
        if replay_bytes != archive_bytes:
            raise VerificationError(
                "Executed replay immutable source differs byte-for-byte: "
                f"{relative.as_posix()}."
            )
        immutable_count += 1

    replay_notebook = _read_replay_json(
        replay_root, NOTEBOOK_RELATIVE, label="executed notebook"
    )
    notebook_summary = _validate_notebook(replay_notebook)
    _validate_replay_inline_helper_sync(replay_root, replay_notebook)

    replay_source_plan = _read_replay_json(
        replay_root, PLAN_RELATIVE, label="source reproducibility plan"
    )
    replay_report = _read_replay_json(
        replay_root, SPECTRAL_REPORT_RELATIVE, label="theorem report"
    )
    _validate_geometry_input_hashes(
        replay_report,
        {
            key: _read_replay_bytes(
                replay_root,
                relative,
                label=f"Hardy-matrix geometry input {key}",
            )
            for key, relative in THEOREM_INPUT_RELATIVES.items()
        },
    )
    replay_effective_plan = _refresh_plan(replay_source_plan, replay_report)
    certificate_summary = _validate_plan_and_report(
        replay_effective_plan, replay_report
    )
    if replay_effective_plan != archive_effective_plan:
        raise VerificationError(
            "Replay effective plan differs from the archive effective plan."
        )
    theorem_field_count = _compare_stable_theorem_report(archive_report, replay_report)

    archive_contour = _archive_payload(
        files, CONTOUR_CERTIFICATE_RELATIVE, label="contour certificate CSV"
    )
    replay_contour = _read_replay_bytes(
        replay_root, CONTOUR_CERTIFICATE_RELATIVE, label="contour certificate CSV"
    )
    contour_row_count = _compare_contour_certificate_csv(
        archive_contour, replay_contour
    )

    archive_phase2 = _archive_payload(
        files, PHASE2_CERTIFIED_ROW_RELATIVE, label="Phase 2 certified row"
    )
    replay_phase2 = _read_replay_bytes(
        replay_root, PHASE2_CERTIFIED_ROW_RELATIVE, label="Phase 2 certified row"
    )
    _compare_exact_single_row_csv(
        archive_phase2,
        replay_phase2,
        label="Phase 2 certified row",
    )

    archive_historical_baseline = _archive_payload(
        files, HISTORICAL_BASELINE_RELATIVE, label="historical baseline CSV"
    )
    replay_historical_baseline = _read_replay_bytes(
        replay_root, HISTORICAL_BASELINE_RELATIVE, label="historical baseline CSV"
    )
    baseline_fields, baseline_row = _compare_exact_single_row_csv(
        archive_historical_baseline,
        replay_historical_baseline,
        label="historical baseline CSV",
    )
    archive_historical_effect = _archive_payload(
        files, HISTORICAL_EFFECT_RELATIVE, label="historical effect CSV"
    )
    replay_historical_effect = _read_replay_bytes(
        replay_root, HISTORICAL_EFFECT_RELATIVE, label="historical effect CSV"
    )
    effect_fields, effect_row = _compare_exact_single_row_csv(
        archive_historical_effect,
        replay_historical_effect,
        label="historical effect CSV",
    )

    archive_historical_report = _json_object(
        _archive_payload(files, HISTORICAL_REPORT_RELATIVE, label="historical report"),
        label="archive historical report",
    )
    replay_historical_report = _read_replay_json(
        replay_root, HISTORICAL_REPORT_RELATIVE, label="historical report"
    )
    archive_historical_stable = _validate_historical_report(
        archive_historical_report,
        baseline_csv_payload=archive_historical_baseline,
        effect_csv_payload=archive_historical_effect,
        label="archive historical report",
    )
    replay_historical_stable = _validate_historical_report(
        replay_historical_report,
        baseline_csv_payload=replay_historical_baseline,
        effect_csv_payload=replay_historical_effect,
        label="replay historical report",
    )
    if replay_historical_stable != archive_historical_stable:
        raise VerificationError(
            "Replay historical report stable fields differ from the archive."
        )

    archive_phase4_payloads: dict[str, bytes] = {}
    replay_phase4_payloads: dict[str, bytes] = {}
    phase4_npz_stable_member_count = 0
    for key, relative in HISTORICAL_PHASE4_DATA_RELATIVES.items():
        archive_payload = _archive_payload(
            files, relative, label=f"historical Phase 4 {key} artifact"
        )
        replay_payload = _read_replay_bytes(
            replay_root, relative, label=f"historical Phase 4 {key} artifact"
        )
        archive_phase4_payloads[key] = archive_payload
        replay_phase4_payloads[key] = replay_payload
        if key in HISTORICAL_PHASE4_NPZ_MEMBER_NAMES:
            phase4_npz_stable_member_count += _compare_historical_phase4_npz(
                archive_payload,
                replay_payload,
                artifact_key=key,
            )
        elif replay_payload != archive_payload:
            raise VerificationError(
                "Replay historical Phase 4 deterministic artifact differs: " f"{key}."
            )

    archive_phase4_report = _json_object(
        _archive_payload(
            files,
            HISTORICAL_PHASE4_REPORT_RELATIVE,
            label="historical Phase 4 report",
        ),
        label="archive historical Phase 4 report",
    )
    replay_phase4_report = _read_replay_json(
        replay_root,
        HISTORICAL_PHASE4_REPORT_RELATIVE,
        label="historical Phase 4 report",
    )
    archive_phase4_stable, archive_phase4_csv_fields = (
        _validate_historical_phase4_report(
            archive_phase4_report,
            artifact_payloads=archive_phase4_payloads,
            label="archive historical Phase 4 report",
        )
    )
    replay_phase4_stable, replay_phase4_csv_fields = _validate_historical_phase4_report(
        replay_phase4_report,
        artifact_payloads=replay_phase4_payloads,
        label="replay historical Phase 4 report",
    )
    if replay_phase4_stable != archive_phase4_stable:
        raise VerificationError(
            "Replay historical Phase 4 report stable fields differ from the archive."
        )
    if replay_phase4_csv_fields != archive_phase4_csv_fields:
        raise VerificationError(
            "Replay historical Phase 4 CSV field counts differ from the archive."
        )

    archive_audit_payloads: dict[str, bytes] = {}
    replay_audit_payloads: dict[str, bytes] = {}
    for key, relative in DIAGNOSTIC_AUDIT_DATA_RELATIVES.items():
        archive_payload = _archive_payload(
            files, relative, label=f"diagnostic audit {key} CSV"
        )
        replay_payload = _read_replay_bytes(
            replay_root, relative, label=f"diagnostic audit {key} CSV"
        )
        if replay_payload != archive_payload:
            raise VerificationError(
                f"Replay diagnostic audit deterministic CSV differs: {key}."
            )
        archive_audit_payloads[key] = archive_payload
        replay_audit_payloads[key] = replay_payload

    archive_audit_report = _json_object(
        _archive_payload(
            files,
            DIAGNOSTIC_AUDIT_REPORT_RELATIVE,
            label="diagnostic audit report",
        ),
        label="archive diagnostic audit report",
    )
    replay_audit_report = _read_replay_json(
        replay_root,
        DIAGNOSTIC_AUDIT_REPORT_RELATIVE,
        label="diagnostic audit report",
    )
    archive_audit_stable, archive_audit_rows = _validate_diagnostic_audit_report(
        archive_audit_report,
        csv_payloads=archive_audit_payloads,
        label="archive diagnostic audit report",
    )
    replay_audit_stable, replay_audit_rows = _validate_diagnostic_audit_report(
        replay_audit_report,
        csv_payloads=replay_audit_payloads,
        label="replay diagnostic audit report",
    )
    if replay_audit_stable != archive_audit_stable:
        raise VerificationError(
            "Replay diagnostic audit report stable fields differ from the archive."
        )
    if replay_audit_rows != archive_audit_rows:
        raise VerificationError(
            "Replay diagnostic audit row counts differ from the archive."
        )

    historical_plot_digests = _compare_plot_group(
        replay_root,
        files=files,
        relatives=HISTORICAL_PLOT_RELATIVES,
        label="historical comparison",
    )
    phase4_plot_digests = _compare_plot_group(
        replay_root,
        files=files,
        relatives=HISTORICAL_PHASE4_PLOT_RELATIVES,
        label="historical Phase 4",
    )
    audit_plot_digests = _compare_plot_group(
        replay_root,
        files=files,
        relatives=DIAGNOSTIC_AUDIT_PLOT_RELATIVES,
        label="diagnostic audit",
    )
    plot_digests = {
        **historical_plot_digests,
        **phase4_plot_digests,
        **audit_plot_digests,
    }

    return {
        "status": "semantic-replay-match",
        "immutable_source_file_count": immutable_count,
        "notebook_validation": {
            **notebook_summary,
            "execution_counts_contiguous": True,
            "inline_helper_sync": True,
        },
        "certificate_validation": certificate_summary,
        "effective_plan_match": True,
        "theorem_report_stable_field_count": theorem_field_count,
        "contour_certificate_row_count": contour_row_count,
        "ignored_timing_fields": sorted(CONTOUR_TIMING_FIELDS),
        "phase2_certified_row_count": 1,
        "historical_comparison": {
            "producer_schema": EXPECTED_HISTORICAL_SCHEMA,
            "legacy_seed_dependency": False,
            "baseline_cells": EXPECTED_HISTORICAL_CONFIGURATION["baseline_cells"],
            "branch_image_cells": EXPECTED_HISTORICAL_CONFIGURATION[
                "branch_image_cells"
            ],
            "baseline_csv_field_count": len(baseline_fields),
            "effect_csv_field_count": len(effect_fields),
            "baseline_csv_row_count": 1 if baseline_row else 0,
            "effect_csv_row_count": 1 if effect_row else 0,
            "plot_count": len(historical_plot_digests),
        },
        "historical_phase4": {
            "producer_schema": EXPECTED_HISTORICAL_PHASE4_SCHEMA,
            "legacy_seed_dependency": False,
            "dataset_count": len(HISTORICAL_PHASE4_DATA_RELATIVES),
            "csv_count": len(HISTORICAL_PHASE4_CSV_ROW_COUNTS),
            "npz_count": len(HISTORICAL_PHASE4_NPZ_MEMBER_NAMES),
            "json_count": 1,
            "npz_stable_member_count": phase4_npz_stable_member_count,
            "csv_row_counts": dict(HISTORICAL_PHASE4_CSV_ROW_COUNTS),
            "csv_field_counts": dict(archive_phase4_csv_fields),
            "plot_count": len(phase4_plot_digests),
            "report_stable_field_count": len(archive_phase4_stable),
        },
        "diagnostic_audits": {
            "producer_schema": EXPECTED_DIAGNOSTIC_AUDIT_SCHEMA,
            "legacy_seed_dependency": False,
            "csv_count": len(DIAGNOSTIC_AUDIT_DATA_RELATIVES),
            "csv_row_counts": dict(archive_audit_rows),
            "plot_count": len(audit_plot_digests),
            "report_stable_field_count": len(archive_audit_stable),
        },
        "png_critical_chunk_visual_digests": plot_digests,
    }


def verify_bundle(
    archive_path: Path,
    *,
    external_manifest_path: Path | None = None,
    checksum_path: Path | None = None,
    compare_replay_root: Path | None = None,
    compare_executed_replay_root: Path | None = None,
) -> dict[str, object]:
    """Verify archive safety, hashes, semantics and optional replay parity."""

    archive_path = Path(archive_path).resolve()
    external_manifest_path = (
        Path(external_manifest_path).resolve()
        if external_manifest_path is not None
        else archive_path.parent / EXTERNAL_MANIFEST_NAME
    )
    checksum_path = (
        Path(checksum_path).resolve()
        if checksum_path is not None
        else archive_path.parent / CHECKSUM_NAME
    )
    if archive_path.name != ARCHIVE_NAME:
        raise VerificationError(f"Unexpected archive name: {archive_path.name}.")

    external = _read_json_object(
        external_manifest_path, label="external reproducibility manifest"
    )
    if external.get("schema_version") != 3:
        raise VerificationError("Unsupported external manifest schema.")
    if external.get("publication_complete") is not True:
        raise VerificationError("The external manifest is not a publication marker.")
    archive_record = external.get("archive")
    bundle = external.get("bundle")
    if not isinstance(archive_record, dict) or not isinstance(bundle, dict):
        raise VerificationError("Malformed external manifest records.")

    archive_hash = _sha256_file(archive_path)
    archive_size = archive_path.stat().st_size
    if archive_record.get("name") != ARCHIVE_NAME:
        raise VerificationError("External manifest archive name mismatch.")
    if archive_record.get("sha256") != archive_hash:
        raise VerificationError("External manifest archive hash mismatch.")
    if archive_record.get("bytes") != archive_size:
        raise VerificationError("External manifest archive size mismatch.")
    if archive_record.get("checksum_name") != CHECKSUM_NAME:
        raise VerificationError("External manifest checksum name mismatch.")
    try:
        checksum_text = checksum_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise VerificationError(f"Cannot read checksum file: {exc}") from exc
    checksum_match = CHECKSUM_RE.fullmatch(checksum_text)
    if checksum_match is None or checksum_match.groups() != (
        archive_hash,
        ARCHIVE_NAME,
    ):
        raise VerificationError("Archive checksum file mismatch.")

    repository = bundle.get("repository")
    if not isinstance(repository, dict):
        raise VerificationError("Missing bundle repository record.")
    commit_epoch = repository.get("commit_epoch")
    if type(commit_epoch) is not int or commit_epoch < 0:
        raise VerificationError("Invalid repository commit epoch.")
    if repository.get("dirty") is not False or repository.get("status_porcelain") != "":
        raise VerificationError("Bundle repository provenance is not clean.")

    files, tar_member_count = _read_safe_archive(
        archive_path, commit_epoch=commit_epoch
    )
    internal_payload = files.get(INTERNAL_MANIFEST_NAME)
    if internal_payload is None:
        raise VerificationError("Archive lacks its internal manifest.")
    if _sha256_bytes(internal_payload) != external.get("bundle_manifest_sha256"):
        raise VerificationError("Internal manifest hash mismatch.")
    internal = _json_object(internal_payload, label="internal reproducibility manifest")
    if internal != bundle:
        raise VerificationError("External and internal bundle manifests differ.")
    if internal.get("schema_version") != 3:
        raise VerificationError("Unsupported bundle manifest schema.")

    records = internal.get("files")
    if not isinstance(records, list):
        raise VerificationError("Bundle file records must be an array.")
    record_paths: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise VerificationError("Bundle file record is not an object.")
        path_text = record.get("archive_path")
        if not isinstance(path_text, str):
            raise VerificationError("Bundle file record lacks archive_path.")
        relative = _safe_relative_path(path_text, label="file-record path")
        label = relative.as_posix()
        if label in record_paths:
            raise VerificationError(f"Duplicate bundle file record: {label}.")
        record_paths.append(label)
        payload = files.get(label)
        if payload is None:
            raise VerificationError(f"Recorded archive file is missing: {label}.")
        if record.get("bytes") != len(payload):
            raise VerificationError(f"Recorded byte size mismatch: {label}.")
        if record.get("sha256") != _sha256_bytes(payload):
            raise VerificationError(f"Recorded hash mismatch: {label}.")
    if record_paths != sorted(record_paths):
        raise VerificationError("Bundle file records are not sorted.")
    if set(files) != set(record_paths) | {INTERNAL_MANIFEST_NAME}:
        raise VerificationError(
            "Archive regular-file closure differs from its records."
        )
    if internal.get("packaged_regular_file_count") != len(files):
        raise VerificationError("Packaged regular-file count mismatch.")

    source_manifest_payload = files.get(SOURCE_MANIFEST_NAME)
    if source_manifest_payload is None:
        raise VerificationError("Archive lacks MANIFEST.sha256.")
    source_entries = _parse_source_manifest(source_manifest_payload)
    source_manifest_record = internal.get("source_manifest")
    if not isinstance(source_manifest_record, dict):
        raise VerificationError("Missing source-manifest metadata.")
    if source_manifest_record.get("entry_count") != len(source_entries):
        raise VerificationError("Source-manifest entry count mismatch.")
    if source_manifest_record.get("sha256") != _sha256_bytes(source_manifest_payload):
        raise VerificationError("Source-manifest digest mismatch.")
    if source_manifest_record.get("archive_path") != SOURCE_MANIFEST_NAME:
        raise VerificationError("Source-manifest path mismatch.")
    if "README.md" not in source_entries or "DEPLOYMENT_README.md" in files:
        raise VerificationError("The source README was not preserved in place.")
    expected_files = (
        set(source_entries)
        | {SOURCE_MANIFEST_NAME, INTERNAL_MANIFEST_NAME}
        | set(GENERATED_FILES)
    )
    if set(files) != expected_files:
        raise VerificationError(
            "Archive is not the exact source-manifest closure plus generated metadata."
        )
    source_only = internal.get("source_only_replay")
    if not isinstance(source_only, dict):
        raise VerificationError("Missing source-only replay metadata.")
    inventory_payload = files[SOURCE_ONLY_INVENTORY_NAME]
    if source_only.get("inventory_path") != SOURCE_ONLY_INVENTORY_NAME:
        raise VerificationError("Source-only replay inventory path mismatch.")
    if source_only.get("inventory_sha256") != _sha256_bytes(inventory_payload):
        raise VerificationError("Source-only replay inventory hash mismatch.")
    if source_only.get("required_for_current_release_acceptance") is not True:
        raise VerificationError("Source-only preparation is not required by the release.")
    source_only_inventory = _json_object(
        inventory_payload, label="source-only replay inventory"
    )
    if (
        source_only_inventory.get("schema_version") != 1
        or source_only_inventory.get("bundle_root_name") != BUNDLE_ROOT
    ):
        raise VerificationError("Source-only replay inventory schema mismatch.")
    inventory_rows = source_only_inventory.get("files")
    if not isinstance(inventory_rows, list):
        raise VerificationError("Source-only replay inventory has no file records.")
    inventory_paths: set[str] = set()
    external_input_count = 0
    for row in inventory_rows:
        if not isinstance(row, dict):
            raise VerificationError("Malformed source-only replay inventory record.")
        relative = _safe_relative_path(row.get("path"), label="source-only inventory path")
        label = relative.as_posix()
        if label in inventory_paths:
            raise VerificationError(f"Duplicate source-only inventory path: {label}.")
        inventory_paths.add(label)
        payload = files.get(label)
        if payload is None:
            raise VerificationError(f"Source-only inventory path is absent: {label}.")
        if row.get("bytes") != len(payload) or row.get("sha256") != _sha256_bytes(payload):
            raise VerificationError(f"Source-only inventory hash or size mismatch: {label}.")
        classification = row.get("classification")
        if classification not in {
            "generated_evidence", "generated_notebook",
            "immutable_source_or_metadata", "immutable_external_input",
        }:
            raise VerificationError(f"Unknown source-only classification: {label}.")
        external_input_count += classification == "immutable_external_input"
    expected_inventory_paths = set(files) - {
        SOURCE_ONLY_INVENTORY_NAME, INTERNAL_MANIFEST_NAME
    }
    if inventory_paths != expected_inventory_paths:
        raise VerificationError("Source-only replay inventory file-set closure mismatch.")
    if source_only.get("immutable_external_input_count") != external_input_count:
        raise VerificationError("Immutable external-input count mismatch.")
    for path_text, expected_hash in source_entries.items():
        payload = files.get(path_text)
        if payload is None or _sha256_bytes(payload) != expected_hash:
            raise VerificationError(f"Source-closure hash mismatch: {path_text}.")

    environment = internal.get("environment")
    if not isinstance(environment, dict):
        raise VerificationError("Missing environment metadata.")
    conda_lock = files[CONDA_LOCK_NAME]
    try:
        conda_lines = conda_lock.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise VerificationError("Conda explicit lock is not UTF-8.") from exc
    if "@EXPLICIT" not in conda_lines:
        raise VerificationError("Conda lock lacks @EXPLICIT marker.")
    if environment.get("conda_explicit_lock_path") != CONDA_LOCK_NAME:
        raise VerificationError("Conda lock path metadata mismatch.")
    if environment.get("conda_explicit_lock_sha256") != _sha256_bytes(conda_lock):
        raise VerificationError("Conda lock hash metadata mismatch.")
    versions = _json_object(
        files[SELECTED_VERSIONS_NAME], label="selected package versions"
    )
    if versions.get("schema_version") != 1 or versions.get(
        "packages"
    ) != environment.get("selected_package_versions"):
        raise VerificationError("Selected package version metadata mismatch.")
    requirements_payload = files.get(PIP_LOCK_NAME)
    if requirements_payload is None:
        raise VerificationError(f"The archive omits {PIP_LOCK_NAME}.")
    if environment.get("pip_requirements_path") != PIP_LOCK_NAME:
        raise VerificationError("Pip requirements path metadata mismatch.")
    if environment.get("pip_requirements_sha256") != _sha256_bytes(
        requirements_payload
    ):
        raise VerificationError("Pip requirements hash metadata mismatch.")
    pip_pins = _exact_pip_pins(requirements_payload)
    if environment.get("pip_requirements") != pip_pins:
        raise VerificationError("Pip requirements metadata mismatch.")
    selected_packages = environment.get("selected_package_versions")
    if not isinstance(selected_packages, dict):
        raise VerificationError("Selected package versions are not a mapping.")
    for name, version in pip_pins.items():
        if selected_packages.get(name) != version:
            raise VerificationError(
                f"Selected package version for {name} does not match its pip pin."
            )

    source_plan_path = internal.get("source_plan_path")
    effective_plan_path = internal.get("effective_plan_path")
    report_path = internal.get("spectral_report_path")
    if source_plan_path != PLAN_RELATIVE.as_posix():
        raise VerificationError("Unexpected source-plan path.")
    if effective_plan_path != EFFECTIVE_PLAN_NAME:
        raise VerificationError("Unexpected effective-plan path.")
    if report_path != SPECTRAL_REPORT_RELATIVE.as_posix():
        raise VerificationError("Unexpected spectral-report path.")
    source_plan = _json_object(files[source_plan_path], label="source plan")
    effective_plan = _json_object(files[effective_plan_path], label="effective plan")
    report = _json_object(files[report_path], label="spectral report")
    _validate_geometry_input_hashes(
        report,
        {
            key: _archive_payload(
                files,
                relative,
                label=f"Hardy-matrix geometry input {key}",
            )
            for key, relative in THEOREM_INPUT_RELATIVES.items()
        },
    )
    recomputed_plan = _refresh_plan(source_plan, report)
    if effective_plan != recomputed_plan:
        raise VerificationError(
            "Effective plan is not the deterministic report refresh of the source plan."
        )
    certificate_summary = _validate_plan_and_report(effective_plan, report)
    if internal.get("precision_settings") != effective_plan.get("precision_settings"):
        raise VerificationError(
            "Internal precision settings differ from effective plan."
        )
    plan_refresh = internal.get("plan_refresh")
    if not isinstance(plan_refresh, dict):
        raise VerificationError("Missing plan refresh provenance.")
    if plan_refresh.get("source_plan_sha256") != _sha256_bytes(files[source_plan_path]):
        raise VerificationError("Source-plan refresh hash mismatch.")
    if plan_refresh.get("effective_plan_sha256") != _sha256_bytes(
        files[effective_plan_path]
    ):
        raise VerificationError("Effective-plan refresh hash mismatch.")

    notebook = _json_object(files[NOTEBOOK_RELATIVE.as_posix()], label="notebook")
    notebook_summary = _validate_notebook(notebook)
    recorded_notebook = internal.get("notebook_validation")
    if not isinstance(recorded_notebook, dict):
        raise VerificationError("Missing notebook validation record.")
    for key, value in notebook_summary.items():
        if recorded_notebook.get(key) != value:
            raise VerificationError(f"Notebook validation mismatch for {key}.")
    if recorded_notebook.get("inline_helper_sync") is not True:
        raise VerificationError("Inline helper sync was not recorded as validated.")

    builder_chain = internal.get("builder_chain")
    if not isinstance(builder_chain, dict):
        raise VerificationError("Missing builder-chain metadata.")
    for path_key, hash_key, expected_path in (
        (
            "locked_template_path",
            "locked_template_sha256",
            TEMPLATE_NOTEBOOK_RELATIVE.as_posix(),
        ),
        (
            "source_notebook_path",
            "source_notebook_sha256",
            SOURCE_NOTEBOOK_RELATIVE.as_posix(),
        ),
        (
            "executed_counterpart_path",
            "executed_counterpart_sha256",
            NOTEBOOK_RELATIVE.as_posix(),
        ),
    ):
        if builder_chain.get(path_key) != expected_path:
            raise VerificationError(f"Builder path mismatch for {path_key}.")
        if builder_chain.get(hash_key) != _sha256_bytes(files[expected_path]):
            raise VerificationError(f"Builder hash mismatch for {hash_key}.")

    artifacts = internal.get("upstream_artifacts")
    effective_names = effective_plan.get("upstream_artifact_names")
    if not isinstance(artifacts, list) or not isinstance(effective_names, list):
        raise VerificationError("Malformed upstream artifact metadata.")
    if len(artifacts) != len(effective_names):
        raise VerificationError("Upstream artifact count mismatch.")
    for name, artifact in zip(effective_names, artifacts, strict=True):
        if not isinstance(name, str) or not isinstance(artifact, dict):
            raise VerificationError("Malformed upstream artifact record.")
        safe_name = _safe_relative_path(name, label="upstream artifact name")
        expected_path = (OUTPUT_RELATIVE / "data" / safe_name).as_posix()
        payload = files.get(expected_path)
        if payload is None:
            raise VerificationError(f"Missing upstream artifact: {expected_path}.")
        if (
            artifact.get("name") != name
            or artifact.get("relative_data_path") != expected_path
        ):
            raise VerificationError("Non-deterministic upstream artifact path.")
        if artifact.get("bytes") != len(payload) or artifact.get(
            "sha256"
        ) != _sha256_bytes(payload):
            raise VerificationError(f"Upstream artifact hash mismatch: {name}.")

    if compare_replay_root is not None:
        _compare_replay_root(
            Path(compare_replay_root), source_manifest_payload, source_entries
        )

    semantic_replay_summary = None
    if compare_executed_replay_root is not None:
        semantic_replay_summary = _compare_executed_replay_root(
            Path(compare_executed_replay_root),
            files=files,
            source_entries=source_entries,
            archive_effective_plan=effective_plan,
            archive_report=report,
        )

    return {
        "archive_path": str(archive_path),
        "archive_sha256": archive_hash,
        "archive_bytes": archive_size,
        "tar_member_count": tar_member_count,
        "regular_file_count": len(files),
        "source_manifest_entry_count": len(source_entries),
        "certificate_schema": certificate_summary["certificate_schema"],
        "target_count": EXPECTED_TARGET_COUNT,
        "total_algebraic_multiplicity": EXPECTED_MULTIPLICITY,
        "notebook_cell_count": notebook_summary["cell_count"],
        "executed_code_cell_count": notebook_summary["executed_code_cell_count"],
        "replay_root_compared": compare_replay_root is not None,
        "executed_replay_root_compared": (compare_executed_replay_root is not None),
        "semantic_replay_summary": semantic_replay_summary,
        "verification_status": "verified",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--external-manifest", type=Path)
    parser.add_argument("--checksum", type=Path)
    parser.add_argument("--compare-replay-root", type=Path)
    parser.add_argument("--compare-executed-replay-root", type=Path)
    arguments = parser.parse_args()
    try:
        result = verify_bundle(
            arguments.archive,
            external_manifest_path=arguments.external_manifest,
            checksum_path=arguments.checksum,
            compare_replay_root=arguments.compare_replay_root,
            compare_executed_replay_root=(arguments.compare_executed_replay_root),
        )
    except VerificationError as exc:
        parser.exit(1, f"Verification failed: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
