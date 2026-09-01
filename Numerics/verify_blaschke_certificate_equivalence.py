"""Verify the theorem-critical meaning of a Blaschke certificate replay.

This verifier intentionally does not compare whole files byte for byte.  It
authenticates the immutable release sources first, then reconstructs the
certificate's discrete and one-sided numerical implications from JSON, CSV,
and the exact-binary Schur cache.  It never imports a bundled project module
and never loads a pickle.

``artifact-only`` mode is a quick consistency review of stored evidence.  It
is not an independent theorem proof.  ``full-replay`` mode additionally
requires the successful clean-room compute receipt and is the only mode that
may emit ``CERTIFICATION_CONFIRMED``.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
from typing import Iterable, Mapping, Sequence


POLICY_RELATIVE = PurePosixPath("release/blaschke-certificate-semantic-policy.json")
RELEASE_INVENTORY_RELATIVE = PurePosixPath("release/source-only-replay-inventory.json")
RELEASE_MANIFEST_RELATIVE = PurePosixPath("release/reproducibility_manifest.json")
ROOT_INVENTORY_RELATIVE = PurePosixPath("source-only-replay-inventory.json")
ROOT_MANIFEST_RELATIVE = PurePosixPath("reproducibility_manifest.json")
COMPUTE_EVIDENCE_RELATIVE = PurePosixPath("clean-room-compute-only-evidence.json")
RUN_ATTESTATION_RELATIVE = PurePosixPath("certificate-replay-run-attestation.json")

CANONICAL_HARDY_RADIUS = "2.473669807791324109321273260"
CANONICAL_Q_GAP_TARGET = "0.927"
EXPECTED_CONFIGURATION = {
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
    "hardy_audit_precision_bits": 1024,
    "hardy_production_precision_bits": 2048,
}
EXPECTED_TARGET_ORDER = (
    "alpha^1", "alpha^2", "mu^1", "alpha^3", "alpha^4", "alpha^5",
    "mu^2", "alpha^6", "alpha^7", "alpha^8", "mu^3", "alpha^9",
    "alpha^10", "alpha^11", "mu^4", "alpha^12", "alpha^13", "mu^5",
    "alpha^14", "alpha^15", "alpha^16", "mu^6", "alpha^17", "alpha^18",
)
EXPECTED_MULTIPLICITIES = (
    1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1,
    1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1,
)
EXPECTED_TRUE_REPORT_GATES = (
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
    "exact_dyadic_schur_below_diagonal_all_zero",
    "exact_dyadic_schur_upper_triangular_certified",
)
EXPECTED_FALSE_REPORT_GATES = (
    "sampled_values_used_in_any_theorem_gate",
    "laurent_digests_used_in_any_theorem_gate",
)
LAURENT_RADIUS_FRACTIONS = {
    "mu^5": Fraction(1, 4),
    "alpha^14": Fraction(9, 50),
    "alpha^15": Fraction(1, 5),
    "alpha^16": Fraction(1, 5),
    "mu^6": Fraction(1, 4),
    "alpha^17": Fraction(1, 4),
    "alpha^18": Fraction(1, 5),
}
EXPECTED_THEOREM_REQUIRED_PATHS = (
    "Numerics/outputs/blaschke_deformation_certifier/data/branch_image_radius_reoptimisation_balanced_highcell_scan.csv",
    "Numerics/outputs/blaschke_deformation_certifier/reports/branch_image_radius_reoptimisation_balanced_highcell_scan.json",
    "Numerics/outputs/blaschke_deformation_certifier/data/branch_image_balanced_candidate_transport_cert_N600.csv",
    "Numerics/outputs/blaschke_deformation_certifier/reports/branch_image_balanced_candidate_transport_cert_N600.json",
    "Numerics/outputs/blaschke_deformation_certifier/data/branch_image_balanced_candidate_transport_inverse_witness_N600.npz",
    "Numerics/outputs/blaschke_deformation_certifier/data/branch_image_balanced_candidate_single_space_row_N600_M610.csv",
    "Numerics/outputs/blaschke_deformation_certifier/reports/branch_image_balanced_candidate_single_space_row_N600_M610.json",
    "Numerics/outputs/blaschke_deformation_certifier/reports/phase2_clean_room_rebuild_manifest.json",
    "Numerics/outputs/blaschke_deformation_certifier/data/branch_image_balanced_candidate_single_space_row_N600_M610_safe_GL.csv",
    "Numerics/outputs/blaschke_deformation_certifier/data/output_response_branch_image_prefactor_interval_cert_balanced_N600.csv",
    "Numerics/outputs/blaschke_deformation_certifier/data/output_response_coherent_packet_interval_cert_balanced_N600.csv",
    "Numerics/outputs/blaschke_deformation_certifier/reports/output_response_branch_image_prefactor_interval_cert_balanced_N600.json",
    "Numerics/outputs/blaschke_deformation_certifier/data/blaschke_deformation_input_tail_certificate_N600.csv",
    "Numerics/outputs/blaschke_deformation_certifier/reports/blaschke_deformation_input_tail_certificate_N600.json",
    "Numerics/outputs/blaschke_deformation_certifier/data/branch_image_balanced_response_prefactor_candidate_row_N600_M610.csv",
    "Numerics/outputs/blaschke_deformation_certifier/data/branch_image_phase2_certified_single_space_row_N600_M610.csv",
    "Numerics/outputs/blaschke_deformation_certifier/data/blaschke_deformation_balanced_hardy_reference_N600_M610_bits2048_certificate.csv",
    "Numerics/outputs/blaschke_deformation_certifier/reports/blaschke_deformation_balanced_hardy_reference_N600_M610_bits2048_certificate.json",
    "Numerics/outputs/blaschke_deformation_certifier/data/blaschke_deformation_balanced_hardy_reference_N600_M610_bits2048.pkl.gz",
    "Numerics/outputs/blaschke_deformation_certifier/data/blaschke_deformation_balanced_hardy_reference_N600_M610_bits2048_diagnostic_midpoint.npz",
    "Numerics/outputs/blaschke_deformation_certifier/data/blaschke_deformation_24_target_N600_M610_contour_plan.csv",
    "Numerics/outputs/blaschke_deformation_certifier/data/blaschke_deformation_24_target_N600_M610_validated_schur.npz",
    "Numerics/outputs/blaschke_deformation_certifier/reports/blaschke_deformation_24_target_N600_M610_validated_schur.json",
    "Numerics/outputs/blaschke_deformation_certifier/data/blaschke_deformation_24_target_N600_M610_spectral_certificate.csv",
    "Numerics/outputs/blaschke_deformation_certifier/reports/blaschke_deformation_24_target_N600_M610_spectral_certificate.json",
    "Numerics/outputs/blaschke_deformation_certifier/data/blaschke_deformation_24_target_N600_M610_laurent_mode_bounds.csv",
    "Numerics/outputs/blaschke_deformation_certifier/data/blaschke_deformation_24_target_N600_M610_laurent_witness_reconstruction.csv",
)

ALLOWED_INVENTORY_CLASSES = frozenset(
    {
        "immutable_source_or_metadata",
        "immutable_external_input",
        "generated_evidence",
        "generated_notebook",
    }
)
IMMUTABLE_INVENTORY_CLASSES = frozenset(
    {"immutable_source_or_metadata", "immutable_external_input"}
)

TRUE_ROW_GATES = (
    "zero_outside_enclosed_region",
    "schur_diagonal_membership_certified",
    "finite_count_matches_expected",
    "A_N_circ_count_transport_certified",
    "mathematical_finite_count_transport_certified",
    "finite_count_certified",
    "certified_small_gain_pass",
    "finite_to_exact_rank_transfer_certified",
    "complete_circle_covered",
    "theorem_certified",
)

PHASE2_FINAL_GATES = (
    "response_boundary_cover_certified",
    "response_coherent_prefix_certified",
    "response_remainder_certified",
    "transport_certified",
    "tail_mismatch_certified",
    "matrix_certified",
    "branch_data_certified",
    "response_prefactor_certified",
    "finite_M_prefactor_certified",
    "certified",
)

SOURCE_PATHS = {
    "blaschke_deformation_contour_certification.py": PurePosixPath(
        "Numerics/blaschke_deformation_contour_certification.py"
    ),
    "blaschke_deformation_spectral_certification.py": PurePosixPath(
        "Numerics/blaschke_deformation_spectral_certification.py"
    ),
    "blaschke_deformation_certification.py": PurePosixPath(
        "Numerics/blaschke_deformation_certification.py"
    ),
    "blaschke_deformation_phase2_final_aggregation.py": PurePosixPath(
        "Numerics/blaschke_deformation_phase2_final_aggregation.py"
    ),
}


class CertificateEquivalenceError(RuntimeError):
    """Raised when a theorem-critical semantic condition does not hold."""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_relative(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CertificateEquivalenceError(f"{label} must be a non-empty path string.")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or value in {".", ".."}
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise CertificateEquivalenceError(f"Unsafe {label}: {value!r}.")
    return relative


def _safe_regular_file(root: Path, relative: PurePosixPath, *, label: str) -> Path:
    candidate = root
    for index, part in enumerate(relative.parts):
        candidate = candidate / part
        if candidate.is_symlink():
            raise CertificateEquivalenceError(
                f"Symlink forbidden for {label}: {relative.as_posix()}."
            )
        if not candidate.exists():
            raise CertificateEquivalenceError(
                f"Missing {label}: {relative.as_posix()}."
            )
        if index < len(relative.parts) - 1 and not candidate.is_dir():
            raise CertificateEquivalenceError(
                f"Non-directory component for {label}: {relative.as_posix()}."
            )
    try:
        mode = candidate.lstat().st_mode
        candidate.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise CertificateEquivalenceError(
            f"{label} escapes or cannot be resolved inside the release root."
        ) from exc
    if not stat.S_ISREG(mode):
        raise CertificateEquivalenceError(
            f"{label} is not a regular file: {relative.as_posix()}."
        )
    return candidate


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CertificateEquivalenceError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CertificateEquivalenceError(f"{label} must be a JSON object.")
    return value


def _load_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise CertificateEquivalenceError(
                    f"{label} has no header or has duplicate columns."
                )
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise CertificateEquivalenceError(f"Cannot read {label}: {exc}") from exc
    if not rows:
        raise CertificateEquivalenceError(f"{label} has no data rows.")
    return rows


def _one_row(rows: Sequence[dict[str, str]], *, label: str) -> dict[str, str]:
    if len(rows) != 1:
        raise CertificateEquivalenceError(
            f"{label} must contain exactly one row; observed {len(rows)}."
        )
    return rows[0]


def _decimal(value: object, *, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise CertificateEquivalenceError(f"{label} is not a decimal scalar.")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CertificateEquivalenceError(f"{label} is not a valid decimal.") from exc
    if not result.is_finite():
        raise CertificateEquivalenceError(f"{label} must be finite.")
    return result


def _integer(value: object, *, label: str) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str) and value:
        try:
            parsed = int(value)
        except ValueError:
            parsed = None
        if parsed is not None and value == str(parsed):
            return parsed
    raise CertificateEquivalenceError(f"{label} must be an exact integer.")


def _json_bool(mapping: Mapping[str, object], key: str, expected: bool) -> None:
    observed = mapping.get(key)
    if type(observed) is not bool or observed is not expected:
        raise CertificateEquivalenceError(
            f"Expected {key}={expected}, observed {observed!r}."
        )


def _csv_bool(mapping: Mapping[str, str], key: str, expected: bool) -> None:
    observed = mapping.get(key)
    expected_text = "True" if expected else "False"
    if observed != expected_text:
        raise CertificateEquivalenceError(
            f"Expected CSV {key}={expected_text}, observed {observed!r}."
        )


def _fraction_decimal(value: object, *, label: str) -> Fraction:
    return Fraction(_decimal(value, label=label))


def _fraction_text(value: object, *, label: str) -> Fraction:
    if not isinstance(value, str) or not value:
        raise CertificateEquivalenceError(f"{label} must be a rational string.")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise CertificateEquivalenceError(f"{label} is not a rational string.") from exc
    return result


def _fraction_decimal_display(value: Fraction, *, upper: bool) -> Decimal:
    """Render an exact rational in a declared one-sided Decimal direction."""

    with localcontext() as context:
        context.prec = 100
        context.rounding = ROUND_CEILING if upper else ROUND_FLOOR
        return Decimal(value.numerator) / Decimal(value.denominator)


def _artifact_relative(policy: Mapping[str, object], name: str) -> PurePosixPath:
    paths = policy.get("artifact_paths")
    if not isinstance(paths, dict) or name not in paths:
        raise CertificateEquivalenceError(f"Semantic policy omits artifact path {name!r}.")
    return _safe_relative(paths[name], label=f"policy artifact path {name}")


def _artifact_file(
    root: Path, policy: Mapping[str, object], name: str, *, label: str | None = None
) -> Path:
    return _safe_regular_file(
        root,
        _artifact_relative(policy, name),
        label=label or name.replace("_", " "),
    )


def _load_policy(root: Path) -> dict[str, object]:
    policy = _load_json(
        _safe_regular_file(root, POLICY_RELATIVE, label="semantic policy"),
        label="semantic policy",
    )
    if (
        policy.get("schema_version") != 1
        or policy.get("policy_id") != "blaschke-certificate-equivalence-v1"
        or policy.get("map_label") != "blaschke_mu_0p3"
    ):
        raise CertificateEquivalenceError("Unsupported semantic policy schema.")
    dimensions = policy.get("dimensions")
    if dimensions != {"N": 600, "M": 610}:
        raise CertificateEquivalenceError("The semantic policy dimensions are invalid.")
    if policy.get("configuration") != EXPECTED_CONFIGURATION:
        raise CertificateEquivalenceError("The semantic configuration policy drifted.")
    if (
        policy.get("canonical_hardy_radius") != CANONICAL_HARDY_RADIUS
        or policy.get("q_gap_target") != CANONICAL_Q_GAP_TARGET
    ):
        raise CertificateEquivalenceError("The canonical radius/q-gap policy drifted.")
    if policy.get("certificate_schemas") != {
        "hardy_matrix": "blaschke-deformation-exact-dyadic-hardy-reference-v2",
        "validated_schur": "blaschke-deformation-exact-dyadic-schur-v2",
        "spectral_contours": "blaschke-deformation-24-contour-hybrid-v4",
    }:
        raise CertificateEquivalenceError("The semantic certificate schema policy drifted.")
    names = policy.get("target_order")
    multiplicities = policy.get("expected_multiplicities")
    if (
        names != list(EXPECTED_TARGET_ORDER)
        or multiplicities != list(EXPECTED_MULTIPLICITIES)
        or policy.get("route_counts")
        != {"Schur-count/Schur-moat": 17, "Schur-count/Laurent-moat": 7}
        or policy.get("true_report_gates") != list(EXPECTED_TRUE_REPORT_GATES)
        or policy.get("false_report_gates") != list(EXPECTED_FALSE_REPORT_GATES)
        or policy.get("theorem_required_generated_paths")
        != list(EXPECTED_THEOREM_REQUIRED_PATHS)
    ):
        raise CertificateEquivalenceError("The fixed semantic theorem policy drifted.")
    return policy


def _release_metadata_paths(root: Path) -> tuple[Path, Path]:
    release_inventory = root.joinpath(*RELEASE_INVENTORY_RELATIVE.parts)
    release_manifest = root.joinpath(*RELEASE_MANIFEST_RELATIVE.parts)
    if release_inventory.is_file() and release_manifest.is_file():
        return release_inventory, release_manifest
    return (
        _safe_regular_file(root, ROOT_INVENTORY_RELATIVE, label="staged inventory"),
        _safe_regular_file(root, ROOT_MANIFEST_RELATIVE, label="staged manifest"),
    )


def _verify_git_identity(root: Path, expected_git_commit: str) -> dict[str, str]:
    if (
        len(expected_git_commit) != 40
        or expected_git_commit.lower() != expected_git_commit
        or any(character not in "0123456789abcdef" for character in expected_git_commit)
    ):
        raise CertificateEquivalenceError(
            "Expected Git commit must be a lowercase 40-hex object ID."
        )
    if not (root / ".git").exists():
        raise CertificateEquivalenceError(
            "Git identity was selected but the release root is not a Git checkout."
        )
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD^{commit}"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CertificateEquivalenceError(f"Cannot authenticate Git identity: {exc}") from exc
    if head != expected_git_commit or dirty:
        raise CertificateEquivalenceError(
            "Git release identity mismatch or non-ignored changes are present."
        )
    return {"authority": "caller-supplied Git commit", "value": head}


def verify_release_integrity(
    root: Path,
    *,
    expected_inventory_sha256: str | None,
    expected_git_commit: str | None,
) -> dict[str, object]:
    """Authenticate release metadata and every immutable source before imports."""

    inventory_path, manifest_path = _release_metadata_paths(root)
    if inventory_path.is_symlink() or manifest_path.is_symlink():
        raise CertificateEquivalenceError("Release metadata must not be symlinks.")
    inventory_payload = inventory_path.read_bytes()
    inventory_sha256 = _sha256_bytes(inventory_payload)
    selected = sum(
        value is not None for value in (expected_inventory_sha256, expected_git_commit)
    )
    if selected != 1:
        raise CertificateEquivalenceError(
            "Exactly one externally trusted inventory digest or Git commit is required."
        )
    if expected_inventory_sha256 is not None:
        if (
            not _is_sha256(expected_inventory_sha256)
            or expected_inventory_sha256 != inventory_sha256
        ):
            raise CertificateEquivalenceError(
                "Release inventory does not match the externally trusted digest."
            )
        external_identity = {
            "authority": "caller-supplied inventory SHA-256",
            "value": inventory_sha256,
        }
    else:
        assert expected_git_commit is not None
        external_identity = _verify_git_identity(root, expected_git_commit)
    try:
        inventory = json.loads(inventory_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CertificateEquivalenceError("Cannot decode release inventory.") from exc
    if not isinstance(inventory, dict):
        raise CertificateEquivalenceError("Release inventory must be a JSON object.")
    manifest = _load_json(manifest_path, label="release reproducibility manifest")

    manifest_schema = manifest.get("schema_version")
    if manifest_schema == 1:
        if (
            manifest.get("release_unit") != "git-tree"
            or manifest.get("inventory_sha256") != inventory_sha256
        ):
            raise CertificateEquivalenceError(
                "The Git-tree release manifest does not authenticate its inventory."
            )
    elif manifest_schema == 3:
        records = manifest.get("files")
        matches = [
            row
            for row in records
            if isinstance(row, dict)
            and row.get("archive_path") == ROOT_INVENTORY_RELATIVE.as_posix()
        ] if isinstance(records, list) else []
        if (
            manifest.get("bundle_format")
            != "blaschke-deformation-certifier-reproducibility-v3"
            or len(matches) != 1
            or matches[0].get("sha256") != inventory_sha256
        ):
            raise CertificateEquivalenceError(
                "The staged replay manifest does not authenticate its inventory."
            )
    else:
        raise CertificateEquivalenceError("Unsupported release manifest schema.")

    if (
        inventory.get("schema_version") != 1
        or inventory.get("bundle_root_name")
        != "blaschke_deformation_certifier_reproducibility"
    ):
        raise CertificateEquivalenceError("Unsupported source-only inventory schema.")
    rows = inventory.get("files")
    if not isinstance(rows, list) or not rows:
        raise CertificateEquivalenceError("The release inventory is empty.")
    seen: set[PurePosixPath] = set()
    immutable_count = 0
    observed_paths: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise CertificateEquivalenceError("Malformed release inventory row.")
        relative = _safe_relative(row.get("path"), label="release inventory path")
        if relative in seen:
            raise CertificateEquivalenceError(
                f"Duplicate release inventory path: {relative.as_posix()}."
            )
        seen.add(relative)
        observed_paths.append(relative.as_posix())
        classification = row.get("classification")
        size = row.get("bytes")
        digest = row.get("sha256")
        if (
            classification not in ALLOWED_INVENTORY_CLASSES
            or type(size) is not int
            or size < 0
            or not _is_sha256(digest)
        ):
            raise CertificateEquivalenceError(
                f"Malformed release inventory record for {relative.as_posix()}."
            )
        if classification in IMMUTABLE_INVENTORY_CLASSES:
            candidate = _safe_regular_file(
                root, relative, label="immutable release source"
            )
            if candidate.stat().st_size != size or _sha256_file(candidate) != digest:
                raise CertificateEquivalenceError(
                    f"Immutable release source drift: {relative.as_posix()}."
                )
            immutable_count += 1
    if observed_paths != sorted(observed_paths):
        raise CertificateEquivalenceError("Release inventory paths are not sorted.")
    policy_path = _safe_regular_file(
        root, PurePosixPath("Numerics/blaschke_source_only_replay_policy.json"),
        label="source-only replay policy",
    )
    if (
        inventory.get("policy_path")
        != "Numerics/blaschke_source_only_replay_policy.json"
        or inventory.get("policy_sha256") != _sha256_file(policy_path)
    ):
        raise CertificateEquivalenceError("Source-only replay policy binding failed.")
    required_immutable = {
        POLICY_RELATIVE,
        PurePosixPath("Numerics/verify_blaschke_certificate_equivalence.py"),
        PurePosixPath("Numerics/run_blaschke_certificate_reproduction.py"),
        PurePosixPath("Numerics/stage_blaschke_certificate_replay.py"),
        PurePosixPath("Numerics/run_blaschke_clean_room_replay.py"),
        PurePosixPath("conda-explicit-lock.txt"),
        PurePosixPath("pip-requirements-lock.txt"),
    }
    missing_required = sorted(path.as_posix() for path in required_immutable - seen)
    if missing_required:
        raise CertificateEquivalenceError(
            f"Release inventory omits required immutable sources: {missing_required}."
        )
    return {
        "inventory_sha256": inventory_sha256,
        "inventory_record_count": len(rows),
        "immutable_record_count": immutable_count,
        "external_identity": external_identity,
    }


def _validate_hash_binding(
    root: Path, relative: PurePosixPath, expected: object, *, label: str
) -> str:
    if not _is_sha256(expected):
        raise CertificateEquivalenceError(f"{label} is not a lowercase SHA-256.")
    observed = _sha256_file(_safe_regular_file(root, relative, label=label))
    if observed != expected:
        raise CertificateEquivalenceError(
            f"Internal hash binding failed for {label}: {observed} != {expected}."
        )
    return observed


def _validate_source_hashes(root: Path, report: Mapping[str, object], *, label: str) -> None:
    records = report.get("source_hashes")
    if not isinstance(records, dict) or not records:
        raise CertificateEquivalenceError(f"{label} has no source_hashes object.")
    for name, digest in records.items():
        if name not in SOURCE_PATHS:
            raise CertificateEquivalenceError(
                f"{label} names an unexpected theorem source: {name!r}."
            )
        _validate_hash_binding(root, SOURCE_PATHS[name], digest, label=f"{label} source {name}")


def _validate_named_output_hashes(
    root: Path,
    records: object,
    *,
    label: str,
) -> int:
    if not isinstance(records, dict) or not records:
        raise CertificateEquivalenceError(f"{label} output binding map is malformed.")
    count = 0
    for value, digest in records.items():
        relative = _safe_relative(
            value, label=f"{label} output path"
        )
        _validate_hash_binding(root, relative, digest, label=f"{label} output")
        count += 1
    return count


def _validate_internal_bindings(
    root: Path,
    policy: Mapping[str, object],
    spectral: Mapping[str, object],
    schur: Mapping[str, object],
    hardy_2048: Mapping[str, object],
    phase2_manifest: Mapping[str, object],
    resolved: Mapping[str, object],
) -> dict[str, object]:
    _validate_source_hashes(root, spectral, label="spectral report")
    _validate_source_hashes(root, schur, label="validated Schur report")
    _validate_source_hashes(root, hardy_2048, label="2048-bit Hardy report")

    _validate_hash_binding(
        root,
        _artifact_relative(policy, "validated_schur_npz"),
        schur.get("cache_sha256"),
        label="validated Schur NPZ",
    )
    if schur.get("matrix_payload_sha256") != hardy_2048.get("payload_sha256"):
        raise CertificateEquivalenceError("Schur/Hardy payload bindings disagree.")
    if schur.get("midpoint_sha256") != hardy_2048.get("midpoint_sha256"):
        raise CertificateEquivalenceError("Schur/Hardy midpoint bindings disagree.")

    for bits, report in ((2048, hardy_2048),):
        _validate_hash_binding(
            root,
            _artifact_relative(policy, f"hardy_{bits}_payload"),
            report.get("payload_sha256"),
            label=f"{bits}-bit Hardy exact-dyadic payload",
        )

    input_hashes = spectral.get("input_hashes")
    geometry_hashes = spectral.get("geometry_input_hashes")
    if not isinstance(input_hashes, dict) or not isinstance(geometry_hashes, dict):
        raise CertificateEquivalenceError("Spectral input hash maps are malformed.")
    fixed_inputs = {
        "epsilon_certificate": _artifact_relative(policy, "phase2_final_candidate"),
        "matrix_midpoint": _artifact_relative(policy, "hardy_2048_midpoint"),
        "matrix_payload": _artifact_relative(policy, "hardy_2048_payload"),
        "matrix_report": _artifact_relative(policy, "hardy_2048_report"),
    }
    for key, relative in fixed_inputs.items():
        _validate_hash_binding(
            root, relative, input_hashes.get(key), label=f"spectral input {key}"
        )
        if key != "epsilon_certificate" and geometry_hashes.get(key) != input_hashes.get(key):
            raise CertificateEquivalenceError(
                f"Spectral geometry/input binding mismatch for {key}."
            )

    phase2_output_count = _validate_named_output_hashes(
        root, phase2_manifest.get("outputs"), label="Phase 2 clean-room manifest"
    )
    resolved_inputs = resolved.get("inputs")
    if not isinstance(resolved_inputs, dict) or not resolved_inputs:
        raise CertificateEquivalenceError("Resolved-response input bindings are malformed.")
    resolved_count = 0
    for value, digest in resolved_inputs.items():
        if not isinstance(value, str):
            raise CertificateEquivalenceError("Resolved-response input path is malformed.")
        if value == "blaschke_deformation_certification.py":
            relative = SOURCE_PATHS[value]
        else:
            relative = _safe_relative(value, label="resolved-response input")
        _validate_hash_binding(root, relative, digest, label="resolved-response input")
        resolved_count += 1
    return {
        "phase2_bound_output_count": phase2_output_count,
        "resolved_bound_input_count": resolved_count,
    }


def _validate_hardy_report(
    root: Path,
    policy: Mapping[str, object],
    report: Mapping[str, object],
    *,
    bits: int,
    expected_ready: bool,
) -> dict[str, object]:
    schemas = policy.get("certificate_schemas")
    assert isinstance(schemas, dict)
    expected_rho = EXPECTED_CONFIGURATION["rho"]
    if (
        report.get("schema") != schemas.get("hardy_matrix")
        or report.get("map_label") != policy.get("map_label")
        or _integer(report.get("N"), label=f"Hardy {bits} N") != 600
        or _integer(report.get("M"), label=f"Hardy {bits} M") != 610
        or _decimal(report.get("rho"), label=f"Hardy {bits} rho")
        != Decimal(str(expected_rho))
        or _integer(report.get("precision_bits"), label=f"Hardy {bits} precision") != bits
        or _integer(report.get("entry_count"), label=f"Hardy {bits} entry count")
        != 360000
    ):
        raise CertificateEquivalenceError(f"Malformed {bits}-bit Hardy certificate report.")
    for gate in (
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
    ):
        _json_bool(report, gate, True)
    _json_bool(report, "spectral_use_ready", expected_ready)
    _json_bool(report, "precision_budget_pass", expected_ready)
    _json_bool(report, "precision_budget_enforced", expected_ready)
    eta = _decimal(report.get("eta_A_upper_text"), label=f"Hardy {bits} eta_A")
    if eta <= 0:
        raise CertificateEquivalenceError(f"Hardy {bits} eta_A is not positive.")
    summary = _one_row(
        _load_csv(
            _artifact_file(root, policy, f"hardy_{bits}_summary"),
            label=f"Hardy {bits} summary",
        ),
        label=f"Hardy {bits} summary",
    )
    for key, expected in (
        ("schema", str(schemas.get("hardy_matrix"))),
        ("map_label", str(policy.get("map_label"))),
        ("N", "600"),
        ("M", "610"),
        ("rho", str(expected_rho)),
        ("precision_bits", str(bits)),
        ("entry_count", "360000"),
    ):
        if summary.get(key) != expected:
            raise CertificateEquivalenceError(
                f"Hardy {bits} report/summary mismatch for {key}."
            )
    if summary.get("payload_sha256") != report.get("payload_sha256"):
        raise CertificateEquivalenceError(f"Hardy {bits} payload summary mismatch.")
    if summary.get("midpoint_sha256") != report.get("midpoint_sha256"):
        raise CertificateEquivalenceError(f"Hardy {bits} midpoint summary mismatch.")
    if (
        _decimal(summary.get("r"), label=f"Hardy {bits} summary radius")
        != _decimal(report.get("r"), label=f"Hardy {bits} report radius")
        or _fraction_decimal(
            summary.get("eta_A_upper_text"), label=f"Hardy {bits} summary eta_A"
        )
        != _fraction_decimal(
            report.get("eta_A_upper_text"), label=f"Hardy {bits} report eta_A"
        )
    ):
        raise CertificateEquivalenceError(
            f"Hardy {bits} report/summary theorem primitive mismatch."
        )
    for gate in (
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
    ):
        _csv_bool(summary, gate, True)
    _csv_bool(summary, "spectral_use_ready", expected_ready)
    _csv_bool(summary, "precision_budget_pass", expected_ready)
    midpoint = _validate_hardy_midpoint_npz(root, policy, report, bits=bits)
    return {
        "eta_A_upper": eta,
        "radius": report.get("r"),
        "midpoint_sha256": midpoint["midpoint_sha256"],
        "diagnostic_midpoint_file_sha256": midpoint["file_sha256"],
    }


def _validate_hardy_midpoint_npz(
    root: Path,
    policy: Mapping[str, object],
    report: Mapping[str, object],
    *,
    bits: int,
) -> dict[str, str]:
    """Bind the safe diagnostic NPZ to the exact-dyadic midpoint identity."""

    try:
        import numpy as np
    except ImportError as exc:
        raise CertificateEquivalenceError(
            "NumPy from the locked environment is required to inspect the Hardy midpoint."
        ) from exc
    path = _artifact_file(root, policy, f"hardy_{bits}_midpoint")
    expected_fields = {
        "A_N_circ_binary64_diagnostic",
        "N",
        "M",
        "rho",
        "r",
        "precision_bits",
        "midpoint_sha256",
        "authoritative_exact_dyadic_payload",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_fields:
                raise CertificateEquivalenceError(
                    f"Hardy {bits} diagnostic midpoint NPZ schema drifted."
                )
            matrix = archive["A_N_circ_binary64_diagnostic"]
            if (
                matrix.shape != (600, 600)
                or matrix.dtype != np.dtype("float64")
                or not np.isfinite(matrix).all()
            ):
                raise CertificateEquivalenceError(
                    f"Hardy {bits} diagnostic midpoint matrix is malformed."
                )

            scalars: dict[str, object] = {}
            for key in expected_fields - {"A_N_circ_binary64_diagnostic"}:
                value = archive[key]
                if value.shape != ():
                    raise CertificateEquivalenceError(
                        f"Hardy {bits} midpoint field {key} is not scalar."
                    )
                scalars[key] = value.item()
    except (OSError, ValueError) as exc:
        raise CertificateEquivalenceError(
            f"Cannot safely inspect Hardy {bits} diagnostic midpoint NPZ: {exc}"
        ) from exc

    digest = scalars["midpoint_sha256"]
    payload_member = scalars["authoritative_exact_dyadic_payload"]
    expected_payload_name = _artifact_relative(
        policy, f"hardy_{bits}_payload"
    ).name
    if (
        _integer(scalars["N"], label=f"Hardy {bits} midpoint N") != 600
        or _integer(scalars["M"], label=f"Hardy {bits} midpoint M") != 610
        or _decimal(scalars["rho"], label=f"Hardy {bits} midpoint rho")
        != Decimal(str(EXPECTED_CONFIGURATION["rho"]))
        or _decimal(scalars["r"], label=f"Hardy {bits} midpoint radius")
        != Decimal(CANONICAL_HARDY_RADIUS)
        or _integer(
            scalars["precision_bits"], label=f"Hardy {bits} midpoint precision"
        )
        != bits
        or not _is_sha256(digest)
        or digest != report.get("midpoint_sha256")
        or not isinstance(payload_member, str)
        or Path(payload_member).name != expected_payload_name
    ):
        raise CertificateEquivalenceError(
            f"Hardy {bits} diagnostic midpoint identity drifted."
        )
    return {
        "midpoint_sha256": str(digest),
        # This outer container hash is deliberately distinct from the digest
        # above.  The current-run 27-artifact closure and the spectral input
        # map authenticate these bytes independently.
        "file_sha256": _sha256_file(path),
    }


def _validate_radius_contract(
    policy: Mapping[str, object], radius_values: Mapping[str, object]
) -> Decimal:
    canonical = _decimal(
        policy.get("canonical_hardy_radius"), label="canonical Hardy radius"
    )
    mismatches: list[str] = []
    for label, value in radius_values.items():
        observed = _decimal(value, label=f"{label} radius")
        if observed != canonical:
            mismatches.append(f"{label}={observed}")
    if mismatches:
        raise CertificateEquivalenceError(
            "Hardy-radius contract mismatch; expected exact Decimal "
            f"{canonical}, observed {mismatches}."
        )
    return canonical


def expected_target_geometry() -> list[dict[str, object]]:
    """Derive the fixed 24 exact rational target circles from alpha and mu."""

    alpha = Fraction(13, 20)
    mu = Fraction(3, 10)
    raw: list[tuple[Fraction, str, str, int, int]] = []
    raw.extend(
        (alpha**power, f"alpha^{power}", "alpha", power, 1)
        for power in range(1, 19)
    )
    raw.extend(
        (mu**power, f"mu^{power}", "mu", power, 2)
        for power in range(1, 7)
    )
    raw.sort(key=lambda item: item[0], reverse=True)
    contours: list[dict[str, object]] = []
    for rank, (centre, name, family, power, multiplicity) in enumerate(raw, 1):
        separation = min(
            abs(centre - other_centre)
            for other_centre, *_ in raw
            if other_centre != centre
        )
        radius_fraction = LAURENT_RADIUS_FRACTIONS.get(name, Fraction(1, 5))
        contours.append(
            {
                "rank": rank,
                "name": name,
                "family": family,
                "power": power,
                "expected_multiplicity": multiplicity,
                "centre": centre,
                "nearest_separation": separation,
                "radius_fraction": radius_fraction,
                "radius": radius_fraction * min(centre, separation),
            }
        )
    if tuple(row["name"] for row in contours) != EXPECTED_TARGET_ORDER:
        raise AssertionError("Internal exact target ordering contract is inconsistent.")
    if tuple(row["expected_multiplicity"] for row in contours) != EXPECTED_MULTIPLICITIES:
        raise AssertionError("Internal exact target multiplicity contract is inconsistent.")
    return contours


def _validate_exact_contour_geometry(
    plan_rows: Sequence[Mapping[str, str]],
    certificate_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    expected = expected_target_geometry()
    if len(plan_rows) != len(expected) or len(certificate_rows) != len(expected):
        raise CertificateEquivalenceError("Expected exactly 24 fixed target circles.")
    for derived, plan, certificate in zip(expected, plan_rows, certificate_rows):
        name = str(derived["name"])
        for row, label in ((plan, "contour plan"), (certificate, "certificate")):
            if (
                row.get("name") != name
                or row.get("family") != derived["family"]
                or _integer(row.get("power"), label=f"{name} {label} power")
                != derived["power"]
                or _fraction_text(row.get("centre_exact"), label=f"{name} centre")
                != derived["centre"]
                or _fraction_text(
                    row.get("radius_fraction_exact"),
                    label=f"{name} radius fraction",
                )
                != derived["radius_fraction"]
                or _fraction_text(row.get("radius_exact"), label=f"{name} radius")
                != derived["radius"]
            ):
                raise CertificateEquivalenceError(
                    f"Exact alpha/mu contour geometry drifted for {name} in {label}."
                )
        if (
            _fraction_text(
                plan.get("nearest_target_separation_exact"),
                label=f"{name} nearest separation",
            )
            != derived["nearest_separation"]
            or _fraction_text(
                plan.get("distance_to_zero_exact"),
                label=f"{name} exact zero distance",
            )
            != derived["centre"] - derived["radius"]
        ):
            raise CertificateEquivalenceError(
                f"Exact separation/zero geometry drifted for {name}."
            )
    for index, left in enumerate(expected):
        for right in expected[index + 1 :]:
            if abs(left["centre"] - right["centre"]) <= left["radius"] + right["radius"]:
                raise CertificateEquivalenceError(
                    f"Exact target contours are not pairwise disjoint: "
                    f"{left['name']} and {right['name']}."
                )
    return expected


def _validate_q_contract(
    final_candidate: Mapping[str, str],
    *,
    radius: Decimal,
    r_tau_value: object,
    q_target: Decimal,
    json_gate: bool = False,
) -> Fraction:
    ratio = _fraction_decimal(r_tau_value, label="r_tau upper") / Fraction(radius)
    target = Fraction(q_target)
    if ratio <= 0 or ratio > target:
        raise CertificateEquivalenceError(
            "The exact derived q ratio is not in (0, q_gap target]."
        )
    if _decimal(final_candidate.get("q_gap_target_text"), label="q-gap target text") != q_target:
        raise CertificateEquivalenceError("Final candidate q-gap target text drifted.")
    numerator = _integer(
        final_candidate.get("q_gap_derived_exact_numerator"),
        label="derived q exact numerator",
    )
    denominator = _integer(
        final_candidate.get("q_gap_derived_exact_denominator"),
        label="derived q exact denominator",
    )
    if denominator <= 0 or Fraction(numerator, denominator) != ratio:
        raise CertificateEquivalenceError("Stored exact q ratio does not match r_tau/r.")
    if json_gate:
        _json_bool(final_candidate, "q_gap_derived_le_target", True)
    else:
        _csv_bool(final_candidate, "q_gap_derived_le_target", True)
    displayed = _decimal(
        final_candidate.get("q_gap_derived_decimal_text"),
        label="derived q display",
    )
    if displayed <= 0 or displayed > q_target:
        raise CertificateEquivalenceError("Derived q display is outside (0, q_target].")
    return ratio


def validate_phase2_inequalities(
    *,
    b_out_value: object,
    b_in_value: object,
    collocation_value: object,
    epsilon_value: object,
    triangle_value: object,
) -> dict[str, Fraction]:
    """Validate Phase 2 RSS and triangle bounds as exact rational inequalities."""

    b_out = _fraction_decimal(b_out_value, label="Phase 2 B_out")
    b_in = _fraction_decimal(b_in_value, label="Phase 2 B_in")
    collocation = _fraction_decimal(
        collocation_value, label="Phase 2 collocation"
    )
    epsilon = _fraction_decimal(epsilon_value, label="Phase 2 epsilon")
    triangle = _fraction_decimal(triangle_value, label="Phase 2 triangle bound")
    if min(b_out, b_in, collocation) < 0 or epsilon < collocation:
        raise CertificateEquivalenceError("Phase 2 upper-bound inputs are invalid.")
    if (epsilon - collocation) ** 2 < b_out**2 + b_in**2:
        raise CertificateEquivalenceError(
            "Phase 2 epsilon fails the exact RSS-square upper-bound inequality."
        )
    if triangle < b_out + b_in + collocation:
        raise CertificateEquivalenceError(
            "Phase 2 triangle check is not a one-sided upper bound."
        )
    if epsilon > triangle:
        raise CertificateEquivalenceError("Phase 2 RSS epsilon exceeds its triangle check.")
    return {
        "B_out": b_out,
        "B_in": b_in,
        "collocation": collocation,
        "epsilon": epsilon,
        "triangle": triangle,
    }


def validate_phase2_candidate_binding(
    final_candidate: Mapping[str, str],
    final_row: Mapping[str, str],
) -> dict[str, Fraction]:
    """Bind the spectrally hashed candidate to the final exact inequalities."""

    epsilon = _fraction_decimal(
        final_row.get("epsilon_upper_text"), label="final epsilon"
    )
    triangle = _fraction_decimal(
        final_row.get("epsilon_triangle_upper_text"), label="final triangle"
    )
    candidate_epsilon = _fraction_decimal(
        final_candidate.get("new_epsilon_response_prefactor_candidate_text"),
        label="final-candidate epsilon",
    )
    candidate_triangle = _fraction_decimal(
        final_candidate.get("epsilon_triangle_check_text"),
        label="final-candidate triangle",
    )
    if candidate_epsilon != epsilon or candidate_triangle != triangle:
        raise CertificateEquivalenceError(
            "Final-candidate/final-certificate epsilon displays disagree."
        )
    final_b_out = _fraction_decimal(final_row.get("B_out"), label="final B_out")
    final_b_in = _fraction_decimal(final_row.get("B_in"), label="final B_in")
    final_collocation = _fraction_decimal(
        final_row.get("collocation"), label="final collocation"
    )
    if _fraction_decimal(
        final_candidate.get("B_out_response_prefactor_cert"),
        label="candidate B_out",
    ) != final_b_out:
        raise CertificateEquivalenceError("Final-candidate B_out binding drifted.")

    candidate_b_in_text = _fraction_decimal(
        final_candidate.get("B_in_selected_cert_text"),
        label="candidate B_in text",
    )
    if candidate_b_in_text != final_b_in:
        raise CertificateEquivalenceError(
            "Final-candidate B_in text does not exactly equal the final B_in primitive."
        )

    for candidate_key, final_value, label in (
        ("B_in_selected_cert_u", final_b_in, "B_in"),
        ("collocation_selected_cert_u", final_collocation, "collocation"),
        (
            "new_epsilon_response_prefactor_candidate",
            epsilon,
            "epsilon",
        ),
    ):
        expected = Decimal(repr(math.nextafter(float(final_value), math.inf)))
        observed = _decimal(
            final_candidate.get(candidate_key), label=f"candidate {label} serialization"
        )
        if observed != expected:
            raise CertificateEquivalenceError(
                f"Final-candidate {label} is not the exact nextafter serialization."
            )
    return validate_phase2_inequalities(
        b_out_value=final_row.get("B_out"),
        b_in_value=final_row.get("B_in"),
        collocation_value=final_row.get("collocation"),
        epsilon_value=final_candidate.get(
            "new_epsilon_response_prefactor_candidate_text"
        ),
        triangle_value=final_candidate.get("epsilon_triangle_check_text"),
    )


def validate_input_tail_configuration(
    policy: Mapping[str, object],
    report_summary: Mapping[str, object],
    csv_summary: Mapping[str, object],
) -> dict[str, object]:
    """Bind the input-tail JSON/CSV on the configuration its producer owns."""

    def normalized(record: Mapping[str, object], *, label: str) -> tuple[object, ...]:
        return (
            record.get("map_label"),
            _integer(record.get("N"), label=f"{label} N"),
            _decimal(record.get("rho"), label=f"{label} rho"),
            _decimal(record.get("r"), label=f"{label} r"),
            _decimal(record.get("mu"), label=f"{label} mu"),
            _integer(record.get("cells"), label=f"{label} cells"),
            _integer(
                record.get("precision_bits"), label=f"{label} precision bits"
            ),
            _integer(record.get("prefix_terms"), label=f"{label} prefix terms"),
            _integer(record.get("J"), label=f"{label} J"),
        )

    report_configuration = normalized(report_summary, label="input-tail report")
    csv_configuration = normalized(csv_summary, label="input-tail CSV")
    expected = (
        policy.get("map_label"),
        600,
        Decimal(str(EXPECTED_CONFIGURATION["rho"])),
        Decimal(CANONICAL_HARDY_RADIUS),
        Decimal("0.3"),
        65536,
        192,
        24,
        623,
    )
    if report_configuration != csv_configuration:
        raise CertificateEquivalenceError(
            "Input-tail JSON/CSV configuration identity failed."
        )
    if report_configuration != expected:
        raise CertificateEquivalenceError("Input-tail configuration drifted.")
    return {
        "map_label": report_configuration[0],
        "N": report_configuration[1],
        "rho": str(report_configuration[2]),
        "r": str(report_configuration[3]),
        "mu": str(report_configuration[4]),
        "cells": report_configuration[5],
        "precision_bits": report_configuration[6],
        "prefix_terms": report_configuration[7],
        "J": report_configuration[8],
    }


def validate_input_tail_provenance(
    root: Path,
    report: Mapping[str, object],
) -> dict[str, str]:
    """Bind input-tail provenance to exact release sources and profile target."""

    if (
        report.get("producer_helper")
        != "blaschke_deformation_phase2_final_aggregation.py"
        or report.get("producer_schema") != "phase2-final-aggregation-v3"
    ):
        raise CertificateEquivalenceError("Input-tail producer provenance drifted.")
    module_path = _safe_regular_file(
        root,
        SOURCE_PATHS["blaschke_deformation_certification.py"],
        label="input-tail certification module",
    )
    source_path = _safe_regular_file(
        root,
        SOURCE_PATHS["blaschke_deformation_phase2_final_aggregation.py"],
        label="input-tail producer source",
    )
    module_sha256 = report.get("module_sha256")
    source_sha256 = report.get("source_sha256")
    if (
        not _is_sha256(module_sha256)
        or module_sha256 != _sha256_file(module_path)
        or not _is_sha256(source_sha256)
        or source_sha256 != _sha256_file(source_path)
    ):
        raise CertificateEquivalenceError("Input-tail source hash binding failed.")

    profile_relative = PurePosixPath(
        "Numerics/outputs/blaschke_deformation_certifier/data/"
        "blaschke_deformation_input_tail_boundary_profile_N600.csv"
    )
    profile_value = report.get("profile_path")
    if profile_value != profile_relative.as_posix():
        raise CertificateEquivalenceError(
            "Input-tail profile path is not the canonical diagnostic member spelling."
        )
    return {
        "producer_helper": str(report["producer_helper"]),
        "producer_schema": str(report["producer_schema"]),
        "module_sha256": str(module_sha256),
        "source_sha256": str(source_sha256),
        "profile_path": profile_relative.as_posix(),
    }


def validate_input_tail_bounds(
    report_summary: Mapping[str, object],
    csv_summary: Mapping[str, object],
    final_candidate: Mapping[str, object],
    resolved_refresh: Mapping[str, object],
    final_row: Mapping[str, object],
) -> dict[str, Fraction]:
    """Bind input-tail primitives to the selected final one-sided B_in bound."""

    for gate in (
        "boundary_cover_certified",
        "exact_prefix_certified",
        "geometric_remainder_certified",
        "branchwise_profile_certified",
        "combined_row_certified",
    ):
        _json_bool(report_summary, gate, True)
        _csv_bool(csv_summary, gate, True)

    fields = (
        "branchwise_profile_cert_u",
        "combined_row_cert_u",
        "maximum_branch_radius_u",
    )
    report_values = {
        field: _fraction_decimal(
            report_summary.get(field), label=f"input-tail report {field}"
        )
        for field in fields
    }
    csv_values = {
        field: _fraction_decimal(
            csv_summary.get(field), label=f"input-tail CSV {field}"
        )
        for field in fields
    }
    if report_values != csv_values:
        raise CertificateEquivalenceError(
            "Input-tail JSON/CSV theorem primitives disagree."
        )
    branchwise = report_values["branchwise_profile_cert_u"]
    combined = report_values["combined_row_cert_u"]
    maximum_radius = report_values["maximum_branch_radius_u"]
    radius = Fraction(Decimal(CANONICAL_HARDY_RADIUS))
    if not (0 <= combined <= branchwise):
        raise CertificateEquivalenceError(
            "Input-tail combined bound is not enclosed by its branchwise bound."
        )
    if not (0 < maximum_radius < radius):
        raise CertificateEquivalenceError(
            "Input-tail maximum branch radius is outside (0, r)."
        )

    if any(
        record.get("B_in_selection") != "coherent_branchwise_intersection"
        for record in (final_candidate, resolved_refresh, final_row)
    ):
        raise CertificateEquivalenceError("Input-tail B_in selection route drifted.")
    final_b_in = _fraction_decimal(final_row.get("B_in"), label="final B_in")
    if (
        final_b_in < combined
        or _fraction_decimal(
            final_row.get("B_in_branchwise"), label="final branchwise B_in"
        )
        != branchwise
        or _fraction_decimal(
            final_row.get("B_in_combined_row"), label="final combined B_in"
        )
        != combined
    ):
        raise CertificateEquivalenceError(
            "Input-tail primitives do not one-sidedly bind the final B_in row."
        )
    candidate_b_in = _fraction_decimal(
        final_candidate.get("B_in_selected_cert_text"),
        label="candidate selected B_in text",
    )
    refresh_b_in = _fraction_decimal(
        resolved_refresh.get("B_in_selected_cert_text"),
        label="resolved selected B_in text",
    )
    if candidate_b_in != refresh_b_in or candidate_b_in != final_b_in:
        raise CertificateEquivalenceError(
            "Input-tail selected bound is not exactly bound across candidate/refresh/final."
        )
    return {
        "branchwise": branchwise,
        "combined": combined,
        "maximum_branch_radius": maximum_radius,
        "selected_B_in": candidate_b_in,
    }


def _validate_phase2(
    root: Path,
    policy: Mapping[str, object],
    phase2_manifest: Mapping[str, object],
    resolved_report: Mapping[str, object],
) -> dict[str, object]:
    safe_row = _one_row(
        _load_csv(_artifact_file(root, policy, "phase2_safe_row"), label="Phase 2 safe row"),
        label="Phase 2 safe row",
    )
    resolved_row = _one_row(
        _load_csv(
            _artifact_file(root, policy, "phase2_resolved_summary"),
            label="resolved-response summary",
        ),
        label="resolved-response summary",
    )
    final_candidate = _one_row(
        _load_csv(
            _artifact_file(root, policy, "phase2_final_candidate"),
            label="Phase 2 final candidate",
        ),
        label="Phase 2 final candidate",
    )
    final_row = _one_row(
        _load_csv(
            _artifact_file(root, policy, "phase2_final_certificate"),
            label="Phase 2 final certificate",
        ),
        label="Phase 2 final certificate",
    )
    input_tail_report = _load_json(
        _artifact_file(root, policy, "input_tail_report"), label="input-tail report"
    )
    input_tail_summary = _one_row(
        _load_csv(
            _artifact_file(root, policy, "input_tail_summary"),
            label="input-tail summary",
        ),
        label="input-tail summary",
    )
    selected_geometry = phase2_manifest.get("selected_geometry")
    resolved_summary = resolved_report.get("summary")
    refresh = resolved_report.get("final_phase2_refresh")
    if (
        not isinstance(selected_geometry, dict)
        or not isinstance(resolved_summary, dict)
        or not isinstance(refresh, dict)
        or not isinstance(input_tail_report.get("summary"), dict)
    ):
        raise CertificateEquivalenceError("Phase 2 theorem reports are malformed.")
    input_tail_json = input_tail_report["summary"]
    input_tail_configuration = validate_input_tail_configuration(
        policy, input_tail_json, input_tail_summary
    )
    input_tail_provenance = validate_input_tail_provenance(root, input_tail_report)
    input_tail_bounds = validate_input_tail_bounds(
        input_tail_json,
        input_tail_summary,
        final_candidate,
        refresh,
        final_row,
    )

    source_plan = _load_json(
        _artifact_file(root, policy, "reproducibility_plan"),
        label="source reproducibility plan",
    )
    precision_settings = source_plan.get("precision_settings")
    if not isinstance(precision_settings, dict):
        raise CertificateEquivalenceError(
            "Source reproducibility plan precision_settings must be an object."
        )
    configuration = phase2_manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise CertificateEquivalenceError("Phase 2 configuration is malformed.")
    if (
        phase2_manifest.get("map_label") != policy.get("map_label")
        or _integer(configuration.get("N"), label="Phase 2 configuration N") != 600
        or _integer(configuration.get("M"), label="Phase 2 configuration M") != 610
        or _decimal(configuration.get("expected_selected_rho"), label="Phase 2 rho")
        != Decimal(str(EXPECTED_CONFIGURATION["rho"]))
        or _decimal(configuration.get("mu"), label="Phase 2 mu") != Decimal("0.3")
        or _decimal(precision_settings.get("rho"), label="source plan rho")
        != Decimal(str(EXPECTED_CONFIGURATION["rho"]))
        or _integer(precision_settings.get("N"), label="source plan N") != 600
        or _integer(precision_settings.get("M"), label="source plan M") != 610
        or precision_settings.get("map_label") != policy.get("map_label")
    ):
        raise CertificateEquivalenceError("Phase 2/source-plan configuration drifted.")
    radius = _validate_radius_contract(
        policy,
        {
            "selected geometry": selected_geometry.get("r_candidate"),
            "safe finite-M row": safe_row.get("r"),
            "resolved report": resolved_summary.get("r"),
            "resolved CSV": resolved_row.get("r"),
            "final candidate": final_candidate.get("r"),
            "final certificate": final_row.get("r"),
            "input-tail report": input_tail_json.get("r"),
            "input-tail CSV": input_tail_summary.get("r"),
            "source reproducibility plan": precision_settings.get("r"),
        },
    )
    q_target = _decimal(policy.get("q_gap_target"), label="q-gap target")
    q_contracts = (
        (
            "selected geometry",
            selected_geometry,
            selected_geometry.get("r_tau_interval_u"),
            True,
        ),
        ("safe finite-M row", safe_row, safe_row.get("r_tau_interval_u"), False),
        (
            "resolved report",
            resolved_summary,
            resolved_summary.get("r_tau_interval_u"),
            True,
        ),
        ("resolved CSV", resolved_row, resolved_row.get("r_tau_interval_u"), False),
        ("final candidate", final_candidate, final_candidate.get("r_tau_interval_u"), False),
        ("final certificate", final_row, final_row.get("r_tau"), False),
    )
    ratios: list[Fraction] = []
    for label, record, r_tau, json_gate in q_contracts:
        try:
            ratios.append(
                _validate_q_contract(
                    record,
                    radius=radius,
                    r_tau_value=r_tau,
                    q_target=q_target,
                    json_gate=json_gate,
                )
            )
        except CertificateEquivalenceError as exc:
            raise CertificateEquivalenceError(f"{label} q-gap contract failed: {exc}") from exc
    if len(set(ratios)) != 1:
        raise CertificateEquivalenceError("Phase 2 artifacts use different exact q ratios.")
    ratio = ratios[0]
    if _decimal(safe_row.get("q_gap"), label="safe-row q_gap") != q_target:
        raise CertificateEquivalenceError("Safe-row q_gap target drifted.")
    if _decimal(final_row.get("q_gap"), label="final-row q_gap") != q_target:
        raise CertificateEquivalenceError("Final-row q_gap target drifted.")

    phase2_bounds = validate_phase2_inequalities(
        b_out_value=final_row.get("B_out"),
        b_in_value=final_row.get("B_in"),
        collocation_value=final_row.get("collocation"),
        epsilon_value=final_row.get("epsilon_upper_text"),
        triangle_value=final_row.get("epsilon_triangle_upper_text"),
    )
    b_in = phase2_bounds["B_in"]
    epsilon = phase2_bounds["epsilon"]
    triangle = phase2_bounds["triangle"]

    validate_phase2_candidate_binding(final_candidate, final_row)

    if _fraction_decimal(
        refresh.get("epsilon_cert_text"), label="resolved epsilon refresh"
    ) != epsilon:
        raise CertificateEquivalenceError("Resolved report/final epsilon mismatch.")
    if (
        _fraction_decimal(
            refresh.get("epsilon_triangle_cert_text"), label="resolved triangle refresh"
        )
        != triangle
    ):
        raise CertificateEquivalenceError("Resolved report/final triangle mismatch.")
    if _fraction_decimal(
        refresh.get("B_in_selected_cert_text"), label="resolved B_in refresh"
    ) != b_in:
        raise CertificateEquivalenceError("Resolved report/final B_in mismatch.")

    for gate in PHASE2_FINAL_GATES:
        _csv_bool(final_row, gate, True)
    for gate in (
        "response_boundary_cover_certified",
        "response_coherent_prefix_certified",
        "response_remainder_certified",
        "response_scaled_legendre_certified",
        "response_whole_ellipse_fallback_certified",
        "response_prefactor_certified",
        "finite_M_prefactor_certified",
    ):
        _json_bool(resolved_summary, gate, True)
    for gate in (
        "input_boundary_cover_certified",
        "input_exact_prefix_certified",
        "input_geometric_remainder_certified",
        "input_branchwise_profile_certified",
        "input_combined_row_certified",
        "input_tail_certified",
        "final_phase2_certified",
    ):
        _csv_bool(final_candidate, gate, True)

    return {
        "radius": radius,
        "q_ratio": ratio,
        "q_target": q_target,
        "epsilon": epsilon,
        "epsilon_triangle": triangle,
        "input_tail_configuration": input_tail_configuration,
        "input_tail_provenance": input_tail_provenance,
        "input_tail_bounds": input_tail_bounds,
        "final_candidate": final_candidate,
        "final_row": final_row,
    }


def _validate_exact_dyadic_schur(
    root: Path,
    policy: Mapping[str, object],
    schur_report: Mapping[str, object],
    spectral_report: Mapping[str, object],
) -> dict[str, object]:
    schemas = policy.get("certificate_schemas")
    assert isinstance(schemas, dict)
    if (
        schur_report.get("certificate_schema") != schemas.get("validated_schur")
        or schur_report.get("map_label") != policy.get("map_label")
        or _integer(schur_report.get("N"), label="Schur N") != 600
        or _integer(schur_report.get("M"), label="Schur M") != 610
        or _decimal(schur_report.get("rho"), label="Schur rho")
        != Decimal(str(EXPECTED_CONFIGURATION["rho"]))
        or _decimal(schur_report.get("r"), label="Schur radius")
        != Decimal(CANONICAL_HARDY_RADIUS)
        or _integer(schur_report.get("precision_bits"), label="Schur precision") != 256
        or schur_report.get("similarity_identity") != "A_N_circ Q = Q T + R"
        or schur_report.get("status")
        != "interval-certified exact-binary Schur similarity"
    ):
        raise CertificateEquivalenceError("Validated Schur configuration/schema drifted.")
    _json_bool(schur_report, "schur_similarity_certified", True)
    if _fraction_decimal(
        schur_report.get("eta_schur_upper_text"), label="Schur eta text"
    ) != _fraction_decimal(
        spectral_report.get("eta_schur_upper_text"), label="spectral eta text"
    ):
        raise CertificateEquivalenceError("Schur/spectral eta_schur text binding drifted.")
    expected_below = 600 * 599 // 2
    for report, label in (
        (schur_report, "validated Schur report"),
        (spectral_report, "spectral report"),
    ):
        if (
            _integer(
                report.get("exact_dyadic_schur_below_diagonal_entry_count"),
                label=f"{label} below-diagonal entry count",
            )
            != expected_below
            or _integer(
                report.get("exact_dyadic_schur_below_diagonal_zero_count"),
                label=f"{label} below-diagonal zero count",
            )
            != expected_below
        ):
            raise CertificateEquivalenceError(
                f"{label} does not record every exact lower-triangle entry as zero."
            )
        _json_bool(report, "exact_dyadic_schur_below_diagonal_all_zero", True)
        _json_bool(report, "exact_dyadic_schur_upper_triangular_certified", True)

    # NumPy is imported only after release/source authentication and report checks.
    try:
        import numpy as np
    except ImportError as exc:
        raise CertificateEquivalenceError(
            "NumPy from the locked environment is required to inspect the Schur NPZ."
        ) from exc
    path = _artifact_file(root, policy, "validated_schur_npz")
    exact_counts: list[int] = []
    plan_rows = _load_csv(
        _artifact_file(root, policy, "contour_plan"), label="contour plan"
    )
    if len(plan_rows) != 24:
        raise CertificateEquivalenceError("Expected 24 contours for exact Schur counts.")
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "T",
                "Q",
                "absolute_T_upper",
                "midpoint_sha256",
                "eta_schur_upper",
                "Q_condition_upper",
            }
            if set(archive.files) != required:
                raise CertificateEquivalenceError("Validated Schur NPZ schema drifted.")
            matrix = archive["T"]
            absolute_upper = archive["absolute_T_upper"]
            q_matrix = archive["Q"]
            if (
                matrix.shape != (600, 600)
                or q_matrix.shape != (600, 600)
                or absolute_upper.shape != (600, 600)
                or matrix.dtype != np.dtype("complex128")
                or q_matrix.dtype != np.dtype("complex128")
                or absolute_upper.dtype != np.dtype("float64")
                or not np.isfinite(matrix).all()
                or not np.isfinite(q_matrix).all()
                or not np.isfinite(absolute_upper).all()
            ):
                raise CertificateEquivalenceError("Validated Schur NPZ arrays are invalid.")
            lower_count = int(np.count_nonzero(np.tril(matrix, k=-1)))
            absolute_lower_count = int(np.count_nonzero(np.tril(absolute_upper, k=-1)))
            if lower_count != 0 or absolute_lower_count != 0:
                raise CertificateEquivalenceError(
                    "Exact-binary Schur cache has a nonzero entry below the diagonal."
                )
            diagonal = np.diag(matrix)
            for row in plan_rows:
                name = row.get("name") or "unnamed target"
                centre = _fraction_text(row.get("centre_exact"), label=f"{name} centre")
                radius = _fraction_text(row.get("radius_exact"), label=f"{name} radius")
                radius_squared = radius * radius
                count = 0
                for value in diagonal:
                    # Binary64 components are decoded as exact rationals.  No
                    # floating-point distance or rounded membership is used.
                    real_part = Fraction.from_float(float(value.real))
                    imaginary_part = Fraction.from_float(float(value.imag))
                    distance_squared = (
                        (real_part - centre) * (real_part - centre)
                        + imaginary_part * imaginary_part
                    )
                    if distance_squared == radius_squared:
                        raise CertificateEquivalenceError(
                            f"An exact-binary Schur diagonal value lies on {name}'s boundary."
                        )
                    count += distance_squared < radius_squared
                expected = _integer(
                    row.get("expected_multiplicity"),
                    label=f"{name} expected multiplicity",
                )
                if count != expected:
                    raise CertificateEquivalenceError(
                        f"Exact Schur-diagonal count for {name} is {count}, expected {expected}."
                    )
                exact_counts.append(count)
            midpoint = archive["midpoint_sha256"]
            if midpoint.shape != () or str(midpoint.item()) != schur_report.get("midpoint_sha256"):
                raise CertificateEquivalenceError("Schur NPZ midpoint binding drifted.")
            eta_scalar = archive["eta_schur_upper"]
            q_scalar = archive["Q_condition_upper"]
            if eta_scalar.shape != () or q_scalar.shape != ():
                raise CertificateEquivalenceError("Schur NPZ theorem scalars are not scalar.")
            eta_npz = _fraction_decimal(eta_scalar.item(), label="Schur NPZ eta")
            q_npz = _fraction_decimal(q_scalar.item(), label="Schur NPZ Q condition")
            eta_report = _fraction_decimal(
                schur_report.get("eta_schur_upper"), label="Schur report eta"
            )
            q_report = _fraction_decimal(
                schur_report.get("Q_condition_upper"), label="Schur report Q condition"
            )
            if eta_npz != eta_report or q_npz != q_report:
                raise CertificateEquivalenceError(
                    "Schur NPZ/report eta_schur or Q_condition binding drifted."
                )
    except (OSError, ValueError) as exc:
        raise CertificateEquivalenceError(f"Cannot safely inspect Schur NPZ: {exc}") from exc
    return {
        "below_diagonal_entry_count": expected_below,
        "below_diagonal_nonzero_count": 0,
        "exact_diagonal_counts": exact_counts,
        "eta_schur_row_upper": eta_report,
        "Q_condition_row_upper": q_report,
    }


def _validate_laurent_support(
    certificate_rows: Sequence[Mapping[str, str]],
    mode_rows: Sequence[Mapping[str, str]],
    reconstruction_rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, Fraction]]:
    modes_by_name: dict[str, list[Mapping[str, str]]] = {}
    for row in mode_rows:
        name = row.get("name")
        if not name:
            raise CertificateEquivalenceError("Laurent mode row has no target name.")
        modes_by_name.setdefault(name, []).append(row)
    reconstruction_by_name = {row.get("name"): row for row in reconstruction_rows}
    if len(reconstruction_by_name) != len(reconstruction_rows):
        raise CertificateEquivalenceError("Duplicate Laurent reconstruction target.")

    laurent_rows = [
        row for row in certificate_rows if row.get("certificate_route") == "Schur-count/Laurent-moat"
    ]
    if set(modes_by_name) != {row["name"] for row in laurent_rows}:
        raise CertificateEquivalenceError("Laurent mode-bound target set drifted.")
    if set(reconstruction_by_name) != {row["name"] for row in laurent_rows}:
        raise CertificateEquivalenceError("Laurent reconstruction target set drifted.")
    derived: dict[str, dict[str, Fraction]] = {}
    for row in laurent_rows:
        name = row["name"]
        minimum = _integer(row.get("minimum_laurent_mode"), label=f"{name} minimum mode")
        maximum = _integer(row.get("maximum_laurent_mode"), label=f"{name} maximum mode")
        sample_count = _integer(row.get("laurent_sample_count"), label=f"{name} sample count")
        expected_modes = list(range(minimum, maximum + 2))
        observed = sorted(
            modes_by_name[name],
            key=lambda item: _integer(item.get("mode"), label=f"{name} mode"),
        )
        observed_modes = [
            _integer(item.get("mode"), label=f"{name} mode") for item in observed
        ]
        if observed_modes != expected_modes or maximum - minimum + 1 != sample_count:
            raise CertificateEquivalenceError(f"Laurent mode support drifted for {name}.")
        for item in observed[:-1]:
            _csv_bool(item, "coefficient_present", True)
        _csv_bool(observed[-1], "coefficient_present", False)
        residual_sum = Fraction(0)
        coefficient_sum = Fraction(0)
        for item in observed:
            if item.get("coefficient_sha256") != row.get("coefficient_sha256"):
                raise CertificateEquivalenceError(f"Laurent coefficient binding drifted for {name}.")
            if _integer(item.get("precision_bits"), label=f"{name} mode precision") != 256:
                raise CertificateEquivalenceError(f"Laurent precision drifted for {name}.")
            residual = _fraction_decimal(
                item.get("residual_frobenius_upper"), label=f"{name} mode residual"
            )
            coefficient = _fraction_decimal(
                item.get("coefficient_frobenius_upper"),
                label=f"{name} mode coefficient",
            )
            if residual < 0 or coefficient < 0:
                raise CertificateEquivalenceError(f"Negative Laurent residual for {name}.")
            residual_sum += residual
            if item is not observed[-1]:
                coefficient_sum += coefficient
        reconstruction = reconstruction_by_name[name]
        for key in (
            "laurent_sample_count",
            "minimum_laurent_mode",
            "maximum_laurent_mode",
            "coefficient_sha256",
        ):
            if reconstruction.get(key) != row.get(key):
                raise CertificateEquivalenceError(
                    f"Laurent certificate/reconstruction mismatch for {name}/{key}."
                )
        for key in (
            "generated_in_recorded_run",
            "candidate_coefficients_validated_exact_dyadic",
            "theorem_certified",
        ):
            _csv_bool(reconstruction, key, True)
        _csv_bool(reconstruction, "digest_used_in_theorem_gate", False)
        aggregate_residual = _fraction_decimal(
            row.get("exact_dyadic_residual_sum_upper"),
            label=f"{name} aggregate exact-dyadic residual",
        )
        aggregate_coefficient = _fraction_decimal(
            row.get("coefficient_frobenius_sum_upper"),
            label=f"{name} aggregate coefficient sum",
        )
        derived[name] = {
            # The two displays are independently outward-rounded.  Taking the
            # larger exact rational value preserves one-sidedness while binding
            # every per-mode witness row to the theorem primitive.
            "residual_upper": max(aggregate_residual, residual_sum),
            "coefficient_upper": max(aggregate_coefficient, coefficient_sum),
            "mode_residual_sum": residual_sum,
            "mode_coefficient_sum": coefficient_sum,
        }
    return derived


def _validate_targets_and_moats(
    root: Path,
    policy: Mapping[str, object],
    spectral_report: Mapping[str, object],
    *,
    epsilon_precise: Fraction,
    schur_primitives: Mapping[str, object],
) -> dict[str, object]:
    plan_rows = _load_csv(
        _artifact_file(root, policy, "contour_plan"), label="contour plan"
    )
    certificate_rows = _load_csv(
        _artifact_file(root, policy, "spectral_certificate"),
        label="spectral certificate table",
    )
    if len(plan_rows) != 24 or len(certificate_rows) != 24:
        raise CertificateEquivalenceError("Expected exactly 24 contour/certificate rows.")
    _validate_exact_contour_geometry(plan_rows, certificate_rows)
    mode_rows = _load_csv(
        _artifact_file(root, policy, "laurent_mode_bounds"), label="Laurent mode bounds"
    )
    reconstruction_rows = _load_csv(
        _artifact_file(root, policy, "laurent_reconstruction"),
        label="Laurent reconstruction table",
    )
    laurent_primitives = _validate_laurent_support(
        certificate_rows, mode_rows, reconstruction_rows
    )
    expected_names = policy["target_order"]
    expected_multiplicities = policy["expected_multiplicities"]
    route_counts = {"Schur-count/Schur-moat": 0, "Schur-count/Laurent-moat": 0}
    derived_lifted: list[Fraction] = []
    derived_gains: list[Fraction] = []
    laurent_exact_dyadic_summaries: list[dict[str, object]] = []
    total_multiplicity = 0

    eta_schur_precise = _fraction_decimal(
        spectral_report.get("eta_schur_upper_text"), label="spectral eta_schur"
    )
    eta_a_precise = _fraction_decimal(
        spectral_report.get("eta_A_upper_text"), label="spectral eta_A"
    )
    if eta_schur_precise <= 0 or eta_a_precise <= 0:
        raise CertificateEquivalenceError("Perturbation radii must be positive.")

    for index, (plan, row) in enumerate(zip(plan_rows, certificate_rows), start=1):
        expected_name = expected_names[index - 1]
        expected_multiplicity = expected_multiplicities[index - 1]
        expected_route = (
            "Schur-count/Schur-moat" if index <= 17 else "Schur-count/Laurent-moat"
        )
        if (
            _integer(plan.get("rank"), label="plan rank") != index
            or _integer(row.get("rank"), label="certificate rank") != index
            or plan.get("name") != expected_name
            or row.get("name") != expected_name
            or _integer(plan.get("expected_multiplicity"), label="plan multiplicity")
            != expected_multiplicity
            or _integer(row.get("expected_multiplicity"), label="certificate multiplicity")
            != expected_multiplicity
            or plan.get("planned_certificate_route") != expected_route
            or row.get("certificate_route") != expected_route
        ):
            raise CertificateEquivalenceError(
                f"Target order/multiplicity/route drift at row {index}."
            )
        if (
            _fraction_text(plan.get("centre_exact"), label=f"{expected_name} centre")
            != _fraction_text(row.get("centre_exact"), label=f"{expected_name} centre")
            or _fraction_text(plan.get("radius_exact"), label=f"{expected_name} radius")
            != _fraction_text(row.get("radius_exact"), label=f"{expected_name} radius")
        ):
            raise CertificateEquivalenceError(f"Contour/certificate geometry mismatch for {expected_name}.")
        centre = _fraction_text(row.get("centre_exact"), label=f"{expected_name} centre")
        radius = _fraction_text(row.get("radius_exact"), label=f"{expected_name} radius")
        exact_zero_distance = abs(centre) - radius
        if radius <= 0 or exact_zero_distance <= 0:
            raise CertificateEquivalenceError(f"Zero is not excluded exactly for {expected_name}.")
        reported_zero_distance = _fraction_decimal(
            row.get("distance_to_zero_lower"), label=f"{expected_name} zero distance"
        )
        if reported_zero_distance <= 0 or reported_zero_distance > exact_zero_distance:
            raise CertificateEquivalenceError(
                f"Reported zero distance is not a valid lower bound for {expected_name}."
            )
        for gate in TRUE_ROW_GATES:
            _csv_bool(row, gate, True)
        _csv_bool(row, "sampled_values_used_in_theorem_gate", False)
        _csv_bool(row, "laurent_used_for_count", False)
        _csv_bool(row, "laurent_used_for_moat", expected_route.endswith("Laurent-moat"))
        if row.get("status") != "theorem_certified":
            raise CertificateEquivalenceError(f"Target {expected_name} is not theorem_certified.")
        if (
            row.get("count_method") != "certified Schur-diagonal algebraic count"
            or row.get("count_reference_matrix")
            != "exact-binary upper-triangular Schur matrix T"
            or _integer(
                row.get("schur_diagonal_algebraic_count"),
                label=f"{expected_name} Schur count",
            )
            != expected_multiplicity
        ):
            raise CertificateEquivalenceError(
                f"The Schur count marker/count is invalid for {expected_name}."
            )

        row_eta_schur = _fraction_decimal(
            row.get("eta_schur_upper"), label=f"{expected_name} eta_schur"
        )
        row_eta_a = _fraction_decimal(
            row.get("eta_A_upper"), label=f"{expected_name} eta_A"
        )
        row_epsilon = _fraction_decimal(
            row.get("epsilon_upper"), label=f"{expected_name} epsilon"
        )
        if (
            row_eta_schur
            != schur_primitives.get("eta_schur_row_upper")
            or _fraction_decimal(
                row.get("Q_condition_upper"), label=f"{expected_name} Q condition"
            )
            != schur_primitives.get("Q_condition_row_upper")
            or row_eta_schur < eta_schur_precise
            or row_eta_a < eta_a_precise
            or row_epsilon < epsilon_precise
        ):
            raise CertificateEquivalenceError(
                f"Rounded theorem upper bound is not conservative for {expected_name}."
            )

        if expected_route == "Schur-count/Schur-moat":
            inverse_upper = _fraction_decimal(
                row.get("triangular_inverse_bound_upper"),
                label=f"{expected_name} triangular inverse",
            )
            q_condition = _fraction_decimal(
                row.get("Q_condition_upper"), label=f"{expected_name} Q condition"
            )
            if inverse_upper <= 0 or q_condition < 1:
                raise CertificateEquivalenceError(
                    f"Invalid Schur primitive upper bounds for {expected_name}."
                )
            base_lower = Fraction(1, 1) / (inverse_upper * q_condition)
            if row.get("moat_method") != "uniform complete-circle triangular Schur resolvent":
                raise CertificateEquivalenceError(f"Invalid Schur moat marker for {expected_name}.")
        else:
            residual_upper = laurent_primitives[expected_name]["residual_upper"]
            coefficient_upper = laurent_primitives[expected_name]["coefficient_upper"]
            q_condition = _fraction_decimal(
                row.get("Q_condition_upper"), label=f"{expected_name} Q condition"
            )
            if (
                residual_upper < 0
                or residual_upper >= 1
                or coefficient_upper <= 0
                or q_condition < 1
            ):
                raise CertificateEquivalenceError(
                    f"Invalid Laurent primitive bounds for {expected_name}."
                )
            base_lower = (Fraction(1, 1) - residual_upper) / (
                coefficient_upper * q_condition
            )
            _csv_bool(row, "candidate_coefficients_validated_exact_dyadic", True)
            _csv_bool(row, "candidate_generated_in_recorded_run", True)
            _csv_bool(row, "coefficient_digest_used_in_theorem_gate", False)
            if (
                row.get("moat_method")
                != "complete-circle Laurent approximate inverse in Schur coordinates"
            ):
                raise CertificateEquivalenceError(f"Invalid Laurent moat marker for {expected_name}.")
            laurent_exact_dyadic_summaries.append(
                {
                    "name": expected_name,
                    "exact_dyadic_residual_sum_upper": str(
                        _fraction_decimal_display(residual_upper, upper=True)
                    ),
                    "coefficient_frobenius_sum_upper": str(
                        _fraction_decimal_display(coefficient_upper, upper=True)
                    ),
                    "Q_condition_upper": str(
                        _fraction_decimal_display(q_condition, upper=True)
                    ),
                    "derived_triangular_laurent_moat_lower": str(
                        _fraction_decimal_display(base_lower, upper=False)
                    ),
                    "candidate_coefficients_validated_exact_dyadic": True,
                    "digest_used_in_theorem_gate": False,
                }
            )

        matrix_lower = base_lower - row_eta_schur - row_eta_a
        zero_lower = _fraction_decimal(
            row.get("distance_to_zero_lower"), label=f"{expected_name} zero lower"
        )
        lifted_lower = min(matrix_lower, zero_lower)
        if lifted_lower <= 0:
            raise CertificateEquivalenceError(
                f"Derived lifted finite-section moat is nonpositive for {expected_name}."
            )
        gain_upper = row_epsilon / lifted_lower
        if gain_upper >= 1:
            raise CertificateEquivalenceError(
                f"Derived small-gain upper bound is not below one for {expected_name}."
            )

        reported_a = _decimal(
            row.get("A_N_circ_moat_lower"), label=f"{expected_name} A_N moat"
        )
        reported_matrix = _decimal(
            row.get("mathematical_finite_matrix_moat_lower"),
            label=f"{expected_name} matrix moat",
        )
        reported_lifted = _decimal(
            row.get("lifted_finite_section_moat_lower"),
            label=f"{expected_name} lifted moat",
        )
        if not (Decimal(0) < reported_lifted <= reported_matrix <= reported_a):
            raise CertificateEquivalenceError(
                f"Reported perturbation-subtraction chain is invalid for {expected_name}."
            )
        # The stored aggregate is a display/diagnostic field.  Its components
        # were rounded outward independently, so it is not used to establish
        # or dominate the theorem gain.  The exact primitive reaggregation
        # above is the sole small-gain decision.
        _decimal(
            row.get("certified_small_gain_product_upper"),
            label=f"{expected_name} displayed gain",
        )
        route_counts[expected_route] += 1
        total_multiplicity += expected_multiplicity
        derived_lifted.append(lifted_lower)
        derived_gains.append(gain_upper)

    if route_counts != policy.get("route_counts") or total_multiplicity != 30:
        raise CertificateEquivalenceError("Route counts or total multiplicity drifted.")
    return {
        "target_count": len(certificate_rows),
        "total_multiplicity": total_multiplicity,
        "route_counts": route_counts,
        "minimum_derived_lifted_moat": _fraction_decimal_display(
            min(derived_lifted), upper=False
        ),
        "maximum_derived_small_gain": _fraction_decimal_display(
            max(derived_gains), upper=True
        ),
        "target_names": list(expected_names),
        "laurent_exact_dyadic_summaries": laurent_exact_dyadic_summaries,
    }


def _validate_global_report(
    policy: Mapping[str, object],
    report: Mapping[str, object],
    *,
    radius: Decimal,
    epsilon: Fraction,
    eta_a: Decimal,
) -> None:
    schemas = policy.get("certificate_schemas")
    assert isinstance(schemas, dict)
    if (
        report.get("certificate_schema") != schemas.get("spectral_contours")
        or report.get("map_label") != policy.get("map_label")
        or _integer(report.get("N"), label="spectral N") != 600
        or _integer(report.get("M"), label="spectral M") != 610
        or _decimal(report.get("rho"), label="spectral rho")
        != Decimal(str(EXPECTED_CONFIGURATION["rho"]))
        or _integer(report.get("alpha_numerator"), label="spectral alpha numerator") != 13
        or _integer(report.get("alpha_denominator"), label="spectral alpha denominator") != 20
        or _integer(report.get("alpha_power_count"), label="spectral alpha powers") != 18
        or _integer(report.get("mu_numerator"), label="spectral mu numerator") != 3
        or _integer(report.get("mu_denominator"), label="spectral mu denominator") != 10
        or _integer(report.get("mu_power_count"), label="spectral mu powers") != 6
        or _integer(
            report.get("default_radius_numerator"),
            label="spectral default radius numerator",
        )
        != 1
        or _integer(
            report.get("default_radius_denominator"),
            label="spectral default radius denominator",
        )
        != 5
        or _integer(report.get("contour_precision_bits"), label="contour precision")
        != 256
        or _integer(report.get("target_count"), label="spectral target count") != 24
        or _integer(
            report.get("total_expected_algebraic_multiplicity"),
            label="expected multiplicity",
        )
        != 30
        or _integer(
            report.get("total_certified_algebraic_multiplicity"),
            label="certified multiplicity",
        )
        != 30
        or _integer(report.get("schur_count_target_count"), label="Schur count targets")
        != 24
        or _integer(
            report.get("schur_triangular_moat_target_count"),
            label="Schur moat targets",
        )
        != 17
        or _integer(report.get("laurent_moat_target_count"), label="Laurent moat targets")
        != 7
    ):
        raise CertificateEquivalenceError("Global spectral count metadata drifted.")
    if _decimal(report.get("r"), label="spectral radius") != radius:
        raise CertificateEquivalenceError("Global spectral radius drifted.")
    if _fraction_decimal(
        report.get("epsilon_upper_text"), label="spectral epsilon"
    ) != epsilon:
        raise CertificateEquivalenceError("Global spectral/Phase 2 epsilon mismatch.")
    if _decimal(report.get("eta_A_upper_text"), label="spectral eta_A") != eta_a:
        raise CertificateEquivalenceError("Global spectral/Hardy eta_A mismatch.")
    for gate in policy.get("true_report_gates", []):
        _json_bool(report, str(gate), True)
    for gate in policy.get("false_report_gates", []):
        _json_bool(report, str(gate), False)
    if not isinstance(report.get("status"), str) or not str(report["status"]).startswith(
        "theorem-certified twenty-four-target Riesz-rank package"
    ):
        raise CertificateEquivalenceError("Unexpected global spectral theorem status.")


def _validate_compute_evidence(
    root: Path,
    *,
    expected_run_nonce: str,
    expected_release_inventory_sha256: str,
) -> dict[str, object]:
    evidence_path = _safe_regular_file(
        root, COMPUTE_EVIDENCE_RELATIVE, label="clean-room compute evidence"
    )
    evidence = _load_json(evidence_path, label="clean-room compute evidence")
    if (
        evidence.get("receipt_schema") != "blaschke-theorem-only-compute-v1"
        or evidence.get("status") != "theorem-only source reconstruction complete"
        or evidence.get("stage") != "theorem-only-compute"
        or evidence.get("kernel_used_for_theorem_compute") is not False
        or evidence.get("historical_phase4_diagnostics_skipped") is not True
        or evidence.get("historical_phase4_diagnostics_authoritative") is not False
        or evidence.get("hardy_1024_diagnostic_skipped") is not True
        or evidence.get("hardy_1024_diagnostic_authoritative") is not False
    ):
        raise CertificateEquivalenceError("Theorem-only compute evidence is not complete.")
    for key in (
        "normalization_run",
        "provenance_refresh_run",
        "published_comparison_run",
    ):
        _json_bool(evidence, key, False)
    forced = evidence.get("forced_rebuild_environment")
    if not isinstance(forced, dict):
        raise CertificateEquivalenceError("Compute evidence has no forced environment.")
    for key in (
        "BLASCHKE_FORCE_HARDY_MATRIX",
        "BLASCHKE_FORCE_CONTOURS",
        "BLASCHKE_SKIP_HARDY_STARTING_AUDIT",
        "BLASCHKE_SKIP_HISTORICAL_PHASE4",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if forced.get(key) != "1":
            raise CertificateEquivalenceError(f"Full replay did not force {key}.")
    authenticated_import_roots = os.pathsep.join(
        (str(root), str(root / "Numerics"))
    )
    if (
        forced.get("PYTHONSAFEPATH") != "1"
        or forced.get("PYTHONPATH") != authenticated_import_roots
        or forced.get("BLASCHKE_AUTHENTICATED_REPLAY_ROOT") != str(root)
    ):
        raise CertificateEquivalenceError(
            "Theorem compute did not use only the authenticated staged import roots."
        )
    if "BLASCHKE_FORCE_HISTORICAL_PHASE4" in forced:
        raise CertificateEquivalenceError("The theorem-only replay forced historical Phase 4.")
    if _integer(
        forced.get("MPMATH_PF_ASSEMBLY_WORKERS"), label="assembly worker count"
    ) < 1:
        raise CertificateEquivalenceError("The theorem-only worker count is invalid.")
    commands = evidence.get("commands")
    if not isinstance(commands, list) or len(commands) != 5:
        raise CertificateEquivalenceError("Clean-room compute command record is incomplete.")
    command_lists: list[list[str]] = []
    for row in commands:
        if not isinstance(row, dict) or row.get("returncode") != 0:
            raise CertificateEquivalenceError("A clean-room compute command did not pass.")
        command = row.get("command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise CertificateEquivalenceError("Malformed clean-room command record.")
        command_lists.append(command)
    python_executable = evidence.get("python_executable")
    if not isinstance(python_executable, str) or any(
        command[0] != python_executable for command in command_lists
    ):
        raise CertificateEquivalenceError("Compute commands use different interpreters.")
    if (
        command_lists[0][:3] != [python_executable, "-B", "-c"]
        or command_lists[1]
        != [python_executable, "-B", "Numerics/build_blaschke_deformation_certifier.py"]
        or command_lists[2]
        != [
            python_executable,
            "-B",
            "Numerics/build_blaschke_deformation_thesis_math_notebook.py",
        ]
        or command_lists[3][:4] != [python_executable, "-u", "-B", "-c"]
        or command_lists[4]
        != [
            python_executable,
            "-u",
            "-B",
            "Numerics/blaschke_deformation_contour_certification.py",
            "--root",
            ".",
            "--force",
        ]
    ):
        raise CertificateEquivalenceError("The theorem-only command sequence drifted.")
    driver_text = command_lists[3][4] if len(command_lists[3]) == 5 else ""
    for required in (
        "rebuild_phase2_inputs",
        "certify_finite_m_completion",
        "certify_resolved_response_completion",
        "certify_final_phase2_aggregation",
        "build_or_load_hardy_matrix_certificate",
    ):
        if required not in driver_text:
            raise CertificateEquivalenceError(
                f"The theorem-only driver omitted required stage {required}."
            )
    if "historical_phase4" in " ".join(" ".join(row) for row in command_lists):
        raise CertificateEquivalenceError("Historical Phase 4 entered theorem-only commands.")
    closure = evidence.get("generated_closure")
    if (
        not isinstance(closure, dict)
        or closure.get("status") != "declared generated output closure complete"
        or closure.get("theorem_only") is not True
        or closure.get("generated_path_count") != 27
        or closure.get("generated_paths") != list(EXPECTED_THEOREM_REQUIRED_PATHS)
        or closure.get("missing") != []
    ):
        raise CertificateEquivalenceError("Generated-output closure did not pass.")
    generated_files = closure.get("generated_files")
    if not isinstance(generated_files, list) or len(generated_files) != 27:
        raise CertificateEquivalenceError("The theorem-generated hash closure is incomplete.")
    run_started_ns = _integer(evidence.get("run_started_ns"), label="compute start time")
    run_finished_ns = _integer(evidence.get("run_finished_ns"), label="compute finish time")
    seen: set[str] = set()
    for record in generated_files:
        if not isinstance(record, dict):
            raise CertificateEquivalenceError("Malformed theorem-generated file record.")
        relative = _safe_relative(record.get("path"), label="theorem-generated path")
        value = relative.as_posix()
        if value in seen or value not in EXPECTED_THEOREM_REQUIRED_PATHS:
            raise CertificateEquivalenceError("Unexpected theorem-generated file record.")
        seen.add(value)
        path = _safe_regular_file(root, relative, label="current theorem output")
        mtime_ns = _integer(record.get("mtime_ns"), label=f"{value} mtime")
        size = _integer(record.get("bytes"), label=f"{value} size")
        if (
            not _is_sha256(record.get("sha256"))
            or path.stat().st_size != size
            or path.stat().st_mtime_ns != mtime_ns
            or _sha256_file(path) != record.get("sha256")
            or not run_started_ns <= mtime_ns <= run_finished_ns
        ):
            raise CertificateEquivalenceError(
                f"Current theorem output hash/time binding failed: {value}."
            )
    if seen != set(EXPECTED_THEOREM_REQUIRED_PATHS):
        raise CertificateEquivalenceError("The theorem-generated path set is incomplete.")

    preparation = evidence.get("preparation")
    if not isinstance(preparation, dict):
        raise CertificateEquivalenceError("Compute evidence has no preparation binding.")
    staged_inventory_sha256 = preparation.get("inventory_sha256")
    if not _is_sha256(staged_inventory_sha256):
        raise CertificateEquivalenceError("Preparation inventory binding is invalid.")
    attestation_path = _safe_regular_file(
        root, RUN_ATTESTATION_RELATIVE, label="current-run attestation"
    )
    attestation = _load_json(attestation_path, label="current-run attestation")
    if (
        attestation.get("receipt_schema")
        != "blaschke-certificate-current-run-attestation-v1"
        or attestation.get("status") != "current theorem-only run hash-bound"
        or attestation.get("run_nonce") != expected_run_nonce
        or attestation.get("release_inventory_sha256")
        != expected_release_inventory_sha256
        or attestation.get("staged_inventory_sha256") != staged_inventory_sha256
        or attestation.get("compute_evidence_path")
        != COMPUTE_EVIDENCE_RELATIVE.as_posix()
        or attestation.get("compute_evidence_sha256") != _sha256_file(evidence_path)
        or attestation.get("preparation_receipt_path")
        != "source-only-replay-preparation.json"
        or attestation.get("theorem_generated_path_count") != 27
        or attestation.get("theorem_generated_files") != generated_files
        or attestation.get("forced_rebuild_environment") != forced
        or not _is_sha256(attestation.get("interpreter_sha256"))
        or attestation.get("historical_diagnostics_skipped") is not True
        or attestation.get("historical_diagnostics_authoritative") is not False
        or attestation.get("hardy_1024_diagnostic_skipped") is not True
        or attestation.get("hardy_1024_diagnostic_authoritative") is not False
    ):
        raise CertificateEquivalenceError("Current-run attestation binding failed.")
    preparation_path = _safe_regular_file(
        root,
        PurePosixPath("source-only-replay-preparation.json"),
        label="source-only preparation receipt",
    )
    if attestation.get("preparation_receipt_sha256") != _sha256_file(preparation_path):
        raise CertificateEquivalenceError("Preparation receipt hash binding failed.")
    interpreter_path = Path(python_executable)
    if (
        interpreter_path.is_symlink()
        or not interpreter_path.is_file()
        or _sha256_file(interpreter_path) != attestation.get("interpreter_sha256")
    ):
        raise CertificateEquivalenceError(
            "Current-run interpreter executable hash binding failed."
        )
    outer_started = _integer(attestation.get("outer_started_ns"), label="outer start")
    outer_finished = _integer(attestation.get("outer_finished_ns"), label="outer finish")
    if not outer_started <= run_started_ns <= run_finished_ns <= outer_finished:
        raise CertificateEquivalenceError("Current-run time ordering is invalid.")
    clean_command = attestation.get("clean_replay_command")
    if (
        not isinstance(clean_command, list)
        or not all(isinstance(item, str) for item in clean_command)
        or not clean_command
        or clean_command[0] != python_executable
        or "--theorem-only-compute" not in clean_command
        or "--compute-only" in clean_command
        or "--raw-comparison-receipt" in clean_command
        or str(staged_inventory_sha256) not in clean_command
        or not any(Path(item).name == "run_blaschke_clean_room_replay.py" for item in clean_command)
    ):
        raise CertificateEquivalenceError("Outer theorem-only invocation binding failed.")
    return {
        "compute_evidence_sha256": _sha256_file(evidence_path),
        "run_attestation_sha256": _sha256_file(attestation_path),
        "current_run_nonce_sha256": _sha256_bytes(expected_run_nonce.encode("ascii")),
        "theorem_generated_path_count": 27,
        "historical_diagnostics_skipped": True,
        "historical_diagnostics_authoritative": False,
        "hardy_1024_diagnostic_skipped": True,
        "hardy_1024_diagnostic_authoritative": False,
    }


def verify_certificate(
    root: Path,
    *,
    mode: str,
    require_release_integrity: bool = True,
    expected_inventory_sha256: str | None = None,
    expected_git_commit: str | None = None,
    expected_run_nonce: str | None = None,
) -> dict[str, object]:
    """Return a canonicalizable semantic receipt or fail closed."""

    if mode not in {"artifact-only", "full-replay"}:
        raise ValueError("mode must be 'artifact-only' or 'full-replay'.")
    root = Path(root)
    if root.is_symlink():
        raise CertificateEquivalenceError("Release root must not be a symlink.")
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise CertificateEquivalenceError(f"Release root does not exist: {root}.") from exc
    if not root.is_dir() or root == Path(root.anchor):
        raise CertificateEquivalenceError("Release root must be a narrow directory.")

    policy = _load_policy(root)
    if mode == "full-replay" and not require_release_integrity:
        raise CertificateEquivalenceError(
            "Full replay cannot certify when release integrity is skipped."
        )
    integrity = (
        verify_release_integrity(
            root,
            expected_inventory_sha256=expected_inventory_sha256,
            expected_git_commit=expected_git_commit,
        )
        if require_release_integrity
        else {
            "inventory_sha256": "TEST_FIXTURE_RELEASE_INTEGRITY_SKIPPED",
            "inventory_record_count": 0,
            "immutable_record_count": 0,
        }
    )

    # Only stdlib parsing occurs before the source-integrity gate above.
    spectral_report = _load_json(
        _artifact_file(root, policy, "spectral_report"), label="spectral report"
    )
    schur_report = _load_json(
        _artifact_file(root, policy, "validated_schur_report"),
        label="validated Schur report",
    )
    hardy_2048 = _load_json(
        _artifact_file(root, policy, "hardy_2048_report"), label="2048-bit Hardy report"
    )
    phase2_manifest = _load_json(
        _artifact_file(root, policy, "phase2_manifest"), label="Phase 2 manifest"
    )
    resolved_report = _load_json(
        _artifact_file(root, policy, "phase2_resolved_report"),
        label="resolved-response report",
    )

    hardy2048_values = _validate_hardy_report(
        root, policy, hardy_2048, bits=2048, expected_ready=True
    )
    phase2 = _validate_phase2(root, policy, phase2_manifest, resolved_report)
    radius = _validate_radius_contract(
        policy,
        {
            "Phase 2": phase2["radius"],
            "Hardy 2048": hardy2048_values["radius"],
            "validated Schur": schur_report.get("r"),
            "spectral contour": spectral_report.get("r"),
        },
    )
    eta_a = hardy2048_values["eta_A_upper"]
    _validate_global_report(
        policy,
        spectral_report,
        radius=radius,
        epsilon=phase2["epsilon"],
        eta_a=eta_a,
    )
    schur_structure = _validate_exact_dyadic_schur(
        root, policy, schur_report, spectral_report
    )
    targets = _validate_targets_and_moats(
        root,
        policy,
        spectral_report,
        epsilon_precise=phase2["epsilon"],
        schur_primitives=schur_structure,
    )
    bindings = _validate_internal_bindings(
        root,
        policy,
        spectral_report,
        schur_report,
        hardy_2048,
        phase2_manifest,
        resolved_report,
    )
    if mode == "full-replay":
        if (
            not isinstance(expected_run_nonce, str)
            or len(expected_run_nonce) != 64
            or any(character not in "0123456789abcdef" for character in expected_run_nonce)
        ):
            raise CertificateEquivalenceError(
                "Full replay requires the current runner's 256-bit nonce."
            )
        full_evidence = _validate_compute_evidence(
            root,
            expected_run_nonce=expected_run_nonce,
            expected_release_inventory_sha256=str(integrity["inventory_sha256"]),
        )
    else:
        full_evidence = {}

    semantic_projection = {
        "policy_id": policy["policy_id"],
        "map_label": policy["map_label"],
        "N": 600,
        "M": 610,
        "hardy_radius": str(radius),
        "q_gap_target": str(phase2["q_target"]),
        "q_gap_exact_numerator": phase2["q_ratio"].numerator,
        "q_gap_exact_denominator": phase2["q_ratio"].denominator,
        "phase2_epsilon_upper": str(
            _fraction_decimal_display(phase2["epsilon"], upper=True)
        ),
        "target_count": targets["target_count"],
        "total_multiplicity": targets["total_multiplicity"],
        "route_counts": targets["route_counts"],
        "target_names": targets["target_names"],
        "minimum_derived_lifted_moat_lower": str(
            targets["minimum_derived_lifted_moat"]
        ),
        "maximum_derived_small_gain_upper": str(
            targets["maximum_derived_small_gain"]
        ),
        "exact_schur_below_diagonal_nonzero_count": schur_structure[
            "below_diagonal_nonzero_count"
        ],
        "exact_schur_diagonal_counts": schur_structure["exact_diagonal_counts"],
        "laurent_exact_dyadic_witness_summary_sha256": _sha256_bytes(
            _canonical_json_bytes(targets["laurent_exact_dyadic_summaries"])
        ),
        "laurent_exact_dyadic_witness_summaries": targets[
            "laurent_exact_dyadic_summaries"
        ],
    }
    receipt = {
        "receipt_schema": "blaschke-certificate-equivalence-receipt-v1",
        "status": (
            "CERTIFICATION_CONFIRMED"
            if mode == "full-replay"
            else "ARTIFACT_SEMANTICS_CONFIRMED_NON_INDEPENDENT"
        ),
        "mode": mode,
        "independent_theorem_recomputation": mode == "full-replay",
        "artifact_only_is_independent_proof": False,
        "hardy_1024_starting_audit_authoritative": False,
        "policy_id": policy["policy_id"],
        "semantic_projection_sha256": _sha256_bytes(
            _canonical_json_bytes(semantic_projection)
        ),
        "semantic_projection": semantic_projection,
        "release_integrity": integrity,
        "internal_bindings": bindings,
        **full_evidence,
    }
    return receipt


def _atomic_write(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--mode",
        choices=("artifact-only", "full-replay"),
        default="artifact-only",
    )
    parser.add_argument("--receipt", type=Path)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--expected-release-inventory-sha256")
    identity.add_argument("--expected-git-commit")
    parser.add_argument("--expected-run-nonce")
    arguments = parser.parse_args()
    try:
        receipt = verify_certificate(
            arguments.root,
            mode=arguments.mode,
            require_release_integrity=True,
            expected_inventory_sha256=arguments.expected_release_inventory_sha256,
            expected_git_commit=arguments.expected_git_commit,
            expected_run_nonce=arguments.expected_run_nonce,
        )
    except CertificateEquivalenceError as exc:
        failure = {
            "receipt_schema": "blaschke-certificate-equivalence-receipt-v1",
            "status": "CERTIFICATION_NOT_CONFIRMED",
            "mode": arguments.mode,
            "error": str(exc),
        }
        payload = _canonical_json_bytes(failure)
        if arguments.receipt is not None:
            _atomic_write(arguments.receipt, payload)
        print(payload.decode("utf-8"), end="")
        return 1
    payload = _canonical_json_bytes(receipt)
    if arguments.receipt is not None:
        _atomic_write(arguments.receipt, payload)
    print(payload.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
