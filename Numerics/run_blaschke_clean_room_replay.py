"""Orchestrate a full source-only replay inside one extracted scratch bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import time
from typing import Mapping


def _load_preparation_helper() -> object:
    """Load the exact sibling helper without trusting ambient import paths."""

    helper_path = Path(__file__).resolve(strict=True).with_name(
        "prepare_blaschke_source_only_replay.py"
    )
    if helper_path.is_symlink() or not helper_path.is_file():
        raise ImportError(f"Replay preparation helper is not a regular file: {helper_path}.")
    specification = importlib.util.spec_from_file_location(
        "_blaschke_authenticated_preparation_helper", helper_path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load replay preparation helper: {helper_path}.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


_PREPARATION_HELPER = _load_preparation_helper()
BUNDLE_ROOT_NAME = _PREPARATION_HELPER.BUNDLE_ROOT_NAME
GENERATED_CLASSES = _PREPARATION_HELPER.GENERATED_CLASSES
IMMUTABLE_CLASSES = _PREPARATION_HELPER.IMMUTABLE_CLASSES
INTERNAL_MANIFEST_NAME = _PREPARATION_HELPER.INTERNAL_MANIFEST_NAME
INVENTORY_NAME = _PREPARATION_HELPER.INVENTORY_NAME
POLICY_RELATIVE = _PREPARATION_HELPER.POLICY_RELATIVE
RECEIPT_NAME = _PREPARATION_HELPER.RECEIPT_NAME
SourceOnlyReplayError = _PREPARATION_HELPER.SourceOnlyReplayError
check_prepared_bundle = _PREPARATION_HELPER.check_prepared_bundle
prepare_bundle = _PREPARATION_HELPER.prepare_bundle
sha256_file = _PREPARATION_HELPER.sha256_file


BLAS_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

THEOREM_REQUIRED_GENERATED_PATHS = (
    # Phase 2 clean upstream reconstruction (7) and its manifest (1).
    "Numerics/outputs/blaschke_deformation_certifier/data/branch_image_radius_reoptimisation_balanced_highcell_scan.csv",
    "Numerics/outputs/blaschke_deformation_certifier/reports/branch_image_radius_reoptimisation_balanced_highcell_scan.json",
    "Numerics/outputs/blaschke_deformation_certifier/data/branch_image_balanced_candidate_transport_cert_N600.csv",
    "Numerics/outputs/blaschke_deformation_certifier/reports/branch_image_balanced_candidate_transport_cert_N600.json",
    "Numerics/outputs/blaschke_deformation_certifier/data/branch_image_balanced_candidate_transport_inverse_witness_N600.npz",
    "Numerics/outputs/blaschke_deformation_certifier/data/branch_image_balanced_candidate_single_space_row_N600_M610.csv",
    "Numerics/outputs/blaschke_deformation_certifier/reports/branch_image_balanced_candidate_single_space_row_N600_M610.json",
    "Numerics/outputs/blaschke_deformation_certifier/reports/phase2_clean_room_rebuild_manifest.json",
    # Safe finite-M, resolved response, unresolved input, and final aggregation (7).
    "Numerics/outputs/blaschke_deformation_certifier/data/branch_image_balanced_candidate_single_space_row_N600_M610_safe_GL.csv",
    "Numerics/outputs/blaschke_deformation_certifier/data/output_response_branch_image_prefactor_interval_cert_balanced_N600.csv",
    "Numerics/outputs/blaschke_deformation_certifier/data/output_response_coherent_packet_interval_cert_balanced_N600.csv",
    "Numerics/outputs/blaschke_deformation_certifier/reports/output_response_branch_image_prefactor_interval_cert_balanced_N600.json",
    "Numerics/outputs/blaschke_deformation_certifier/data/blaschke_deformation_input_tail_certificate_N600.csv",
    "Numerics/outputs/blaschke_deformation_certifier/reports/blaschke_deformation_input_tail_certificate_N600.json",
    "Numerics/outputs/blaschke_deformation_certifier/data/branch_image_balanced_response_prefactor_candidate_row_N600_M610.csv",
    "Numerics/outputs/blaschke_deformation_certifier/data/branch_image_phase2_certified_single_space_row_N600_M610.csv",
    # Authoritative 2048-bit Hardy enclosure (4).
    "Numerics/outputs/blaschke_deformation_certifier/data/blaschke_deformation_balanced_hardy_reference_N600_M610_bits2048_certificate.csv",
    "Numerics/outputs/blaschke_deformation_certifier/reports/blaschke_deformation_balanced_hardy_reference_N600_M610_bits2048_certificate.json",
    "Numerics/outputs/blaschke_deformation_certifier/data/blaschke_deformation_balanced_hardy_reference_N600_M610_bits2048.pkl.gz",
    "Numerics/outputs/blaschke_deformation_certifier/data/blaschke_deformation_balanced_hardy_reference_N600_M610_bits2048_diagnostic_midpoint.npz",
    # Exact contour geometry, Schur proof, spectral rows, and Laurent witnesses (7).
    "Numerics/outputs/blaschke_deformation_certifier/data/blaschke_deformation_24_target_N600_M610_contour_plan.csv",
    "Numerics/outputs/blaschke_deformation_certifier/data/blaschke_deformation_24_target_N600_M610_validated_schur.npz",
    "Numerics/outputs/blaschke_deformation_certifier/reports/blaschke_deformation_24_target_N600_M610_validated_schur.json",
    "Numerics/outputs/blaschke_deformation_certifier/data/blaschke_deformation_24_target_N600_M610_spectral_certificate.csv",
    "Numerics/outputs/blaschke_deformation_certifier/reports/blaschke_deformation_24_target_N600_M610_spectral_certificate.json",
    "Numerics/outputs/blaschke_deformation_certifier/data/blaschke_deformation_24_target_N600_M610_laurent_mode_bounds.csv",
    "Numerics/outputs/blaschke_deformation_certifier/data/blaschke_deformation_24_target_N600_M610_laurent_witness_reconstruction.csv",
)


def final_certificate_compatibility_row(
    final_certificate: Mapping[str, object],
) -> dict[str, object]:
    """Build the notebook-compatible final Phase 2 row as a safe superset."""

    def compatibility_value(value: object) -> object:
        if value.__class__.__module__.startswith("mpmath"):
            # Preserve the exact decimal serialization used by the Phase 2
            # aggregation.  Binary64 conversion is reserved for the separate
            # outward ``*_u`` diagnostic fields produced upstream.
            return str(value)
        return value

    required = ("kappa_hat", "tail_components_interval", "certified")
    missing = [key for key in required if key not in final_certificate]
    if missing:
        raise SourceOnlyReplayError(
            f"Final Phase 2 certificate lacks compatibility fields: {missing}."
        )
    row = {
        key: compatibility_value(value)
        for key, value in final_certificate.items()
    }
    row.update(
        {
            "C_tr": row["kappa_hat"],
            "output_tail_mode": "response-prefactor branch-image certificate",
            "transport_route": "certified finite Chebyshev-gauge transport factor",
            "tail_mismatch_certified": bool(row["tail_components_interval"]),
            "branch_data_certified": True,
            "certified": bool(row["certified"]),
        }
    )
    return row


PROJECT_IMPORT_PREFLIGHT = r"""
import importlib.util
import os
from pathlib import Path

authenticated_root = Path(os.environ["BLASCHKE_AUTHENTICATED_REPLAY_ROOT"]).resolve(strict=True)
expected_origins = {
    "Numerics.blaschke_deformation_phase2_pipeline": authenticated_root / "Numerics/blaschke_deformation_phase2_pipeline.py",
    "Numerics.blaschke_deformation_spectral_certification": authenticated_root / "Numerics/blaschke_deformation_spectral_certification.py",
    "Numerics.blaschke_deformation_contour_certification": authenticated_root / "Numerics/blaschke_deformation_contour_certification.py",
    "blaschke_deformation_phase2_geometry": authenticated_root / "Numerics/blaschke_deformation_phase2_geometry.py",
}
authenticated_project_import_origins = {}
for module_name, expected_origin in expected_origins.items():
    specification = importlib.util.find_spec(module_name)
    if specification is None or specification.origin is None:
        raise RuntimeError(f"Cannot resolve authenticated project module {module_name!r}")
    observed_origin = Path(specification.origin).resolve(strict=True)
    if observed_origin != expected_origin.resolve(strict=True):
        raise RuntimeError(
            f"Project module {module_name!r} resolved outside the authenticated staged root: "
            f"{observed_origin}"
        )
    authenticated_project_import_origins[module_name] = str(observed_origin)
""".strip()

THEOREM_ONLY_DRIVER = r"""
from pathlib import Path
import os
import pandas as pd
from Numerics.blaschke_deformation_phase2_pipeline import Phase2RebuildConfig, rebuild_phase2_inputs
from Numerics.blaschke_deformation_phase2_finite_m import Phase2FiniteMConfig, certify_finite_m_completion
from Numerics.blaschke_deformation_phase2_resolved_response import Phase2ResolvedResponseConfig, certify_resolved_response_completion
from Numerics.blaschke_deformation_phase2_final_aggregation import Phase2FinalAggregationConfig, certify_final_phase2_aggregation
from Numerics.blaschke_deformation_spectral_certification import HardyMatrixCertificateConfig, build_or_load_hardy_matrix_certificate
from Numerics.run_blaschke_clean_room_replay import final_certificate_compatibility_row

root = Path.cwd()
output = root / "Numerics/outputs/blaschke_deformation_certifier"
data = output / "data"
reports = output / "reports"
rebuild_phase2_inputs(Phase2RebuildConfig.production_n600_m610(), data_dir=data, report_dir=reports, process_workers=int(__import__("os").environ["MPMATH_PF_ASSEMBLY_WORKERS"]))
certify_finite_m_completion(Phase2FiniteMConfig.production_n600_m610(), output_dir=output)
certify_resolved_response_completion(Phase2ResolvedResponseConfig.production_n600_m610(), output_dir=output)
final = certify_final_phase2_aggregation(Phase2FinalAggregationConfig.production_n600_m610(), output_dir=output)
final_path = data / "branch_image_phase2_certified_single_space_row_N600_M610.csv"
final_temporary = final_path.with_suffix(final_path.suffix + ".tmp")
certificate = final_certificate_compatibility_row(final.final_certificate)
pd.DataFrame([certificate]).to_csv(final_temporary, index=False)
os.replace(final_temporary, final_path)
hardy = build_or_load_hardy_matrix_certificate(
    config=HardyMatrixCertificateConfig(),
    output_dir=output,
    source_files=(root / "Numerics/blaschke_deformation_spectral_certification.py",),
    force=True,
)
if not bool(hardy.get("spectral_use_ready")):
    raise RuntimeError("The authoritative 2048-bit Hardy certificate is not ready.")
""".strip()

BLAS_RUNTIME_PREFLIGHT = PROJECT_IMPORT_PREFLIGHT + "\n" + r"""
import json
import numpy as np
import scipy.linalg as spla
from threadpoolctl import threadpool_info

spla.svdvals(np.eye(2, dtype=np.float64), check_finite=False)
records = [
    record for record in threadpool_info()
    if record.get("user_api") == "blas"
]
invalid = [
    record for record in records
    if int(record.get("num_threads") or 0) != 1
]
if not records or invalid:
    raise RuntimeError(
        "Official replay requires every loaded BLAS runtime to use exactly "
        f"one thread; observed {records!r}."
    )
print(json.dumps({
    "status": "single-threaded BLAS runtime verified",
    "blas": records,
    "authenticated_project_import_origins": authenticated_project_import_origins,
}, sort_keys=True))
""".strip()


def _run(command: list[str], *, root: Path, environment: dict[str, str]) -> dict[str, object]:
    started = time.time()
    subprocess.run(command, cwd=root, env=environment, check=True)
    return {
        "command": command,
        "elapsed_seconds": time.time() - started,
        "returncode": 0,
    }


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceOnlyReplayError(f"Cannot read {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceOnlyReplayError(f"{label} must be a JSON object.")
    return value


def _canonical_relative_path(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise SourceOnlyReplayError(f"{label} must be a non-empty path string.")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or value in {".", ".."}
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise SourceOnlyReplayError(f"Unsafe {label}: {value!r}.")
    return relative


def _regular_bundle_file(
    root: Path,
    relative: PurePosixPath,
    *,
    label: str,
    allow_missing: bool = False,
) -> Path | None:
    """Return a regular in-bundle file without following a symlink component."""

    candidate = root
    for index, part in enumerate(relative.parts):
        candidate = candidate / part
        if candidate.is_symlink():
            raise SourceOnlyReplayError(
                f"Symlinks are forbidden for {label}: {relative.as_posix()}."
            )
        if not candidate.exists():
            if allow_missing:
                return None
            raise SourceOnlyReplayError(
                f"Missing {label}: {relative.as_posix()}."
            )
        if index < len(relative.parts) - 1 and not candidate.is_dir():
            raise SourceOnlyReplayError(
                f"Non-directory path component for {label}: {relative.as_posix()}."
            )
    try:
        mode = candidate.lstat().st_mode
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise SourceOnlyReplayError(
            f"{label} escapes or cannot be resolved inside the bundle: "
            f"{relative.as_posix()}."
        ) from exc
    if not stat.S_ISREG(mode):
        raise SourceOnlyReplayError(
            f"{label} is not a regular file: {relative.as_posix()}."
        )
    return candidate


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _authenticate_inventory_before_preparation(
    *,
    root: Path,
    expected_inventory_sha256: object,
) -> str:
    """Authenticate untouched inventory bytes before any preparation mutation."""

    if not _is_sha256(expected_inventory_sha256):
        raise ValueError(
            "A lowercase 64-hex --expected-inventory-sha256 from an "
            "authenticated external authority is required for compute/full replay."
        )
    inventory_path = _regular_bundle_file(
        root,
        PurePosixPath(INVENTORY_NAME),
        label="untouched source-only replay inventory",
    )
    assert inventory_path is not None
    try:
        inventory_payload = inventory_path.read_bytes()
    except OSError as exc:
        raise SourceOnlyReplayError(
            f"Cannot read untouched replay inventory at {inventory_path}: {exc}"
        ) from exc
    observed = hashlib.sha256(inventory_payload).hexdigest()
    if observed != expected_inventory_sha256:
        raise SourceOnlyReplayError(
            "The untouched replay inventory does not match the externally "
            "authenticated expected SHA-256; refusing source-only preparation."
        )
    return expected_inventory_sha256


def _verify_generated_closure(
    *,
    root: Path,
    preparation: dict[str, object],
    expected_inventory_sha256: str,
    theorem_only: bool = False,
    not_before_ns: int | None = None,
) -> dict[str, object]:
    """Require every archive-declared generated member before finalization."""

    inventory_relative = PurePosixPath(INVENTORY_NAME)
    receipt_relative = PurePosixPath(RECEIPT_NAME)
    manifest_relative = PurePosixPath(INTERNAL_MANIFEST_NAME)
    inventory_path = _regular_bundle_file(
        root, inventory_relative, label="source-only replay inventory"
    )
    receipt_path = _regular_bundle_file(
        root, receipt_relative, label="source-only preparation receipt"
    )
    manifest_path = _regular_bundle_file(
        root, manifest_relative, label="internal reproducibility manifest"
    )
    policy_path = _regular_bundle_file(
        root, POLICY_RELATIVE, label="source-only replay policy"
    )
    assert inventory_path is not None
    assert receipt_path is not None
    assert manifest_path is not None
    assert policy_path is not None

    if not _is_sha256(expected_inventory_sha256):
        raise SourceOnlyReplayError(
            "The externally authenticated expected inventory SHA-256 is invalid."
        )
    try:
        inventory_payload = inventory_path.read_bytes()
    except OSError as exc:
        raise SourceOnlyReplayError(
            f"Cannot read source-only replay inventory at {inventory_path}: {exc}"
        ) from exc
    inventory_sha256 = hashlib.sha256(inventory_payload).hexdigest()
    if inventory_sha256 != expected_inventory_sha256:
        raise SourceOnlyReplayError(
            "The replay inventory does not match the externally authenticated "
            "expected SHA-256."
        )

    try:
        receipt_payload = receipt_path.read_bytes()
        on_disk_receipt = json.loads(receipt_payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceOnlyReplayError(
            f"Cannot read source-only preparation receipt at {receipt_path}: {exc}"
        ) from exc
    if not isinstance(on_disk_receipt, dict):
        raise SourceOnlyReplayError(
            "The source-only preparation receipt must be a JSON object."
        )
    if on_disk_receipt != preparation:
        raise SourceOnlyReplayError(
            "The in-memory and on-disk source-only preparation receipts differ."
        )
    try:
        inventory = json.loads(inventory_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceOnlyReplayError(
            f"Cannot decode source-only replay inventory at {inventory_path}: {exc}"
        ) from exc
    if not isinstance(inventory, dict):
        raise SourceOnlyReplayError(
            "The source-only replay inventory must be a JSON object."
        )

    if (
        type(inventory.get("schema_version")) is not int
        or inventory.get("schema_version") != 1
        or inventory.get("bundle_root_name") != BUNDLE_ROOT_NAME
        or inventory.get("policy_path") != POLICY_RELATIVE.as_posix()
        or inventory.get("complete_file_set_excludes")
        != [INVENTORY_NAME, INTERNAL_MANIFEST_NAME]
    ):
        raise SourceOnlyReplayError(
            "The source-only replay inventory schema is invalid."
        )
    if inventory.get("policy_sha256") != sha256_file(policy_path):
        raise SourceOnlyReplayError(
            "The source-only replay policy hash has drifted."
        )

    rows = inventory.get("files")
    if not isinstance(rows, list) or not rows:
        raise SourceOnlyReplayError(
            "The source-only replay inventory has no file records."
        )
    classifications = GENERATED_CLASSES | IMMUTABLE_CLASSES
    seen: set[PurePosixPath] = set()
    generated: list[PurePosixPath] = []
    observed_counts = {
        classification: 0 for classification in sorted(classifications)
    }
    for row in rows:
        if not isinstance(row, dict):
            raise SourceOnlyReplayError("Malformed source-only replay record.")
        relative = _canonical_relative_path(
            row.get("path"), label="source-only inventory path"
        )
        if relative in seen:
            raise SourceOnlyReplayError(
                f"Duplicate source-only inventory path: {relative.as_posix()}."
            )
        seen.add(relative)
        classification = row.get("classification")
        if (
            not isinstance(classification, str)
            or classification not in classifications
        ):
            raise SourceOnlyReplayError(
                f"Unknown source-only classification for {relative.as_posix()}."
            )
        size = row.get("bytes")
        if (
            not _is_sha256(row.get("sha256"))
            or type(size) is not int
            or size < 0
        ):
            raise SourceOnlyReplayError(
                f"Malformed hash or size for {relative.as_posix()}."
            )
        observed_counts[classification] += 1
        if classification in GENERATED_CLASSES:
            generated.append(relative)

    counts = inventory.get("counts")
    if (
        not isinstance(counts, dict)
        or set(counts) != classifications
        or any(type(value) is not int or value < 0 for value in counts.values())
        or counts != observed_counts
    ):
        raise SourceOnlyReplayError(
            "The source-only replay inventory counts are malformed or inconsistent."
        )
    all_generated_paths = sorted(relative.as_posix() for relative in generated)
    if not all_generated_paths:
        raise SourceOnlyReplayError(
            "The source-only replay inventory declares no generated members."
        )
    if theorem_only:
        generated_paths = list(THEOREM_REQUIRED_GENERATED_PATHS)
        if len(generated_paths) != 27 or len(set(generated_paths)) != 27:
            raise AssertionError("The fixed theorem-only closure must contain 27 paths.")
        omitted = sorted(set(all_generated_paths) - set(generated_paths))
        missing_declarations = sorted(set(generated_paths) - set(all_generated_paths))
        if missing_declarations:
            raise SourceOnlyReplayError(
                "The source-only inventory omits theorem-required generated paths: "
                f"{missing_declarations}."
            )
    else:
        generated_paths = all_generated_paths
        omitted = []

    if (
        type(on_disk_receipt.get("schema_version")) is not int
        or on_disk_receipt.get("schema_version") != 1
        or on_disk_receipt.get("status") != "source-only replay root prepared"
        or on_disk_receipt.get("bundle_root_name") != BUNDLE_ROOT_NAME
        or on_disk_receipt.get("removed_paths") != all_generated_paths
        or type(on_disk_receipt.get("removed_file_count")) is not int
        or on_disk_receipt.get("removed_file_count") != len(all_generated_paths)
        or type(on_disk_receipt.get("immutable_external_input_count")) is not int
        or on_disk_receipt.get("immutable_external_input_count")
        != observed_counts["immutable_external_input"]
    ):
        raise SourceOnlyReplayError(
            "The source-only preparation receipt does not match the inventory."
        )
    if (
        not _is_sha256(on_disk_receipt.get("inventory_sha256"))
        or on_disk_receipt.get("inventory_sha256") != inventory_sha256
    ):
        raise SourceOnlyReplayError(
            "The source-only preparation receipt does not authenticate the inventory."
        )

    manifest = _load_json_object(
        manifest_path, label="internal reproducibility manifest"
    )
    manifest_rows = manifest.get("files")
    inventory_bindings = [
        row
        for row in manifest_rows
        if isinstance(row, dict) and row.get("archive_path") == INVENTORY_NAME
    ] if isinstance(manifest_rows, list) else []
    if (
        manifest.get("bundle_format")
        != "blaschke-deformation-certifier-reproducibility-v3"
        or len(inventory_bindings) != 1
        or inventory_bindings[0].get("sha256") != inventory_sha256
    ):
        raise SourceOnlyReplayError(
            "The internal manifest does not authenticate the replay inventory."
        )

    missing: list[str] = []
    generated_files: list[dict[str, object]] = []
    for value in generated_paths:
        relative = PurePosixPath(value)
        path = _regular_bundle_file(
            root,
            relative,
            label="declared generated member",
            allow_missing=True,
        )
        if path is None:
            missing.append(value)
            continue
        stat_result = path.stat()
        if not_before_ns is not None and stat_result.st_mtime_ns < not_before_ns:
            raise SourceOnlyReplayError(
                f"The theorem output was not freshly recreated in this run: {value}."
            )
        generated_files.append(
            {
                "path": value,
                "bytes": stat_result.st_size,
                "sha256": sha256_file(path),
                "mtime_ns": stat_result.st_mtime_ns,
            }
        )
    if missing:
        raise SourceOnlyReplayError(
            f"Generated output closure is incomplete; missing={missing}."
        )

    fingerprint_payload = {
        "count": len(generated_paths),
        "paths": generated_paths,
    }
    fingerprint_bytes = json.dumps(
        fingerprint_payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    fingerprint = hashlib.sha256(fingerprint_bytes).hexdigest()
    return {
        "closure_schema": "blaschke-declared-generated-closure-v1",
        "status": "declared generated output closure complete",
        "expected_inventory_authority": (
            "caller-supplied from authenticated external release manifest or "
            "externally recorded prelaunch preparation"
        ),
        "expected_inventory_sha256": expected_inventory_sha256,
        "inventory_sha256": inventory_sha256,
        "preparation_receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
        "generated_path_count": len(generated_paths),
        "generated_paths": generated_paths,
        "generated_paths_sha256": fingerprint,
        "generated_files": generated_files,
        "theorem_only": theorem_only,
        "omitted_generated_paths": omitted,
        "historical_diagnostics_authoritative": False if theorem_only else None,
        "fingerprint_serialization": (
            "UTF-8 JSON object with count and paths; ensure_ascii=true; "
            "separators=(',',':'); sort_keys=true"
        ),
        "missing": [],
        "undeclared_file_policy": (
            "Undeclared transient row-block and cache files are not closure members; "
            "this gate neither enumerates them nor counts them as declared output."
        ),
    }


def run_replay(
    *,
    bundle_root: Path,
    kernel_name: str,
    assembly_workers: int,
    surface_workers: int,
    prepare_only: bool,
    published_archive: Path | None,
    compute_only: bool = False,
    theorem_only_compute: bool = False,
    expected_inventory_sha256: str | None = None,
    raw_comparison_receipt: Path | None = None,
) -> dict[str, object]:
    root = Path(bundle_root).resolve(strict=True)
    if root.name != BUNDLE_ROOT_NAME or (root / ".git").exists():
        raise RuntimeError(
            "The replay root must be the exact extracted scratch bundle, never a Git working tree."
        )
    compute_stop = compute_only or theorem_only_compute
    if compute_stop and published_archive is not None:
        raise ValueError(
            "Compute-only replay cannot compare a published archive before "
            "normalization and provenance refresh."
        )
    if compute_stop and raw_comparison_receipt is not None:
        raise ValueError(
            "Compute-only replay stops before raw comparison and cannot consume "
            "a comparison receipt."
        )
    if not prepare_only and not compute_stop and raw_comparison_receipt is None:
        raise ValueError(
            "Full publication replay requires an external hash-bound raw "
            "comparison receipt."
        )
    if not prepare_only:
        expected_inventory_sha256 = _authenticate_inventory_before_preparation(
            root=root,
            expected_inventory_sha256=expected_inventory_sha256,
        )
    receipt_path = root / "source-only-replay-preparation.json"
    preparation = (
        check_prepared_bundle(root) if receipt_path.is_file() else prepare_bundle(root)
    )
    if prepare_only:
        return {"status": "prepared", "preparation": preparation, "commands": []}
    assert expected_inventory_sha256 is not None

    if theorem_only_compute:
        stale = [
            value
            for value in THEOREM_REQUIRED_GENERATED_PATHS
            if _regular_bundle_file(
                root,
                PurePosixPath(value),
                label="pre-compute theorem output",
                allow_missing=True,
            )
            is not None
        ]
        if stale:
            raise SourceOnlyReplayError(
                "Theorem-only replay did not begin from absent generated evidence: "
                f"{stale}."
            )

    if assembly_workers < 1 or surface_workers < 1:
        raise ValueError("Worker counts must be positive.")
    environment = dict(os.environ)
    environment.update({
        "BLASCHKE_FORCE_HARDY_MATRIX": "1",
        "BLASCHKE_FORCE_CONTOURS": "1",
        "MPMATH_PF_ASSEMBLY_WORKERS": str(assembly_workers),
        # PYTHONSAFEPATH prevents the interpreter from implicitly trusting the
        # current working directory or a script directory.  These are the only
        # two project import roots supplied to child processes, and ``root`` has
        # already been bound to the caller-authenticated replay inventory.
        "PYTHONSAFEPATH": "1",
        "PYTHONPATH": os.pathsep.join((str(root), str(root / "Numerics"))),
        "BLASCHKE_AUTHENTICATED_REPLAY_ROOT": str(root),
        **{key: "1" for key in BLAS_THREAD_ENVIRONMENT},
    })
    if theorem_only_compute:
        environment["BLASCHKE_SKIP_HARDY_STARTING_AUDIT"] = "1"
        environment["BLASCHKE_SKIP_HISTORICAL_PHASE4"] = "1"
        environment.pop("BLASCHKE_FORCE_HISTORICAL_PHASE4", None)
    else:
        environment["BLASCHKE_FORCE_HISTORICAL_PHASE4"] = "1"
    python = sys.executable
    if theorem_only_compute:
        commands = [
            [python, "-B", "-c", BLAS_RUNTIME_PREFLIGHT],
            [python, "-B", "Numerics/build_blaschke_deformation_certifier.py"],
            [python, "-B", "Numerics/build_blaschke_deformation_thesis_math_notebook.py"],
            [python, "-u", "-B", "-c", THEOREM_ONLY_DRIVER],
            [
                python,
                "-u",
                "-B",
                "Numerics/blaschke_deformation_contour_certification.py",
                "--root",
                ".",
                "--force",
            ],
        ]
    else:
        commands = [
            [python, "-B", "-c", BLAS_RUNTIME_PREFLIGHT],
            [python, "-B", "Numerics/build_blaschke_deformation_certifier.py"],
            [python, "-B", "Numerics/build_blaschke_deformation_thesis_math_notebook.py"],
            [
                python, "-u", "-B",
                "Numerics/blaschke_deformation_historical_phase4.py",
                "--production", "--assembly-workers", str(assembly_workers),
                "--surface-workers", str(surface_workers), "--force",
            ],
            [
                python, "-u", "-B", "Numerics/execute_notebook_incremental.py",
                "Numerics/blaschke_deformation_certifier_thesis_math.ipynb",
                "--kernel-name", kernel_name,
            ],
        ]
    run_started_ns = time.time_ns()
    command_records = [
        _run(command, root=root, environment=environment) for command in commands
    ]
    plan_source = root / "Numerics/blaschke_deformation_reproducibility_plan.json"
    plan_alias = root / (
        "Numerics/outputs/blaschke_deformation_certifier/reports/"
        "blaschke_deformation_reproducibility_plan.json"
    )
    plan_alias.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(plan_source, plan_alias)
    generated_closure = _verify_generated_closure(
        root=root,
        preparation=preparation,
        expected_inventory_sha256=expected_inventory_sha256,
        theorem_only=theorem_only_compute,
        not_before_ns=run_started_ns if theorem_only_compute else None,
    )
    forced_keys = [
        "BLASCHKE_FORCE_HARDY_MATRIX",
        "BLASCHKE_FORCE_CONTOURS",
        "MPMATH_PF_ASSEMBLY_WORKERS",
        "PYTHONSAFEPATH",
        "PYTHONPATH",
        "BLASCHKE_AUTHENTICATED_REPLAY_ROOT",
        *BLAS_THREAD_ENVIRONMENT,
    ]
    if theorem_only_compute:
        forced_keys.extend(
            (
                "BLASCHKE_SKIP_HARDY_STARTING_AUDIT",
                "BLASCHKE_SKIP_HISTORICAL_PHASE4",
            )
        )
    else:
        forced_keys.append("BLASCHKE_FORCE_HISTORICAL_PHASE4")
    forced_environment = {key: environment[key] for key in forced_keys}
    if compute_stop:
        receipt_path = root / "clean-room-compute-only-evidence.json"
        result = {
            "receipt_schema": (
                "blaschke-theorem-only-compute-v1"
                if theorem_only_compute
                else "blaschke-clean-room-compute-only-v1"
            ),
            "status": (
                "theorem-only source reconstruction complete"
                if theorem_only_compute
                else "compute-only source reconstruction complete"
            ),
            "stage": "theorem-only-compute" if theorem_only_compute else "compute-only",
            "normalization_run": False,
            "provenance_refresh_run": False,
            "published_comparison_run": False,
            "preparation": preparation,
            "python_executable": python,
            "kernel_name": kernel_name,
            "kernel_used_for_theorem_compute": False if theorem_only_compute else True,
            "forced_rebuild_environment": forced_environment,
            "historical_phase4_diagnostics_skipped": theorem_only_compute,
            "historical_phase4_diagnostics_authoritative": False,
            "hardy_1024_diagnostic_skipped": theorem_only_compute,
            "hardy_1024_diagnostic_authoritative": False,
            "run_started_ns": run_started_ns,
            "run_finished_ns": time.time_ns(),
            "commands": command_records,
            "reproducibility_plan_alias": plan_alias.relative_to(root).as_posix(),
            "generated_closure": generated_closure,
            "compute_receipt": receipt_path.name,
        }
        temporary_receipt = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
        temporary_receipt.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        temporary_receipt.replace(receipt_path)
        return result

    command_records.append(
        _run(
            [
                python,
                "-B",
                "Numerics/normalize_blaschke_publication.py",
                "--root",
                ".",
                "--raw-comparison-receipt",
                str(Path(raw_comparison_receipt).resolve(strict=True)),
            ],
            root=root,
            environment=environment,
        )
    )

    if published_archive is not None:
        archive = Path(published_archive).resolve(strict=True)
        verify_command = [
            python, "-B", "Numerics/verify_blaschke_deformation_reproducibility.py",
            str(archive), "--compare-executed-replay-root", ".",
        ]
        command_records.append(
            _run(verify_command, root=root, environment=environment)
        )
    result = {
        "status": (
            "source-only replay and published comparison complete"
            if published_archive is not None
            else "source-only replay complete; published comparison not requested"
        ),
        "preparation": preparation,
        "python_executable": python,
        "kernel_name": kernel_name,
        "forced_rebuild_environment": forced_environment,
        "commands": command_records,
        "generated_closure": generated_closure,
    }
    evidence_path = root / "clean-room-replay-evidence.json"
    evidence_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=Path.cwd())
    parser.add_argument("--kernel-name", default="blaschke-replay")
    parser.add_argument("--assembly-workers", type=int, default=24)
    parser.add_argument("--surface-workers", type=int, default=6)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--prepare-only", action="store_true")
    stage.add_argument(
        "--compute-only",
        action="store_true",
        help=(
            "Stop after the source-only numerical reconstruction and write an "
            "explicit receipt, before normalization, provenance, or comparison."
        ),
    )
    stage.add_argument(
        "--theorem-only-compute",
        action="store_true",
        help=(
            "Recompute only the 27 theorem-required artifacts, skipping historical "
            "Phase 4 and the non-authoritative 1024-bit starting audit."
        ),
    )
    parser.add_argument("--published-archive", type=Path)
    parser.add_argument(
        "--raw-comparison-receipt",
        type=Path,
        help=(
            "External PASS receipt from comparison of this replay's raw, "
            "pre-normalization members; required for full publication replay."
        ),
    )
    parser.add_argument(
        "--expected-inventory-sha256",
        help=(
            "Exact source-only inventory digest from the authenticated external "
            "release manifest, or from an externally recorded prepare-only launch."
        ),
    )
    args = parser.parse_args()
    result = run_replay(
        bundle_root=args.bundle_root,
        kernel_name=args.kernel_name,
        assembly_workers=args.assembly_workers,
        surface_workers=args.surface_workers,
        prepare_only=args.prepare_only,
        published_archive=args.published_archive,
        compute_only=args.compute_only,
        theorem_only_compute=args.theorem_only_compute,
        expected_inventory_sha256=args.expected_inventory_sha256,
        raw_comparison_receipt=args.raw_comparison_receipt,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
