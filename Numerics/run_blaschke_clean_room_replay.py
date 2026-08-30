"""Orchestrate a full source-only replay inside one extracted scratch bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import time

from prepare_blaschke_source_only_replay import (
    BUNDLE_ROOT_NAME,
    GENERATED_CLASSES,
    IMMUTABLE_CLASSES,
    INTERNAL_MANIFEST_NAME,
    INVENTORY_NAME,
    POLICY_RELATIVE,
    RECEIPT_NAME,
    SourceOnlyReplayError,
    check_prepared_bundle,
    prepare_bundle,
    sha256_file,
)


BLAS_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

BLAS_RUNTIME_PREFLIGHT = r"""
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


def _verify_generated_closure(
    *,
    root: Path,
    preparation: dict[str, object],
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

    on_disk_receipt = _load_json_object(
        receipt_path, label="source-only preparation receipt"
    )
    if on_disk_receipt != preparation:
        raise SourceOnlyReplayError(
            "The in-memory and on-disk source-only preparation receipts differ."
        )
    inventory = _load_json_object(
        inventory_path, label="source-only replay inventory"
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
    generated_paths = sorted(relative.as_posix() for relative in generated)
    if not generated_paths:
        raise SourceOnlyReplayError(
            "The source-only replay inventory declares no generated members."
        )

    if (
        type(on_disk_receipt.get("schema_version")) is not int
        or on_disk_receipt.get("schema_version") != 1
        or on_disk_receipt.get("status") != "source-only replay root prepared"
        or on_disk_receipt.get("bundle_root_name") != BUNDLE_ROOT_NAME
        or on_disk_receipt.get("removed_paths") != generated_paths
        or type(on_disk_receipt.get("removed_file_count")) is not int
        or on_disk_receipt.get("removed_file_count") != len(generated_paths)
        or type(on_disk_receipt.get("immutable_external_input_count")) is not int
        or on_disk_receipt.get("immutable_external_input_count")
        != observed_counts["immutable_external_input"]
    ):
        raise SourceOnlyReplayError(
            "The source-only preparation receipt does not match the inventory."
        )
    inventory_sha256 = sha256_file(inventory_path)
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
    for value in generated_paths:
        relative = PurePosixPath(value)
        if _regular_bundle_file(
            root,
            relative,
            label="declared generated member",
            allow_missing=True,
        ) is None:
            missing.append(value)
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
        "inventory_sha256": inventory_sha256,
        "preparation_receipt_sha256": sha256_file(receipt_path),
        "generated_path_count": len(generated_paths),
        "generated_paths": generated_paths,
        "generated_paths_sha256": fingerprint,
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
) -> dict[str, object]:
    root = Path(bundle_root).resolve(strict=True)
    if root.name != BUNDLE_ROOT_NAME or (root / ".git").exists():
        raise RuntimeError(
            "The replay root must be the exact extracted scratch bundle, never a Git working tree."
        )
    receipt_path = root / "source-only-replay-preparation.json"
    preparation = (
        check_prepared_bundle(root) if receipt_path.is_file() else prepare_bundle(root)
    )
    if prepare_only:
        return {"status": "prepared", "preparation": preparation, "commands": []}
    if compute_only and published_archive is not None:
        raise ValueError(
            "Compute-only replay cannot compare a published archive before "
            "normalization and provenance refresh."
        )

    if assembly_workers < 1 or surface_workers < 1:
        raise ValueError("Worker counts must be positive.")
    environment = dict(os.environ)
    environment.update({
        "BLASCHKE_FORCE_HARDY_MATRIX": "1",
        "BLASCHKE_FORCE_CONTOURS": "1",
        "BLASCHKE_FORCE_HISTORICAL_PHASE4": "1",
        "MPMATH_PF_ASSEMBLY_WORKERS": str(assembly_workers),
        **{key: "1" for key in BLAS_THREAD_ENVIRONMENT},
    })
    python = sys.executable
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
    )
    forced_environment = {
        key: environment[key]
        for key in (
            "BLASCHKE_FORCE_HARDY_MATRIX",
            "BLASCHKE_FORCE_CONTOURS",
            "BLASCHKE_FORCE_HISTORICAL_PHASE4",
            "MPMATH_PF_ASSEMBLY_WORKERS",
            *BLAS_THREAD_ENVIRONMENT,
        )
    }
    if compute_only:
        receipt_path = root / "clean-room-compute-only-evidence.json"
        result = {
            "receipt_schema": "blaschke-clean-room-compute-only-v1",
            "status": "compute-only source reconstruction complete",
            "stage": "compute-only",
            "normalization_run": False,
            "provenance_refresh_run": False,
            "published_comparison_run": False,
            "preparation": preparation,
            "python_executable": python,
            "kernel_name": kernel_name,
            "forced_rebuild_environment": forced_environment,
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
    parser.add_argument("--published-archive", type=Path)
    args = parser.parse_args()
    result = run_replay(
        bundle_root=args.bundle_root,
        kernel_name=args.kernel_name,
        assembly_workers=args.assembly_workers,
        surface_workers=args.surface_workers,
        prepare_only=args.prepare_only,
        published_archive=args.published_archive,
        compute_only=args.compute_only,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
