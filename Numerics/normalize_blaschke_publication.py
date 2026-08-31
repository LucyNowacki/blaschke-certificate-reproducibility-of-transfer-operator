"""Normalize retained Blaschke evidence for a portable public bundle.

The numerical producers need absolute paths while they are running.  Those
runtime locations are not evidence, however, and must not be persisted in the
published closure.  This module performs the narrow post-execution
normalization: executable artifact references are made relative to the
notebook directory, display-only paths are redacted, volatile cache locations
are replaced by content-addressed labels, and notebook runtime metadata is
made environment-neutral.

No numerical array, certificate endpoint, theorem gate, cell source, or
execution count is changed.  Hash fields which directly attest a normalized
member are refreshed after that member is rewritten.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import pickletools
import re
from typing import Any, Callable, Iterable
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class PublicationPortabilityError(RuntimeError):
    """Raised when the retained publication closure is not portable."""


OUTPUT_DIRECTORY_RELATIVE = "outputs/blaschke_deformation_certifier"
LOCKED_PYTHON_VERSION = "3.13.2"
PORTABLE_KERNELSPEC = {
    "display_name": f"Python {LOCKED_PYTHON_VERSION} (reproducibility)",
    "language": "python",
    "name": "blaschke-replay",
}

RAW_COMPARISON_RECEIPT_SCHEMA = (
    "numerics1-raw-reconstruction-comparison-v2-retirement-policy"
)
SOURCE_SYNC_SCHEMA = "numerics1-source-sync-after-execution-v1"

NOTEBOOK_RELATIVES = (
    "Numerics/blaschke_deformation_certifier_template.ipynb",
    "Numerics/blaschke_deformation_certifier.ipynb",
    "Numerics/blaschke_deformation_certifier_thesis_math.ipynb",
)

HARDY_STEMS = (
    "blaschke_deformation_balanced_hardy_reference_N600_M610_bits1024",
    "blaschke_deformation_balanced_hardy_reference_N600_M610_bits2048",
)

SURFACE_RELATIVES = {
    "local_surface": (
        "Numerics/outputs/blaschke_deformation_certifier/data/"
        "branch_image_wide_candidate_mu2_hardy_moat_surface_N600_M610_grid25.npz"
    ),
    "global_surface": (
        "Numerics/outputs/blaschke_deformation_certifier/data/"
        "branch_image_wide_candidate_first15_global_hardy_moat_surface_"
        "N600_M610_grid45.npz"
    ),
    "zoom_surface": (
        "Numerics/outputs/blaschke_deformation_certifier/data/"
        "branch_image_wide_candidate_first15_zoom_hardy_moat_surface_"
        "N600_M610_grid161.npz"
    ),
    "deep_zoom_surface": (
        "Numerics/outputs/blaschke_deformation_certifier/data/"
        "branch_image_wide_candidate_first15_deep_zoom_hardy_moat_surface_"
        "N600_M610_grid201.npz"
    ),
}

SPECTRAL_REPORT_RELATIVE = (
    "Numerics/outputs/blaschke_deformation_certifier/reports/"
    "blaschke_deformation_24_target_N600_M610_spectral_certificate.json"
)
SCHUR_REPORT_RELATIVE = (
    "Numerics/outputs/blaschke_deformation_certifier/reports/"
    "blaschke_deformation_24_target_N600_M610_validated_schur.json"
)
HISTORICAL_TRANSPORT_CSV_RELATIVE = (
    "Numerics/outputs/blaschke_deformation_certifier/data/"
    "historical_comparison_internal/"
    "branch_image_balanced_candidate_transport_cert_N600.csv"
)
HISTORICAL_TRANSPORT_REPORT_RELATIVE = (
    "Numerics/outputs/blaschke_deformation_certifier/reports/"
    "historical_comparison_internal/"
    "branch_image_balanced_candidate_transport_cert_N600.json"
)
HISTORICAL_PHASE2_REPORT_RELATIVE = (
    "Numerics/outputs/blaschke_deformation_certifier/reports/"
    "historical_phase2_comparison_rebuild.json"
)
HISTORICAL_PHASE4_REPORT_RELATIVE = (
    "Numerics/outputs/blaschke_deformation_certifier/reports/"
    "blaschke_deformation_historical_phase4_rebuild_N600_M610.json"
)

_ABSOLUTE_OUTPUT_PATTERN = re.compile(
    r"/[^,\r\n\"']*?/Numerics/outputs/blaschke_deformation_certifier/"
    r"[^,\r\n\"']+"
)
_INLINE_HELPER_PATTERN = re.compile(
    r"/[^'\"\r\n<>]*/blaschke_thesis_math_[^/'\"\r\n<>]+/"
    r"([A-Za-z0-9_.-]+\.py)"
)
_BUNDLE_DISPLAY_PATTERN = re.compile(r"/[^'\"\r\n<>]*/Numerics/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == int(length)
        and re.fullmatch(r"[0-9a-f]+", value) is not None
    )


def _safe_receipt_member(root: Path, value: object) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise PublicationPortabilityError(
            "Raw-comparison member paths must be non-empty strings."
        )
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise PublicationPortabilityError(
            f"Unsafe raw-comparison member path: {value!r}."
        )
    candidate = root.joinpath(*relative.parts)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PublicationPortabilityError(
                f"Raw-comparison member traverses a symlink: {value!r}."
            )
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise PublicationPortabilityError(
            f"Raw-comparison member is missing or escapes the bundle: {value!r}."
        ) from exc
    if not resolved.is_file():
        raise PublicationPortabilityError(
            f"Raw-comparison member is not a regular file: {value!r}."
        )
    return relative.as_posix(), resolved


def _load_raw_comparison_receipt(
    receipt_path: Path,
    *,
    root: Path,
) -> dict[str, object]:
    """Validate a raw receipt and derive hash-bound notebook metadata.

    Validation happens before any publication member is rewritten.  The
    receipt must live outside the candidate bundle and enumerate the exact raw
    members it compared, so a receipt from another replay cannot be reused.
    """

    root = Path(root).resolve(strict=True)
    receipt_path = Path(receipt_path)
    if receipt_path.is_symlink():
        raise PublicationPortabilityError(
            "The raw-comparison receipt must not be a symlink."
        )
    try:
        receipt_path = receipt_path.resolve(strict=True)
    except OSError as exc:
        raise PublicationPortabilityError(
            f"Cannot resolve the raw-comparison receipt: {receipt_path}."
        ) from exc
    try:
        receipt_path.relative_to(root)
    except ValueError:
        pass
    else:
        raise PublicationPortabilityError(
            "The raw-comparison receipt must be external to the candidate bundle."
        )
    if not receipt_path.is_file():
        raise PublicationPortabilityError(
            "The raw-comparison receipt is not a regular file."
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationPortabilityError(
            "The raw-comparison receipt is not valid UTF-8 JSON."
        ) from exc
    if not isinstance(receipt, dict):
        raise PublicationPortabilityError(
            "The raw-comparison receipt must be a JSON object."
        )
    required_state = {
        "schema": RAW_COMPARISON_RECEIPT_SCHEMA,
        "phase": "raw-pre-normalization",
        "status": "PASS",
        "comparison_run": True,
        "normalization_run": False,
        "provenance_refresh_run": False,
        "production_identity_checked": True,
        "failures": [],
    }
    if any(receipt.get(key) != value for key, value in required_state.items()):
        raise PublicationPortabilityError(
            "The raw-comparison receipt did not pass in raw pre-normalization form."
        )

    identity = receipt.get("candidate_identity")
    if not isinstance(identity, dict):
        raise PublicationPortabilityError(
            "The raw-comparison receipt lacks a candidate identity."
        )
    if not (
        _is_lower_hex(identity.get("commit"), 40)
        and _is_lower_hex(identity.get("tree"), 40)
        and _is_lower_hex(identity.get("manifest_sha256"), 64)
        and _is_lower_hex(
            identity.get("source_inventory_expected_sha256"), 64
        )
        and _is_lower_hex(
            identity.get("source_inventory_observed_sha256"), 64
        )
        and identity.get("source_inventory_expected_sha256")
        == identity.get("source_inventory_observed_sha256")
    ):
        raise PublicationPortabilityError(
            "The raw-comparison candidate identity is malformed or unauthenticated."
        )

    for key in (
        "comparator_sha256",
        "comparison_rules_sha256",
        "theorem_projection_sha256",
        "candidate_raw_members_sha256",
    ):
        if not _is_lower_hex(receipt.get(key), 64):
            raise PublicationPortabilityError(
                f"The raw-comparison receipt has an invalid {key}."
            )

    member_hashes = receipt.get("candidate_raw_member_sha256")
    generated_paths = receipt.get("generated_member_paths")
    if not isinstance(member_hashes, dict) or not member_hashes:
        raise PublicationPortabilityError(
            "The raw-comparison receipt has no hash-bound candidate members."
        )
    if not isinstance(generated_paths, list) or not generated_paths:
        raise PublicationPortabilityError(
            "The raw-comparison receipt has no generated-member set."
        )
    if generated_paths != sorted(generated_paths) or len(generated_paths) != len(
        set(generated_paths)
    ):
        raise PublicationPortabilityError(
            "The generated raw-comparison member paths are not unique and sorted."
        )
    if receipt.get("generated_member_count") != len(generated_paths):
        raise PublicationPortabilityError(
            "The raw-comparison generated-member count is inconsistent."
        )

    observed_member_hashes: dict[str, str] = {}
    for value, expected_digest in sorted(member_hashes.items()):
        if not _is_lower_hex(expected_digest, 64):
            raise PublicationPortabilityError(
                f"Malformed raw-comparison member digest for {value!r}."
            )
        relative, candidate = _safe_receipt_member(root, value)
        observed_digest = sha256_file(candidate)
        if observed_digest != expected_digest:
            raise PublicationPortabilityError(
                f"Raw-comparison member digest drifted: {relative}."
            )
        observed_member_hashes[relative] = observed_digest
    if list(member_hashes) != sorted(member_hashes):
        raise PublicationPortabilityError(
            "Raw-comparison candidate members must be serialized in path order."
        )
    if any(path not in observed_member_hashes for path in generated_paths):
        raise PublicationPortabilityError(
            "A generated raw-comparison path is absent from the member hashes."
        )
    required_members = {
        "MANIFEST.sha256",
        "source-only-replay-inventory.json",
        "reproducibility_manifest.json",
        *NOTEBOOK_RELATIVES,
    }
    if not required_members <= set(observed_member_hashes):
        raise PublicationPortabilityError(
            "The raw-comparison receipt omits required identity or notebook members."
        )
    if not {
        "Numerics/blaschke_deformation_certifier.ipynb",
        "Numerics/blaschke_deformation_certifier_thesis_math.ipynb",
    } <= set(generated_paths):
        raise PublicationPortabilityError(
            "The raw-comparison receipt did not classify both rebuilt notebooks as generated."
        )
    if (
        observed_member_hashes["MANIFEST.sha256"]
        != identity["manifest_sha256"]
        or observed_member_hashes["source-only-replay-inventory.json"]
        != identity["source_inventory_observed_sha256"]
    ):
        raise PublicationPortabilityError(
            "The raw-comparison candidate identity is not bound to this bundle."
        )
    fingerprint = _canonical_sha256(
        {"candidate_raw_member_sha256": observed_member_hashes}
    )
    if fingerprint != receipt["candidate_raw_members_sha256"]:
        raise PublicationPortabilityError(
            "The raw-comparison member-set fingerprint is invalid."
        )

    return {
        "schema": SOURCE_SYNC_SCHEMA,
        "status": "raw reconstruction comparison passed before normalization",
        "execution_binding": (
            "fresh candidate sources and stored outputs were compared in raw "
            "pre-normalization form"
        ),
        "raw_comparison": {
            "receipt_schema": receipt["schema"],
            "receipt_sha256": sha256_file(receipt_path),
            "phase": receipt["phase"],
            "status": receipt["status"],
            "production_identity_checked": True,
            "candidate_identity": dict(identity),
            "candidate_raw_members_sha256": fingerprint,
            "candidate_raw_member_count": len(observed_member_hashes),
            "generated_member_count": len(generated_paths),
            "comparator_sha256": receipt["comparator_sha256"],
            "comparison_rules_sha256": receipt["comparison_rules_sha256"],
            "theorem_projection_sha256": receipt[
                "theorem_projection_sha256"
            ],
        },
        "publication_normalization": {
            "role": "portable paths and runtime metadata only",
            "numerical_execution_performed": False,
            "numerical_arrays_or_theorem_fields_changed": False,
        },
    }


def _portable_output_path(value: object) -> str:
    text = str(value).replace("\\", "/")
    if text.startswith(f"{OUTPUT_DIRECTORY_RELATIVE}/"):
        return text
    numerics_relative = f"Numerics/{OUTPUT_DIRECTORY_RELATIVE}/"
    if text.startswith(numerics_relative):
        return text[len("Numerics/") :]
    marker = f"/Numerics/{OUTPUT_DIRECTORY_RELATIVE}/"
    marker_index = text.find(marker)
    if marker_index >= 0:
        return f"{OUTPUT_DIRECTORY_RELATIVE}/{text[marker_index + len(marker):]}"
    raise PublicationPortabilityError(
        f"Artifact path is outside the publication output tree: {value!r}."
    )


def _normalise_output_paths_in_text(text: str) -> str:
    return _ABSOLUTE_OUTPUT_PATTERN.sub(
        lambda match: _portable_output_path(match.group(0)), text
    )


def _normalise_display_text(text: str) -> str:
    # Image payloads dominate the executed notebook and cannot contain either
    # path form unless their decoded text marker is present.  Guarding the two
    # regular expressions keeps this pass linear in ordinary notebook size.
    normalised = text
    if "blaschke_thesis_math_" in normalised:
        normalised = _INLINE_HELPER_PATTERN.sub(
            lambda match: f"<inline-helper>/{match.group(1)}", normalised
        )
    if "/Numerics/" in normalised:
        normalised = _BUNDLE_DISPLAY_PATTERN.sub("Numerics/", normalised)
    return normalised


def _walk_strings(value: Any, transform: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return transform(value)
    if isinstance(value, list):
        return [_walk_strings(item, transform) for item in value]
    if isinstance(value, dict):
        return {
            key: _walk_strings(item, transform) for key, item in value.items()
        }
    return value


def _json_bytes(value: object, *, indent: int, sort_keys: bool) -> bytes:
    return (
        json.dumps(
            value,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes_if_changed(path: Path, payload: bytes) -> bool:
    if path.read_bytes() == payload:
        return False
    path.write_bytes(payload)
    return True


def _normalise_notebook(
    path: Path,
    source_sync_after_execution: dict[str, object],
) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    metadata = notebook.setdefault("metadata", {})
    metadata["kernelspec"] = dict(PORTABLE_KERNELSPEC)
    language_info = dict(metadata.get("language_info", {}))
    language_info.update(
        {
            "name": "python",
            "version": LOCKED_PYTHON_VERSION,
        }
    )
    metadata["language_info"] = language_info
    metadata["source_sync_after_execution"] = json.loads(
        json.dumps(source_sync_after_execution)
    )
    for cell in notebook.get("cells", []):
        if isinstance(cell, dict) and "outputs" in cell:
            cell["outputs"] = _walk_strings(
                cell.get("outputs", []), _normalise_display_text
            )
    return _write_bytes_if_changed(
        path, _json_bytes(notebook, indent=1, sort_keys=False)
    )


def _normalise_text_file(path: Path) -> bool:
    original = path.read_bytes()
    text = original.decode("utf-8")
    payload = _normalise_output_paths_in_text(text).encode("utf-8")
    if original == payload:
        return False
    path.write_bytes(payload)
    return True


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PublicationPortabilityError(f"Expected a JSON object at {path}.")
    return value


def _write_json_if_changed(path: Path, value: dict[str, Any]) -> bool:
    return _write_bytes_if_changed(
        path, _json_bytes(value, indent=2, sort_keys=True)
    )


def _normalise_path_fields(
    record: dict[str, Any], keys: Iterable[str]
) -> None:
    for key in keys:
        if key not in record:
            raise PublicationPortabilityError(f"Missing artifact path field {key!r}.")
        record[key] = _portable_output_path(record[key])


def _deterministic_npz_bytes(arrays: list[tuple[str, np.ndarray]]) -> bytes:
    archive_stream = io.BytesIO()
    with zipfile.ZipFile(
        archive_stream,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for name, array in arrays:
            member_stream = io.BytesIO()
            np.lib.format.write_array(
                member_stream,
                np.asarray(array),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, member_stream.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    return archive_stream.getvalue()


def _rewrite_npz_strings(
    path: Path, replacements: dict[str, Callable[[dict[str, np.ndarray]], str]]
) -> bool:
    with np.load(path, allow_pickle=False) as payload:
        arrays = [(name, np.array(payload[name], copy=True)) for name in payload.files]
    indexed = {name: array for name, array in arrays}
    changed = False
    for key, replacement in replacements.items():
        if key not in indexed:
            raise PublicationPortabilityError(f"Missing NPZ metadata field {key!r}.")
        array = indexed[key]
        if array.shape != () or array.dtype.kind not in "US":
            raise PublicationPortabilityError(
                f"NPZ metadata field {key!r} is not a scalar string."
            )
        desired = replacement(indexed)
        if str(array.item()) != desired:
            indexed[key] = np.asarray(desired)
            changed = True
    if not changed:
        return False
    ordered = [(name, indexed[name]) for name, _ in arrays]
    return _write_bytes_if_changed(path, _deterministic_npz_bytes(ordered))


def _hardy_paths(root: Path, stem: str) -> dict[str, Path]:
    output = root / "Numerics" / OUTPUT_DIRECTORY_RELATIVE
    return {
        "payload": output / "data" / f"{stem}.pkl.gz",
        "midpoint": output / "data" / f"{stem}_diagnostic_midpoint.npz",
        "summary": output / "data" / f"{stem}_certificate.csv",
        "report": output / "reports" / f"{stem}_certificate.json",
    }


def _normalise_hardy_members(root: Path, changed: list[str]) -> None:
    for stem in HARDY_STEMS:
        paths = _hardy_paths(root, stem)
        if _normalise_text_file(paths["summary"]):
            changed.append(str(paths["summary"].relative_to(root)))
        report = _load_json(paths["report"])
        _normalise_path_fields(
            report, ("payload_path", "midpoint_path", "summary_path", "report_path")
        )
        if _write_json_if_changed(paths["report"], report):
            changed.append(str(paths["report"].relative_to(root)))
        if _rewrite_npz_strings(
            paths["midpoint"],
            {
                "authoritative_exact_dyadic_payload": (
                    lambda _arrays: _portable_output_path(paths["payload"])
                )
            },
        ):
            changed.append(str(paths["midpoint"].relative_to(root)))


def _normalise_spectral_members(root: Path, changed: list[str]) -> None:
    schur_path = root / SCHUR_REPORT_RELATIVE
    schur = _load_json(schur_path)
    _normalise_path_fields(schur, ("cache_path",))
    if _write_json_if_changed(schur_path, schur):
        changed.append(SCHUR_REPORT_RELATIVE)

    report_path = root / SPECTRAL_REPORT_RELATIVE
    report = _load_json(report_path)
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise PublicationPortabilityError("Spectral artifact inventory is missing.")
    _normalise_path_fields(artifacts, tuple(artifacts))
    matrix = _hardy_paths(root, HARDY_STEMS[-1])
    for hash_group in ("geometry_input_hashes", "input_hashes"):
        values = report.get(hash_group)
        if not isinstance(values, dict):
            raise PublicationPortabilityError(
                f"Spectral report field {hash_group!r} is missing."
            )
        values["matrix_midpoint"] = sha256_file(matrix["midpoint"])
        values["matrix_report"] = sha256_file(matrix["report"])
    if _write_json_if_changed(report_path, report):
        changed.append(SPECTRAL_REPORT_RELATIVE)


def _normalise_historical_phase2(root: Path, changed: list[str]) -> None:
    csv_path = root / HISTORICAL_TRANSPORT_CSV_RELATIVE
    if _normalise_text_file(csv_path):
        changed.append(HISTORICAL_TRANSPORT_CSV_RELATIVE)

    internal_path = root / HISTORICAL_TRANSPORT_REPORT_RELATIVE
    internal = _load_json(internal_path)
    certificate = internal.get("certificate")
    if not isinstance(certificate, dict):
        raise PublicationPortabilityError("Historical transport certificate is missing.")
    _normalise_path_fields(certificate, ("inverse_witness_path",))
    _normalise_path_fields(internal, ("csv_path", "inverse_witness_path"))
    internal["csv_sha256"] = sha256_file(csv_path)
    if _write_json_if_changed(internal_path, internal):
        changed.append(HISTORICAL_TRANSPORT_REPORT_RELATIVE)

    outer_path = root / HISTORICAL_PHASE2_REPORT_RELATIVE
    outer = _load_json(outer_path)
    transport = outer.get("transport")
    if not isinstance(transport, dict) or not isinstance(transport.get("record"), dict):
        raise PublicationPortabilityError("Historical Phase 2 transport record is missing.")
    _normalise_path_fields(transport["record"], ("inverse_witness_path",))
    transport["csv_sha256"] = sha256_file(csv_path)
    transport["report_sha256"] = sha256_file(internal_path)
    if _write_json_if_changed(outer_path, outer):
        changed.append(HISTORICAL_PHASE2_REPORT_RELATIVE)


def _normalise_historical_phase4(root: Path, changed: list[str]) -> None:
    for relative in SURFACE_RELATIVES.values():
        path = root / relative
        if _rewrite_npz_strings(
            path,
            {
                "matrix_path": lambda arrays: (
                    "transient-cache/A_X_"
                    f"{str(np.asarray(arrays['A_X_sha256']).item())}.npy"
                ),
                "row_block_dir": lambda arrays: _portable_output_path(
                    np.asarray(arrays["row_block_dir"]).item()
                ),
            },
        ):
            changed.append(relative)

    report_path = root / HISTORICAL_PHASE4_REPORT_RELATIVE
    report = _load_json(report_path)
    outputs = report.get("outputs")
    if not isinstance(outputs, dict):
        raise PublicationPortabilityError("Historical Phase 4 outputs are missing.")
    for key, relative in SURFACE_RELATIVES.items():
        record = outputs.get(key)
        if not isinstance(record, dict):
            raise PublicationPortabilityError(
                f"Historical Phase 4 output {key!r} is missing."
            )
        record["sha256"] = sha256_file(root / relative)
    if _write_json_if_changed(report_path, report):
        changed.append(HISTORICAL_PHASE4_REPORT_RELATIVE)


def normalize_publication(
    root: Path,
    *,
    raw_comparison_receipt: Path,
) -> list[str]:
    """Normalize the exact retained publication members and dependent hashes."""

    root = Path(root).resolve(strict=True)
    source_sync = _load_raw_comparison_receipt(
        raw_comparison_receipt,
        root=root,
    )
    changed: list[str] = []
    for relative in NOTEBOOK_RELATIVES:
        if _normalise_notebook(root / relative, source_sync):
            changed.append(relative)
    _normalise_hardy_members(root, changed)
    _normalise_spectral_members(root, changed)
    _normalise_historical_phase2(root, changed)
    _normalise_historical_phase4(root, changed)
    assert_publication_portable(root)
    return sorted(set(changed))


def _iter_string_arrays(path: Path) -> Iterable[tuple[str, str]]:
    with np.load(path, allow_pickle=False) as payload:
        for key in payload.files:
            array = np.asarray(payload[key])
            if array.dtype.kind not in "US":
                continue
            for index, value in np.ndenumerate(array):
                yield f"{key}{index}", str(value)


def _forbidden_markers() -> tuple[str, ...]:
    return (
        "/" + "tmp",
        "/" + "home" + "/",
        "/" + "users" + "/",
    )


def _contains_forbidden_host_path(value: str | bytes) -> bool:
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return False
    else:
        text = value
    normalized = text.replace("\\", "/").lower()
    return any(marker in normalized for marker in _forbidden_markers())


_BINARY_NOTEBOOK_MIME_TYPES = frozenset(
    {
        "application/octet-stream",
        "application/pdf",
        "image/bmp",
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/webp",
    }
)


def _iter_notebook_text_values(
    value: Any, path: tuple[str, ...] = ()
) -> Iterable[tuple[str, str]]:
    """Yield semantic notebook strings, excluding opaque binary MIME payloads."""

    if isinstance(value, str):
        yield "/".join(path), value
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_notebook_text_values(item, path + (str(index),))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            yield "/".join(path + (f"key:{key_text}",)), key_text
            if key_text in _BINARY_NOTEBOOK_MIME_TYPES:
                continue
            yield from _iter_notebook_text_values(item, path + (key_text,))


def _parquet_column_can_contain_text(data_type: pa.DataType) -> bool:
    if (
        pa.types.is_string(data_type)
        or pa.types.is_large_string(data_type)
        or pa.types.is_binary(data_type)
        or pa.types.is_large_binary(data_type)
        or pa.types.is_fixed_size_binary(data_type)
    ):
        return True
    if pa.types.is_dictionary(data_type):
        return _parquet_column_can_contain_text(data_type.value_type)
    if pa.types.is_list(data_type) or pa.types.is_large_list(data_type):
        return _parquet_column_can_contain_text(data_type.value_type)
    if pa.types.is_fixed_size_list(data_type):
        return _parquet_column_can_contain_text(data_type.value_type)
    if pa.types.is_struct(data_type):
        return any(
            _parquet_column_can_contain_text(field.type) for field in data_type
        )
    if pa.types.is_map(data_type):
        return _parquet_column_can_contain_text(
            data_type.key_type
        ) or _parquet_column_can_contain_text(data_type.item_type)
    return False


def _iter_nested_text_values(value: Any) -> Iterable[str | bytes]:
    if isinstance(value, (str, bytes)):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_nested_text_values(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_nested_text_values(key)
            yield from _iter_nested_text_values(item)


def _iter_parquet_text_values(path: Path) -> Iterable[tuple[str, str | bytes]]:
    table = pq.read_table(path)
    for field, column in zip(table.schema, table.columns, strict=True):
        if not _parquet_column_can_contain_text(field.type):
            continue
        for row_index, value in enumerate(column.to_pylist()):
            for nested in _iter_nested_text_values(value):
                yield f"{field.name}[{row_index}]", nested


def _authoritative_paths(root: Path) -> tuple[Path, ...]:
    manifest = root / "MANIFEST.sha256"
    paths: list[Path] = []
    seen: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise PublicationPortabilityError("Malformed source manifest entry.")
        candidate_relative = Path(relative)
        if (
            candidate_relative.is_absolute()
            or ".." in candidate_relative.parts
            or relative in seen
        ):
            raise PublicationPortabilityError(
                f"Unsafe or duplicate source manifest path: {relative!r}."
            )
        candidate = root / candidate_relative
        try:
            candidate.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as exc:
            raise PublicationPortabilityError(
                f"Source manifest path escapes or is missing: {relative!r}."
            ) from exc
        if candidate.is_symlink() or not candidate.is_file():
            raise PublicationPortabilityError(
                f"Source manifest path is not a regular file: {relative!r}."
            )
        seen.add(relative)
        paths.append(candidate)
    # The manifest is excluded from its own checksum closure, but its relative
    # filenames are still public text and therefore part of the privacy scan.
    paths.append(manifest)
    return tuple(paths)


def _assert_zero_host_markers(root: Path) -> None:
    text_suffixes = {
        "",
        ".csv",
        ".ipynb",
        ".json",
        ".md",
        ".py",
        ".sha256",
        ".svg",
        ".toml",
        ".txt",
    }
    failures: list[str] = []
    for path in _authoritative_paths(root):
        if path.suffix == ".ipynb":
            try:
                notebook = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PublicationPortabilityError(
                    f"Cannot inspect notebook JSON in {path.relative_to(root)}."
                ) from exc
            for label, value in _iter_notebook_text_values(notebook):
                if _contains_forbidden_host_path(value):
                    failures.append(f"{path.relative_to(root)}::{label}")
        elif path.suffix in text_suffixes:
            text = path.read_text(encoding="utf-8", errors="strict")
            if _contains_forbidden_host_path(text):
                failures.append(str(path.relative_to(root)))
        elif path.suffix == ".npz":
            for label, value in _iter_string_arrays(path):
                if _contains_forbidden_host_path(value):
                    failures.append(f"{path.relative_to(root)}::{label}")
        elif path.suffix == ".parquet":
            for label, value in _iter_parquet_text_values(path):
                if _contains_forbidden_host_path(value):
                    failures.append(f"{path.relative_to(root)}::{label}")
        elif path.name.endswith(".pkl.gz"):
            with gzip.open(path, "rb") as stream:
                payload = stream.read()
            try:
                operations = pickletools.genops(payload)
                for opcode, argument, position in operations:
                    if isinstance(argument, (str, bytes)) and (
                        _contains_forbidden_host_path(argument)
                    ):
                        failures.append(
                            f"{path.relative_to(root)}::pickle-"
                            f"{opcode.name}@{position}"
                        )
            except ValueError as exc:
                raise PublicationPortabilityError(
                    f"Cannot inspect pickle opcodes in {path.relative_to(root)}."
                ) from exc
    if failures:
        raise PublicationPortabilityError(
            "Host-specific path markers remain in publication members: "
            + ", ".join(failures)
        )


def _validated_source_sync(value: object, *, relative: str) -> dict[str, object]:
    if not isinstance(value, dict) or value.get("schema") != SOURCE_SYNC_SCHEMA:
        raise PublicationPortabilityError(
            f"Notebook source-sync metadata is missing or invalid: {relative}."
        )
    if value.get("status") != (
        "raw reconstruction comparison passed before normalization"
    ):
        raise PublicationPortabilityError(
            f"Notebook source-sync status is invalid: {relative}."
        )
    raw = value.get("raw_comparison")
    if not isinstance(raw, dict):
        raise PublicationPortabilityError(
            f"Notebook raw-comparison binding is missing: {relative}."
        )
    identity = raw.get("candidate_identity")
    if not isinstance(identity, dict) or not (
        raw.get("receipt_schema") == RAW_COMPARISON_RECEIPT_SCHEMA
        and raw.get("phase") == "raw-pre-normalization"
        and raw.get("status") == "PASS"
        and raw.get("production_identity_checked") is True
        and _is_lower_hex(raw.get("receipt_sha256"), 64)
        and _is_lower_hex(raw.get("candidate_raw_members_sha256"), 64)
        and type(raw.get("candidate_raw_member_count")) is int
        and type(raw.get("generated_member_count")) is int
        and raw["candidate_raw_member_count"] >= raw["generated_member_count"] > 0
        and _is_lower_hex(raw.get("comparator_sha256"), 64)
        and _is_lower_hex(raw.get("comparison_rules_sha256"), 64)
        and _is_lower_hex(raw.get("theorem_projection_sha256"), 64)
        and _is_lower_hex(identity.get("commit"), 40)
        and _is_lower_hex(identity.get("tree"), 40)
        and _is_lower_hex(identity.get("manifest_sha256"), 64)
        and _is_lower_hex(
            identity.get("source_inventory_expected_sha256"), 64
        )
        and identity.get("source_inventory_expected_sha256")
        == identity.get("source_inventory_observed_sha256")
    ):
        raise PublicationPortabilityError(
            f"Notebook raw-comparison binding is malformed: {relative}."
        )
    expected_normalization = {
        "role": "portable paths and runtime metadata only",
        "numerical_execution_performed": False,
        "numerical_arrays_or_theorem_fields_changed": False,
    }
    if value.get("publication_normalization") != expected_normalization:
        raise PublicationPortabilityError(
            f"Notebook publication-normalization role is invalid: {relative}."
        )
    return value


def _assert_notebook_metadata(root: Path) -> None:
    common_source_sync: dict[str, object] | None = None
    for relative in NOTEBOOK_RELATIVES:
        notebook = json.loads((root / relative).read_text(encoding="utf-8"))
        metadata = notebook.get("metadata", {})
        if metadata.get("kernelspec") != PORTABLE_KERNELSPEC:
            raise PublicationPortabilityError(
                f"Notebook kernelspec is not portable: {relative}."
            )
        if metadata.get("language_info", {}).get("version") != LOCKED_PYTHON_VERSION:
            raise PublicationPortabilityError(
                f"Notebook Python version is not locked: {relative}."
            )
        source_sync = _validated_source_sync(
            metadata.get("source_sync_after_execution"),
            relative=relative,
        )
        if common_source_sync is None:
            common_source_sync = source_sync
        elif source_sync != common_source_sync:
            raise PublicationPortabilityError(
                "Notebook source-sync metadata does not share one raw receipt."
            )


def _assert_relative_artifacts_resolve(root: Path) -> None:
    notebook_directory = root / "Numerics"
    fields: list[tuple[str, str]] = []
    for stem in HARDY_STEMS:
        report = _load_json(_hardy_paths(root, stem)["report"])
        fields.extend(
            (f"{stem}.{key}", str(report[key]))
            for key in ("payload_path", "midpoint_path", "summary_path", "report_path")
        )
    schur = _load_json(root / SCHUR_REPORT_RELATIVE)
    fields.append(("schur.cache_path", str(schur["cache_path"])))
    spectral = _load_json(root / SPECTRAL_REPORT_RELATIVE)
    artifacts = spectral.get("artifacts", {})
    fields.extend((f"spectral.{key}", str(value)) for key, value in artifacts.items())
    internal = _load_json(root / HISTORICAL_TRANSPORT_REPORT_RELATIVE)
    fields.extend(
        (
            (
                "historical.certificate.inverse_witness_path",
                str(internal["certificate"]["inverse_witness_path"]),
            ),
            ("historical.csv_path", str(internal["csv_path"])),
            ("historical.inverse_witness_path", str(internal["inverse_witness_path"])),
        )
    )
    for label, value in fields:
        _resolve_bundle_artifact(notebook_directory, label=label, value=value)


def _resolve_bundle_artifact(
    notebook_directory: Path, *, label: str, value: str
) -> Path:
    notebook_directory = Path(notebook_directory).resolve(strict=True)
    candidate = Path(value)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise PublicationPortabilityError(
            f"Portable artifact field is not a safe relative path: {label}={value!r}."
        )
    unresolved = notebook_directory / candidate
    cursor = notebook_directory
    for part in candidate.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PublicationPortabilityError(
                f"Portable artifact field traverses a symlink: {label}={value!r}."
            )
    try:
        resolved = unresolved.resolve(strict=True)
        resolved.relative_to(notebook_directory)
    except (OSError, ValueError) as exc:
        raise PublicationPortabilityError(
            "Portable artifact field does not resolve inside the notebook "
            f"directory: {label}={value!r}."
        ) from exc
    if not resolved.is_file():
        raise PublicationPortabilityError(
            f"Portable artifact field is not a regular file: {label}={value!r}."
        )
    return resolved


def _assert_dependent_hashes(root: Path) -> None:
    matrix = _hardy_paths(root, HARDY_STEMS[-1])
    spectral = _load_json(root / SPECTRAL_REPORT_RELATIVE)
    for group in ("geometry_input_hashes", "input_hashes"):
        values = spectral[group]
        if values["matrix_midpoint"] != sha256_file(matrix["midpoint"]):
            raise PublicationPortabilityError("Stale spectral midpoint digest.")
        if values["matrix_report"] != sha256_file(matrix["report"]):
            raise PublicationPortabilityError("Stale spectral report digest.")

    csv_path = root / HISTORICAL_TRANSPORT_CSV_RELATIVE
    internal_path = root / HISTORICAL_TRANSPORT_REPORT_RELATIVE
    internal = _load_json(internal_path)
    outer = _load_json(root / HISTORICAL_PHASE2_REPORT_RELATIVE)
    if internal["csv_sha256"] != sha256_file(csv_path):
        raise PublicationPortabilityError("Stale historical transport CSV digest.")
    if outer["transport"]["csv_sha256"] != sha256_file(csv_path):
        raise PublicationPortabilityError("Stale outer historical transport CSV digest.")
    if outer["transport"]["report_sha256"] != sha256_file(internal_path):
        raise PublicationPortabilityError("Stale outer historical transport report digest.")

    phase4 = _load_json(root / HISTORICAL_PHASE4_REPORT_RELATIVE)
    for key, relative in SURFACE_RELATIVES.items():
        if phase4["outputs"][key]["sha256"] != sha256_file(root / relative):
            raise PublicationPortabilityError(
                f"Stale historical Phase 4 surface digest for {key}."
            )


def assert_publication_portable(root: Path) -> None:
    """Fail unless all retained text and compressed metadata are portable."""

    root = Path(root).resolve(strict=True)
    _assert_zero_host_markers(root)
    _assert_notebook_metadata(root)
    _assert_relative_artifacts_resolve(root)
    _assert_dependent_hashes(root)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--raw-comparison-receipt",
        type=Path,
        help=(
            "External hash-bound PASS receipt for the raw pre-normalization "
            "reconstruction comparison; required when normalization is run."
        ),
    )
    arguments = parser.parse_args()
    if arguments.check:
        assert_publication_portable(arguments.root)
        print("Publication portability check passed.")
        return
    if arguments.raw_comparison_receipt is None:
        parser.error(
            "--raw-comparison-receipt is required unless --check is used"
        )
    changed = normalize_publication(
        arguments.root,
        raw_comparison_receipt=arguments.raw_comparison_receipt,
    )
    print(json.dumps({"changed": changed, "changed_count": len(changed)}, indent=2))


if __name__ == "__main__":
    _main()
