"""Authenticate the Git-tree release and stage a source-only replay bundle.

The committed release inventory is the authority for the GitHub release unit.
This script verifies that finite tree without importing project code, copies it
to a fresh narrow scratch directory, and creates the two root-level metadata
aliases required by the existing clean-room compute orchestrator.
"""

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
from typing import Mapping


BUNDLE_ROOT_NAME = "blaschke_deformation_certifier_reproducibility"
RELEASE_INVENTORY = PurePosixPath("release/source-only-replay-inventory.json")
RELEASE_MANIFEST = PurePosixPath("release/reproducibility_manifest.json")
RELEASE_REPLAY = PurePosixPath("release/REPLAY.md")
ROOT_INVENTORY = PurePosixPath("source-only-replay-inventory.json")
ROOT_MANIFEST = PurePosixPath("reproducibility_manifest.json")
ROOT_REPLAY = PurePosixPath("REPLAY.md")
SOURCE_ONLY_POLICY = PurePosixPath("Numerics/blaschke_source_only_replay_policy.json")

METADATA_EXCLUDES = frozenset({RELEASE_INVENTORY, RELEASE_MANIFEST})
WALK_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".certificate-replay-env",
        ".certificate-replay-work",
        "__pycache__",
    }
)
WALK_EXCLUDED_PREFIXES = (
    PurePosixPath("Numerics/outputs/blaschke_deformation_certifier/reproducibility"),
)
ALLOWED_CLASSES = frozenset(
    {
        "immutable_source_or_metadata",
        "immutable_external_input",
        "generated_evidence",
        "generated_notebook",
    }
)


class StageError(RuntimeError):
    """Raised when release validation or safe staging fails."""


def _canonical_bytes(value: object) -> bytes:
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
        raise StageError(f"{label} must be a non-empty path string.")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or value in {".", ".."}
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise StageError(f"Unsafe {label}: {value!r}.")
    return relative


def _root(root: Path) -> Path:
    if root.is_symlink():
        raise StageError("Source release root must not be a symlink.")
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise StageError(f"Source release root does not exist: {root}.") from exc
    if not resolved.is_dir() or resolved == Path(resolved.anchor):
        raise StageError("Source release root must be a narrow directory.")
    return resolved


def _safe_file(root: Path, relative: PurePosixPath, *, label: str) -> Path:
    candidate = root
    for index, part in enumerate(relative.parts):
        candidate = candidate / part
        if candidate.is_symlink():
            raise StageError(f"Symlink forbidden for {label}: {relative.as_posix()}.")
        if not candidate.exists():
            raise StageError(f"Missing {label}: {relative.as_posix()}.")
        if index < len(relative.parts) - 1 and not candidate.is_dir():
            raise StageError(f"Non-directory component for {label}: {relative.as_posix()}.")
    try:
        mode = candidate.lstat().st_mode
        candidate.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise StageError(f"{label} escapes the release root.") from exc
    if not stat.S_ISREG(mode):
        raise StageError(f"{label} is not a regular file: {relative.as_posix()}.")
    return candidate


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise StageError(f"{label} must be a JSON object.")
    return value


def _under_prefix(relative: PurePosixPath, prefix: PurePosixPath) -> bool:
    return relative == prefix or prefix in relative.parents


def _actual_release_files(root: Path) -> set[PurePosixPath]:
    files: set[PurePosixPath] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        relative_directory = PurePosixPath(current.relative_to(root).as_posix())
        directory_names[:] = [
            name
            for name in directory_names
            if name not in WALK_EXCLUDED_DIRECTORIES
            and not any(
                _under_prefix(
                    PurePosixPath(
                        (relative_directory / name).as_posix()
                        if relative_directory != PurePosixPath(".")
                        else name
                    ),
                    prefix,
                )
                for prefix in WALK_EXCLUDED_PREFIXES
            )
        ]
        for name in (*directory_names, *file_names):
            if (current / name).is_symlink():
                raise StageError(f"Symlink forbidden in release tree: {current / name}.")
        for name in file_names:
            relative = PurePosixPath((current / name).relative_to(root).as_posix())
            if relative in METADATA_EXCLUDES or any(
                _under_prefix(relative, prefix) for prefix in WALK_EXCLUDED_PREFIXES
            ):
                continue
            if name.endswith(".pyc"):
                continue
            files.add(relative)
    return files


def _load_policy(root: Path) -> dict[str, object]:
    policy = _load_json(
        _safe_file(root, SOURCE_ONLY_POLICY, label="source-only replay policy"),
        label="source-only replay policy",
    )
    if (
        policy.get("schema_version") != 1
        or policy.get("bundle_root_name") != BUNDLE_ROOT_NAME
    ):
        raise StageError("Unsupported source-only replay policy.")
    return policy


def _classify(relative: PurePosixPath, policy: Mapping[str, object]) -> str:
    generated_files = {
        _safe_relative(value, label="generated file")
        for value in policy.get("generated_files", [])
    }
    generated_prefixes = tuple(
        _safe_relative(value, label="generated prefix")
        for value in policy.get("generated_prefixes", [])
    )
    external_inputs = {
        _safe_relative(value, label="immutable external input")
        for value in policy.get("immutable_external_inputs", [])
    }
    if relative in generated_files:
        return "generated_notebook"
    if any(_under_prefix(relative, prefix) for prefix in generated_prefixes):
        return "generated_evidence"
    if relative in external_inputs:
        return "immutable_external_input"
    return "immutable_source_or_metadata"


def _record(root: Path, relative: PurePosixPath, policy: Mapping[str, object]) -> dict[str, object]:
    path = _safe_file(root, relative, label="release member")
    return {
        "path": relative.as_posix(),
        "classification": _classify(relative, policy),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def build_release_metadata(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Build deterministic release metadata from the current finite tree."""

    root = _root(root)
    policy = _load_policy(root)
    files = sorted(_actual_release_files(root), key=PurePosixPath.as_posix)
    rows = [_record(root, relative, policy) for relative in files]
    counts = {
        classification: sum(row["classification"] == classification for row in rows)
        for classification in sorted(ALLOWED_CLASSES)
    }
    inventory = {
        "schema_version": 1,
        "bundle_root_name": BUNDLE_ROOT_NAME,
        "release_unit": "git-tree",
        "policy_path": SOURCE_ONLY_POLICY.as_posix(),
        "policy_sha256": _sha256_file(root.joinpath(*SOURCE_ONLY_POLICY.parts)),
        "complete_file_set_excludes": [
            RELEASE_INVENTORY.as_posix(),
            RELEASE_MANIFEST.as_posix(),
        ],
        "counts": counts,
        "files": rows,
        "external_input_classification": (
            "No immutable external numerical inputs: the map, constants, and exact "
            "targets are source encoded; Numerics/outputs is generated evidence."
        ),
    }
    inventory_payload = _canonical_bytes(inventory)
    tree_projection = [
        {
            "path": row["path"],
            "classification": row["classification"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in rows
    ]
    manifest = {
        "schema_version": 1,
        "release_unit": "git-tree",
        "release_policy": "certificate-equivalent source replay",
        "inventory_path": RELEASE_INVENTORY.as_posix(),
        "inventory_sha256": _sha256_bytes(inventory_payload),
        "tree_file_count": len(rows) + 2,
        "inventory_record_count": len(rows),
        "tree_projection_sha256": _sha256_bytes(_canonical_bytes(tree_projection)),
        "environment_locks": {
            "conda_explicit": {
                "path": "conda-explicit-lock.txt",
                "sha256": _sha256_file(root / "conda-explicit-lock.txt"),
                "platform": "linux-64",
            },
            "pip_overrides": {
                "path": "pip-requirements-lock.txt",
                "sha256": _sha256_file(root / "pip-requirements-lock.txt"),
                "require_hashes": True,
            },
        },
        "full_replay_verdict": "CERTIFICATION_CONFIRMED",
        "artifact_only_verdict": "ARTIFACT_SEMANTICS_CONFIRMED_NON_INDEPENDENT",
    }
    return inventory, manifest


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _copy_authenticated_file(
    source: Path,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> None:
    """Copy one member and immediately recheck the trusted row on destination."""

    if destination.exists() or destination.is_symlink():
        raise StageError(f"Refusing to replace staged {label}: {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for parent in (destination.parent, *destination.parent.parents):
        if parent.is_symlink():
            raise StageError(f"Symlink forbidden in staged {label} path: {parent}.")
        if parent == destination.anchor:
            break
    shutil.copyfile(source, destination)
    try:
        mode = destination.lstat().st_mode
        observed_size = destination.stat().st_size
        observed_sha256 = _sha256_file(destination)
    except OSError as exc:
        raise StageError(f"Cannot authenticate copied {label}: {destination}.") from exc
    if (
        not stat.S_ISREG(mode)
        or observed_size != expected_size
        or observed_sha256 != expected_sha256
    ):
        raise StageError(
            f"Copied {label} does not match its authenticated release row: {destination}."
        )


def _empty_workspace(path: Path) -> Path:
    """Require an existing, empty directory with no symlink in its full path."""

    raw = path if path.is_absolute() else Path.cwd() / path
    if ".." in raw.parts:
        raise StageError("Replay workspace path must not contain parent traversal.")
    candidate = Path(raw.anchor)
    for part in raw.parts[1:]:
        candidate = candidate / part
        if candidate.is_symlink():
            raise StageError(f"Symlink forbidden in replay workspace path: {candidate}.")
        if not candidate.exists():
            raise StageError(
                "Replay workspace must be a freshly created existing directory."
            )
        if not candidate.is_dir():
            raise StageError(f"Replay workspace component is not a directory: {candidate}.")
    resolved = raw.resolve(strict=True)
    try:
        first_member = next(resolved.iterdir())
    except StopIteration:
        first_member = None
    if first_member is not None:
        raise StageError(
            "Replay workspace must be empty and fresh; found " f"{first_member.name!r}."
        )
    return resolved


def write_release_metadata(root: Path) -> dict[str, object]:
    root = _root(root)
    inventory, manifest = build_release_metadata(root)
    _atomic_write(root.joinpath(*RELEASE_INVENTORY.parts), _canonical_bytes(inventory))
    _atomic_write(root.joinpath(*RELEASE_MANIFEST.parts), _canonical_bytes(manifest))
    return {
        "status": "release metadata written",
        "inventory_sha256": manifest["inventory_sha256"],
        "inventory_record_count": manifest["inventory_record_count"],
    }


def _validate_external_identity(
    root: Path,
    *,
    inventory_sha256: str,
    expected_inventory_sha256: str | None,
    expected_git_commit: str | None,
) -> dict[str, str]:
    selected = sum(value is not None for value in (expected_inventory_sha256, expected_git_commit))
    if selected != 1:
        raise StageError(
            "Exactly one caller-supplied expected inventory SHA-256 or Git commit is required."
        )
    if expected_inventory_sha256 is not None:
        if not _is_sha256(expected_inventory_sha256) or expected_inventory_sha256 != inventory_sha256:
            raise StageError("Release inventory does not match the caller-supplied digest.")
        return {"authority": "caller-supplied inventory SHA-256", "value": inventory_sha256}
    assert expected_git_commit is not None
    if (
        len(expected_git_commit) != 40
        or expected_git_commit.lower() != expected_git_commit
        or any(character not in "0123456789abcdef" for character in expected_git_commit)
    ):
        raise StageError("Expected Git commit must be a lowercase 40-hex object ID.")
    if not (root / ".git").exists():
        raise StageError("Caller selected Git identity but the release is not a Git checkout.")
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
        raise StageError(f"Cannot authenticate the Git release identity: {exc}") from exc
    if head != expected_git_commit or dirty:
        raise StageError(
            "Git release identity mismatch or non-ignored working-tree changes are present."
        )
    return {"authority": "caller-supplied Git commit", "value": head}


def validate_release(
    root: Path,
    *,
    expected_inventory_sha256: str | None,
    expected_git_commit: str | None,
) -> dict[str, object]:
    root = _root(root)
    inventory_path = _safe_file(root, RELEASE_INVENTORY, label="release inventory")
    manifest_path = _safe_file(root, RELEASE_MANIFEST, label="release manifest")
    inventory_payload = inventory_path.read_bytes()
    inventory = _load_json(inventory_path, label="release inventory")
    manifest = _load_json(manifest_path, label="release manifest")
    inventory_sha256 = _sha256_bytes(inventory_payload)
    external_identity = _validate_external_identity(
        root,
        inventory_sha256=inventory_sha256,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_git_commit=expected_git_commit,
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("release_unit") != "git-tree"
        or manifest.get("inventory_path") != RELEASE_INVENTORY.as_posix()
        or manifest.get("inventory_sha256") != inventory_sha256
    ):
        raise StageError("Release manifest does not authenticate its inventory.")
    if (
        inventory.get("schema_version") != 1
        or inventory.get("bundle_root_name") != BUNDLE_ROOT_NAME
        or inventory.get("release_unit") != "git-tree"
        or inventory.get("policy_path") != SOURCE_ONLY_POLICY.as_posix()
    ):
        raise StageError("Unsupported release inventory schema.")
    policy = _load_policy(root)
    if inventory.get("policy_sha256") != _sha256_file(root.joinpath(*SOURCE_ONLY_POLICY.parts)):
        raise StageError("Release inventory policy hash drifted.")
    rows = inventory.get("files")
    if not isinstance(rows, list) or not rows:
        raise StageError("Release inventory is empty.")
    seen: set[PurePosixPath] = set()
    observed_paths: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise StageError("Malformed release inventory row.")
        relative = _safe_relative(row.get("path"), label="release inventory path")
        if relative in seen:
            raise StageError(f"Duplicate release inventory path: {relative.as_posix()}.")
        seen.add(relative)
        observed_paths.append(relative.as_posix())
        classification = row.get("classification")
        size = row.get("bytes")
        digest = row.get("sha256")
        path = _safe_file(root, relative, label="release member")
        if (
            classification not in ALLOWED_CLASSES
            or classification != _classify(relative, policy)
            or type(size) is not int
            or size < 0
            or not _is_sha256(digest)
            or path.stat().st_size != size
            or _sha256_file(path) != digest
        ):
            raise StageError(f"Release member binding failed: {relative.as_posix()}.")
    if observed_paths != sorted(observed_paths):
        raise StageError("Release inventory paths are not sorted.")
    actual = _actual_release_files(root)
    if actual != seen:
        missing = sorted(path.as_posix() for path in seen - actual)
        extra = sorted(path.as_posix() for path in actual - seen)
        raise StageError(f"Git-tree release closure drifted; missing={missing}, extra={extra}.")
    if manifest.get("inventory_record_count") != len(rows):
        raise StageError("Release manifest inventory count drifted.")
    lock_records = manifest.get("environment_locks")
    if not isinstance(lock_records, dict):
        raise StageError("Release manifest environment locks are malformed.")
    for name, relative in (
        ("conda_explicit", PurePosixPath("conda-explicit-lock.txt")),
        ("pip_overrides", PurePosixPath("pip-requirements-lock.txt")),
    ):
        record = lock_records.get(name)
        if (
            not isinstance(record, dict)
            or record.get("path") != relative.as_posix()
            or record.get("sha256")
            != _sha256_file(_safe_file(root, relative, label=f"{name} lock"))
        ):
            raise StageError(f"Release {name} lock binding failed.")
    conda_text = (root / "conda-explicit-lock.txt").read_text(encoding="utf-8")
    pip_text = (root / "pip-requirements-lock.txt").read_text(encoding="utf-8")
    if (
        "# platform: linux-64" not in conda_text
        or "@EXPLICIT" not in conda_text
        or "python-3.13.2-" not in conda_text
        or "numpy-2.3.2-" not in conda_text
        or "scipy-1.16.0-" not in conda_text
        or "libopenblas-0.3.30-pthreads" not in conda_text
    ):
        raise StageError("Conda explicit lock lacks the required Linux runtime pins.")
    pip_rows = [
        line for line in pip_text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(pip_rows) != 7 or any(" --hash=sha256:" not in line for line in pip_rows):
        raise StageError("Pip override lock must contain seven exact hashed requirements.")
    return {
        "status": "git-tree release validated",
        "inventory_sha256": inventory_sha256,
        "inventory_record_count": len(rows),
        "records": rows,
        "policy": policy,
        "external_identity": external_identity,
        "release_metadata_records": {
            RELEASE_INVENTORY.as_posix(): {
                "bytes": len(inventory_payload),
                "sha256": inventory_sha256,
            },
            RELEASE_MANIFEST.as_posix(): {
                "bytes": manifest_path.stat().st_size,
                "sha256": _sha256_file(manifest_path),
            },
        },
    }


def _stage_inventory(root: Path, policy: Mapping[str, object]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in (*directory_names, *file_names):
            if (current / name).is_symlink():
                raise StageError(f"Symlink forbidden in staged tree: {current / name}.")
        for name in file_names:
            relative = PurePosixPath((current / name).relative_to(root).as_posix())
            if relative in {ROOT_INVENTORY, ROOT_MANIFEST}:
                continue
            path = current / name
            rows.append(
                {
                    "path": relative.as_posix(),
                    "classification": _classify(relative, policy),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    rows.sort(key=lambda row: str(row["path"]))
    counts = {
        classification: sum(row["classification"] == classification for row in rows)
        for classification in sorted(ALLOWED_CLASSES)
    }
    return {
        "schema_version": 1,
        "bundle_root_name": BUNDLE_ROOT_NAME,
        "policy_path": SOURCE_ONLY_POLICY.as_posix(),
        "policy_sha256": _sha256_file(root.joinpath(*SOURCE_ONLY_POLICY.parts)),
        "complete_file_set_excludes": [ROOT_INVENTORY.as_posix(), ROOT_MANIFEST.as_posix()],
        "counts": counts,
        "files": rows,
        "external_input_classification": (
            "No external numerical inputs; generated evidence is removed before replay."
        ),
    }


def stage_release(
    source_root: Path,
    workspace: Path,
    *,
    expected_inventory_sha256: str | None,
    expected_git_commit: str | None,
) -> dict[str, object]:
    source_root = _root(source_root)
    validation = validate_release(
        source_root,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_git_commit=expected_git_commit,
    )
    workspace = _empty_workspace(Path(workspace))
    _require_separate_workspace(source_root, workspace)
    staged_root = workspace / BUNDLE_ROOT_NAME
    if staged_root.exists() or staged_root.is_symlink():
        raise StageError(f"Refusing to replace existing staged root: {staged_root}.")
    staged_root.mkdir()
    try:
        records_by_path = {str(row["path"]): row for row in validation["records"]}
        for row in validation["records"]:
            relative = _safe_relative(row["path"], label="release inventory path")
            source = _safe_file(source_root, relative, label="release member")
            destination = staged_root.joinpath(*relative.parts)
            _copy_authenticated_file(
                source,
                destination,
                expected_size=int(row["bytes"]),
                expected_sha256=str(row["sha256"]),
                label=f"release member {relative.as_posix()}",
            )
        for relative in (RELEASE_INVENTORY, RELEASE_MANIFEST):
            source = _safe_file(source_root, relative, label="release metadata")
            destination = staged_root.joinpath(*relative.parts)
            record = validation["release_metadata_records"][relative.as_posix()]
            _copy_authenticated_file(
                source,
                destination,
                expected_size=int(record["bytes"]),
                expected_sha256=str(record["sha256"]),
                label=f"release metadata {relative.as_posix()}",
            )
        replay_record = records_by_path[RELEASE_REPLAY.as_posix()]
        _copy_authenticated_file(
            _safe_file(source_root, RELEASE_REPLAY, label="release replay guide"),
            staged_root.joinpath(*ROOT_REPLAY.parts),
            expected_size=int(replay_record["bytes"]),
            expected_sha256=str(replay_record["sha256"]),
            label="root replay-guide alias",
        )
        policy = _load_policy(staged_root)
        inventory = _stage_inventory(staged_root, policy)
        inventory_payload = _canonical_bytes(inventory)
        inventory_sha256 = _sha256_bytes(inventory_payload)
        _atomic_write(staged_root.joinpath(*ROOT_INVENTORY.parts), inventory_payload)
        file_records = [
            {
                "archive_path": row["path"],
                "sha256": row["sha256"],
                "bytes": row["bytes"],
                "role": row["classification"],
            }
            for row in inventory["files"]
        ]
        file_records.append(
            {
                "archive_path": ROOT_INVENTORY.as_posix(),
                "sha256": inventory_sha256,
                "bytes": len(inventory_payload),
                "role": "source-only replay inventory",
            }
        )
        file_records.sort(key=lambda row: str(row["archive_path"]))
        manifest = {
            "schema_version": 3,
            "bundle_format": "blaschke-deformation-certifier-reproducibility-v3",
            "release_unit": "authenticated Git tree staged for semantic replay",
            "release_inventory_sha256": validation["inventory_sha256"],
            "files": file_records,
        }
        _atomic_write(staged_root.joinpath(*ROOT_MANIFEST.parts), _canonical_bytes(manifest))
    except Exception:
        # Keep a partially staged tree for forensic inspection; never mutate source_root.
        raise
    return {
        "status": "fresh replay tree staged",
        "staged_root": str(staged_root),
        "release_inventory_sha256": validation["inventory_sha256"],
        "staged_inventory_sha256": inventory_sha256,
        "staged_inventory_record_count": len(inventory["files"]),
        "tracked_source_bytes_mutated": False,
        "external_release_identity": validation["external_identity"],
    }


def _require_separate_workspace(source_root: Path, workspace: Path) -> None:
    """Reject either direction of source/workspace containment."""

    overlaps = False
    try:
        workspace.relative_to(source_root)
        overlaps = True
    except ValueError:
        pass
    try:
        source_root.relative_to(workspace)
        overlaps = True
    except ValueError:
        pass
    if workspace == Path(workspace.anchor) or overlaps:
        raise StageError("Replay workspace must be narrow and separate from the source tree.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--write-release-metadata", action="store_true")
    identity = parser.add_mutually_exclusive_group()
    identity.add_argument("--expected-inventory-sha256")
    identity.add_argument("--expected-git-commit")
    arguments = parser.parse_args()
    selected = sum(
        (
            arguments.check_only,
            arguments.write_release_metadata,
            arguments.workspace is not None,
        )
    )
    if selected != 1:
        parser.error(
            "select exactly one of --check-only, --write-release-metadata, or --workspace"
        )
    try:
        if arguments.write_release_metadata:
            result = write_release_metadata(arguments.source_root)
        elif arguments.check_only:
            result = validate_release(
                arguments.source_root,
                expected_inventory_sha256=arguments.expected_inventory_sha256,
                expected_git_commit=arguments.expected_git_commit,
            )
            result = {
                key: value
                for key, value in result.items()
                if key not in {"records", "policy", "release_metadata_records"}
            }
        else:
            assert arguments.workspace is not None
            result = stage_release(
                arguments.source_root,
                arguments.workspace,
                expected_inventory_sha256=arguments.expected_inventory_sha256,
                expected_git_commit=arguments.expected_git_commit,
            )
    except StageError as exc:
        print(
            _canonical_bytes(
                {
                    "status": "STAGING_FAILED",
                    "error": str(exc),
                }
            ).decode("utf-8"),
            end="",
        )
        return 1
    print(_canonical_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
