"""Refresh or verify the checksum manifest for the current deployment tree.

The authoritative set consists of tracked files plus untracked, non-ignored
deployment sources and stored evidence.  Two explicitly historical notebook
copies are excluded, as is the separately packaged reproducibility directory.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile


EXCLUDED_PATHS = frozenset(
    {
        "MANIFEST.sha256",
        "Numerics/blaschke_deformation_certifier_thesis_math_backup_pre_appendix_20260823.ipynb",
        "Numerics/blaschke_deformation_certifier_thesis_math_dist.ipynb",
    }
)
EXCLUDED_PREFIXES = (
    "Numerics/outputs/blaschke_deformation_certifier/reproducibility/",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authoritative_paths(root: Path) -> tuple[Path, ...]:
    """Return the deterministic tracked-plus-untracked deployment file set."""

    completed = subprocess.run(
        ("git", "ls-files", "-co", "--exclude-standard", "-z"),
        cwd=root,
        check=True,
        capture_output=True,
    )
    relative_paths = completed.stdout.decode("utf-8").split("\0")
    selected: list[Path] = []
    for relative_text in relative_paths:
        if not relative_text:
            continue
        relative = Path(relative_text)
        normalised = relative.as_posix()
        if normalised in EXCLUDED_PATHS:
            continue
        if any(normalised.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            continue
        absolute = root / relative
        if absolute.is_file():
            selected.append(relative)
    return tuple(sorted(set(selected), key=lambda path: path.as_posix()))


def manifest_text(root: Path) -> str:
    return "".join(
        f"{sha256_file(root / relative)}  {relative.as_posix()}\n"
        for relative in authoritative_paths(root)
    )


def refresh_manifest(root: Path, manifest_path: Path) -> int:
    payload = manifest_text(root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, manifest_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return payload.count("\n")


def check_manifest(root: Path, manifest_path: Path) -> int:
    expected = manifest_text(root)
    observed = manifest_path.read_text(encoding="utf-8")
    if observed != expected:
        raise SystemExit(
            "MANIFEST.sha256 does not match the current authoritative deployment file set."
        )
    return expected.count("\n")


def main() -> None:
    root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "MANIFEST.sha256"
    if args.check:
        count = check_manifest(root, manifest_path)
        print(f"Verified {count} deployment files against {manifest_path}.")
    else:
        count = refresh_manifest(root, manifest_path)
        print(f"Recorded {count} deployment files in {manifest_path}.")


if __name__ == "__main__":
    main()
