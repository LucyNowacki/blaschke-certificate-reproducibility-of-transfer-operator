"""Build a referee-level reproducibility bundle for the Blaschke certifier."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Iterable, Mapping


PACKAGE_NAMES = (
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "mpmath",
    "python-flint",
    "nbformat",
    "nbclient",
    "jupyter",
    "ipykernel",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _copy_with_record(
    source: Path,
    destination: Path,
    archive_root: Path,
    repo_root: Path,
    records: list[dict[str, object]],
    role: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    try:
        source_label = str(source.relative_to(repo_root))
    except ValueError:
        source_label = str(source)
    records.append(
        {
            "archive_path": str(destination.relative_to(archive_root)),
            "source_path": source_label,
            "role": role,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
    )


def _record_generated(
    path: Path,
    archive_root: Path,
    records: list[dict[str, object]],
    role: str,
) -> None:
    """Record a generated archive member after its bytes are final."""

    records.append(
        {
            "archive_path": str(path.relative_to(archive_root)),
            "source_path": "generated within reproducibility bundle",
            "role": role,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    )


def _validate_inline_helper_provenance(repo_root: Path, notebook_path: Path) -> None:
    """Reject an archive whose inline helpers differ from standalone sources."""

    builder_path = (
        repo_root / "Numerics" / "build_blaschke_deformation_thesis_math_notebook.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_blaschke_thesis_math_builder_provenance",
        builder_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load inline-helper validator from {builder_path}.")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    builder.validate_inline_helper_sync(notebook)


def build_reproducibility_bundle(
    *,
    repo_root: Path,
    notebook_path: Path,
    output_dir: Path,
    precision_settings: Mapping[str, object],
    upstream_artifact_names: Iterable[str],
) -> dict[str, object]:
    """Archive code, evidence, environment and provenance for one executed run."""

    repo_root = Path(repo_root).resolve()
    notebook_path = Path(notebook_path).resolve()
    output_dir = Path(output_dir).resolve()
    commit = _git_value(repo_root, "rev-parse", "HEAD")
    status = _git_value(repo_root, "status", "--porcelain")
    if commit is None:
        raise RuntimeError(
            f"The deployment root is not a Git worktree: {repo_root}"
        )
    if status:
        raise RuntimeError(
            "The final reproducibility bundle requires a clean deployment "
            f"commit; Git reports:\n{status}"
        )
    reproducibility_dir = output_dir / "reproducibility"
    reproducibility_dir.mkdir(parents=True, exist_ok=True)
    archive_path = reproducibility_dir / "blaschke_deformation_certifier_reproducibility.tar.gz"
    manifest_path = reproducibility_dir / "blaschke_deformation_certifier_reproducibility_manifest.json"
    checksum_path = reproducibility_dir / "blaschke_deformation_certifier_reproducibility.tar.gz.sha256"

    _validate_inline_helper_provenance(repo_root, notebook_path)

    local_sources = (
        repo_root / ".final_deployment_generated",
        repo_root / ".gitattributes",
        repo_root / ".gitignore",
        repo_root / "README.md",
        repo_root / "requirements.txt",
        repo_root / "MANIFEST.sha256",
        notebook_path,
        repo_root / "Numerics" / "blaschke_deformation_certifier.ipynb",
        repo_root / "Numerics" / "blaschke_deformation_certifier_template.ipynb",
        repo_root / "Numerics" / "blaschke_deformation_certification.py",
        repo_root / "Numerics" / "blaschke_deformation_spectral_certification.py",
        repo_root / "Numerics" / "blaschke_deformation_contour_certification.py",
        repo_root / "Numerics" / "blaschke_deformation_reproducibility.py",
        repo_root / "Numerics" / "mpmath_pf_raw.py",
        repo_root / "Numerics" / "hardy_moat_surface_worker.py",
        repo_root / "Numerics" / "transfer_spectrum_certification.py",
        repo_root / "Numerics" / "build_blaschke_deformation_certifier.py",
        repo_root / "Numerics" / "build_blaschke_deformation_thesis_math_notebook.py",
        repo_root / "Numerics" / "execute_notebook_incremental.py",
        repo_root / "Numerics" / "transfer_lab_hyperparameters.json",
        repo_root / "Numerics" / "test_blaschke_deformation_contour_certification.py",
        repo_root / "Numerics" / "test_inline_helper_provenance.py",
    )
    missing_sources = tuple(path for path in local_sources if not path.is_file())
    if missing_sources:
        raise FileNotFoundError(
            "Missing reproducibility source files: "
            + ", ".join(str(path) for path in missing_sources)
        )

    upstream_names = tuple(str(name) for name in upstream_artifact_names)
    evidence_roots = tuple(output_dir / name for name in ("data", "reports", "figures"))
    generated_at = datetime.now(timezone.utc).isoformat()

    with tempfile.TemporaryDirectory(prefix="blaschke-certifier-repro-") as temporary:
        staging_root = Path(temporary) / "blaschke_deformation_certifier_reproducibility"
        file_records: list[dict[str, object]] = []

        for source in local_sources:
            relative = source.relative_to(repo_root)
            if source == repo_root / "README.md":
                relative = Path("DEPLOYMENT_README.md")
            _copy_with_record(
                source,
                staging_root / relative,
                staging_root,
                repo_root,
                file_records,
                "executed notebook" if source == notebook_path else "local source",
            )

        for evidence_root in evidence_roots:
            if not evidence_root.exists():
                continue
            for source in sorted(path for path in evidence_root.rglob("*") if path.is_file()):
                relative = source.relative_to(repo_root)
                _copy_with_record(
                    source,
                    staging_root / relative,
                    staging_root,
                    repo_root,
                    file_records,
                    "stored numerical evidence",
                )

        environment = {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "package_versions": _package_versions(),
        }
        manifest = {
            "schema_version": 2,
            "generated_at_utc": generated_at,
            "map_label": "blaschke_mu_0p3",
            "repository_commit": commit,
            "repository_dirty": bool(status),
            "repository_status_porcelain": status or "",
            "builder_chain": {
                "locked_template_sha256": sha256_file(
                    repo_root
                    / "Numerics"
                    / "blaschke_deformation_certifier_template.ipynb"
                ),
                "source_notebook_sha256": sha256_file(
                    repo_root
                    / "Numerics"
                    / "blaschke_deformation_certifier.ipynb"
                ),
                "executed_counterpart_sha256": sha256_file(notebook_path),
            },
            "precision_settings": dict(precision_settings),
            "upstream_artifact_names": list(upstream_names),
            "environment": environment,
            "execution_command": (
                "python -u -B Numerics/execute_notebook_incremental.py "
                "Numerics/blaschke_deformation_certifier_thesis_math.ipynb"
            ),
            "files": file_records,
        }
        requirements_path = staging_root / "requirements_versions.txt"
        requirements_path.write_text(
            "\n".join(
                f"{name}=={version}"
                for name, version in environment["package_versions"].items()
                if version is not None
            )
            + "\n",
            encoding="utf-8",
        )
        readme_path = staging_root / "README.md"
        readme_path.write_text(
            "# Blaschke deformation certifier reproducibility bundle\n\n"
            "This archive contains the rebuild source notebook, the executed "
            "thesis-mathematics notebook, its local Python modules, "
            "the tracked deployment instructions in `DEPLOYMENT_README.md`, "
            "the source manifest, the stored CSV, "
            "JSON, NPZ, Parquet, report and figure evidence, exact package "
            "versions, precision settings and repository provenance.\n\n"
            "First run `python -B Numerics/"
            "build_blaschke_deformation_certifier.py`, then `python -B "
            "Numerics/build_blaschke_deformation_thesis_math_notebook.py`. "
            "The builders "
            "reject any mismatch between a standalone helper, its displayed "
            "digest and its inline source. Then run the notebook from the archive "
            "root in the `lucy` Conda environment with the command recorded in "
            "`reproducibility_manifest.json`; "
            "set `BLASCHKE_SKIP_ARCHIVE=1` during that execution, commit the "
            "resulting notebook and evidence, and invoke "
            "`blaschke_deformation_reproducibility.py` from the clean commit. "
            "The notebook is restricted to `blaschke_mu_0p3`.\n",
            encoding="utf-8",
        )
        _record_generated(
            readme_path,
            staging_root,
            file_records,
            "generated archive instructions",
        )
        _record_generated(
            requirements_path,
            staging_root,
            file_records,
            "generated exact package versions",
        )
        staged_manifest = staging_root / "reproducibility_manifest.json"
        staged_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(staging_root, arcname=staging_root.name)

    archive_sha256 = sha256_file(archive_path)
    checksum_path.write_text(
        f"{archive_sha256}  {archive_path.name}\n", encoding="utf-8"
    )
    external_manifest = dict(manifest)
    external_manifest["archive_path"] = str(archive_path.relative_to(repo_root))
    external_manifest["archive_sha256"] = archive_sha256
    external_manifest["archive_bytes"] = archive_path.stat().st_size
    manifest_path.write_text(json.dumps(external_manifest, indent=2), encoding="utf-8")

    return {
        "archive_path": str(archive_path),
        "manifest_path": str(manifest_path),
        "checksum_path": str(checksum_path),
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_path.stat().st_size,
        "packaged_file_count": len(file_records),
        "repository_commit": commit,
        "repository_dirty": bool(status),
        "execution_status": "clean_commit_bundle_created",
    }


def main() -> None:
    root_default = Path(__file__).resolve().parents[1]
    output_default = (
        root_default
        / "Numerics"
        / "outputs"
        / "blaschke_deformation_certifier"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root_default)
    parser.add_argument(
        "--notebook",
        type=Path,
        default=(
            root_default
            / "Numerics"
            / "blaschke_deformation_certifier_thesis_math.ipynb"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=output_default)
    parser.add_argument(
        "--plan",
        type=Path,
        default=(
            output_default
            / "reports"
            / "blaschke_deformation_reproducibility_plan.json"
        ),
    )
    arguments = parser.parse_args()
    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    result = build_reproducibility_bundle(
        repo_root=arguments.repo_root,
        notebook_path=arguments.notebook,
        output_dir=arguments.output_dir,
        precision_settings=dict(plan["precision_settings"]),
        upstream_artifact_names=tuple(plan["upstream_artifact_names"]),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
