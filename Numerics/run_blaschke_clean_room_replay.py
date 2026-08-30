"""Orchestrate a full source-only replay inside one extracted scratch bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from prepare_blaschke_source_only_replay import (
    BUNDLE_ROOT_NAME,
    check_prepared_bundle,
    prepare_bundle,
)


def _run(command: list[str], *, root: Path, environment: dict[str, str]) -> dict[str, object]:
    started = time.time()
    subprocess.run(command, cwd=root, env=environment, check=True)
    return {
        "command": command,
        "elapsed_seconds": time.time() - started,
        "returncode": 0,
    }


def run_replay(
    *,
    bundle_root: Path,
    kernel_name: str,
    assembly_workers: int,
    surface_workers: int,
    prepare_only: bool,
    published_archive: Path | None,
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

    if assembly_workers < 1 or surface_workers < 1:
        raise ValueError("Worker counts must be positive.")
    environment = dict(os.environ)
    environment.update({
        "BLASCHKE_FORCE_HARDY_MATRIX": "1",
        "BLASCHKE_FORCE_CONTOURS": "1",
        "BLASCHKE_FORCE_HISTORICAL_PHASE4": "1",
        "MPMATH_PF_ASSEMBLY_WORKERS": str(assembly_workers),
    })
    python = sys.executable
    commands = [
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
        "forced_rebuild_environment": {
            key: environment[key]
            for key in (
                "BLASCHKE_FORCE_HARDY_MATRIX",
                "BLASCHKE_FORCE_CONTOURS",
                "BLASCHKE_FORCE_HISTORICAL_PHASE4",
                "MPMATH_PF_ASSEMBLY_WORKERS",
            )
        },
        "commands": command_records,
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
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--published-archive", type=Path)
    args = parser.parse_args()
    result = run_replay(
        bundle_root=args.bundle_root,
        kernel_name=args.kernel_name,
        assembly_workers=args.assembly_workers,
        surface_workers=args.surface_workers,
        prepare_only=args.prepare_only,
        published_archive=args.published_archive,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
