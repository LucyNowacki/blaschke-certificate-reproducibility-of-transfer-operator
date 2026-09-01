"""Run a clean certificate-equivalent Blaschke source replay.

The runner stages the authenticated Git release into a fresh scratch tree,
validates the exact Linux interpreter/package/BLAS identity, calls the existing
compute-only clean-room replay with every theorem producer forced, and finally
invokes the semantic verifier.  It never runs the byte comparator,
normalization, publication provenance refresh, or archive comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import tempfile
import time


STAGE_SCRIPT = Path("Numerics/stage_blaschke_certificate_replay.py")
CLEAN_REPLAY_SCRIPT = Path("Numerics/run_blaschke_clean_room_replay.py")
SEMANTIC_VERIFIER = Path("Numerics/verify_blaschke_certificate_equivalence.py")
SEMANTIC_POLICY = Path("release/blaschke-certificate-semantic-policy.json")
COMPUTE_EVIDENCE = Path("clean-room-compute-only-evidence.json")
RUN_ATTESTATION = Path("certificate-replay-run-attestation.json")
PREPARATION_RECEIPT = Path("source-only-replay-preparation.json")
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

INTERPRETER_PREFLIGHT = r"""
import importlib.metadata as md
import json
from pathlib import Path
import platform
import sys

if sys.version_info[:3] != (3, 13, 2):
    raise RuntimeError(f"Expected Python 3.13.2, observed {sys.version_info[:3]!r}")
if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
    raise RuntimeError(f"Expected Linux x86_64, observed {platform.platform()!r}")

import numpy as np
import scipy
import scipy.linalg as spla
from threadpoolctl import threadpool_info

if np.__version__ != "2.3.2" or scipy.__version__ != "1.16.0":
    raise RuntimeError(
        f"Locked NumPy/SciPy mismatch: numpy={np.__version__}, scipy={scipy.__version__}"
    )
expected = {
    "jupyter-server": "2.20.0",
    "mpmath": "1.3.0",
    "pandas": "2.3.3",
    "pip": "26.1.2",
    "pyarrow": "24.0.0",
    "python-flint": "0.8.0",
    "threadpoolctl": "3.6.0",
    "ipykernel": "6.30.1",
    "jupyter-client": "8.6.3",
}
observed = {name: md.version(name) for name in expected}
if observed != expected:
    raise RuntimeError(f"Locked pip override mismatch: {observed!r}")
spla.svdvals(np.eye(2, dtype=np.float64), check_finite=False)
blas = [row for row in threadpool_info() if row.get("user_api") == "blas"]
if not blas:
    raise RuntimeError("No loaded BLAS runtime was detected")
for row in blas:
    if (
        row.get("internal_api") != "openblas"
        or row.get("version") != "0.3.30"
        or row.get("threading_layer") != "pthreads"
        or int(row.get("num_threads") or 0) != 1
    ):
        raise RuntimeError(f"Locked single-thread OpenBLAS identity mismatch: {blas!r}")
    filepath = row.get("filepath")
    if not isinstance(filepath, str):
        raise RuntimeError(f"BLAS runtime has no library path: {row!r}")
    prefix = Path(sys.executable).resolve().parent.parent
    try:
        Path(filepath).resolve(strict=True).relative_to(prefix)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"BLAS runtime is outside the validated interpreter prefix: {filepath!r}"
        ) from exc
print(json.dumps({
    "status": "locked interpreter verified",
    "python": platform.python_version(),
    "machine": platform.machine(),
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "pip_overrides": observed,
    "blas": blas,
}, sort_keys=True))
""".strip()


class ReproductionError(RuntimeError):
    """Raised when staging, environment validation, replay, or verification fails."""


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_root(path: Path) -> Path:
    if path.is_symlink():
        raise ReproductionError("Source root must not be a symlink.")
    try:
        root = path.resolve(strict=True)
    except OSError as exc:
        raise ReproductionError(f"Source root does not exist: {path}.") from exc
    if not root.is_dir() or root == Path(root.anchor):
        raise ReproductionError("Source root must be a narrow directory.")
    for relative in (STAGE_SCRIPT, CLEAN_REPLAY_SCRIPT, SEMANTIC_VERIFIER):
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise ReproductionError(f"Missing regular release entry point: {relative}.")
    return root


def _python(path: Path | None) -> Path:
    candidate = Path(sys.executable) if path is None else Path(path)
    if candidate.is_symlink():
        # Conda's bin/python can itself be a controlled relative symlink.  Resolve
        # it and bind the receipt to the actual regular interpreter.
        candidate = candidate.resolve(strict=True)
    else:
        candidate = candidate.resolve(strict=True)
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise ReproductionError(f"Python interpreter is not executable: {candidate}.")
    return candidate


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "JUPYTER_PATH",
        "JUPYTER_CONFIG_DIR",
        "JUPYTER_DATA_DIR",
    ):
        environment.pop(key, None)
    environment.update({key: "1" for key in THREAD_ENVIRONMENT})
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONSAFEPATH"] = "1"
    return environment


def _run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except subprocess.CalledProcessError as exc:
        detail = ""
        if capture:
            detail = f"\nstdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
        raise ReproductionError(
            f"Command failed with exit {exc.returncode}: {command!r}{detail}"
        ) from exc


def _json_stdout(completed: subprocess.CompletedProcess[str], *, label: str) -> dict[str, object]:
    try:
        value = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReproductionError(f"{label} did not return one JSON object.") from exc
    if not isinstance(value, dict):
        raise ReproductionError(f"{label} JSON result is not an object.")
    return value


def validate_locked_interpreter(
    python: Path,
    preflight_cwd: Path,
    environment: dict[str, str],
    kernel_name: str,
    kernel_spec_path: Path,
) -> dict[str, object]:
    result = _json_stdout(
        _run(
            [str(python), "-I", "-B", "-c", INTERPRETER_PREFLIGHT],
            cwd=preflight_cwd,
            environment=environment,
            capture=True,
        ),
        label="interpreter preflight",
    )
    pip_check = _run(
        [str(python), "-I", "-B", "-m", "pip", "check"],
        cwd=preflight_cwd,
        environment=environment,
        capture=True,
    )
    if pip_check.stdout.strip() != "No broken requirements found.":
        raise ReproductionError(
            "pip check did not return the expected dependency-consistency PASS."
        )
    validate_ephemeral_kernel_spec(kernel_spec_path, python)
    return {
        "python_sha256": _sha256_file(python),
        "python_version": result.get("python"),
        "machine": result.get("machine"),
        "numpy": result.get("numpy"),
        "scipy": result.get("scipy"),
        "pip_overrides": result.get("pip_overrides"),
        "blas": result.get("blas"),
        "kernel_name": kernel_name,
        "kernel_spec_path": str(kernel_spec_path),
        "kernel_spec_sha256": _sha256_file(kernel_spec_path),
        "pip_check": "PASS: No broken requirements found.",
    }


def validate_ephemeral_kernel_spec(kernel_spec_path: Path, python: Path) -> None:
    """Require the exact isolated kernel argv and environment contract."""

    try:
        spec = json.loads(kernel_spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReproductionError(f"Cannot read ephemeral kernelspec: {exc}") from exc
    expected_env = {
        **{key: "1" for key in THREAD_ENVIRONMENT},
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
    }
    expected_argv = [
        str(python),
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    if (
        not isinstance(spec, dict)
        or spec.get("argv") != expected_argv
        or spec.get("env") != expected_env
        or spec.get("language") != "python"
        or spec.get("display_name") != "Blaschke certificate replay (ephemeral)"
    ):
        raise ReproductionError(
            "Ephemeral kernelspec argv/env does not match the exact replay contract."
        )


def _stage(
    python: Path,
    source_root: Path,
    workspace: Path,
    environment: dict[str, str],
    expected_inventory_sha256: str | None,
    expected_git_commit: str | None,
) -> dict[str, object]:
    identity_arguments = (
        ["--expected-inventory-sha256", expected_inventory_sha256]
        if expected_inventory_sha256 is not None
        else ["--expected-git-commit", str(expected_git_commit)]
    )
    completed = _run(
        [
            str(python),
            "-I",
            "-B",
            str(source_root / STAGE_SCRIPT),
            "--source-root",
            str(source_root),
            "--workspace",
            str(workspace),
            *identity_arguments,
        ],
        cwd=source_root,
        environment=environment,
        capture=True,
    )
    result = _json_stdout(completed, label="release stager")
    if result.get("status") != "fresh replay tree staged":
        raise ReproductionError(f"Release staging did not complete: {result!r}.")
    staged_root = Path(str(result.get("staged_root"))).resolve(strict=True)
    if staged_root.name != "blaschke_deformation_certifier_reproducibility":
        raise ReproductionError("Stager returned an unexpected replay root.")
    result["staged_root"] = str(staged_root)
    return result


def _provision_ephemeral_kernel(
    *,
    workspace: Path,
    python: Path,
    kernel_name: str,
    environment: dict[str, str],
) -> Path:
    if not kernel_name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in kernel_name):
        raise ReproductionError("Kernel name contains unsafe characters.")
    workspace = workspace.resolve(strict=True)
    ephemeral_root = workspace / "ephemeral-jupyter"
    config_root = workspace / "ephemeral-jupyter-config"
    data_root = workspace / "ephemeral-jupyter-data"
    for path in (ephemeral_root, config_root, data_root):
        if path.exists() or path.is_symlink():
            raise ReproductionError(
                f"Refusing pre-existing ephemeral replay path: {path}."
            )
    ephemeral_root.mkdir()
    kernels_root = ephemeral_root / "kernels"
    kernels_root.mkdir()
    kernel_root = kernels_root / kernel_name
    kernel_root.mkdir()
    kernel_path = kernel_root / "kernel.json"
    expected_env = {
        **{key: "1" for key in THREAD_ENVIRONMENT},
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
    }
    spec = {
        "argv": [
            str(python),
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ],
        "display_name": "Blaschke certificate replay (ephemeral)",
        "language": "python",
        "metadata": {"debugger": False},
        "env": expected_env,
    }
    _atomic_write(kernel_path, _canonical_bytes(spec))
    environment["JUPYTER_PATH"] = str(ephemeral_root)
    environment["JUPYTER_CONFIG_DIR"] = str(config_root)
    environment["JUPYTER_DATA_DIR"] = str(data_root)
    return kernel_path


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReproductionError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReproductionError(f"{label} must be a JSON object.")
    return value


def _attest_current_theorem_run(
    *,
    staged_root: Path,
    staging: dict[str, object],
    interpreter: dict[str, object],
    clean_command: list[str],
    run_nonce: str,
    outer_started_ns: int,
) -> dict[str, object]:
    evidence_path = staged_root / COMPUTE_EVIDENCE
    preparation_path = staged_root / PREPARATION_RECEIPT
    policy = _load_json_object(staged_root / SEMANTIC_POLICY, label="semantic policy")
    expected_paths = policy.get("theorem_required_generated_paths")
    if (
        not isinstance(expected_paths, list)
        or len(expected_paths) != 27
        or not all(isinstance(value, str) and value for value in expected_paths)
        or len(set(expected_paths)) != 27
    ):
        raise ReproductionError("Semantic policy does not declare the fixed 27-file closure.")
    evidence = _load_json_object(evidence_path, label="theorem-only compute evidence")
    if (
        evidence.get("receipt_schema") != "blaschke-theorem-only-compute-v1"
        or evidence.get("status") != "theorem-only source reconstruction complete"
        or evidence.get("stage") != "theorem-only-compute"
        or evidence.get("historical_phase4_diagnostics_skipped") is not True
        or evidence.get("historical_phase4_diagnostics_authoritative") is not False
        or evidence.get("hardy_1024_diagnostic_skipped") is not True
        or evidence.get("hardy_1024_diagnostic_authoritative") is not False
    ):
        raise ReproductionError("Theorem-only compute evidence has the wrong contract.")
    forced = evidence.get("forced_rebuild_environment")
    required_forced = {
        "BLASCHKE_FORCE_HARDY_MATRIX": "1",
        "BLASCHKE_FORCE_CONTOURS": "1",
        "BLASCHKE_SKIP_HARDY_STARTING_AUDIT": "1",
        "BLASCHKE_SKIP_HISTORICAL_PHASE4": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONPATH": os.pathsep.join(
            (str(staged_root), str(staged_root / "Numerics"))
        ),
        "BLASCHKE_AUTHENTICATED_REPLAY_ROOT": str(staged_root),
    }
    if not isinstance(forced, dict) or any(forced.get(key) != value for key, value in required_forced.items()):
        raise ReproductionError("The theorem-only force/skip environment is incomplete.")
    if "BLASCHKE_FORCE_HISTORICAL_PHASE4" in forced:
        raise ReproductionError("Historical diagnostics were forced in theorem-only mode.")
    preparation = evidence.get("preparation")
    on_disk_preparation = _load_json_object(
        preparation_path, label="source-only preparation receipt"
    )
    if (
        not isinstance(preparation, dict)
        or on_disk_preparation != preparation
        or preparation.get("inventory_sha256") != staging.get("staged_inventory_sha256")
    ):
        raise ReproductionError("The current preparation receipt is not bound to staging.")
    closure = evidence.get("generated_closure")
    if (
        not isinstance(closure, dict)
        or closure.get("theorem_only") is not True
        or closure.get("generated_paths") != expected_paths
        or closure.get("generated_path_count") != 27
        or closure.get("missing") != []
    ):
        raise ReproductionError("The fixed theorem-generated closure did not pass.")
    records = closure.get("generated_files")
    if not isinstance(records, list) or len(records) != 27:
        raise ReproductionError("The theorem-generated hash records are incomplete.")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ReproductionError("Malformed theorem-generated hash record.")
        relative = record["path"]
        if relative in seen or relative not in expected_paths or ".." in Path(relative).parts:
            raise ReproductionError("Unsafe or duplicate theorem-generated path.")
        seen.add(relative)
        path = staged_root / relative
        stat_result = path.stat()
        if (
            not path.is_file()
            or path.is_symlink()
            or stat_result.st_size != record.get("bytes")
            or _sha256_file(path) != record.get("sha256")
            or stat_result.st_mtime_ns != record.get("mtime_ns")
            or stat_result.st_mtime_ns < outer_started_ns
        ):
            raise ReproductionError(f"Current theorem output binding failed: {relative}.")
    if seen != set(expected_paths):
        raise ReproductionError("The current theorem output set is incomplete.")
    commands = evidence.get("commands")
    if not isinstance(commands, list) or len(commands) != 5:
        raise ReproductionError("The theorem-only command record is incomplete.")
    finished_ns = time.time_ns()
    attestation = {
        "receipt_schema": "blaschke-certificate-current-run-attestation-v1",
        "status": "current theorem-only run hash-bound",
        "run_nonce": run_nonce,
        "outer_started_ns": outer_started_ns,
        "outer_finished_ns": finished_ns,
        "release_inventory_sha256": staging["release_inventory_sha256"],
        "staged_inventory_sha256": staging["staged_inventory_sha256"],
        "external_release_identity": staging["external_release_identity"],
        "clean_replay_command": clean_command,
        "compute_evidence_path": COMPUTE_EVIDENCE.as_posix(),
        "compute_evidence_sha256": _sha256_file(evidence_path),
        "preparation_receipt_path": PREPARATION_RECEIPT.as_posix(),
        "preparation_receipt_sha256": _sha256_file(preparation_path),
        "theorem_generated_files": records,
        "theorem_generated_path_count": 27,
        "forced_rebuild_environment": forced,
        "interpreter_sha256": interpreter["python_sha256"],
        "historical_diagnostics_skipped": True,
        "historical_diagnostics_authoritative": False,
        "hardy_1024_diagnostic_skipped": True,
        "hardy_1024_diagnostic_authoritative": False,
    }
    _atomic_write(staged_root / RUN_ATTESTATION, _canonical_bytes(attestation))
    return attestation


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_reproduction(
    *,
    source_root: Path,
    workspace: Path | None,
    python_path: Path | None,
    kernel_name: str,
    assembly_workers: int,
    surface_workers: int,
    dry_run: bool,
    receipt_path: Path | None,
    expected_inventory_sha256: str | None,
    expected_git_commit: str | None,
) -> dict[str, object]:
    source_root = _source_root(source_root)
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise ReproductionError("The locked replay supports Linux x86_64 only.")
    if assembly_workers < 1 or surface_workers < 1:
        raise ReproductionError("Worker counts must be positive.")
    python = _python(python_path)
    environment = _environment()
    if workspace is None:
        workspace = Path(tempfile.mkdtemp(prefix="blaschke-certificate-replay-"))
    staging = _stage(
        python,
        source_root,
        workspace,
        environment,
        expected_inventory_sha256,
        expected_git_commit,
    )
    workspace = Path(workspace).resolve(strict=True)
    staged_root = Path(str(staging["staged_root"]))
    kernel_spec_path = _provision_ephemeral_kernel(
        workspace=workspace,
        python=python,
        kernel_name=kernel_name,
        environment=environment,
    )
    preflight_cwd = workspace / "isolated-runtime-preflight"
    preflight_cwd.mkdir()
    interpreter = validate_locked_interpreter(
        python,
        preflight_cwd,
        environment,
        kernel_name,
        kernel_spec_path,
    )

    # Validate every CLI after staging.  This executes no project producer.
    for relative in (CLEAN_REPLAY_SCRIPT, SEMANTIC_VERIFIER):
        _run(
            [str(python), "-B", str(staged_root / relative), "--help"],
            cwd=staged_root,
            environment=environment,
            capture=True,
        )
    if dry_run:
        result = {
            "receipt_schema": "blaschke-certificate-reproduction-dry-run-v1",
            "status": "DRY_RUN_READY_NO_CERTIFICATION",
            "certification_confirmed": False,
            "compute_started": False,
            "tracked_source_bytes_mutated": False,
            "release_inventory_sha256": staging["release_inventory_sha256"],
            "staged_inventory_sha256": staging["staged_inventory_sha256"],
            "interpreter": interpreter,
            "external_release_identity": staging["external_release_identity"],
        }
        payload = _canonical_bytes(result)
        if receipt_path is not None:
            _atomic_write(receipt_path, payload)
        return result

    for relative in (COMPUTE_EVIDENCE, RUN_ATTESTATION):
        if (staged_root / relative).exists():
            raise ReproductionError(
                f"Fresh staged replay unexpectedly contains {relative.as_posix()}."
            )
    run_nonce = secrets.token_hex(32)
    outer_started_ns = time.time_ns()
    clean_command = [
            str(python),
            "-u",
            "-B",
            str(staged_root / CLEAN_REPLAY_SCRIPT),
            "--bundle-root",
            str(staged_root),
            "--kernel-name",
            kernel_name,
            "--assembly-workers",
            str(assembly_workers),
            "--surface-workers",
            str(surface_workers),
            "--expected-inventory-sha256",
            str(staging["staged_inventory_sha256"]),
            "--theorem-only-compute",
        ]
    _run(
        clean_command,
        cwd=staged_root,
        environment=environment,
    )
    run_attestation = _attest_current_theorem_run(
        staged_root=staged_root,
        staging=staging,
        interpreter=interpreter,
        clean_command=clean_command,
        run_nonce=run_nonce,
        outer_started_ns=outer_started_ns,
    )
    semantic_receipt = staged_root / "certificate-equivalence-receipt.json"
    completed = _run(
        [
            str(python),
            "-B",
            str(staged_root / SEMANTIC_VERIFIER),
            "--root",
            str(staged_root),
            "--mode",
            "full-replay",
            "--expected-release-inventory-sha256",
            str(staging["release_inventory_sha256"]),
            "--expected-run-nonce",
            run_nonce,
            "--receipt",
            str(semantic_receipt),
        ],
        cwd=staged_root,
        environment=environment,
        capture=True,
    )
    result = _json_stdout(completed, label="semantic verifier")
    if (
        result.get("status") != "CERTIFICATION_CONFIRMED"
        or result.get("independent_theorem_recomputation") is not True
    ):
        raise ReproductionError("Semantic verifier did not confirm the certification.")
    result["replay_provenance"] = {
        "release_inventory_sha256": staging["release_inventory_sha256"],
        "staged_inventory_sha256": staging["staged_inventory_sha256"],
        "tracked_source_bytes_mutated": False,
        "raw_byte_comparator_run": False,
        "normalization_run": False,
        "interpreter": interpreter,
        "external_release_identity": staging["external_release_identity"],
        "run_attestation_sha256": _sha256_file(staged_root / RUN_ATTESTATION),
        "run_nonce": run_attestation["run_nonce"],
    }
    payload = _canonical_bytes(result)
    _atomic_write(semantic_receipt, payload)
    if receipt_path is not None:
        _atomic_write(receipt_path, payload)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--existing-python",
        type=Path,
        help=(
            "Reuse an already authenticated pinned interpreter.  The exact package, "
            "Python, BLAS and kernel identities are still revalidated."
        ),
    )
    parser.add_argument("--kernel-name", default="blaschke-certificate-replay")
    parser.add_argument("--assembly-workers", type=int, default=24)
    parser.add_argument("--surface-workers", type=int, default=6)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate release, locks, staging, environment and CLIs without computation.",
    )
    parser.add_argument("--receipt", type=Path)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--expected-release-inventory-sha256")
    identity.add_argument("--expected-git-commit")
    arguments = parser.parse_args()
    try:
        result = run_reproduction(
            source_root=arguments.source_root,
            workspace=arguments.workspace,
            python_path=arguments.existing_python,
            kernel_name=arguments.kernel_name,
            assembly_workers=arguments.assembly_workers,
            surface_workers=arguments.surface_workers,
            dry_run=arguments.dry_run,
            receipt_path=arguments.receipt,
            expected_inventory_sha256=arguments.expected_release_inventory_sha256,
            expected_git_commit=arguments.expected_git_commit,
        )
    except (OSError, ReproductionError) as exc:
        failure = {
            "receipt_schema": "blaschke-certificate-reproduction-v1",
            "status": "CERTIFICATION_NOT_CONFIRMED",
            "error": str(exc),
        }
        payload = _canonical_bytes(failure)
        if arguments.receipt is not None:
            _atomic_write(arguments.receipt, payload)
        print(payload.decode("utf-8"), end="")
        return 1
    print(_canonical_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
