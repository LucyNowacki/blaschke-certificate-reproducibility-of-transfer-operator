"""Build a fail-closed reproducibility bundle for the Blaschke certifier."""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Callable, Iterable, Mapping, Sequence

try:
    from .prepare_blaschke_source_only_replay import make_archive_inventory
except ImportError:
    from prepare_blaschke_source_only_replay import make_archive_inventory


PACKAGE_NAMES = (
    "pip",
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "matplotlib",
    "tqdm",
    "ipywidgets",
    "threadpoolctl",
    "mpmath",
    "python-flint",
    "nbformat",
    "nbclient",
    "jupyter",
    "ipykernel",
)
PIP_LOCK_PACKAGES = ("mpmath", "pip", "python-flint", "threadpoolctl")

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

OUTPUT_RELATIVE = PurePosixPath(
    "Numerics/outputs/blaschke_deformation_certifier"
)
NOTEBOOK_RELATIVE = PurePosixPath(
    "Numerics/blaschke_deformation_certifier_thesis_math.ipynb"
)
# The verifier's v3 archive schema still names the generated reports-path plan.
# Package that path as a compatibility alias of this source-controlled plan.
SOURCE_CONTROLLED_PLAN_RELATIVE = PurePosixPath(
    "Numerics/blaschke_deformation_reproducibility_plan.json"
)
PLAN_RELATIVE = OUTPUT_RELATIVE / "reports/blaschke_deformation_reproducibility_plan.json"
SPECTRAL_REPORT_RELATIVE = OUTPUT_RELATIVE / (
    "reports/blaschke_deformation_24_target_N600_M610_spectral_certificate.json"
)
HISTORICAL_PHASE2_REPORT_RELATIVE = OUTPUT_RELATIVE / (
    "reports/historical_phase2_comparison_rebuild.json"
)
HISTORICAL_PHASE4_REPORT_RELATIVE = OUTPUT_RELATIVE / (
    "reports/blaschke_deformation_historical_phase4_rebuild_N600_M610.json"
)
DIAGNOSTIC_AUDIT_REPORT_RELATIVE = OUTPUT_RELATIVE / (
    "reports/blaschke_deformation_diagnostic_audits_rebuild.json"
)
PHASE1_DIAGNOSTIC_REPORT_RELATIVE = OUTPUT_RELATIVE / (
    "reports/blaschke_deformation_phase1_diagnostics_rebuild.json"
)
SAMPLED_SCHUR_DIAGNOSTIC_REPORT_RELATIVE = OUTPUT_RELATIVE / (
    "reports/blaschke_deformation_sampled_schur_diagnostics_rebuild.json"
)
TEMPLATE_NOTEBOOK_RELATIVE = PurePosixPath(
    "Numerics/blaschke_deformation_certifier_template.ipynb"
)
SOURCE_NOTEBOOK_RELATIVE = PurePosixPath(
    "Numerics/blaschke_deformation_certifier.ipynb"
)
PHASE1_DIAGNOSTIC_PRODUCER_RELATIVE = PurePosixPath(
    "Numerics/blaschke_deformation_phase1_diagnostics.py"
)
SAMPLED_SCHUR_DIAGNOSTIC_PRODUCER_RELATIVE = PurePosixPath(
    "Numerics/blaschke_deformation_sampled_schur_diagnostics.py"
)
PHASE1_WORKER_RELATIVE = PurePosixPath("Numerics/mpmath_pf_raw.py")
SAMPLED_SCHUR_ENVELOPE_RELATIVE = PurePosixPath(
    "Numerics/transfer_spectrum_certification.py"
)
THESIS_NOTEBOOK_BUILDER_RELATIVE = PurePosixPath(
    "Numerics/build_blaschke_deformation_thesis_math_notebook.py"
)

AUTHORITATIVE_EXCLUDED_PATHS = frozenset(
    {
        SOURCE_MANIFEST_NAME,
        "Numerics/blaschke_deformation_certifier_thesis_math_backup_pre_appendix_20260823.ipynb",
        "Numerics/blaschke_deformation_certifier_thesis_math_dist.ipynb",
    }
)
AUTHORITATIVE_EXCLUDED_PREFIXES = (
    f"{OUTPUT_RELATIVE.as_posix()}/reproducibility/",
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
EXPECTED_HISTORICAL_PHASE2_SCHEMA = "phase2-historical-comparisons-v1"
EXPECTED_HISTORICAL_PHASE4_SCHEMA = "historical-wide-phase4-source-rebuild-v2"
EXPECTED_HISTORICAL_PHASE4_DOC_REFRESH_SCHEMA = (
    "historical-phase4-docstring-only-source-refresh-v1"
)
EXPECTED_DIAGNOSTIC_AUDIT_SCHEMA = "blaschke-deformation-diagnostic-audits-v1"
EXPECTED_DIAGNOSTIC_SCHEMA_VERSION = "2.0.0"
EXPECTED_PHASE1_PRODUCER_SCHEMA = "blaschke-deformation-phase1-diagnostics-v1"
EXPECTED_SAMPLED_SCHUR_PRODUCER_SCHEMA = (
    "blaschke-deformation-sampled-schur-diagnostics-v1"
)
EXPECTED_PHASE1_CLOUD_PAIRS = ((12, 12), (12, 24), (20, 20), (20, 30))
EXPECTED_PHASE1_OUTPUT_NAMES = frozenset(
    {
        *(
            f"phase1_eigencloud_{kind}_N{n_value}_M{m_value}"
            for n_value, m_value in EXPECTED_PHASE1_CLOUD_PAIRS
            for kind in ("match", "eigenvalues")
        ),
        "phase1_raw_error_vs_M_N25",
        "phase1_raw_NM_heatmap_data_near_square",
    }
)
EXPECTED_SAMPLED_SCHUR_N = (30, 40, 50, 60, 80, 100)
EXPECTED_SAMPLED_SCHUR_OVERSAMPLING = 6
EXPECTED_SAMPLED_SCHUR_FILENAME = (
    "transfer_lab_blaschke_mu_0p3_generic_sampled_schur_envelope.csv"
)
EXPECTED_SAMPLED_SCHUR_COLUMNS = frozenset(
    {
        "map_label",
        "N",
        "M",
        "epsilon_schur_diagnostic",
        "transported_Bmat_diagnostic",
        "status",
        "geometry_status",
    }
)

REMOVED_UNPRODUCED_UPSTREAM_ARTIFACT_NAMES = frozenset(
    {
        "branch_image_wide_candidate_single_space_row_N600_M610.csv",
        "branch_image_phase1_raw_eigs_N600_M610.csv",
        "phase4_diag_A_X_N600_M610.npz",
        "phase4_diag_hardy_gauge_eigenvalues_N600_M610.csv",
        "phase4_hp_A_X_N600_M610.npz",
    }
)
EXPECTED_UPSTREAM_ARTIFACT_NAMES = (
    "final_blaschke_N600_schur_certificate.csv",
    "branch_image_input_tail_interval_effect_N600.csv",
    "branch_image_balanced_candidate_single_space_row_N600_M610.csv",
    "branch_image_balanced_candidate_single_space_row_N600_M610_safe_GL.csv",
    "branch_image_balanced_response_prefactor_candidate_row_N600_M610.csv",
    "output_response_branch_image_prefactor_interval_cert_balanced_N600.csv",
    "branch_image_balanced_candidate_transport_cert_N600.csv",
    "branch_image_radius_reoptimisation_balanced_highcell_scan.csv",
    "branch_image_wide_candidate_first15_sampled_spectral_packets_N600_M610.csv",
    "branch_image_wide_candidate_A_X_from_mp_scaled_N600_M610_r2p225974769705636.npz",
    "branch_image_wide_candidate_first15_contour_moats_N600_M610_J128.csv",
    "branch_image_wide_candidate_first15_fragile_robustness_N600_M610.csv",
    "branch_image_wide_candidate_first15_contour_profile_summary_N600_M610_J128.csv",
    "branch_image_wide_candidate_first15_contour_profiles_N600_M610_J128.csv",
    "final_blaschke_branch-image_first15_contour_moats_N600_M610.csv",
    "branch_image_wide_candidate_mu2_hardy_moat_surface_N600_M610.csv",
    "branch_image_wide_candidate_first15_global_hardy_moat_surface_N600_M610_grid45.npz",
    "branch_image_wide_candidate_first15_deep_zoom_hardy_moat_surface_N600_M610_grid201.npz",
    "branch_image_wide_candidate_first15_zoom_hardy_moat_surface_N600_M610_grid161.npz",
    "phase4_hp_hardy_gauge_eigenvalues_N600_M610.csv",
)

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

MANIFEST_LINE_RE = re.compile(r"([0-9a-f]{64})  (.+)")


class ReproducibilityError(RuntimeError):
    """Raised when a fail-closed packaging invariant does not hold."""


@dataclass(frozen=True)
class GitSnapshot:
    root: Path
    head: str
    status: str
    commit_epoch: int
    commit_time: str


@dataclass(frozen=True)
class ManifestEntry:
    relative_path: PurePosixPath
    sha256: str
    size: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _DocstringStripper(ast.NodeTransformer):
    """Remove only leading Python docstrings from semantic-AST comparisons."""

    def _strip(self, node: ast.AST) -> ast.AST:
        self.generic_visit(node)
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]
        return node

    visit_Module = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip
    visit_ClassDef = _strip


def _semantic_ast_sha256(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise ReproducibilityError(
            f"Cannot parse documentation-refresh source {path}."
        ) from exc
    stripped = _DocstringStripper().visit(tree)
    ast.fix_missing_locations(stripped)
    payload = ast.dump(
        stripped,
        annotate_fields=True,
        include_attributes=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _source_manifest_bytes(entries: Iterable[ManifestEntry]) -> bytes:
    return "".join(
        f"{entry.sha256}  {entry.relative_path.as_posix()}\n"
        for entry in entries
    ).encode("utf-8")


def _expect_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ReproducibilityError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _run_checked(
    command: Sequence[str], *, cwd: Path, text: bool = True
) -> str | bytes:
    try:
        completed = subprocess.run(
            tuple(command),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=text,
        )
    except OSError as exc:
        raise ReproducibilityError(
            f"Cannot execute {command[0]!r} in {cwd}: {exc}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        detail = (stderr or "").strip() or "no stderr"
        raise ReproducibilityError(
            f"Command failed ({exc.returncode}): {' '.join(command)}\n{detail}"
        ) from exc
    return completed.stdout


def _git_text(repo_root: Path, *args: str) -> str:
    return str(_run_checked(("git", *args), cwd=repo_root, text=True)).strip()


def _git_bytes(repo_root: Path, *args: str) -> bytes:
    output = _run_checked(("git", *args), cwd=repo_root, text=False)
    if not isinstance(output, bytes):
        raise AssertionError("Binary Git command unexpectedly returned text.")
    return output


def _capture_git_snapshot(repo_root: Path) -> GitSnapshot:
    root = Path(repo_root).resolve()
    top_level = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise ReproducibilityError(
            "The supplied deployment root is not the exact Git repository root: "
            f"supplied {root}, Git reports {top_level}."
        )
    head = _git_text(root, "rev-parse", "--verify", "HEAD^{commit}")
    status = _git_text(
        root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if status:
        raise ReproducibilityError(
            "The final reproducibility bundle requires a clean deployment "
            f"commit; Git reports:\n{status}"
        )
    commit_epoch_text = _git_text(root, "show", "-s", "--format=%ct", head)
    commit_time = _git_text(root, "show", "-s", "--format=%cI", head)
    try:
        commit_epoch = int(commit_epoch_text)
    except ValueError as exc:
        raise ReproducibilityError(
            f"Git returned an invalid commit timestamp: {commit_epoch_text!r}."
        ) from exc
    return GitSnapshot(root, head, status, commit_epoch, commit_time)


def _assert_same_clean_snapshot(repo_root: Path, expected: GitSnapshot) -> None:
    if _capture_git_snapshot(repo_root) != expected:
        raise ReproducibilityError(
            "The deployment Git root, HEAD, or clean status changed while the "
            "reproducibility bundle was being staged."
        )


def _select_source_plan_relative(repo_root: Path) -> PurePosixPath:
    source_controlled = repo_root.joinpath(*SOURCE_CONTROLLED_PLAN_RELATIVE.parts)
    if source_controlled.exists():
        _absolute_repo_file(repo_root, SOURCE_CONTROLLED_PLAN_RELATIVE)
        tracked = _git_text(
            repo_root,
            "ls-files",
            "--",
            SOURCE_CONTROLLED_PLAN_RELATIVE.as_posix(),
        )
        if tracked != SOURCE_CONTROLLED_PLAN_RELATIVE.as_posix():
            raise ReproducibilityError(
                "The authoritative reproducibility plan must be source-controlled: "
                f"{SOURCE_CONTROLLED_PLAN_RELATIVE.as_posix()}."
            )
        return SOURCE_CONTROLLED_PLAN_RELATIVE
    return PLAN_RELATIVE


def _safe_relative_path(value: str, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ReproducibilityError(f"{label} must be a non-empty string.")
    if "\\" in value or any(character in value for character in "\x00\n\r"):
        raise ReproducibilityError(f"Unsafe {label}: {value!r}.")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ReproducibilityError(f"Unsafe {label}: {value!r}.")
    relative = PurePosixPath(value)
    if relative.is_absolute() or relative.as_posix() != value:
        raise ReproducibilityError(f"Unsafe {label}: {value!r}.")
    return relative


def _absolute_repo_file(repo_root: Path, relative: PurePosixPath) -> Path:
    candidate = repo_root.joinpath(*relative.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise ReproducibilityError(
            f"Authoritative path is not a regular file: {relative.as_posix()}."
        )
    try:
        candidate.resolve(strict=True).relative_to(repo_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ReproducibilityError(
            f"Authoritative path escapes the deployment root: {relative.as_posix()}."
        ) from exc
    return candidate


def _authoritative_relative_paths(repo_root: Path) -> tuple[PurePosixPath, ...]:
    output = _git_bytes(
        repo_root, "ls-files", "-co", "--exclude-standard", "-z"
    )
    try:
        labels = output.decode("utf-8").split("\0")
    except UnicodeDecodeError as exc:
        raise ReproducibilityError(
            "Git returned a non-UTF-8 authoritative deployment path."
        ) from exc
    selected: set[PurePosixPath] = set()
    for label in labels:
        if not label:
            continue
        relative = _safe_relative_path(label, label="authoritative Git path")
        normalised = relative.as_posix()
        if normalised in AUTHORITATIVE_EXCLUDED_PATHS:
            continue
        if any(
            normalised.startswith(prefix)
            for prefix in AUTHORITATIVE_EXCLUDED_PREFIXES
        ):
            continue
        if repo_root.joinpath(*relative.parts).is_file():
            selected.add(relative)
    return tuple(sorted(selected, key=lambda path: path.as_posix()))


def _validate_source_manifest(repo_root: Path) -> tuple[ManifestEntry, ...]:
    manifest_path = repo_root / SOURCE_MANIFEST_NAME
    try:
        text = manifest_path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ReproducibilityError(
            f"Cannot read UTF-8 source manifest {manifest_path}: {exc}"
        ) from exc
    if not text or not text.endswith("\n"):
        raise ReproducibilityError(
            f"{SOURCE_MANIFEST_NAME} must be non-empty and newline-terminated."
        )

    entries: list[ManifestEntry] = []
    seen: set[PurePosixPath] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = MANIFEST_LINE_RE.fullmatch(line)
        if match is None:
            raise ReproducibilityError(
                f"Malformed {SOURCE_MANIFEST_NAME} line {line_number}: {line!r}."
            )
        expected_hash, path_text = match.groups()
        relative = _safe_relative_path(path_text, label="manifest path")
        if relative.as_posix() == SOURCE_MANIFEST_NAME:
            raise ReproducibilityError(
                f"{SOURCE_MANIFEST_NAME} must not list itself."
            )
        if relative in seen:
            raise ReproducibilityError(
                f"Duplicate manifest path: {relative.as_posix()}."
            )
        seen.add(relative)
        source = _absolute_repo_file(repo_root, relative)
        observed_hash = sha256_file(source)
        if observed_hash != expected_hash:
            raise ReproducibilityError(
                f"Checksum mismatch for {relative.as_posix()}: expected "
                f"{expected_hash}, observed {observed_hash}."
            )
        entries.append(
            ManifestEntry(relative, expected_hash, source.stat().st_size)
        )

    manifest_paths = tuple(entry.relative_path for entry in entries)
    if manifest_paths != tuple(
        sorted(manifest_paths, key=lambda path: path.as_posix())
    ):
        raise ReproducibilityError(
            f"{SOURCE_MANIFEST_NAME} paths are not in deterministic sorted order."
        )
    authoritative_paths = _authoritative_relative_paths(repo_root)
    if manifest_paths != authoritative_paths:
        missing = sorted(
            path.as_posix() for path in set(authoritative_paths) - set(manifest_paths)
        )
        extra = sorted(
            path.as_posix() for path in set(manifest_paths) - set(authoritative_paths)
        )
        raise ReproducibilityError(
            f"{SOURCE_MANIFEST_NAME} does not match the current authoritative "
            f"deployment paths; missing={missing}, extra={extra}."
        )
    return tuple(entries)


def _archive_manifest_entries(
    entries: Iterable[ManifestEntry],
    *,
    source_plan_relative: PurePosixPath,
) -> tuple[ManifestEntry, ...]:
    """Alias the source-controlled plan at the verifier's v3 archive path."""

    indexed = {entry.relative_path: entry for entry in entries}
    source_entry = indexed.get(source_plan_relative)
    if source_entry is None:
        raise ReproducibilityError(
            "The authoritative reproducibility plan is absent from "
            f"{SOURCE_MANIFEST_NAME}: {source_plan_relative.as_posix()}."
        )
    if source_plan_relative != PLAN_RELATIVE:
        indexed[PLAN_RELATIVE] = ManifestEntry(
            PLAN_RELATIVE,
            source_entry.sha256,
            source_entry.size,
        )
    return tuple(
        sorted(indexed.values(), key=lambda entry: entry.relative_path.as_posix())
    )


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _validate_pip_requirements(
    repo_root: Path, selected_versions: Mapping[str, str | None]
) -> dict[str, str]:
    """Require exact pins for packages omitted by ``conda list --explicit``."""

    path = repo_root / PIP_LOCK_NAME
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ReproducibilityError(f"Cannot read exact pip requirements: {exc}") from exc
    pins: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
        if match is None:
            raise ReproducibilityError(
                f"Every pip requirement must be an exact pin, observed {line!r}."
            )
        name = match.group(1).lower().replace("_", "-")
        if name in pins:
            raise ReproducibilityError(f"Duplicate pip requirement for {name}.")
        pins[name] = match.group(2)
    if set(pins) != set(PIP_LOCK_PACKAGES):
        raise ReproducibilityError(
            "The exact pip requirements must contain precisely "
            f"{PIP_LOCK_PACKAGES!r}, observed {tuple(sorted(pins))!r}."
        )
    for name, version in pins.items():
        if selected_versions.get(name) != version:
            raise ReproducibilityError(
                f"The installed {name} version {selected_versions.get(name)!r} "
                f"does not match the exact pip pin {version!r}."
            )
    return dict(sorted(pins.items()))


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReproducibilityError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReproducibilityError(f"{label} must be a JSON object: {path}.")
    return value


def _expect_exact_int(mapping: Mapping[str, object], key: str, expected: int) -> None:
    observed = mapping.get(key)
    if type(observed) is not int or observed != expected:
        raise ReproducibilityError(
            f"Expected {key}={expected}, observed {observed!r}."
        )


def _expect_bool(mapping: Mapping[str, object], key: str, expected: bool) -> None:
    observed = mapping.get(key)
    if type(observed) is not bool or observed is not expected:
        raise ReproducibilityError(
            f"Expected {key}={expected}, observed {observed!r}."
        )


def _validate_source_controlled_plan_inventory(
    plan: Mapping[str, object],
) -> None:
    names = plan.get("upstream_artifact_names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ReproducibilityError(
            "The source-controlled upstream artifact inventory is malformed."
        )
    observed = tuple(names)
    if observed != EXPECTED_UPSTREAM_ARTIFACT_NAMES:
        missing = sorted(set(EXPECTED_UPSTREAM_ARTIFACT_NAMES) - set(observed))
        extra = sorted(set(observed) - set(EXPECTED_UPSTREAM_ARTIFACT_NAMES))
        raise ReproducibilityError(
            "The source-controlled upstream artifact inventory drifted; "
            f"missing={missing}, extra={extra}, order_matches=False."
        )
    retained_unproduced = sorted(
        REMOVED_UNPRODUCED_UPSTREAM_ARTIFACT_NAMES.intersection(observed)
    )
    if retained_unproduced:
        raise ReproducibilityError(
            "The source-controlled plan retains unproduced artifacts: "
            f"{retained_unproduced}."
        )


def validate_plan_and_spectral_report(
    plan: Mapping[str, object], report: Mapping[str, object]
) -> dict[str, object]:
    """Validate the reproducibility plan against theorem-facing report gates."""

    if set(plan) != {"precision_settings", "upstream_artifact_names"}:
        raise ReproducibilityError(
            "The reproducibility plan must contain exactly precision_settings "
            "and upstream_artifact_names."
        )
    precision = plan.get("precision_settings")
    if not isinstance(precision, dict):
        raise ReproducibilityError("precision_settings must be a JSON object.")
    artifact_names = plan.get("upstream_artifact_names")
    if not isinstance(artifact_names, list) or not artifact_names:
        raise ReproducibilityError(
            "upstream_artifact_names must be a non-empty JSON array."
        )

    if precision.get("map_label") != EXPECTED_MAP_LABEL:
        raise ReproducibilityError(
            f"The plan map_label must be {EXPECTED_MAP_LABEL!r}."
        )
    if report.get("map_label") != EXPECTED_MAP_LABEL:
        raise ReproducibilityError(
            f"The spectral report map_label must be {EXPECTED_MAP_LABEL!r}."
        )
    _expect_exact_int(precision, "N", 600)
    _expect_exact_int(precision, "M", 610)
    _expect_exact_int(report, "N", 600)
    _expect_exact_int(report, "M", 610)
    for key in ("rho", "r"):
        if precision.get(key) != report.get(key):
            raise ReproducibilityError(
                f"Plan/report mismatch for {key}: {precision.get(key)!r} != "
                f"{report.get(key)!r}."
            )

    schema = precision.get("contour_certificate_schema")
    if not isinstance(schema, str) or not schema:
        raise ReproducibilityError(
            "The plan contour_certificate_schema must be a non-empty string."
        )
    if report.get("certificate_schema") != schema:
        raise ReproducibilityError(
            "Plan/report certificate schema mismatch: "
            f"{schema!r} != {report.get('certificate_schema')!r}."
        )

    _expect_exact_int(precision, "contour_target_count", EXPECTED_TARGET_COUNT)
    _expect_exact_int(
        precision, "contour_total_multiplicity", EXPECTED_MULTIPLICITY
    )
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
    _expect_exact_int(
        report, "laurent_moat_target_count", EXPECTED_LAURENT_MOAT_COUNT
    )
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
        raise ReproducibilityError(
            f"Unexpected spectral report status: {report.get('status')!r}."
        )
    for key in TRUE_THEOREM_GATES:
        _expect_bool(report, key, True)
    for key in FALSE_THEOREM_GATES:
        _expect_bool(report, key, False)

    linked_fields = (
        ("contour_arb_bits", "contour_precision_bits"),
        ("contour_eta_schur_upper", "eta_schur_upper_text"),
        ("hardy_matrix_eta_A_upper", "eta_A_upper_text"),
        ("hardy_matrix_midpoint_sha256", "matrix_midpoint_sha256"),
    )
    for plan_key, report_key in linked_fields:
        if precision.get(plan_key) != report.get(report_key):
            raise ReproducibilityError(
                f"Plan/report mismatch for {plan_key}/{report_key}: "
                f"{precision.get(plan_key)!r} != {report.get(report_key)!r}."
            )

    return {
        "certificate_schema": schema,
        "map_label": EXPECTED_MAP_LABEL,
        "target_count": EXPECTED_TARGET_COUNT,
        "total_algebraic_multiplicity": EXPECTED_MULTIPLICITY,
        "schur_count_target_count": EXPECTED_TARGET_COUNT,
        "schur_triangular_moat_target_count": EXPECTED_SCHUR_MOAT_COUNT,
        "laurent_moat_target_count": EXPECTED_LAURENT_MOAT_COUNT,
        "true_theorem_gates": list(TRUE_THEOREM_GATES),
        "false_theorem_gates": list(FALSE_THEOREM_GATES),
    }


def refresh_reproducibility_plan(
    source_plan: Mapping[str, object], report: Mapping[str, object]
) -> dict[str, object]:
    """Derive the effective plan without executing excluded notebook Cell 104."""

    if set(source_plan) != {"precision_settings", "upstream_artifact_names"}:
        raise ReproducibilityError(
            "The source reproducibility plan has an unexpected top-level schema."
        )
    source_precision = source_plan.get("precision_settings")
    source_artifacts = source_plan.get("upstream_artifact_names")
    if not isinstance(source_precision, dict) or not isinstance(
        source_artifacts, list
    ):
        raise ReproducibilityError(
            "The source reproducibility plan fields have invalid JSON types."
        )

    precision = dict(source_precision)
    report_updates = {
        "map_label": report.get("map_label"),
        "N": report.get("N"),
        "M": report.get("M"),
        "rho": report.get("rho"),
        "r": report.get("r"),
        "hardy_matrix_eta_A_upper": report.get("eta_A_upper_text"),
        "hardy_matrix_midpoint_sha256": report.get("matrix_midpoint_sha256"),
        "contour_certificate_schema": report.get("certificate_schema"),
        "contour_schur_count_target_count": report.get(
            "schur_count_target_count"
        ),
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
    precision.update(report_updates)
    effective_plan = {
        "precision_settings": precision,
        "upstream_artifact_names": list(source_artifacts),
    }
    validate_plan_and_spectral_report(effective_plan, report)
    return effective_plan


def _validate_inline_helper_provenance(
    repo_root: Path, notebook: Mapping[str, object]
) -> None:
    builder_path = (
        repo_root / "Numerics" / "build_blaschke_deformation_thesis_math_notebook.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_blaschke_thesis_math_builder_provenance",
        builder_path,
    )
    if spec is None or spec.loader is None:
        raise ReproducibilityError(
            f"Cannot load inline-helper validator from {builder_path}."
        )
    builder = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(builder)
        builder.validate_curated_counterpart(notebook)
    except Exception as exc:
        raise ReproducibilityError(
            f"Inline helper provenance validation failed: {exc}"
        ) from exc


def validate_executed_notebook(
    repo_root: Path, notebook_path: Path
) -> dict[str, object]:
    notebook = _load_json_object(notebook_path, label="executed notebook")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or len(cells) != EXPECTED_NOTEBOOK_CELLS:
        observed = len(cells) if isinstance(cells, list) else None
        raise ReproducibilityError(
            f"Executed notebook must contain {EXPECTED_NOTEBOOK_CELLS} cells; "
            f"observed {observed!r}."
        )
    code_cells = [
        cell
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "code"
    ]
    if len(code_cells) != EXPECTED_NOTEBOOK_CODE_CELLS:
        raise ReproducibilityError(
            "Executed notebook must contain exactly "
            f"{EXPECTED_NOTEBOOK_CODE_CELLS} code cells; observed "
            f"{len(code_cells)}."
        )
    execution_counts: list[int] = []
    error_count = 0
    for ordinal, cell in enumerate(code_cells, start=1):
        count = cell.get("execution_count")
        if type(count) is not int:
            raise ReproducibilityError(
                f"Code cell {ordinal} is not executed: execution_count={count!r}."
            )
        execution_counts.append(count)
        outputs = cell.get("outputs")
        if not isinstance(outputs, list):
            raise ReproducibilityError(
                f"Code cell {ordinal} outputs must be a JSON array."
            )
        error_count += sum(
            isinstance(output, dict) and output.get("output_type") == "error"
            for output in outputs
        )
    expected_counts = list(range(1, EXPECTED_NOTEBOOK_CODE_CELLS + 1))
    if execution_counts != expected_counts:
        raise ReproducibilityError(
            "Executed notebook code-cell counts must be contiguous 1 through "
            f"{EXPECTED_NOTEBOOK_CODE_CELLS}; observed {execution_counts}."
        )
    if error_count:
        raise ReproducibilityError(
            f"Executed notebook contains {error_count} stored error outputs."
        )
    _validate_inline_helper_provenance(repo_root, notebook)
    return {
        "path": NOTEBOOK_RELATIVE.as_posix(),
        "cell_count": EXPECTED_NOTEBOOK_CELLS,
        "code_cell_count": EXPECTED_NOTEBOOK_CODE_CELLS,
        "executed_code_cell_count": EXPECTED_NOTEBOOK_CODE_CELLS,
        "execution_counts_contiguous": True,
        "stored_error_output_count": 0,
        "inline_helper_sync": True,
    }


def _manifest_entry_for(
    manifest_by_path: Mapping[str, ManifestEntry], relative: PurePosixPath
) -> ManifestEntry:
    entry = manifest_by_path.get(relative.as_posix())
    if entry is None:
        raise ReproducibilityError(
            f"Source-generated evidence is absent from {SOURCE_MANIFEST_NAME}: "
            f"{relative.as_posix()}."
        )
    return entry


def _validate_report_source_records(
    records: object,
    *,
    manifest_by_path: Mapping[str, ManifestEntry],
    label: str,
) -> int:
    if not isinstance(records, dict) or not records:
        raise ReproducibilityError(f"{label} source records must be non-empty.")
    validated = 0
    for source_label, record in records.items():
        if not isinstance(source_label, str) or not isinstance(record, dict):
            raise ReproducibilityError(f"Malformed {label} source record.")
        path_text = record.get("path")
        digest = record.get("sha256")
        relative = _safe_relative_path(path_text, label=f"{label} source path")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ReproducibilityError(
                f"Malformed digest for {label} source {source_label}."
            )
        entry = _manifest_entry_for(manifest_by_path, relative)
        if entry.sha256 != digest:
            raise ReproducibilityError(
                f"{label} source digest differs from the authoritative manifest: "
                f"{relative.as_posix()}."
            )
        validated += 1
    return validated


def _validate_historical_phase4_documentation_refresh(
    repo_root: Path,
    report: Mapping[str, object],
) -> dict[str, object] | None:
    refresh = report.get("documentation_only_provenance_refresh")
    if refresh is None:
        return None
    if not isinstance(refresh, dict):
        raise ReproducibilityError(
            "Historical Phase 4 documentation refresh is malformed."
        )
    if refresh.get("schema") != EXPECTED_HISTORICAL_PHASE4_DOC_REFRESH_SCHEMA:
        raise ReproducibilityError(
            "Historical Phase 4 documentation refresh schema is invalid."
        )
    if refresh.get("numerical_outputs_reused") is not True:
        raise ReproducibilityError(
            "Historical Phase 4 documentation refresh must identify reused outputs."
        )
    expected_method = (
        "Python AST equality after removing module, class, function, and "
        "async-function docstrings and ignoring source-location attributes"
    )
    if refresh.get("verification_method") != expected_method:
        raise ReproducibilityError(
            "Historical Phase 4 documentation refresh method is invalid."
        )

    current_records = report.get("producer_sources")
    runtime_versions = report.get("runtime_versions")
    sources = refresh.get("sources")
    if (
        not isinstance(current_records, dict)
        or not isinstance(runtime_versions, dict)
        or not isinstance(sources, dict)
        or set(sources) != set(current_records)
    ):
        raise ReproducibilityError(
            "Historical Phase 4 documentation source inventory is inconsistent."
        )

    previous_records: dict[str, dict[str, str]] = {}
    verified_sources: dict[str, dict[str, str]] = {}
    for label, current_record in current_records.items():
        source = sources.get(label)
        if not isinstance(current_record, dict) or not isinstance(source, dict):
            raise ReproducibilityError(
                "Historical Phase 4 documentation source record is malformed."
            )
        relative = _safe_relative_path(
            source.get("path"),
            label=f"historical Phase 4 documentation source {label}",
        )
        path_text = relative.as_posix()
        current_sha = _expect_sha256(
            source.get("current_sha256"),
            label=f"historical Phase 4 current source {label}",
        )
        previous_sha = _expect_sha256(
            source.get("previous_sha256"),
            label=f"historical Phase 4 previous source {label}",
        )
        current_semantic = _expect_sha256(
            source.get("current_semantic_ast_sha256"),
            label=f"historical Phase 4 current semantic source {label}",
        )
        previous_semantic = _expect_sha256(
            source.get("previous_semantic_ast_sha256"),
            label=f"historical Phase 4 previous semantic source {label}",
        )
        if current_record != {"path": path_text, "sha256": current_sha}:
            raise ReproducibilityError(
                "Historical Phase 4 refreshed source record differs from its attestation."
            )
        source_path = repo_root.joinpath(*relative.parts)
        if sha256_file(source_path) != current_sha:
            raise ReproducibilityError(
                "Historical Phase 4 refreshed source hash is stale."
            )
        observed_semantic = _semantic_ast_sha256(source_path)
        if not (
            observed_semantic == current_semantic == previous_semantic
        ):
            raise ReproducibilityError(
                "Historical Phase 4 source differs beyond Python docstrings."
            )
        previous_records[str(label)] = {
            "path": path_text,
            "sha256": previous_sha,
        }
        verified_sources[str(label)] = {
            "path": path_text,
            "previous_sha256": previous_sha,
            "current_sha256": current_sha,
            "semantic_ast_sha256": observed_semantic,
        }

    previous_source_digest = _canonical_digest({
        "producer_sources": previous_records,
        "runtime_versions": runtime_versions,
    })
    current_source_digest = _canonical_digest({
        "producer_sources": current_records,
        "runtime_versions": runtime_versions,
    })
    if not (
        report.get("source_digest")
        == refresh.get("previous_source_digest")
        == previous_source_digest
    ):
        raise ReproducibilityError(
            "Historical Phase 4 previous source digest is inconsistent."
        )
    if refresh.get("current_source_digest") != current_source_digest:
        raise ReproducibilityError(
            "Historical Phase 4 current source digest is inconsistent."
        )

    configuration_digest = report.get("configuration_digest")
    producer_schema = report.get("producer_schema")
    previous_cache_key = _canonical_digest({
        "producer_schema": producer_schema,
        "configuration_digest": configuration_digest,
        "source_digest": previous_source_digest,
    })
    current_cache_key = _canonical_digest({
        "producer_schema": producer_schema,
        "configuration_digest": configuration_digest,
        "source_digest": current_source_digest,
    })
    if not (
        report.get("cache_key")
        == refresh.get("previous_cache_key")
        == previous_cache_key
    ):
        raise ReproducibilityError(
            "Historical Phase 4 previous cache identity is inconsistent."
        )
    if refresh.get("current_cache_key_if_rebuilt") != current_cache_key:
        raise ReproducibilityError(
            "Historical Phase 4 prospective cache identity is inconsistent."
        )
    return {
        "schema": EXPECTED_HISTORICAL_PHASE4_DOC_REFRESH_SCHEMA,
        "numerical_outputs_reused": True,
        "verified_source_count": len(verified_sources),
        "previous_source_digest": previous_source_digest,
        "current_source_digest": current_source_digest,
        "sources": verified_sources,
    }


def _validate_generated_output_hashes(
    repo_root: Path,
    outputs: object,
    *,
    manifest_by_path: Mapping[str, ManifestEntry],
    label: str,
    expected_count: int,
) -> int:
    if not isinstance(outputs, dict) or len(outputs) != expected_count:
        observed = len(outputs) if isinstance(outputs, dict) else None
        raise ReproducibilityError(
            f"{label} must describe {expected_count} generated outputs; "
            f"observed {observed!r}."
        )
    output_root = repo_root.joinpath(*OUTPUT_RELATIVE.parts)
    validated = 0
    for output_key, value in outputs.items():
        if not isinstance(output_key, str):
            raise ReproducibilityError(f"Malformed {label} output key.")
        if isinstance(value, str):
            filename = output_key
            digest = value
        elif isinstance(value, dict):
            filename_value = value.get("path", output_key)
            filename = Path(str(filename_value)).name
            digest = value.get("sha256")
        else:
            raise ReproducibilityError(f"Malformed {label} output record.")
        if Path(filename).name != filename or filename in {"", ".", ".."}:
            raise ReproducibilityError(f"Unsafe {label} output filename: {filename!r}.")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ReproducibilityError(
                f"Malformed digest for {label} output {output_key}."
            )
        candidates = tuple(
            path
            for path in (
                output_root / "data" / filename,
                output_root / "reports" / filename,
            )
            if path.is_file() and not path.is_symlink()
        )
        if len(candidates) != 1:
            raise ReproducibilityError(
                f"{label} output {filename} must resolve to exactly one regular file."
            )
        relative = PurePosixPath(candidates[0].relative_to(repo_root).as_posix())
        entry = _manifest_entry_for(manifest_by_path, relative)
        observed = sha256_file(candidates[0])
        if observed != digest or entry.sha256 != digest:
            raise ReproducibilityError(
                f"{label} output digest mismatch for {relative.as_posix()}."
            )
        validated += 1
    return validated


def _validate_manifest_output(
    repo_root: Path,
    manifest_by_path: Mapping[str, ManifestEntry],
    *,
    filename: str,
    digest: object,
    label: str,
) -> Path:
    expected_hash = _expect_sha256(digest, label=f"{label} sha256")
    relative = OUTPUT_RELATIVE / "data" / filename
    path = _absolute_repo_file(repo_root, relative)
    entry = _manifest_entry_for(manifest_by_path, relative)
    observed = sha256_file(path)
    if observed != expected_hash or entry.sha256 != expected_hash:
        raise ReproducibilityError(
            f"{label} hash differs from its report or {SOURCE_MANIFEST_NAME}."
        )
    return path


def _csv_header_and_rows(path: Path, *, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ReproducibilityError(f"{label} has no CSV header.")
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ReproducibilityError(f"Cannot parse {label}: {exc}") from exc
    if any(None in row for row in rows):
        raise ReproducibilityError(f"{label} contains rows outside its CSV schema.")
    return list(reader.fieldnames), rows


def _validate_diagnostic_report_preamble(
    report: Mapping[str, object], *, label: str, producer_schema: str
) -> None:
    if report.get("producer_schema") != producer_schema:
        raise ReproducibilityError(f"Unexpected {label} producer schema.")
    if report.get("schema_version") != EXPECTED_DIAGNOSTIC_SCHEMA_VERSION:
        raise ReproducibilityError(f"Unexpected {label} report schema.")
    if report.get("map_label") != EXPECTED_MAP_LABEL:
        raise ReproducibilityError(f"Unexpected map label in {label} report.")
    _expect_bool(report, "diagnostic_only", True)
    _expect_bool(report, "theorem_gate", False)
    _expect_bool(report, "legacy_seed_dependency", False)


def _validate_canonical_callable_lineage(
    record: object,
    *,
    manifest_by_path: Mapping[str, ManifestEntry],
    canonical_relative: PurePosixPath,
    label: str,
) -> dict[str, str]:
    if not isinstance(record, dict):
        raise ReproducibilityError(f"{label} lineage record is missing.")
    module = record.get("module")
    qualname = record.get("qualname")
    if not isinstance(module, str) or not module:
        raise ReproducibilityError(f"{label} lineage module is missing.")
    if not isinstance(qualname, str) or not qualname:
        raise ReproducibilityError(f"{label} lineage qualname is missing.")
    digest = _expect_sha256(
        record.get("source_sha256"), label=f"{label} source_sha256"
    )
    canonical = _manifest_entry_for(manifest_by_path, canonical_relative)
    if digest != canonical.sha256:
        raise ReproducibilityError(
            f"{label} lineage is not anchored to "
            f"{canonical_relative.as_posix()}."
        )
    if record.get("source_scope") != "complete module file":
        raise ReproducibilityError(
            f"{label} lineage must hash the complete canonical module file."
        )
    return {
        "module": module,
        "qualname": qualname,
        "source_path": canonical_relative.as_posix(),
        "source_sha256": digest,
    }


def _builder_callable_source_hashes(
    repo_root: Path, *, constant_name: str, function_name: str
) -> frozenset[str]:
    builder_path = _absolute_repo_file(repo_root, THESIS_NOTEBOOK_BUILDER_RELATIVE)
    try:
        builder_source = builder_path.read_text(encoding="utf-8")
        builder_tree = ast.parse(builder_source, filename=str(builder_path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise ReproducibilityError(
            f"Cannot inspect notebook-builder lineage: {exc}"
        ) from exc
    cell_source: str | None = None
    for node in builder_tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        if not any(
            isinstance(target, ast.Name) and target.id == constant_name
            for target in targets
        ):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            cell_source = value.value
            break
    if cell_source is None:
        raise ReproducibilityError(
            f"Notebook builder lacks the {constant_name} source block."
        )
    try:
        cell_tree = ast.parse(cell_source, filename=constant_name)
    except SyntaxError as exc:
        raise ReproducibilityError(
            f"Cannot parse notebook-builder source block {constant_name}: {exc}"
        ) from exc
    function = next(
        (
            node
            for node in cell_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    if function is None or function.end_lineno is None:
        raise ReproducibilityError(
            f"Notebook builder lacks the {function_name} callable source."
        )
    lines = cell_source.splitlines(keepends=True)
    source = "".join(lines[function.lineno - 1 : function.end_lineno])
    variants = {source, source.rstrip("\r\n"), source.rstrip("\r\n") + "\n"}
    return frozenset(
        hashlib.sha256(value.encode("utf-8")).hexdigest() for value in variants
    )


def _validate_builder_callable_lineage(
    repo_root: Path,
    record: object,
    *,
    manifest_by_path: Mapping[str, ManifestEntry],
    constant_name: str,
    function_name: str,
    label: str,
) -> dict[str, str]:
    _manifest_entry_for(manifest_by_path, THESIS_NOTEBOOK_BUILDER_RELATIVE)
    if not isinstance(record, dict):
        raise ReproducibilityError(f"{label} lineage record is missing.")
    module = record.get("module")
    qualname = record.get("qualname")
    if not isinstance(module, str) or not module:
        raise ReproducibilityError(f"{label} lineage module is missing.")
    if qualname != function_name:
        raise ReproducibilityError(
            f"{label} lineage qualname must be {function_name!r}."
        )
    digest = _expect_sha256(
        record.get("source_sha256"), label=f"{label} source_sha256"
    )
    scope = record.get("source_scope")
    if scope == "callable source":
        expected_hashes = _builder_callable_source_hashes(
            repo_root,
            constant_name=constant_name,
            function_name=function_name,
        )
    elif scope == "stable callable identity":
        identity = f"{module}:{qualname}".encode("utf-8")
        expected_hashes = frozenset({hashlib.sha256(identity).hexdigest()})
    else:
        raise ReproducibilityError(
            f"{label} lineage must use stable callable-source identity."
        )
    if digest not in expected_hashes:
        raise ReproducibilityError(
            f"{label} lineage digest differs from the source-controlled builder."
        )
    return {
        "module": module,
        "qualname": qualname,
        "source_path": THESIS_NOTEBOOK_BUILDER_RELATIVE.as_posix(),
        "source_sha256": digest,
    }


def _validate_phase1_diagnostic_producer(
    repo_root: Path,
    manifest_by_path: Mapping[str, ManifestEntry],
) -> dict[str, object]:
    report_path = _absolute_repo_file(repo_root, PHASE1_DIAGNOSTIC_REPORT_RELATIVE)
    report_entry = _manifest_entry_for(
        manifest_by_path, PHASE1_DIAGNOSTIC_REPORT_RELATIVE
    )
    if sha256_file(report_path) != report_entry.sha256:
        raise ReproducibilityError("Phase 1 diagnostic report hash mismatch.")
    report = _load_json_object(report_path, label="Phase 1 diagnostic report")
    _validate_diagnostic_report_preamble(
        report,
        label="Phase 1 diagnostic",
        producer_schema=EXPECTED_PHASE1_PRODUCER_SCHEMA,
    )
    _expect_sha256(report.get("generation_id"), label="Phase 1 generation_id")

    config = report.get("config")
    if not isinstance(config, dict):
        raise ReproducibilityError("Phase 1 diagnostic config is missing.")
    cloud_pairs = tuple(
        tuple(pair) if isinstance(pair, list) else ()
        for pair in config.get("cloud_pairs", ())
    )
    if cloud_pairs != EXPECTED_PHASE1_CLOUD_PAIRS:
        raise ReproducibilityError("Phase 1 cloud-pair schedule has drifted.")
    _expect_exact_int(config, "fixed_n", 25)
    _expect_exact_int(config, "expected_target_count", EXPECTED_TARGET_COUNT)

    schedules = report.get("schedules")
    if not isinstance(schedules, dict) or tuple(
        tuple(pair) if isinstance(pair, list) else ()
        for pair in schedules.get("cloud_pairs", ())
    ) != EXPECTED_PHASE1_CLOUD_PAIRS:
        raise ReproducibilityError("Phase 1 persisted schedule has drifted.")

    lineage = report.get("lineage")
    if not isinstance(lineage, dict):
        raise ReproducibilityError("Phase 1 diagnostic lineage is missing.")
    for key in ("map_spec_sha256", "expected_target_names_sha256"):
        _expect_sha256(lineage.get(key), label=f"Phase 1 {key}")
    reference_digest = lineage.get("reference_clusters_sha256")
    if reference_digest is not None:
        _expect_sha256(reference_digest, label="Phase 1 reference_clusters_sha256")
    raw_sweep_lineage = _validate_canonical_callable_lineage(
        lineage.get("raw_sweep"),
        manifest_by_path=manifest_by_path,
        canonical_relative=PHASE1_WORKER_RELATIVE,
        label="Phase 1 raw sweep",
    )
    producer_entry = _manifest_entry_for(
        manifest_by_path, PHASE1_DIAGNOSTIC_PRODUCER_RELATIVE
    )

    outputs = report.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != EXPECTED_PHASE1_OUTPUT_NAMES:
        observed = sorted(outputs) if isinstance(outputs, dict) else None
        raise ReproducibilityError(
            "Phase 1 diagnostic output schema has drifted; "
            f"observed={observed!r}."
        )
    validated_files = 0
    for name in sorted(outputs):
        record = outputs[name]
        if not isinstance(record, dict):
            raise ReproducibilityError(f"Malformed Phase 1 output record {name}.")
        csv_name = f"{name}.csv"
        parquet_name = f"{name}.parquet"
        if Path(str(record.get("path"))).name != csv_name:
            raise ReproducibilityError(f"Unexpected Phase 1 CSV path for {name}.")
        if Path(str(record.get("parquet_path"))).name != parquet_name:
            raise ReproducibilityError(
                f"Phase 1 output {name} lacks its mandatory Parquet companion."
            )
        csv_path = _validate_manifest_output(
            repo_root,
            manifest_by_path,
            filename=csv_name,
            digest=record.get("sha256"),
            label=f"Phase 1 CSV {name}",
        )
        parquet_path = _validate_manifest_output(
            repo_root,
            manifest_by_path,
            filename=parquet_name,
            digest=record.get("parquet_sha256"),
            label=f"Phase 1 Parquet {name}",
        )
        if parquet_path.stat().st_size == 0:
            raise ReproducibilityError(
                f"Phase 1 Parquet companion is empty for {name}."
            )
        columns = record.get("columns")
        row_count = record.get("rows")
        if (
            not isinstance(columns, list)
            or not columns
            or not all(isinstance(column, str) and column for column in columns)
            or len(set(columns)) != len(columns)
            or type(row_count) is not int
            or row_count < 1
        ):
            raise ReproducibilityError(f"Malformed Phase 1 schema for {name}.")
        header, rows = _csv_header_and_rows(csv_path, label=f"Phase 1 CSV {name}")
        if header != columns or len(rows) != row_count or "map_name" not in header:
            raise ReproducibilityError(f"Phase 1 CSV schema mismatch for {name}.")
        map_labels = {row.get("map_name", "") for row in rows}
        if map_labels != {EXPECTED_MAP_LABEL}:
            raise ReproducibilityError(f"Phase 1 CSV map lock failed for {name}.")
        validated_files += 2

    return {
        "producer_schema": EXPECTED_PHASE1_PRODUCER_SCHEMA,
        "schema_version": EXPECTED_DIAGNOSTIC_SCHEMA_VERSION,
        "map_label": EXPECTED_MAP_LABEL,
        "diagnostic_only": True,
        "theorem_gate": False,
        "report_path": PHASE1_DIAGNOSTIC_REPORT_RELATIVE.as_posix(),
        "report_sha256": report_entry.sha256,
        "producer_source_path": PHASE1_DIAGNOSTIC_PRODUCER_RELATIVE.as_posix(),
        "producer_source_sha256": producer_entry.sha256,
        "raw_sweep_lineage": raw_sweep_lineage,
        "dataset_count": len(outputs),
        "validated_file_count": validated_files,
        "mandatory_parquet_companions": True,
    }


def _validate_sampled_schur_diagnostic_producer(
    repo_root: Path,
    manifest_by_path: Mapping[str, ManifestEntry],
) -> dict[str, object]:
    report_path = _absolute_repo_file(
        repo_root, SAMPLED_SCHUR_DIAGNOSTIC_REPORT_RELATIVE
    )
    report_entry = _manifest_entry_for(
        manifest_by_path, SAMPLED_SCHUR_DIAGNOSTIC_REPORT_RELATIVE
    )
    if sha256_file(report_path) != report_entry.sha256:
        raise ReproducibilityError("Sampled Schur diagnostic report hash mismatch.")
    report = _load_json_object(report_path, label="sampled Schur diagnostic report")
    _validate_diagnostic_report_preamble(
        report,
        label="sampled Schur diagnostic",
        producer_schema=EXPECTED_SAMPLED_SCHUR_PRODUCER_SCHEMA,
    )

    config = report.get("config")
    if not isinstance(config, dict):
        raise ReproducibilityError("Sampled Schur diagnostic config is missing.")
    n_values = config.get("n_values")
    if not isinstance(n_values, list) or tuple(n_values) != EXPECTED_SAMPLED_SCHUR_N:
        raise ReproducibilityError("Sampled Schur N schedule has drifted.")
    _expect_exact_int(
        config, "oversampling", EXPECTED_SAMPLED_SCHUR_OVERSAMPLING
    )

    output = report.get("output")
    if not isinstance(output, dict):
        raise ReproducibilityError("Sampled Schur output record is missing.")
    if Path(str(output.get("path"))).name != EXPECTED_SAMPLED_SCHUR_FILENAME:
        raise ReproducibilityError("Unexpected sampled Schur output path.")
    csv_path = _validate_manifest_output(
        repo_root,
        manifest_by_path,
        filename=EXPECTED_SAMPLED_SCHUR_FILENAME,
        digest=output.get("sha256"),
        label="sampled Schur CSV",
    )
    columns = output.get("columns")
    row_count = output.get("rows")
    if (
        not isinstance(columns, list)
        or not all(isinstance(column, str) and column for column in columns)
        or not EXPECTED_SAMPLED_SCHUR_COLUMNS.issubset(columns)
        or row_count != len(EXPECTED_SAMPLED_SCHUR_N)
    ):
        raise ReproducibilityError("Sampled Schur output schema has drifted.")
    header, rows = _csv_header_and_rows(csv_path, label="sampled Schur CSV")
    if header != columns or len(rows) != row_count:
        raise ReproducibilityError("Sampled Schur CSV schema mismatch.")
    try:
        observed_n = tuple(int(row["N"]) for row in rows)
        observed_m = tuple(int(row["M"]) for row in rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise ReproducibilityError(
            "Sampled Schur CSV contains invalid dimensions."
        ) from exc
    if observed_n != EXPECTED_SAMPLED_SCHUR_N or observed_m != tuple(
        n_value + EXPECTED_SAMPLED_SCHUR_OVERSAMPLING
        for n_value in EXPECTED_SAMPLED_SCHUR_N
    ):
        raise ReproducibilityError("Sampled Schur CSV schedule has drifted.")
    if {row.get("map_label", "") for row in rows} != {EXPECTED_MAP_LABEL}:
        raise ReproducibilityError("Sampled Schur CSV map lock failed.")

    lineage = report.get("lineage")
    if not isinstance(lineage, dict):
        raise ReproducibilityError("Sampled Schur diagnostic lineage is missing.")
    envelope_lineage = _validate_canonical_callable_lineage(
        lineage.get("sampled_schur_envelope"),
        manifest_by_path=manifest_by_path,
        canonical_relative=SAMPLED_SCHUR_ENVELOPE_RELATIVE,
        label="sampled Schur envelope",
    )
    kappa_lineage = _validate_builder_callable_lineage(
        repo_root,
        lineage.get("kappa"),
        manifest_by_path=manifest_by_path,
        constant_name="SAMPLED_SCHUR_DIAGNOSTIC_REBUILD",
        function_name="_sampled_schur_kappa",
        label="sampled Schur kappa",
    )
    producer_entry = _manifest_entry_for(
        manifest_by_path, SAMPLED_SCHUR_DIAGNOSTIC_PRODUCER_RELATIVE
    )
    return {
        "producer_schema": EXPECTED_SAMPLED_SCHUR_PRODUCER_SCHEMA,
        "schema_version": EXPECTED_DIAGNOSTIC_SCHEMA_VERSION,
        "map_label": EXPECTED_MAP_LABEL,
        "diagnostic_only": True,
        "theorem_gate": False,
        "report_path": SAMPLED_SCHUR_DIAGNOSTIC_REPORT_RELATIVE.as_posix(),
        "report_sha256": report_entry.sha256,
        "producer_source_path": (
            SAMPLED_SCHUR_DIAGNOSTIC_PRODUCER_RELATIVE.as_posix()
        ),
        "producer_source_sha256": producer_entry.sha256,
        "envelope_lineage": envelope_lineage,
        "kappa_lineage": kappa_lineage,
        "dataset_count": 1,
        "validated_file_count": 1,
    }


def validate_source_generated_diagnostics(
    repo_root: Path,
    manifest_by_path: Mapping[str, ManifestEntry],
) -> dict[str, object]:
    """Require every retained diagnostic dataset to have a source rebuild proof."""

    reports: dict[str, tuple[PurePosixPath, str]] = {
        "historical_phase2": (
            HISTORICAL_PHASE2_REPORT_RELATIVE,
            EXPECTED_HISTORICAL_PHASE2_SCHEMA,
        ),
        "historical_phase4": (
            HISTORICAL_PHASE4_REPORT_RELATIVE,
            EXPECTED_HISTORICAL_PHASE4_SCHEMA,
        ),
        "diagnostic_audits": (
            DIAGNOSTIC_AUDIT_REPORT_RELATIVE,
            EXPECTED_DIAGNOSTIC_AUDIT_SCHEMA,
        ),
    }
    loaded: dict[str, dict[str, object]] = {}
    report_hashes: dict[str, str] = {}
    for name, (relative, schema) in reports.items():
        path = _absolute_repo_file(repo_root, relative)
        entry = _manifest_entry_for(manifest_by_path, relative)
        report = _load_json_object(path, label=f"{name} source-rebuild report")
        if report.get("producer_schema") != schema:
            raise ReproducibilityError(f"Unexpected producer schema in {name} report.")
        if report.get("map_label") != EXPECTED_MAP_LABEL:
            raise ReproducibilityError(f"Unexpected map label in {name} report.")
        if report.get("legacy_seed_dependency") is not False:
            raise ReproducibilityError(
                f"{name} report does not exclude retained legacy seed inputs."
            )
        if sha256_file(path) != entry.sha256:
            raise ReproducibilityError(f"Manifest mismatch for {name} report.")
        loaded[name] = report
        report_hashes[name] = entry.sha256

    phase2 = loaded["historical_phase2"]
    if phase2.get("diagnostic_status") != (
        "retained source-rebuilt historical-design comparison; not a theorem gate"
    ):
        raise ReproducibilityError("Historical Phase 2 status is not diagnostic-only.")
    phase2_output_count = _validate_generated_output_hashes(
        repo_root,
        phase2.get("outputs"),
        manifest_by_path=manifest_by_path,
        label="historical Phase 2",
        expected_count=2,
    )
    phase2_source_count = _validate_report_source_records(
        phase2.get("producer_sources"),
        manifest_by_path=manifest_by_path,
        label="historical Phase 2",
    )

    phase4 = loaded["historical_phase4"]
    configuration = phase4.get("configuration")
    if not isinstance(configuration, dict):
        raise ReproducibilityError("Historical Phase 4 configuration is missing.")
    _expect_exact_int(configuration, "N", 600)
    _expect_exact_int(configuration, "M", 610)
    if configuration.get("mode") != "production":
        raise ReproducibilityError("Historical Phase 4 was not rebuilt in production mode.")
    for key, expected in (
        ("authoritative_for_current_thesis", False),
        ("theorem_gate_eligible", False),
        ("contour_interval_certified", False),
    ):
        if phase4.get(key) is not expected:
            raise ReproducibilityError(f"Unexpected historical Phase 4 gate {key}.")
    if phase4.get("retained_csv_inputs") != [] or phase4.get("retained_npz_inputs") != []:
        raise ReproducibilityError("Historical Phase 4 reports retained seed inputs.")
    regression = phase4.get("historical_regression")
    if not isinstance(regression, dict):
        raise ReproducibilityError("Historical Phase 4 regression record is missing.")
    if regression.get("contract") != "source-rederived historical sampled comparison":
        raise ReproducibilityError("Historical Phase 4 regression contract is invalid.")
    if regression.get("sampled_validation_passes") != 15:
        raise ReproducibilityError("Historical Phase 4 did not reproduce all sampled passes.")
    worst_product = regression.get("worst_finite_epsilon_m_gamma")
    if (
        isinstance(worst_product, bool)
        or not isinstance(worst_product, (int, float))
        or not 0.45 <= float(worst_product) <= 0.55
    ):
        raise ReproducibilityError(
            "Historical Phase 4 sampled product left its regression regime."
        )
    process_execution = phase4.get("process_execution")
    if not isinstance(process_execution, dict) or not all(
        process_execution.get(key) is True
        for key in ("row_block_assembly", "surface_row_blocks")
    ):
        raise ReproducibilityError("Historical Phase 4 process execution is incomplete.")
    phase4_output_count = _validate_generated_output_hashes(
        repo_root,
        phase4.get("outputs"),
        manifest_by_path=manifest_by_path,
        label="historical Phase 4",
        expected_count=20,
    )
    phase4_source_count = _validate_report_source_records(
        phase4.get("producer_sources"),
        manifest_by_path=manifest_by_path,
        label="historical Phase 4",
    )
    phase4_documentation_refresh = (
        _validate_historical_phase4_documentation_refresh(repo_root, phase4)
    )

    audits = loaded["diagnostic_audits"]
    if audits.get("diagnostic_only") is not True:
        raise ReproducibilityError("Diagnostic audit report is not diagnostic-only.")
    checks = audits.get("checks")
    if not isinstance(checks, dict) or not all(
        checks.get(key) is True
        for key in (
            "map_lock",
            "schema_alignment",
            "target_alignment",
            "sampled_claims_remain_diagnostic",
        )
    ):
        raise ReproducibilityError("Diagnostic audit source checks are incomplete.")
    audit_output_count = _validate_generated_output_hashes(
        repo_root,
        audits.get("outputs"),
        manifest_by_path=manifest_by_path,
        label="diagnostic audits",
        expected_count=2,
    )
    source_extraction = audits.get("source_extraction")
    if not isinstance(source_extraction, dict):
        raise ReproducibilityError("Diagnostic audit source extraction is missing.")
    audit_source_count = _validate_report_source_records(
        source_extraction,
        manifest_by_path=manifest_by_path,
        label="diagnostic audits",
    )

    phase1_source_present = (
        PHASE1_DIAGNOSTIC_PRODUCER_RELATIVE.as_posix() in manifest_by_path
    )
    phase1_report_present = (
        PHASE1_DIAGNOSTIC_REPORT_RELATIVE.as_posix() in manifest_by_path
    )
    if not phase1_source_present:
        raise ReproducibilityError(
            "The Phase 1 diagnostic producer source is absent from the manifest."
        )
    if not phase1_report_present:
        raise ReproducibilityError(
            "The Phase 1 diagnostic report is absent from the manifest."
        )
    phase1_validation: dict[str, object] = {
        "required": True,
        "validated": True,
        **_validate_phase1_diagnostic_producer(repo_root, manifest_by_path),
    }
    report_hashes["phase1_diagnostics"] = str(
        phase1_validation["report_sha256"]
    )

    sampled_source_present = (
        SAMPLED_SCHUR_DIAGNOSTIC_PRODUCER_RELATIVE.as_posix()
        in manifest_by_path
    )
    sampled_report_present = (
        SAMPLED_SCHUR_DIAGNOSTIC_REPORT_RELATIVE.as_posix()
        in manifest_by_path
    )
    if not sampled_source_present:
        raise ReproducibilityError(
            "The sampled Schur diagnostic producer source is absent from the manifest."
        )
    if not sampled_report_present:
        raise ReproducibilityError(
            "The sampled Schur diagnostic report is absent from the manifest."
        )
    sampled_validation: dict[str, object] = {
        "required": True,
        "validated": True,
        **_validate_sampled_schur_diagnostic_producer(
            repo_root, manifest_by_path
        ),
    }
    report_hashes["sampled_schur_diagnostics"] = str(
        sampled_validation["report_sha256"]
    )

    phase1_file_count = int(phase1_validation.get("validated_file_count", 0))
    sampled_file_count = int(sampled_validation.get("validated_file_count", 0))
    producer_source_count = 5

    return {
        "legacy_seed_dependency": False,
        "report_sha256": report_hashes,
        "validated_output_count": (
            phase2_output_count
            + phase4_output_count
            + audit_output_count
            + phase1_file_count
            + sampled_file_count
        ),
        "validated_source_count": (
            phase2_source_count
            + phase4_source_count
            + audit_source_count
            + producer_source_count
        ),
        "historical_phase2_output_count": phase2_output_count,
        "historical_phase4_output_count": phase4_output_count,
        "historical_phase4_documentation_refresh": (
            phase4_documentation_refresh
        ),
        "diagnostic_audit_output_count": audit_output_count,
        "phase1_diagnostics": phase1_validation,
        "sampled_schur_diagnostics": sampled_validation,
    }


def _conda_explicit_lock(repo_root: Path) -> str:
    conda_executable = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda_executable:
        raise ReproducibilityError(
            "Cannot capture a full Conda explicit lock because conda is unavailable."
        )
    output = _run_checked(
        (
            conda_executable,
            "list",
            "--explicit",
            "--prefix",
            sys.prefix,
        ),
        cwd=repo_root,
        text=True,
    )
    if not isinstance(output, str):
        raise AssertionError("Conda explicit lock unexpectedly returned bytes.")
    if "@EXPLICIT" not in output.splitlines():
        raise ReproducibilityError(
            "conda list --explicit did not return an @EXPLICIT environment lock."
        )
    return output if output.endswith("\n") else output + "\n"


def _manifest_index(
    entries: Iterable[ManifestEntry],
) -> dict[str, ManifestEntry]:
    return {entry.relative_path.as_posix(): entry for entry in entries}


def _validate_upstream_artifacts(
    *,
    repo_root: Path,
    names: Iterable[str],
    manifest_entries: Mapping[str, ManifestEntry],
) -> list[dict[str, object]]:
    if isinstance(names, (str, bytes)):
        raise ReproducibilityError(
            "upstream_artifact_names must be an iterable of relative data paths."
        )
    resolved: list[dict[str, object]] = []
    seen: set[PurePosixPath] = set()
    for value in names:
        artifact_relative = _safe_relative_path(
            value, label="upstream artifact data path"
        )
        if artifact_relative in seen:
            raise ReproducibilityError(
                f"Duplicate upstream artifact: {artifact_relative.as_posix()}."
            )
        seen.add(artifact_relative)
        repo_relative = OUTPUT_RELATIVE / "data" / artifact_relative
        repo_label = repo_relative.as_posix()
        source = _absolute_repo_file(repo_root, repo_relative)
        manifest_entry = manifest_entries.get(repo_label)
        if manifest_entry is None:
            raise ReproducibilityError(
                f"Upstream artifact is outside the manifest closure: {repo_label}."
            )
        observed_hash = sha256_file(source)
        if observed_hash != manifest_entry.sha256:
            raise ReproducibilityError(
                f"Upstream artifact hash mismatch for {repo_label}."
            )
        resolved.append(
            {
                "name": artifact_relative.as_posix(),
                "relative_data_path": repo_label,
                "bytes": manifest_entry.size,
                "sha256": manifest_entry.sha256,
            }
        )
    if not resolved:
        raise ReproducibilityError("At least one upstream artifact is required.")
    return resolved


def _copy_and_record(
    *,
    source: Path,
    destination: Path,
    archive_path: str,
    source_path: str | None,
    role: str,
    expected_hash: str | None = None,
) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    observed_hash = sha256_file(destination)
    if expected_hash is not None and observed_hash != expected_hash:
        raise ReproducibilityError(
            f"Staged bytes changed for {archive_path}: expected {expected_hash}, "
            f"observed {observed_hash}."
        )
    return {
        "archive_path": archive_path,
        "source_path": source_path,
        "role": role,
        "bytes": destination.stat().st_size,
        "sha256": observed_hash,
    }


def _write_generated_and_record(
    *,
    path: Path,
    payload: bytes,
    archive_path: str,
    role: str,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "archive_path": archive_path,
        "source_path": None,
        "role": role,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _replay_text() -> str:
    return """# Replay the Blaschke deformation certificate

`README.md` is the unchanged deployment README from the source commit. This
file records the archive-specific replay sequence.

1. Recreate the recorded environment with `conda create --name blaschke-replay
   --file conda-explicit-lock.txt` and activate it. Install the exact pip-only
   installer with `python -m ensurepip --upgrade`, then install the exact
   pip-only lock with `python -m pip install --no-deps --only-binary=:all:
   --requirement pip-requirements-lock.txt`. Register its interpreter with
   `python -m ipykernel install --user --name blaschke-replay
   --display-name "Python (blaschke-replay)"`.
2. Before running any producer, remove only inventory-declared generated
   evidence and generated notebook counterparts with `python -B
   Numerics/prepare_blaschke_source_only_replay.py --bundle-root .`. This
   command verifies the exact extracted bundle layout, hashes every immutable
   source/input record, rejects inventory drift or traversal, and writes
   `source-only-replay-preparation.json`. For this benchmark there are no
   external numerical input files: map data and exact targets are encoded in
   immutable sources, while every file below `Numerics/outputs` is generated.
3. Rebuild the output-free notebooks with `python -B
   Numerics/build_blaschke_deformation_certifier.py` and `python -B
   Numerics/build_blaschke_deformation_thesis_math_notebook.py`.
4. Rebuild the retained historical Phase 4 diagnostics from source with
   `python -u -B Numerics/blaschke_deformation_historical_phase4.py
   --production --assembly-workers 24 --surface-workers 6 --force`.
5. Execute `BLASCHKE_FORCE_HARDY_MATRIX=1 BLASCHKE_FORCE_CONTOURS=1 python -u -B
   Numerics/execute_notebook_incremental.py
   Numerics/blaschke_deformation_certifier_thesis_math.ipynb
   --kernel-name blaschke-replay`. The rebuilt Hardy-matrix digest invalidates
   the dependent contour cache, so this pass also rebuilds the Schur and
   Laurent contour certificates. The source-only Phase 1 and sampled-Schur
   diagnostic producer cells must also regenerate their reports and every
   mandatory Phase 1 CSV/Parquet pair.
6. Restore the verifier-v3 compatibility alias with `cp
   Numerics/blaschke_deformation_reproducibility_plan.json
   Numerics/outputs/blaschke_deformation_certifier/reports/blaschke_deformation_reproducibility_plan.json`.
   The source-controlled file is the authoritative plan.
7. Compare the forced executed replay with the published archive using
   `python -B Numerics/verify_blaschke_deformation_reproducibility.py ARCHIVE
   --compare-executed-replay-root .`. The verifier requires exact immutable
   source identity and semantic equality of the theorem-facing and retained
   historical outputs; volatile timings, temporary paths and container metadata
   are not treated as mathematical evidence.

The theorem-facing report is
`Numerics/outputs/blaschke_deformation_certifier/reports/blaschke_deformation_24_target_N600_M610_spectral_certificate.json`.
"""


def _add_deterministic_tar_member(
    archive: tarfile.TarFile,
    *,
    source: Path,
    archive_name: str,
    mtime: int,
) -> None:
    info = tarfile.TarInfo(archive_name)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = mtime
    if source.is_dir():
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
        info.size = 0
        archive.addfile(info)
        return
    if source.is_symlink() or not source.is_file():
        raise ReproducibilityError(
            f"Cannot archive non-regular staged path: {source}."
        )
    info.type = tarfile.REGTYPE
    info.mode = 0o755 if source.stat().st_mode & 0o111 else 0o644
    info.size = source.stat().st_size
    with source.open("rb") as stream:
        archive.addfile(info, stream)


def _create_deterministic_archive(
    *, staging_root: Path, archive_path: Path, mtime: int
) -> None:
    with archive_path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
        ) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
            ) as archive:
                _add_deterministic_tar_member(
                    archive,
                    source=staging_root,
                    archive_name=BUNDLE_ROOT,
                    mtime=mtime,
                )
                staged_paths = sorted(
                    staging_root.rglob("*"),
                    key=lambda path: path.relative_to(staging_root).as_posix(),
                )
                for source in staged_paths:
                    relative = source.relative_to(staging_root).as_posix()
                    _add_deterministic_tar_member(
                        archive,
                        source=source,
                        archive_name=f"{BUNDLE_ROOT}/{relative}",
                        mtime=mtime,
                    )
        raw.flush()
        os.fsync(raw.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_atomically(
    publications: Sequence[tuple[Path, Path]],
    *,
    post_publish_check: Callable[[], None],
) -> None:
    """Publish a validated set with atomic replacements and rollback."""

    if not publications:
        raise ReproducibilityError("No reproducibility artifacts were staged.")
    parent = publications[0][1].parent
    if any(target.parent != parent for _, target in publications):
        raise ReproducibilityError("Atomic publication targets must share a directory.")
    backup_dir = publications[0][0].parent / ".publication-backups"
    backup_dir.mkdir()
    backups: dict[Path, Path | None] = {}
    published: list[Path] = []
    try:
        for _, target in publications:
            if target.exists():
                if target.is_symlink() or not target.is_file():
                    raise ReproducibilityError(
                        f"Refusing to replace non-regular release path: {target}."
                    )
                backup = backup_dir / target.name
                try:
                    os.link(target, backup)
                except OSError:
                    shutil.copyfile(target, backup)
                backups[target] = backup
            else:
                backups[target] = None
        for staged, target in publications:
            os.replace(staged, target)
            published.append(target)
        _fsync_directory(parent)
        post_publish_check()
    except Exception:
        for target in reversed(published):
            backup = backups.get(target)
            if backup is None:
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass
            elif backup.exists():
                os.replace(backup, target)
        _fsync_directory(parent)
        raise


def _require_exact_deployment_paths(
    *, repo_root: Path, notebook_path: Path, output_dir: Path
) -> None:
    expected_notebook = repo_root.joinpath(*NOTEBOOK_RELATIVE.parts).resolve()
    expected_output = repo_root.joinpath(*OUTPUT_RELATIVE.parts).resolve()
    try:
        expected_notebook.relative_to(repo_root)
        expected_output.relative_to(repo_root)
    except ValueError as exc:
        raise ReproducibilityError(
            "An authoritative deployment path resolves outside the Git root."
        ) from exc
    if notebook_path != expected_notebook:
        raise ReproducibilityError(
            f"The executed notebook must be {expected_notebook}, not {notebook_path}."
        )
    if output_dir != expected_output:
        raise ReproducibilityError(
            f"The certifier output directory must be {expected_output}, not {output_dir}."
        )


def build_reproducibility_bundle(
    *,
    repo_root: Path,
    notebook_path: Path,
    output_dir: Path,
    precision_settings: Mapping[str, object],
    upstream_artifact_names: Iterable[str],
) -> dict[str, object]:
    """Validate, stage and atomically publish one clean-commit bundle."""

    repo_root = Path(repo_root).resolve()
    notebook_path = Path(notebook_path).resolve()
    output_dir = Path(output_dir).resolve()
    initial_git = _capture_git_snapshot(repo_root)
    _require_exact_deployment_paths(
        repo_root=repo_root,
        notebook_path=notebook_path,
        output_dir=output_dir,
    )

    source_plan_relative = _select_source_plan_relative(repo_root)
    repository_manifest_entries = _validate_source_manifest(repo_root)
    manifest_entries = _archive_manifest_entries(
        repository_manifest_entries,
        source_plan_relative=source_plan_relative,
    )
    manifest_by_path = _manifest_index(manifest_entries)
    notebook_entry = manifest_by_path.get(NOTEBOOK_RELATIVE.as_posix())
    if notebook_entry is None:
        raise ReproducibilityError(
            "The executed notebook is absent from the source manifest closure."
        )
    notebook_validation = validate_executed_notebook(repo_root, notebook_path)

    supplied_names = list(upstream_artifact_names)
    supplied_plan = {
        "precision_settings": dict(precision_settings),
        "upstream_artifact_names": supplied_names,
    }
    plan_path = repo_root.joinpath(*source_plan_relative.parts)
    stored_plan = _load_json_object(plan_path, label="reproducibility plan")
    if source_plan_relative == SOURCE_CONTROLLED_PLAN_RELATIVE:
        _validate_source_controlled_plan_inventory(stored_plan)
    if stored_plan.get("upstream_artifact_names") != supplied_names:
        raise ReproducibilityError(
            "The supplied upstream artifacts do not exactly match the stored "
            f"plan at {source_plan_relative.as_posix()}."
        )
    report_path = repo_root.joinpath(*SPECTRAL_REPORT_RELATIVE.parts)
    report = _load_json_object(report_path, label="spectral certificate report")
    effective_plan = refresh_reproducibility_plan(stored_plan, report)
    effective_precision = effective_plan["precision_settings"]
    if not isinstance(effective_precision, dict):
        raise AssertionError("The refreshed plan precision settings are not a dict.")
    if supplied_plan["precision_settings"] not in (
        stored_plan.get("precision_settings"),
        effective_precision,
    ):
        raise ReproducibilityError(
            "The supplied precision settings match neither the stored source "
            "plan nor the report-refreshed effective plan."
        )
    certificate_validation = validate_plan_and_spectral_report(
        effective_plan, report
    )
    upstream_names = effective_plan["upstream_artifact_names"]
    if not isinstance(upstream_names, list):
        raise AssertionError("The refreshed plan artifact names are not a list.")
    upstream_artifacts = _validate_upstream_artifacts(
        repo_root=repo_root,
        names=upstream_names,
        manifest_entries=manifest_by_path,
    )
    diagnostic_rebuild_validation = validate_source_generated_diagnostics(
        repo_root,
        manifest_by_path,
    )

    selected_versions = _package_versions()
    pip_requirements = _validate_pip_requirements(repo_root, selected_versions)
    conda_lock = _conda_explicit_lock(repo_root)
    repository_manifest_path = repo_root / SOURCE_MANIFEST_NAME
    repository_manifest_hash = sha256_file(repository_manifest_path)
    source_manifest_payload = _source_manifest_bytes(manifest_entries)
    source_manifest_hash = hashlib.sha256(source_manifest_payload).hexdigest()

    reproducibility_dir = output_dir / "reproducibility"
    if reproducibility_dir.is_symlink() or (
        reproducibility_dir.exists() and not reproducibility_dir.is_dir()
    ):
        raise ReproducibilityError(
            f"Reproducibility output path is not a real directory: {reproducibility_dir}."
        )
    reproducibility_dir.mkdir(parents=True, exist_ok=True)
    try:
        reproducibility_dir.resolve(strict=True).relative_to(repo_root)
    except (OSError, ValueError) as exc:
        raise ReproducibilityError(
            "The reproducibility output directory escapes the Git root."
        ) from exc
    archive_path = reproducibility_dir / ARCHIVE_NAME
    checksum_path = reproducibility_dir / CHECKSUM_NAME
    external_manifest_path = reproducibility_dir / EXTERNAL_MANIFEST_NAME

    with tempfile.TemporaryDirectory(
        prefix="blaschke-certifier-stage-"
    ) as stage_directory, tempfile.TemporaryDirectory(
        prefix=".publish-", dir=reproducibility_dir
    ) as publish_directory:
        staging_root = Path(stage_directory) / BUNDLE_ROOT
        staging_root.mkdir()
        file_records: list[dict[str, object]] = []

        for entry in manifest_entries:
            relative_label = entry.relative_path.as_posix()
            source_relative = entry.relative_path
            if (
                source_plan_relative != PLAN_RELATIVE
                and entry.relative_path == PLAN_RELATIVE
            ):
                source_relative = source_plan_relative
            source = repo_root.joinpath(*source_relative.parts)
            destination = staging_root.joinpath(*entry.relative_path.parts)
            if entry.relative_path == NOTEBOOK_RELATIVE:
                role = "executed notebook"
            elif source_relative == SOURCE_CONTROLLED_PLAN_RELATIVE:
                role = (
                    "source-controlled reproducibility plan"
                    if entry.relative_path == source_relative
                    else "verifier-v3 compatibility alias of source-controlled plan"
                )
            else:
                role = "authoritative manifest-closure file"
            file_records.append(
                _copy_and_record(
                    source=source,
                    destination=destination,
                    archive_path=relative_label,
                    source_path=source_relative.as_posix(),
                    role=role,
                    expected_hash=entry.sha256,
                )
            )

        file_records.append(
            _write_generated_and_record(
                path=staging_root / SOURCE_MANIFEST_NAME,
                payload=source_manifest_payload,
                archive_path=SOURCE_MANIFEST_NAME,
                role=(
                    "archive source checksum manifest with source-controlled "
                    "plan compatibility alias"
                ),
            )
        )
        file_records.append(
            _write_generated_and_record(
                path=staging_root / REPLAY_NAME,
                payload=_replay_text().encode("utf-8"),
                archive_path=REPLAY_NAME,
                role="generated replay instructions",
            )
        )
        lock_payload = conda_lock.encode("utf-8")
        file_records.append(
            _write_generated_and_record(
                path=staging_root / CONDA_LOCK_NAME,
                payload=lock_payload,
                archive_path=CONDA_LOCK_NAME,
                role="full Conda explicit environment lock",
            )
        )
        selected_versions_payload = _json_bytes(
            {"schema_version": 1, "packages": selected_versions}
        )
        file_records.append(
            _write_generated_and_record(
                path=staging_root / SELECTED_VERSIONS_NAME,
                payload=selected_versions_payload,
                archive_path=SELECTED_VERSIONS_NAME,
                role="selected Python package versions",
            )
        )
        effective_plan_payload = _json_bytes(effective_plan)
        file_records.append(
            _write_generated_and_record(
                path=staging_root / EFFECTIVE_PLAN_NAME,
                payload=effective_plan_payload,
                archive_path=EFFECTIVE_PLAN_NAME,
                role="report-refreshed effective reproducibility plan",
            )
        )
        source_only_inventory = make_archive_inventory(
            staging_root=staging_root,
            file_records=file_records,
        )
        source_only_inventory_payload = _json_bytes(source_only_inventory)
        file_records.append(
            _write_generated_and_record(
                path=staging_root / SOURCE_ONLY_INVENTORY_NAME,
                payload=source_only_inventory_payload,
                archive_path=SOURCE_ONLY_INVENTORY_NAME,
                role=(
                    "archive-specific generated-evidence versus immutable-source "
                    "inventory for fail-closed source-only replay preparation"
                ),
            )
        )
        file_records.sort(key=lambda record: str(record["archive_path"]))

        for builder_path in (
            TEMPLATE_NOTEBOOK_RELATIVE,
            SOURCE_NOTEBOOK_RELATIVE,
            NOTEBOOK_RELATIVE,
        ):
            if builder_path.as_posix() not in manifest_by_path:
                raise ReproducibilityError(
                    f"Builder-chain path is absent from the manifest: {builder_path}."
                )

        bundle_manifest: dict[str, object] = {
            "schema_version": 3,
            "bundle_format": "blaschke-deformation-certifier-reproducibility-v3",
            "map_label": EXPECTED_MAP_LABEL,
            "repository": {
                "commit": initial_git.head,
                "commit_time": initial_git.commit_time,
                "commit_epoch": initial_git.commit_epoch,
                "dirty": False,
                "status_porcelain": "",
                "root_name": repo_root.name,
            },
            "source_manifest": {
                "archive_path": SOURCE_MANIFEST_NAME,
                "sha256": source_manifest_hash,
                "entry_count": len(manifest_entries),
                "closure_policy": (
                    "clean committed authoritative deployment files with the "
                    "verifier-v3 plan path aliased to the source-controlled plan"
                ),
                "repository_manifest_sha256": repository_manifest_hash,
                "repository_manifest_entry_count": len(
                    repository_manifest_entries
                ),
                "plan_compatibility_alias_path": PLAN_RELATIVE.as_posix(),
            },
            "builder_chain": {
                "locked_template_path": TEMPLATE_NOTEBOOK_RELATIVE.as_posix(),
                "locked_template_sha256": manifest_by_path[
                    TEMPLATE_NOTEBOOK_RELATIVE.as_posix()
                ].sha256,
                "source_notebook_path": SOURCE_NOTEBOOK_RELATIVE.as_posix(),
                "source_notebook_sha256": manifest_by_path[
                    SOURCE_NOTEBOOK_RELATIVE.as_posix()
                ].sha256,
                "executed_counterpart_path": NOTEBOOK_RELATIVE.as_posix(),
                "executed_counterpart_sha256": notebook_entry.sha256,
            },
            "notebook_validation": notebook_validation,
            "certificate_validation": certificate_validation,
            "diagnostic_rebuild_validation": diagnostic_rebuild_validation,
            "source_plan_path": PLAN_RELATIVE.as_posix(),
            "source_controlled_plan_path": source_plan_relative.as_posix(),
            "effective_plan_path": EFFECTIVE_PLAN_NAME,
            "plan_refresh": {
                "reason": (
                    "curated computational body excludes source Cell 104; refresh from "
                    "the current theorem-facing spectral report"
                ),
                "source_plan_sha256": manifest_by_path[
                    source_plan_relative.as_posix()
                ].sha256,
                "effective_plan_sha256": hashlib.sha256(
                    effective_plan_payload
                ).hexdigest(),
            },
            "spectral_report_path": SPECTRAL_REPORT_RELATIVE.as_posix(),
            "precision_settings": effective_precision,
            "upstream_artifacts": upstream_artifacts,
            "environment": {
                "python_version": sys.version,
                "python_implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "selected_package_versions_path": SELECTED_VERSIONS_NAME,
                "selected_package_versions": selected_versions,
                "pip_requirements_path": PIP_LOCK_NAME,
                "pip_requirements": pip_requirements,
                "pip_requirements_sha256": sha256_file(
                    repo_root / PIP_LOCK_NAME
                ),
                "conda_explicit_lock_path": CONDA_LOCK_NAME,
                "conda_explicit_lock_sha256": hashlib.sha256(
                    lock_payload
                ).hexdigest(),
            },
            "execution_commands": (
                "python -B Numerics/prepare_blaschke_source_only_replay.py "
                "--bundle-root .",
                "python -u -B "
                "Numerics/blaschke_deformation_historical_phase4.py "
                "--production --assembly-workers 24 --surface-workers 6 --force",
                "BLASCHKE_FORCE_HARDY_MATRIX=1 BLASCHKE_FORCE_CONTOURS=1 "
                "python -u -B "
                "Numerics/execute_notebook_incremental.py "
                "Numerics/blaschke_deformation_certifier_thesis_math.ipynb "
                "--kernel-name blaschke-replay",
            ),
            "source_only_replay": {
                "inventory_path": SOURCE_ONLY_INVENTORY_NAME,
                "inventory_sha256": hashlib.sha256(
                    source_only_inventory_payload
                ).hexdigest(),
                "preparation_script": (
                    "Numerics/prepare_blaschke_source_only_replay.py"
                ),
                "orchestrator": "Numerics/run_blaschke_clean_room_replay.py",
                "required_for_current_release_acceptance": True,
                "immutable_external_input_count": source_only_inventory["counts"][
                    "immutable_external_input"
                ],
            },
            "replay_instructions_path": REPLAY_NAME,
            "files": file_records,
            "packaged_regular_file_count": len(file_records) + 1,
        }
        internal_manifest_payload = _json_bytes(bundle_manifest)
        (staging_root / INTERNAL_MANIFEST_NAME).write_bytes(
            internal_manifest_payload
        )

        publish_root = Path(publish_directory)
        staged_archive = publish_root / ARCHIVE_NAME
        _create_deterministic_archive(
            staging_root=staging_root,
            archive_path=staged_archive,
            mtime=initial_git.commit_epoch,
        )
        archive_hash = sha256_file(staged_archive)
        staged_checksum = publish_root / CHECKSUM_NAME
        staged_checksum.write_text(
            f"{archive_hash}  {ARCHIVE_NAME}\n", encoding="utf-8", newline=""
        )
        with staged_checksum.open("rb") as stream:
            os.fsync(stream.fileno())

        external_manifest = {
            "schema_version": 3,
            "publication_complete": True,
            "archive": {
                "name": ARCHIVE_NAME,
                "sha256": archive_hash,
                "bytes": staged_archive.stat().st_size,
                "checksum_name": CHECKSUM_NAME,
            },
            "bundle_manifest_sha256": hashlib.sha256(
                internal_manifest_payload
            ).hexdigest(),
            "bundle": bundle_manifest,
        }
        staged_external_manifest = publish_root / EXTERNAL_MANIFEST_NAME
        staged_external_manifest.write_bytes(_json_bytes(external_manifest))
        with staged_external_manifest.open("rb") as stream:
            os.fsync(stream.fileno())

        verifier_path = repo_root / "Numerics" / (
            "verify_blaschke_deformation_reproducibility.py"
        )
        _absolute_repo_file(
            repo_root,
            PurePosixPath(
                "Numerics/verify_blaschke_deformation_reproducibility.py"
            ),
        )
        _run_checked(
            (
                sys.executable,
                "-B",
                str(verifier_path),
                str(staged_archive),
                "--external-manifest",
                str(staged_external_manifest),
                "--checksum",
                str(staged_checksum),
            ),
            cwd=repo_root,
            text=True,
        )
        _assert_same_clean_snapshot(repo_root, initial_git)
        _publish_atomically(
            (
                (staged_archive, archive_path),
                (staged_checksum, checksum_path),
                (staged_external_manifest, external_manifest_path),
            ),
            post_publish_check=lambda: _assert_same_clean_snapshot(
                repo_root, initial_git
            ),
        )

    return {
        "archive_path": str(archive_path),
        "manifest_path": str(external_manifest_path),
        "checksum_path": str(checksum_path),
        "archive_sha256": archive_hash,
        "archive_bytes": archive_path.stat().st_size,
        "packaged_file_count": len(file_records) + 1,
        "source_manifest_entry_count": len(manifest_entries),
        "repository_manifest_entry_count": len(repository_manifest_entries),
        "source_controlled_plan_path": source_plan_relative.as_posix(),
        "repository_commit": initial_git.head,
        "repository_dirty": False,
        "execution_status": "clean_commit_bundle_created",
    }


def main() -> None:
    root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root_default)
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan", type=Path)
    arguments = parser.parse_args()

    repo_root = arguments.repo_root.resolve()
    notebook_path = (
        arguments.notebook.resolve()
        if arguments.notebook is not None
        else repo_root.joinpath(*NOTEBOOK_RELATIVE.parts)
    )
    output_dir = (
        arguments.output_dir.resolve()
        if arguments.output_dir is not None
        else repo_root.joinpath(*OUTPUT_RELATIVE.parts)
    )
    plan_path = (
        arguments.plan.resolve()
        if arguments.plan is not None
        else repo_root.joinpath(*SOURCE_CONTROLLED_PLAN_RELATIVE.parts)
    )
    expected_plan_path = repo_root.joinpath(
        *SOURCE_CONTROLLED_PLAN_RELATIVE.parts
    ).resolve()
    if plan_path != expected_plan_path:
        raise ReproducibilityError(
            f"The plan must be the authoritative path {expected_plan_path}."
        )
    plan = _load_json_object(plan_path, label="reproducibility plan")
    if set(plan) != {"precision_settings", "upstream_artifact_names"}:
        raise ReproducibilityError(
            "The reproducibility plan has an unexpected top-level schema."
        )
    precision_settings = plan["precision_settings"]
    upstream_names = plan["upstream_artifact_names"]
    if not isinstance(precision_settings, dict) or not isinstance(
        upstream_names, list
    ):
        raise ReproducibilityError(
            "The reproducibility plan fields have invalid JSON types."
        )
    result = build_reproducibility_bundle(
        repo_root=repo_root,
        notebook_path=notebook_path,
        output_dir=output_dir,
        precision_settings=precision_settings,
        upstream_artifact_names=upstream_names,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
