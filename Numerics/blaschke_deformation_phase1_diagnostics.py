"""Source-regenerate the retained Phase 1 diagnostic datasets.

The routines in this module rebuild the empirical Legendre--Gauss sweeps used
by the thesis notebook.  They never accept their destination files as inputs:
every output is derived from the selected map specification, the raw mpmath
transfer-block worker and the explicit schedules below.

These products are diagnostic only.  They do not enter the theorem-facing
single-space perturbation or contour-certification gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
import os
import platform
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPORT_FILENAME = "blaschke_deformation_phase1_diagnostics_rebuild.json"
LOCKED_MAP_LABEL = "blaschke_mu_0p3"
PRODUCER_SCHEMA = "blaschke-deformation-phase1-diagnostics-v1"


@dataclass(frozen=True)
class Phase1DiagnosticsConfig:
    """Lock the empirical schedules and high-precision assembly controls."""

    cloud_pairs: tuple[tuple[int, int], ...] = (
        (12, 12),
        (12, 24),
        (20, 20),
        (20, 30),
    )
    fixed_n: int = 25
    fixed_m_values: tuple[int, ...] = tuple(range(10, 40))
    heat_n_values: tuple[int, ...] = (6, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30)
    heat_m_values: tuple[int, ...] = (
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        25,
        26,
        28,
        30,
        32,
        36,
        40,
        48,
    )
    dps: int = 180
    max_power: int = 25
    max_clusters: int = 24
    expected_target_count: int = 24
    assembly_workers: int = 24
    row_block_size: int | None = None
    progress: bool = True


@dataclass(frozen=True)
class Phase1DiagnosticsResult:
    """Return generated frames and their persisted source records."""

    cloud_match_frames: Mapping[tuple[int, int], pd.DataFrame]
    cloud_eigenvalue_frames: Mapping[tuple[int, int], pd.DataFrame]
    fixed_m_frame: pd.DataFrame
    heatmap_frame: pd.DataFrame
    csv_paths: Mapping[str, Path]
    parquet_paths: Mapping[str, Path]
    report_path: Path
    report_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _callable_source_record(function: Callable[..., Any]) -> dict[str, Any]:
    module = sys.modules.get(getattr(function, "__module__", ""))
    inline_sha256 = getattr(module, "__source_sha256__", None)
    if inline_sha256:
        return {
            "module": getattr(function, "__module__", None),
            "qualname": getattr(function, "__qualname__", None),
            "module_path": f"inline:{getattr(function, '__module__', 'module')}",
            "source_scope": "complete inline module source",
            "source_sha256": str(inline_sha256),
            "worker_api_version": getattr(module, "WORKER_API_VERSION", None),
        }
    module_path_value = getattr(module, "__file__", None)
    module_path = Path(module_path_value).resolve() if module_path_value else None
    if module_path is not None and module_path.is_file():
        source_sha256 = _sha256(module_path)
        source_path = f"Numerics/{module_path.name}"
        source_scope = "complete module file"
    else:
        source = inspect.getsource(function).encode("utf-8")
        source_sha256 = _sha256_bytes(source)
        source_path = None
        source_scope = "callable source"
    return {
        "module": getattr(function, "__module__", None),
        "qualname": getattr(function, "__qualname__", None),
        "module_path": source_path,
        "source_scope": source_scope,
        "source_sha256": source_sha256,
        "worker_api_version": getattr(module, "WORKER_API_VERSION", None),
    }


def _runtime_record() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    _atomic_write_text(path, frame.to_csv(index=False))


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp.parquet",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
        return True
    except (ImportError, ModuleNotFoundError, ValueError):
        temporary.unlink(missing_ok=True)
        return False
    finally:
        temporary.unlink(missing_ok=True)


def _error_to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan


def _with_oversampling_columns(frame: pd.DataFrame) -> pd.DataFrame:
    '''Explanation: For fixed truncation N, increasing the quadrature order M should remove integration error while leaving truncation error. Comparing with the largest M reveals whether the displayed Phase 1 spectrum has reached that stable regime.
Functionality: Add finite-M excess and normalised error columns relative to the largest sampled quadrature order at each N.'''
    result = frame.copy()
    result["N"] = result["N"].astype(int)
    result["M"] = result["M"].astype(int)
    result["m"] = result["M"] - result["N"]
    result["is_reference_M"] = False
    result["error_float"] = result["error"].map(_error_to_float)

    reference_lookup: dict[tuple[int, str], float] = {}
    for (n_value, name), group in result.groupby(["N", "name"], sort=False):
        finite = group[
            np.isfinite(group["error_float"].astype(float))
            & (group["error_float"].astype(float) > 0)
        ]
        if not finite.empty:
            reference = finite.loc[finite["M"].idxmax()]
            reference_lookup[(int(n_value), str(name))] = float(
                reference["error_float"]
            )

    references: list[float] = []
    ratios: list[float] = []
    logarithms: list[float] = []
    for row in result.itertuples(index=False):
        reference = reference_lookup.get((int(row.N), str(row.name)), math.nan)
        error = _error_to_float(row.error)
        references.append(reference)
        if math.isfinite(error) and math.isfinite(reference) and error > 0 and reference > 0:
            ratio = error / reference
            ratios.append(ratio)
            logarithms.append(math.log10(ratio))
        else:
            ratios.append(math.nan)
            logarithms.append(math.nan)

    result["reference_error_at_max_M"] = references
    result["excess_ratio_to_max_M"] = ratios
    result["log10_excess_ratio_to_max_M"] = logarithms
    return result


def _normalise_pairs(pairs: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    normalised = tuple(sorted({(int(n_value), int(m_value)) for n_value, m_value in pairs}))
    if not normalised or any(n_value < 1 or m_value < 1 for n_value, m_value in normalised):
        raise ValueError("Phase 1 schedules require positive N and M values.")
    return normalised


def _expected_target_names(
    *,
    raw_sweep: Callable[..., Any],
    map_spec: Mapping[str, Any],
    reference_clusters: Sequence[Mapping[str, Any]] | None,
    max_power: int,
    max_clusters: int,
) -> tuple[str, ...]:
    if reference_clusters is not None:
        clusters = list(reference_clusters)
    else:
        module = sys.modules.get(getattr(raw_sweep, "__module__", ""))
        exact_clusters = getattr(module, "exact_clusters_for_map", None)
        if exact_clusters is None:
            raise RuntimeError(
                "The Phase 1 producer cannot reconstruct the exact target packet "
                "from the supplied raw-sweep module."
            )
        clusters = list(
            exact_clusters(
                dict(map_spec),
                max_power=int(max_power),
                include_trivial=False,
                full_multiplicity=True,
            )
        )
    names = tuple(str(cluster["name"]) for cluster in clusters[: int(max_clusters)])
    if len(names) != len(set(names)):
        raise RuntimeError("The source target packet contains duplicate names.")
    return names


def _validate_raw_frames(
    rows: pd.DataFrame,
    eigenvalues: pd.DataFrame,
    *,
    expected_pairs: Sequence[tuple[int, int]],
    expected_target_names: Sequence[str],
    expected_map_label: str,
) -> None:
    required_rows = {"N", "M", "name", "error", "selected"}
    required_eigenvalues = {"N", "M", "eig"}
    missing_rows = required_rows.difference(rows.columns)
    missing_eigenvalues = required_eigenvalues.difference(eigenvalues.columns)
    if missing_rows:
        raise ValueError(f"Raw Phase 1 rows lack columns: {sorted(missing_rows)}")
    if missing_eigenvalues:
        raise ValueError(
            f"Raw Phase 1 eigenvalues lack columns: {sorted(missing_eigenvalues)}"
        )
    if rows.empty or eigenvalues.empty:
        raise ValueError("The raw Phase 1 sweep returned an empty table.")

    expected_pair_set = set(expected_pairs)
    row_pair_set = {
        (int(n_value), int(m_value))
        for n_value, m_value in rows[["N", "M"]].drop_duplicates().itertuples(index=False)
    }
    eigenvalue_pair_set = {
        (int(n_value), int(m_value))
        for n_value, m_value in eigenvalues[["N", "M"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    if row_pair_set != expected_pair_set:
        raise RuntimeError(
            "The Phase 1 target rows do not match the locked pair schedule: "
            f"missing={sorted(expected_pair_set - row_pair_set)}, "
            f"unexpected={sorted(row_pair_set - expected_pair_set)}."
        )
    if eigenvalue_pair_set != expected_pair_set:
        raise RuntimeError(
            "The Phase 1 eigenvalue rows do not match the locked pair schedule: "
            f"missing={sorted(expected_pair_set - eigenvalue_pair_set)}, "
            f"unexpected={sorted(eigenvalue_pair_set - expected_pair_set)}."
        )

    target_names = tuple(map(str, expected_target_names))
    target_name_set = set(target_names)
    for n_value, m_value in expected_pairs:
        pair_rows = rows[
            (rows["N"].astype(int) == int(n_value))
            & (rows["M"].astype(int) == int(m_value))
        ]
        pair_eigenvalues = eigenvalues[
            (eigenvalues["N"].astype(int) == int(n_value))
            & (eigenvalues["M"].astype(int) == int(m_value))
        ]
        observed_names = tuple(map(str, pair_rows["name"]))
        if len(pair_rows) != len(target_names) or set(observed_names) != target_name_set:
            raise RuntimeError(
                f"The Phase 1 target packet is incomplete for N={n_value}, M={m_value}: "
                f"expected {len(target_names)} unique targets, received "
                f"{len(pair_rows)} rows and {pair_rows['name'].nunique()} unique names."
            )
        if pair_rows["name"].duplicated().any():
            raise RuntimeError(
                f"The Phase 1 target packet contains duplicate names for N={n_value}, M={m_value}."
            )
        if len(pair_eigenvalues) != int(n_value):
            raise RuntimeError(
                f"The Phase 1 eigenvalue cloud for N={n_value}, M={m_value} "
                f"contains {len(pair_eigenvalues)} rows instead of exactly N={n_value}."
            )
        if "eig_index" in pair_eigenvalues and pair_eigenvalues["eig_index"].nunique() != int(n_value):
            raise RuntimeError(
                f"The Phase 1 eigenvalue indices are incomplete for N={n_value}, M={m_value}."
            )

    for frame_name, frame in (("target", rows), ("eigenvalue", eigenvalues)):
        if "map_name" not in frame:
            raise ValueError(f"Raw Phase 1 {frame_name} rows lack the map_name column.")
        observed_labels = set(map(str, frame["map_name"].dropna().unique()))
        if observed_labels != {expected_map_label}:
            raise RuntimeError(
                f"Raw Phase 1 {frame_name} rows violate the map lock: {sorted(observed_labels)}."
            )


def rebuild_phase1_diagnostics(
    config: Phase1DiagnosticsConfig,
    *,
    map_spec: Mapping[str, Any],
    raw_sweep: Callable[..., tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]],
    reference_clusters: Sequence[Mapping[str, Any]] | None,
    data_dir: Path,
    report_path: Path | None = None,
) -> Phase1DiagnosticsResult:
    '''Explanation: Phase 1 asks whether high-precision Legendre--Gauss matrices converge towards the formula spectral packets. Rebuilding every matrix and cluster match from the map makes that convergence study reproducible, while keeping it separate from the later theorem certificate.
Functionality: Rebuild every retained Phase 1 diagnostic input from source mathematics.'''

    if int(config.dps) < 50:
        raise ValueError("Phase 1 diagnostic assembly requires at least 50 decimal digits.")
    if not 1 <= int(config.assembly_workers) <= 24:
        raise ValueError("Phase 1 process row-block assembly requires 1 to 24 workers.")
    if int(config.max_clusters) < 1 or int(config.max_power) < 1:
        raise ValueError("Phase 1 target limits must be positive.")

    map_payload = dict(map_spec)
    map_label = str(map_payload.get("params", {}).get("label", "")).strip()
    if map_label != LOCKED_MAP_LABEL:
        raise ValueError(
            f"This deployment producer is locked to {LOCKED_MAP_LABEL}, not {map_label or 'an unlabeled map'}."
        )

    cloud_pairs = _normalise_pairs(config.cloud_pairs)
    fixed_pairs = _normalise_pairs(
        (int(config.fixed_n), int(m_value)) for m_value in config.fixed_m_values
    )
    heat_pairs = _normalise_pairs(
        (int(n_value), int(m_value))
        for n_value in config.heat_n_values
        for m_value in config.heat_m_values
        if int(m_value) >= int(n_value)
    )
    all_pairs = _normalise_pairs((*cloud_pairs, *fixed_pairs, *heat_pairs))
    expected_target_names = _expected_target_names(
        raw_sweep=raw_sweep,
        map_spec=map_payload,
        reference_clusters=reference_clusters,
        max_power=int(config.max_power),
        max_clusters=int(config.max_clusters),
    )
    if len(expected_target_names) != int(config.expected_target_count):
        raise RuntimeError(
            "The locked Phase 1 target packet has drifted: "
            f"expected {config.expected_target_count}, found {len(expected_target_names)}."
        )

    rows, result = raw_sweep(
        all_pairs,
        map_spec=map_payload,
        dps=int(config.dps),
        # Pair-level parallelism stays disabled because each transfer block is
        # assembled by the configured process-based row-block worker pool.
        workers=1,
        max_power=int(config.max_power),
        max_clusters=int(config.max_clusters),
        include_trivial=False,
        full_multiplicity=True,
        progress=bool(config.progress),
        reference_clusters=reference_clusters,
        assembly_workers=int(config.assembly_workers),
        row_block_size=config.row_block_size,
    )
    raw_rows = pd.DataFrame(rows)
    raw_eigenvalues = pd.DataFrame(result.get("eig_rows", ()))
    _validate_raw_frames(
        raw_rows,
        raw_eigenvalues,
        expected_pairs=all_pairs,
        expected_target_names=expected_target_names,
        expected_map_label=map_label,
    )

    jobs = tuple(result.get("jobs", ()))
    if len(jobs) != len(all_pairs):
        raise RuntimeError(
            f"The raw Phase 1 sweep reported {len(jobs)} jobs for {len(all_pairs)} locked pairs."
        )
    job_pairs = {(int(job["N"]), int(job["M"])) for job in jobs}
    if job_pairs != set(all_pairs):
        raise RuntimeError("The raw Phase 1 job metadata do not match the locked pair schedule.")
    expected_assembly_mode = (
        "process_row_blocks" if int(config.assembly_workers) > 1 else "serial"
    )
    for job in jobs:
        if not bool(job.get("ok", True)):
            raise RuntimeError("The raw Phase 1 metadata contain a failed job.")
        if int(job.get("assembly_workers", 0)) != int(config.assembly_workers):
            raise RuntimeError("The raw Phase 1 worker count differs from the locked configuration.")
        if str(job.get("assembly_mode")) != expected_assembly_mode:
            raise RuntimeError("The raw Phase 1 assembly mode differs from the locked configuration.")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path or data_dir.parent / "reports" / REPORT_FILENAME)
    csv_paths: dict[str, Path] = {}
    parquet_paths: dict[str, Path] = {}
    output_frames: dict[str, pd.DataFrame] = {}
    cloud_match_frames: dict[tuple[int, int], pd.DataFrame] = {}
    cloud_eigenvalue_frames: dict[tuple[int, int], pd.DataFrame] = {}

    def register(name: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            raise RuntimeError(f"Refusing to persist an empty Phase 1 dataset: {name}.")
        output_frames[name] = frame.copy()

    for n_value, m_value in cloud_pairs:
        match_frame = raw_rows[
            (raw_rows["N"].astype(int) == n_value)
            & (raw_rows["M"].astype(int) == m_value)
        ].copy()
        eigenvalue_frame = raw_eigenvalues[
            (raw_eigenvalues["N"].astype(int) == n_value)
            & (raw_eigenvalues["M"].astype(int) == m_value)
        ].copy()
        if match_frame.empty or eigenvalue_frame.empty:
            raise RuntimeError(
                f"The Phase 1 cloud producer returned no data for N={n_value}, M={m_value}."
            )
        cloud_match_frames[(n_value, m_value)] = match_frame
        cloud_eigenvalue_frames[(n_value, m_value)] = eigenvalue_frame
        register(f"phase1_eigencloud_match_N{n_value}_M{m_value}", match_frame)
        register(
            f"phase1_eigencloud_eigenvalues_N{n_value}_M{m_value}",
            eigenvalue_frame,
        )

    row_pairs = pd.MultiIndex.from_arrays(
        (raw_rows["N"].astype(int), raw_rows["M"].astype(int))
    )
    fixed_mask = row_pairs.isin(pd.MultiIndex.from_tuples(fixed_pairs))
    heat_mask = row_pairs.isin(pd.MultiIndex.from_tuples(heat_pairs))
    fixed_m_frame = _with_oversampling_columns(raw_rows.loc[fixed_mask].copy())
    heatmap_frame = _with_oversampling_columns(raw_rows.loc[heat_mask].copy())
    register(f"phase1_raw_error_vs_M_N{int(config.fixed_n)}", fixed_m_frame)
    register("phase1_raw_NM_heatmap_data_near_square", heatmap_frame)

    source_record = _callable_source_record(raw_sweep)
    lineage = {
        "map_spec_sha256": _canonical_sha256(map_payload),
        "reference_clusters_sha256": (
            _canonical_sha256(reference_clusters)
            if reference_clusters is not None
            else None
        ),
        "expected_target_names_sha256": _canonical_sha256(expected_target_names),
        "raw_sweep": source_record,
        "runtime": _runtime_record(),
    }
    generation_id = _canonical_sha256(
        {
            "config": asdict(config),
            "lineage": lineage,
            "pairs": all_pairs,
        }
    )

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".phase1-diagnostics-generation.",
        dir=data_dir.parent,
    ) as temporary_directory:
        staging = Path(temporary_directory)
        staged_csv: dict[str, Path] = {}
        staged_parquet: dict[str, Path] = {}
        for name, frame in output_frames.items():
            csv_path = staging / f"{name}.csv"
            frame.to_csv(csv_path, index=False)
            staged_csv[name] = csv_path
            parquet_path = staging / f"{name}.parquet"
            try:
                frame.to_parquet(parquet_path, index=False)
            except (ImportError, ModuleNotFoundError, ValueError) as exc:
                raise RuntimeError(
                    "The pinned deployment requires a Parquet engine for every "
                    f"Phase 1 dataset; failed while writing {name}."
                ) from exc
            staged_parquet[name] = parquet_path

        for name, staged_path in staged_csv.items():
            destination = data_dir / staged_path.name
            os.replace(staged_path, destination)
            csv_paths[name] = destination
        for name in output_frames:
            destination = data_dir / f"{name}.parquet"
            os.replace(staged_parquet[name], destination)
            parquet_paths[name] = destination

    report = {
        "producer_schema": PRODUCER_SCHEMA,
        "schema_version": "2.0.0",
        "generation_id": generation_id,
        "map_label": map_label,
        "diagnostic_only": True,
        "theorem_gate": False,
        "legacy_seed_dependency": False,
        "commit_protocol": (
            "all datasets are staged first; canonical payloads are replaced next; "
            "this report is written last and is the generation commit marker"
        ),
        "config": asdict(config),
        "lineage": lineage,
        "jobs": [
            {
                key: job.get(key)
                for key in (
                    "N",
                    "M",
                    "ok",
                    "assembly_mode",
                    "assembly_workers",
                    "row_block_size",
                    "gram_error",
                )
            }
            for job in jobs
        ],
        "schedules": {
            "cloud_pairs": [list(pair) for pair in cloud_pairs],
            "fixed_pairs": [list(pair) for pair in fixed_pairs],
            "heat_pairs": [list(pair) for pair in heat_pairs],
            "unique_pairs": [list(pair) for pair in all_pairs],
        },
        "outputs": {
            name: {
                "path": str(path),
                "rows": int(output_frames[name].shape[0]),
                "columns": list(map(str, output_frames[name].columns)),
                "sha256": _sha256(path),
                "parquet_path": (
                    str(parquet_paths[name]) if name in parquet_paths else None
                ),
                "parquet_sha256": (
                    _sha256(parquet_paths[name]) if name in parquet_paths else None
                ),
            }
            for name, path in csv_paths.items()
        },
    }
    _atomic_write_text(
        report_path,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    report_sha256 = _sha256(report_path)
    return Phase1DiagnosticsResult(
        cloud_match_frames=cloud_match_frames,
        cloud_eigenvalue_frames=cloud_eigenvalue_frames,
        fixed_m_frame=fixed_m_frame,
        heatmap_frame=heatmap_frame,
        csv_paths=csv_paths,
        parquet_paths=parquet_paths,
        report_path=report_path,
        report_sha256=report_sha256,
    )


__all__ = [
    "LOCKED_MAP_LABEL",
    "PRODUCER_SCHEMA",
    "REPORT_FILENAME",
    "Phase1DiagnosticsConfig",
    "Phase1DiagnosticsResult",
    "rebuild_phase1_diagnostics",
]
