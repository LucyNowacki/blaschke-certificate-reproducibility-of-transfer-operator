"""Source-regenerate the retained sampled Schur diagnostic rows.

This module rebuilds the six map-specific diagnostic rows consumed by the
universal certification ladder.  The rows are not theorem-facing: the
underlying Schur--Frobenius expression uses sampled geometry and fixes the
auxiliary finite-order factor ``D_M`` to one.  The destination CSV is never
accepted as an input or cache.
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
from typing import Any, Callable, Iterable, Mapping

import pandas as pd


CSV_FILENAME_TEMPLATE = "transfer_lab_{map_label}_generic_sampled_schur_envelope.csv"
REPORT_FILENAME = "blaschke_deformation_sampled_schur_diagnostics_rebuild.json"
LOCKED_MAP_LABEL = "blaschke_mu_0p3"
PRODUCER_SCHEMA = "blaschke-deformation-sampled-schur-diagnostics-v1"


@dataclass(frozen=True)
class SampledSchurDiagnosticsConfig:
    """Lock the retained diagnostic dimensions and oversampling offset."""

    n_values: tuple[int, ...] = (30, 40, 50, 60, 80, 100)
    oversampling: int = 6


@dataclass(frozen=True)
class SampledSchurDiagnosticsResult:
    """Return the rebuilt diagnostic table and persisted source record."""

    frame: pd.DataFrame
    csv_path: Path
    report_path: Path
    csv_sha256: str
    report_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
        }
    module_path_value = getattr(module, "__file__", None)
    module_path = Path(module_path_value).resolve() if module_path_value else None
    if module_path is not None and module_path.is_file():
        source_sha256 = _sha256(module_path)
        source_path = f"Numerics/{module_path.name}"
        source_scope = "complete module file"
    else:
        try:
            source = inspect.getsource(function)
            source_scope = "callable source"
        except (OSError, TypeError):
            source = (
                f"{getattr(function, '__module__', '')}:"
                f"{getattr(function, '__qualname__', '')}"
            )
            source_scope = "stable callable identity"
        source_sha256 = _sha256_bytes(source.encode("utf-8"))
        source_path = None
    return {
        "module": getattr(function, "__module__", None),
        "qualname": getattr(function, "__qualname__", None),
        "module_path": source_path,
        "source_scope": source_scope,
        "source_sha256": source_sha256,
    }


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
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


def _normalise_geometry(geometry: Mapping[str, Any]) -> dict[str, float]:
    aliases = {
        "rho": ("rho",),
        "r_tau": ("r_tau", "r_tau_sampled"),
        "r": ("r", "r_target"),
        "Phi": ("Phi", "Phi_sampled"),
    }
    normalised: dict[str, float] = {}
    for target, candidates in aliases.items():
        source = next((name for name in candidates if name in geometry), None)
        if source is None:
            raise ValueError(
                f"Sampled Schur geometry lacks the required {target} value."
            )
        normalised[target] = float(geometry[source])
    if not 1 < normalised["r_tau"] < normalised["r"] < normalised["rho"]:
        raise ValueError("Sampled Schur geometry does not satisfy 1 < r_tau < r < rho.")
    if normalised["Phi"] <= 0:
        raise ValueError("The sampled branch-weight factor must be positive.")
    return normalised


def _normalise_dimensions(values: Iterable[int]) -> tuple[int, ...]:
    dimensions = tuple(int(value) for value in values)
    if not dimensions or any(value < 1 for value in dimensions):
        raise ValueError("Sampled Schur dimensions must be positive.")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("Sampled Schur dimensions must be distinct.")
    return dimensions


def rebuild_sampled_schur_diagnostics(
    config: SampledSchurDiagnosticsConfig,
    *,
    map_label: str,
    geometry: Mapping[str, Any],
    sampled_schur_envelope: Callable[..., pd.DataFrame],
    kappa: Callable[[int, float], float],
    data_dir: Path,
    report_path: Path | None = None,
) -> SampledSchurDiagnosticsResult:
    '''Explanation: Sampled Schur envelopes show how finite nonnormality and contour separation interact across selected designs. They help interpret parameter choices, but the final theorem uses complete-circle interval moats instead of these samples.
Functionality: Rebuild the six retained sampled rows without reading prior outputs.'''

    dimensions = _normalise_dimensions(config.n_values)
    if int(config.oversampling) < 0:
        raise ValueError("The sampled Schur oversampling offset cannot be negative.")
    label = str(map_label).strip()
    if label != LOCKED_MAP_LABEL:
        raise ValueError(
            f"This deployment producer is locked to {LOCKED_MAP_LABEL}, not {label or 'an unlabeled map'}."
        )
    normalised_geometry = _normalise_geometry(geometry)

    frame = sampled_schur_envelope(
        N_values=dimensions,
        oversampling=int(config.oversampling),
        geometry=normalised_geometry,
        kappa=kappa,
    ).copy()
    required = {
        "N",
        "M",
        "epsilon_schur_diagnostic",
        "transported_Bmat_diagnostic",
        "status",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Sampled Schur output lacks columns: {sorted(missing)}")
    if tuple(frame["N"].astype(int)) != dimensions:
        raise RuntimeError("Sampled Schur output does not preserve the locked dimensions.")
    expected_m = tuple(value + int(config.oversampling) for value in dimensions)
    if tuple(frame["M"].astype(int)) != expected_m:
        raise RuntimeError("Sampled Schur output does not preserve the locked M=N+6 law.")
    if frame.empty or frame.loc[:, sorted(required)].isna().all().all():
        raise RuntimeError("The sampled Schur diagnostic producer returned no usable rows.")
    if len(frame) != len(dimensions) or frame["N"].astype(int).nunique() != len(dimensions):
        raise RuntimeError("The sampled Schur diagnostic producer returned duplicate or missing rows.")
    for column in ("epsilon_schur_diagnostic", "transported_Bmat_diagnostic"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError(f"The sampled Schur column {column} is not finite.")
        if (values < 0).any():
            raise RuntimeError(f"The sampled Schur column {column} contains a negative value.")
    if frame["status"].astype(str).str.strip().eq("").any():
        raise RuntimeError("The sampled Schur diagnostic producer returned an empty status.")

    frame.insert(0, "map_label", label)
    frame["geometry_status"] = str(
        geometry.get(
            "geometry_status",
            "source-regenerated sampled geometry; diagnostic only",
        )
    )

    data_dir = Path(data_dir)
    csv_path = data_dir / CSV_FILENAME_TEMPLATE.format(map_label=label)
    report_path = Path(report_path or data_dir.parent / "reports" / REPORT_FILENAME)
    _atomic_write_text(csv_path, frame.to_csv(index=False))
    csv_sha256 = _sha256(csv_path)
    kappa_runtime_evidence = list(
        getattr(kappa, "runtime_evidence", ())
    )
    report = {
        "producer_schema": PRODUCER_SCHEMA,
        "schema_version": "2.0.0",
        "map_label": label,
        "diagnostic_only": True,
        "theorem_gate": False,
        "legacy_seed_dependency": False,
        "commit_protocol": (
            "the CSV is replaced atomically and this report is written last as "
            "the generation commit marker"
        ),
        "config": asdict(config),
        "geometry": normalised_geometry,
        "lineage": {
            "sampled_schur_envelope": _callable_source_record(sampled_schur_envelope),
            "kappa": _callable_source_record(kappa),
            "runtime": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
            },
            "kappa_binary64_blas_runtime_evidence": kappa_runtime_evidence,
            "kappa_binary64_blas_runtime_evidence_role": (
                "portable live evidence for the six diagnostic SVD calls; "
                "never a theorem gate"
            ),
        },
        "output": {
            "path": str(csv_path),
            "rows": int(frame.shape[0]),
            "columns": list(map(str, frame.columns)),
            "sha256": csv_sha256,
        },
    }
    _atomic_write_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return SampledSchurDiagnosticsResult(
        frame=frame,
        csv_path=csv_path,
        report_path=report_path,
        csv_sha256=csv_sha256,
        report_sha256=_sha256(report_path),
    )


__all__ = [
    "CSV_FILENAME_TEMPLATE",
    "LOCKED_MAP_LABEL",
    "PRODUCER_SCHEMA",
    "REPORT_FILENAME",
    "SampledSchurDiagnosticsConfig",
    "SampledSchurDiagnosticsResult",
    "rebuild_sampled_schur_diagnostics",
]
