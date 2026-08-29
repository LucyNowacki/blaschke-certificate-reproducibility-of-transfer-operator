"""Rebuild the retained Blaschke diagnostic audit tables from explicit inputs.

The pure DataFrame generators below extract the arithmetic and status logic from
source-notebook Cells 100 and 102.  The small universal-audit routine needed by
those cells is included here so the deployed producer has no hidden helper
dependency.  These outputs are historical diagnostics: sampled geometry,
Schur rows, contour minima, counts, and small-gain checks are never promoted to
theorem certificates here.

Neither retained audit CSV is accepted as an input.  Callers must supply the
generated target, geometry, sampled-Schur, and sampled-moat DataFrames (or paths
to those generated frames) explicitly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, TypeAlias

import numpy as np
import pandas as pd


MAP_LABEL = "blaschke_mu_0p3"
PRODUCER_SCHEMA = "blaschke-deformation-diagnostic-audits-v1"
UNIVERSAL_AUDIT_FILENAME = (
    "transfer_lab_blaschke_mu_0p3_universal_certification_audit.csv"
)
FIRST14_AUDIT_FILENAME = "transfer_lab_blaschke_mu_0p3_first14_packet_audit.csv"
REPORT_FILENAME = "blaschke_deformation_diagnostic_audits_rebuild.json"

FrameInput: TypeAlias = pd.DataFrame | str | os.PathLike[str]


UNIVERSAL_AUDIT_COLUMNS = (
    "map_label",
    "audit_item",
    "status",
    "evidence",
)

FIRST14_AUDIT_COLUMNS = (
    "name",
    "family_packet",
    "power",
    "multiplicity_packet",
    "packet_centre",
    "target_mode",
    "rank",
    "target",
    "centre",
    "multiplicity_moat",
    "radius",
    "finite_eigenvalue_count",
    "cluster_radius",
    "nearest_outside_distance",
    "radius_status",
    "s_gamma_H",
    "theta_min",
    "d_Gamma0",
    "m_gamma_X",
    "epsilon_X",
    "epsilon_m_gamma",
    "pass_sampled_small_gain",
    "selected_count",
    "selected_eigenvalues",
    "max_numeric_error",
    "pass_sampled_packet",
    "family_moat",
    "count_ok",
    "sampled_validation_pass",
    "contour_interval_certified",
    "validation_status",
    "small_gain_diagnostic",
    "target_identity_status",
    "finite_section_status",
    "contour_role",
    "finite_count_certified",
    "finite_count_matches",
    "sampled_small_gain_pass",
    "small_gain_pass",
    "certified_small_gain_pass",
    "riesz_rank_status",
)

_MOAT_COLUMNS = (
    "rank",
    "target",
    "centre",
    "multiplicity",
    "radius",
    "finite_eigenvalue_count",
    "cluster_radius",
    "nearest_outside_distance",
    "radius_status",
    "s_gamma_H",
    "theta_min",
    "d_Gamma0",
    "m_gamma_X",
    "epsilon_X",
    "epsilon_m_gamma",
    "pass_sampled_small_gain",
    "selected_count",
    "selected_eigenvalues",
    "max_numeric_error",
    "pass_sampled_packet",
    "family",
    "count_ok",
    "sampled_validation_pass",
    "contour_interval_certified",
    "validation_status",
)

_MOAT_ALIASES = {
    "target": ("target", "name"),
    "multiplicity": ("multiplicity", "expected_multiplicity"),
    "radius": ("radius", "contour_radius"),
    "s_gamma_H": ("s_gamma_H", "s_min_gamma"),
}

_EXPECTED_TARGETS = (
    ("alpha^1", "alpha", 1, 1, 0.65),
    ("alpha^2", "alpha", 2, 1, 0.4225),
    ("mu^1", "mu", 1, 2, 0.3),
    ("alpha^3", "alpha", 3, 1, 0.274625),
    ("alpha^4", "alpha", 4, 1, 0.17850625),
    ("alpha^5", "alpha", 5, 1, 0.1160290625),
    ("mu^2", "mu", 2, 2, 0.09),
    ("alpha^6", "alpha", 6, 1, 0.075418890625),
    ("alpha^7", "alpha", 7, 1, 0.04902227890625),
    ("alpha^8", "alpha", 8, 1, 0.0318644812890625),
    ("mu^3", "mu", 3, 2, 0.027),
    ("alpha^9", "alpha", 9, 1, 0.020711912837890624),
    ("alpha^10", "alpha", 10, 1, 0.013462743344628906),
    ("alpha^11", "alpha", 11, 1, 0.00875078317400879),
)

_EXPECTED_SCHUR_N = (30, 40, 50, 60, 80, 100)
_GEOMETRY_LOCK = {
    "rho": 2.725,
    "r_tau": 2.2926584027370747,
    "r": 2.47367,
}
_LEGACY_AUDIT_FILENAMES = {
    UNIVERSAL_AUDIT_FILENAME,
    FIRST14_AUDIT_FILENAME,
}


@dataclass(frozen=True)
class CertificationCapability:
    """Truth-status fields used by the retained diagnostic ladder."""

    target_mode: str
    target_status: str
    geometry_status: str
    deterministic_status: str
    contour_status: str
    exact_model_adapter: str | None = None
    notes: str = ""

    def as_record(self, map_label: str) -> dict[str, Any]:
        return {
            "map_label": str(map_label),
            "target_mode": self.target_mode,
            "target_status": self.target_status,
            "geometry_status": self.geometry_status,
            "deterministic_status": self.deterministic_status,
            "contour_status": self.contour_status,
            "exact_model_adapter": self.exact_model_adapter,
            "notes": self.notes,
        }


_DIAGNOSTIC_CAPABILITY = CertificationCapability(
    target_mode="exact_targets",
    target_status="theorem_certified",
    geometry_status="sampled_not_interval_certified",
    deterministic_status="sampled_not_theorem_certified",
    contour_status="sampled_not_interval_certified",
    exact_model_adapter="symmetric_interval_blaschke",
    notes=(
        "Exact target identities are retained. Geometry, Schur rows, contour "
        "minima, finite counts, and small-gain values in these two tables are "
        "diagnostic only."
    ),
)


class DiagnosticAuditValidationError(ValueError):
    """Raised when an explicit input violates the locked audit contract."""


@dataclass(frozen=True)
class DiagnosticAuditResult:
    """Paths, frames, and report digest returned by the atomic writer."""

    universal_audit: pd.DataFrame
    first14_audit: pd.DataFrame
    universal_csv_path: Path
    first14_csv_path: Path
    report_path: Path
    report_digest: str


@dataclass(frozen=True)
class _LoadedFrame:
    frame: pd.DataFrame
    source: dict[str, Any]


@dataclass(frozen=True)
class _PreparedInputs:
    targets: pd.DataFrame
    geometry: dict[str, Any]
    schur_rows: pd.DataFrame
    moat_rows: pd.DataFrame


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _deployment_root() -> Path:
    candidates = (
        Path(__file__).resolve().parents[1],
        Path.cwd().resolve(),
        Path.cwd().resolve().parent,
    )
    marker = Path("Numerics") / "blaschke_deformation_diagnostic_audits.py"
    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate
    raise FileNotFoundError(
        "Cannot locate the Final Deployment root containing the diagnostic-audit source."
    )


def _canonical_numerics_file(filename: str) -> Path:
    path = _deployment_root() / "Numerics" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical deployment source: {path}")
    return path


def _portable_path(path: Path) -> str:
    path = Path(path).resolve()
    deployment_root = _deployment_root()
    try:
        return path.relative_to(deployment_root).as_posix()
    except ValueError:
        return str(path)


def _source_record(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": _portable_path(path),
        "sha256": _sha256(path),
    }


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        if path.exists():
            os.fchmod(descriptor, path.stat().st_mode & 0o777)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _require_map_lock(map_label: str) -> None:
    if str(map_label) != MAP_LABEL:
        raise DiagnosticAuditValidationError(
            f"map lock requires {MAP_LABEL!r}, received {map_label!r}"
        )


def _require_unique_columns(frame: pd.DataFrame, label: str) -> None:
    duplicates = frame.columns[frame.columns.duplicated()].astype(str).tolist()
    if duplicates:
        raise DiagnosticAuditValidationError(
            f"{label} has duplicate columns: {duplicates}"
        )


def _require_columns(
    frame: pd.DataFrame, required: tuple[str, ...], label: str
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DiagnosticAuditValidationError(
            f"{label} is missing required columns: {missing}"
        )


def _validate_map_column(frame: pd.DataFrame, label: str) -> None:
    if "map_label" not in frame.columns:
        return
    values = set(frame["map_label"].dropna().astype(str))
    if values != {MAP_LABEL} or frame["map_label"].isna().any():
        raise DiagnosticAuditValidationError(
            f"{label} violates the {MAP_LABEL!r} map lock: {sorted(values)}"
        )


def _numeric_series(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric.notna() & np.isfinite(numeric)
    if not bool(finite.all()):
        raise DiagnosticAuditValidationError(
            f"{label} must contain only finite numeric values"
        )
    return numeric


def _integer_series(series: pd.Series, label: str) -> pd.Series:
    numeric = _numeric_series(series, label)
    if not bool((numeric == np.floor(numeric)).all()):
        raise DiagnosticAuditValidationError(f"{label} must contain integers")
    return numeric.astype(int)


def _boolean_series(series: pd.Series, label: str) -> pd.Series:
    def convert(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1"}:
            return True
        if text in {"false", "0"}:
            return False
        raise DiagnosticAuditValidationError(
            f"{label} contains a non-boolean value: {value!r}"
        )

    if series.isna().any():
        raise DiagnosticAuditValidationError(f"{label} contains missing values")
    return series.map(convert).astype(bool)


def _complex_value(value: Any, label: str) -> complex:
    try:
        if isinstance(value, complex):
            result = value
        elif isinstance(value, (float, int, np.floating, np.integer)):
            result = complex(float(value), 0.0)
        else:
            text = str(value).strip().replace("I", "j").replace("i", "j")
            result = complex(text)
    except (TypeError, ValueError) as exc:
        raise DiagnosticAuditValidationError(
            f"{label} contains an invalid complex value: {value!r}"
        ) from exc
    if not (math.isfinite(result.real) and math.isfinite(result.imag)):
        raise DiagnosticAuditValidationError(
            f"{label} contains a non-finite complex value: {value!r}"
        )
    return result


def _normalise_targets(frame: pd.DataFrame) -> pd.DataFrame:
    label = "targets"
    frame = frame.copy(deep=True)
    _require_unique_columns(frame, label)
    _validate_map_column(frame, label)
    _require_columns(
        frame,
        ("name", "family", "power", "multiplicity", "target_mode"),
        label,
    )
    if len(frame) < len(_EXPECTED_TARGETS):
        raise DiagnosticAuditValidationError(
            f"targets requires at least {len(_EXPECTED_TARGETS)} rows"
        )

    selected = frame.head(len(_EXPECTED_TARGETS)).copy()
    if "centre" in selected.columns:
        centres = [
            _complex_value(value, "targets.centre") for value in selected["centre"]
        ]
    elif "target_real_float" in selected.columns:
        real = _numeric_series(
            selected["target_real_float"], "targets.target_real_float"
        )
        imag = (
            _numeric_series(selected["target_imag_float"], "targets.target_imag_float")
            if "target_imag_float" in selected.columns
            else pd.Series(0.0, index=selected.index)
        )
        centres = [complex(re, im) for re, im in zip(real, imag)]
    elif "target_float" in selected.columns:
        centres = [
            complex(value, 0.0)
            for value in _numeric_series(
                selected["target_float"], "targets.target_float"
            )
        ]
    else:
        raise DiagnosticAuditValidationError(
            "targets requires centre, target_real_float, or target_float"
        )

    out = pd.DataFrame(
        {
            "name": selected["name"].astype(str).to_numpy(),
            "family": selected["family"].astype(str).to_numpy(),
            "power": _integer_series(selected["power"], "targets.power").to_numpy(),
            "multiplicity": _integer_series(
                selected["multiplicity"], "targets.multiplicity"
            ).to_numpy(),
            "centre": centres,
            "target_mode": selected["target_mode"].astype(str).to_numpy(),
        }
    )
    if out["name"].duplicated().any():
        raise DiagnosticAuditValidationError(
            "targets contains duplicate identities in the first fourteen rows"
        )

    for index, expected in enumerate(_EXPECTED_TARGETS):
        name, family, power, multiplicity, centre = expected
        row = out.iloc[index]
        observed = complex(row["centre"])
        if (
            row["name"] != name
            or row["family"] != family
            or int(row["power"]) != power
            or int(row["multiplicity"]) != multiplicity
            or row["target_mode"] != "exact"
            or not math.isclose(observed.real, centre, rel_tol=0.0, abs_tol=5.0e-16)
            or abs(observed.imag) > 5.0e-16
        ):
            raise DiagnosticAuditValidationError(
                "targets do not match the locked first-fourteen "
                f"{MAP_LABEL} sequence at row {index + 1}"
            )
    return out


def _single_alias(
    frame: pd.DataFrame,
    canonical: str,
    aliases: tuple[str, ...],
    label: str,
) -> pd.Series:
    present = [column for column in aliases if column in frame.columns]
    if len(present) != 1:
        raise DiagnosticAuditValidationError(
            f"{label} requires exactly one source column for {canonical!r}; "
            f"found {present}"
        )
    return frame[present[0]]


def _normalise_geometry(frame: pd.DataFrame) -> dict[str, Any]:
    label = "geometry"
    frame = frame.copy(deep=True)
    _require_unique_columns(frame, label)
    _validate_map_column(frame, label)
    if len(frame) != 1:
        raise DiagnosticAuditValidationError(
            f"geometry requires exactly one row, received {len(frame)}"
        )

    aliases = {
        "rho": ("rho",),
        "r_tau": ("r_tau", "r_tau_sampled"),
        "r": ("r", "r_target"),
        "Phi": ("Phi", "Phi_sampled"),
        "Phi_star": ("Phi_star", "Phi_star_sampled"),
        "q_out": ("q_out",),
        "q_gap": ("q_gap",),
        "q_star": ("q_star",),
    }
    geometry: dict[str, Any] = {}
    for canonical, candidates in aliases.items():
        source = _single_alias(frame, canonical, candidates, label)
        geometry[canonical] = float(
            _numeric_series(source, f"geometry.{canonical}").iloc[0]
        )

    for key, expected in _GEOMETRY_LOCK.items():
        if not math.isclose(geometry[key], expected, rel_tol=2.0e-15, abs_tol=5.0e-16):
            raise DiagnosticAuditValidationError(
                f"geometry violates the {MAP_LABEL} lock for {key}: "
                f"expected {expected!r}, received {geometry[key]!r}"
            )
    if not math.isclose(
        geometry["q_out"],
        geometry["r"] / geometry["rho"],
        rel_tol=2.0e-15,
        abs_tol=5.0e-16,
    ) or not math.isclose(
        geometry["q_gap"],
        geometry["r_tau"] / geometry["r"],
        rel_tol=2.0e-15,
        abs_tol=5.0e-16,
    ):
        raise DiagnosticAuditValidationError(
            "geometry q_out/q_gap values do not align with rho, r_tau, and r"
        )
    geometry["status"] = "sampled_geometry_not_interval_certified"
    geometry["method"] = "explicit sampled Blaschke geometry input"
    return geometry


def _normalise_schur_rows(
    frame: pd.DataFrame, geometry: dict[str, Any]
) -> pd.DataFrame:
    label = "schur_rows"
    frame = frame.copy(deep=True)
    _require_unique_columns(frame, label)
    _validate_map_column(frame, label)
    _require_columns(
        frame,
        ("N", "M", "epsilon_schur_diagnostic", "status"),
        label,
    )
    if len(frame) != len(_EXPECTED_SCHUR_N):
        raise DiagnosticAuditValidationError(
            f"schur_rows requires {len(_EXPECTED_SCHUR_N)} sampled rows"
        )
    n_values = tuple(_integer_series(frame["N"], "schur_rows.N"))
    m_values = tuple(_integer_series(frame["M"], "schur_rows.M"))
    if n_values != _EXPECTED_SCHUR_N or m_values != tuple(
        value + 6 for value in _EXPECTED_SCHUR_N
    ):
        raise DiagnosticAuditValidationError(
            "schur_rows do not match the locked N and M sampling schedule"
        )
    epsilon = _numeric_series(
        frame["epsilon_schur_diagnostic"],
        "schur_rows.epsilon_schur_diagnostic",
    )
    if not bool((epsilon > 0).all()):
        raise DiagnosticAuditValidationError(
            "schur_rows epsilon_schur_diagnostic values must be positive"
        )
    statuses = frame["status"].astype(str)
    if not bool(
        statuses.str.contains("diagnostic", case=False, regex=False).all()
        and statuses.str.contains(
            "not_theorem_certified", case=False, regex=False
        ).all()
    ):
        raise DiagnosticAuditValidationError(
            "schur_rows must remain sampled diagnostics, not theorem certificates"
        )

    geometry_columns = {
        "rho": "rho",
        "r_tau_sampled": "r_tau",
        "r": "r",
    }
    for column, geometry_key in geometry_columns.items():
        if column not in frame.columns:
            continue
        values = _numeric_series(frame[column], f"schur_rows.{column}")
        if not bool(
            np.isclose(
                values.to_numpy(dtype=float),
                geometry[geometry_key],
                rtol=2.0e-15,
                atol=5.0e-16,
            ).all()
        ):
            raise DiagnosticAuditValidationError(
                f"schur_rows.{column} does not align with geometry.{geometry_key}"
            )
    return frame


def _normalise_moat_rows(frame: pd.DataFrame) -> pd.DataFrame:
    label = "moat_rows"
    frame = frame.copy(deep=True)
    _require_unique_columns(frame, label)
    _validate_map_column(frame, label)
    if {"target_identity_status", "riesz_rank_status"} & set(frame.columns):
        raise DiagnosticAuditValidationError(
            "moat_rows appears to be a retained audit output, not a generated moat input"
        )
    if len(frame) < len(_EXPECTED_TARGETS):
        raise DiagnosticAuditValidationError(
            f"moat_rows requires at least {len(_EXPECTED_TARGETS)} rows"
        )
    for prohibited in ("finite_count_certified", "certified_small_gain_pass"):
        if prohibited in frame.columns and bool(
            _boolean_series(frame[prohibited], f"moat_rows.{prohibited}").any()
        ):
            raise DiagnosticAuditValidationError(
                f"moat_rows cannot promote sampled evidence via {prohibited}"
            )

    selected = frame.head(len(_EXPECTED_TARGETS)).copy()
    columns: dict[str, pd.Series] = {}
    for canonical in _MOAT_COLUMNS:
        candidates = _MOAT_ALIASES.get(canonical, (canonical,))
        columns[canonical] = _single_alias(
            selected, canonical, candidates, label
        ).reset_index(drop=True)
    out = pd.DataFrame(columns)

    for column in (
        "rank",
        "multiplicity",
        "finite_eigenvalue_count",
        "selected_count",
    ):
        out[column] = _integer_series(out[column], f"moat_rows.{column}")
    for column in (
        "centre",
        "radius",
        "cluster_radius",
        "nearest_outside_distance",
        "s_gamma_H",
        "theta_min",
        "d_Gamma0",
        "m_gamma_X",
        "epsilon_X",
        "epsilon_m_gamma",
        "max_numeric_error",
    ):
        out[column] = _numeric_series(out[column], f"moat_rows.{column}")
    for column in (
        "pass_sampled_small_gain",
        "pass_sampled_packet",
        "count_ok",
        "sampled_validation_pass",
        "contour_interval_certified",
    ):
        out[column] = _boolean_series(out[column], f"moat_rows.{column}")
    for column in (
        "target",
        "radius_status",
        "selected_eigenvalues",
        "family",
        "validation_status",
    ):
        out[column] = out[column].astype(str)

    if tuple(out["rank"]) != tuple(range(1, len(_EXPECTED_TARGETS) + 1)):
        raise DiagnosticAuditValidationError(
            "moat_rows ranks must be the contiguous first-fourteen order"
        )
    if out["target"].duplicated().any():
        raise DiagnosticAuditValidationError(
            "moat_rows contains duplicate first-fourteen target identities"
        )
    if bool(out["contour_interval_certified"].any()):
        raise DiagnosticAuditValidationError(
            "moat_rows is sampled and cannot set contour_interval_certified"
        )
    if set(out["validation_status"]) != {"sampled_not_interval_certified"}:
        raise DiagnosticAuditValidationError(
            "moat_rows validation_status must remain sampled_not_interval_certified"
        )
    if not bool((out["radius"] > 0).all() and (out["s_gamma_H"] > 0).all()):
        raise DiagnosticAuditValidationError(
            "moat_rows requires positive sampled radii and singular-value minima"
        )

    count_matches = out["finite_eigenvalue_count"] == out["multiplicity"]
    if not bool((out["count_ok"] == count_matches).all()):
        raise DiagnosticAuditValidationError(
            "moat_rows count_ok does not match finite and expected packet counts"
        )
    sampled_pass = out["epsilon_m_gamma"] < 1.0
    if not bool((out["pass_sampled_small_gain"] == sampled_pass).all()):
        raise DiagnosticAuditValidationError(
            "moat_rows pass_sampled_small_gain disagrees with epsilon_m_gamma"
        )

    epsilon_x = float(out["epsilon_X"].iloc[0])
    if not bool(
        np.isclose(
            out["epsilon_X"].to_numpy(dtype=float),
            epsilon_x,
            rtol=2.0e-15,
            atol=0.0,
        ).all()
    ):
        raise DiagnosticAuditValidationError(
            "moat_rows must use one aligned epsilon_X value"
        )
    recomputed = out["m_gamma_X"].to_numpy(dtype=float) * epsilon_x
    if not bool(
        np.isclose(
            out["epsilon_m_gamma"].to_numpy(dtype=float),
            recomputed,
            rtol=2.0e-15,
            atol=0.0,
        ).all()
    ):
        raise DiagnosticAuditValidationError(
            "moat_rows epsilon_m_gamma does not align with m_gamma_X and epsilon_X"
        )
    out["small_gain_diagnostic"] = pd.to_numeric(out["epsilon_m_gamma"], errors="raise")
    return out


def _validate_target_alignment(targets: pd.DataFrame, moat_rows: pd.DataFrame) -> None:
    target_names = tuple(targets["name"])
    moat_names = tuple(moat_rows["target"])
    if target_names != moat_names:
        raise DiagnosticAuditValidationError(
            "targets and moat_rows are not aligned by first-fourteen identity and order"
        )
    if not bool(
        (
            targets["multiplicity"].to_numpy(dtype=int)
            == moat_rows["multiplicity"].to_numpy(dtype=int)
        ).all()
    ):
        raise DiagnosticAuditValidationError(
            "targets and moat_rows have misaligned multiplicities"
        )
    target_centres = np.asarray(targets["centre"], dtype=np.complex128)
    moat_centres = moat_rows["centre"].to_numpy(dtype=float)
    if not bool(
        np.isclose(target_centres.real, moat_centres, rtol=0.0, atol=5.0e-16).all()
        and np.isclose(target_centres.imag, 0.0, rtol=0.0, atol=5.0e-16).all()
    ):
        raise DiagnosticAuditValidationError(
            "targets and moat_rows have misaligned contour centres"
        )


def _prepare_inputs(
    *,
    map_label: str,
    targets: pd.DataFrame,
    geometry: pd.DataFrame,
    schur_rows: pd.DataFrame,
    moat_rows: pd.DataFrame,
) -> _PreparedInputs:
    _require_map_lock(map_label)
    for value, label in (
        (targets, "targets"),
        (geometry, "geometry"),
        (schur_rows, "schur_rows"),
        (moat_rows, "moat_rows"),
    ):
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"{label} must be an explicit pandas DataFrame")
    prepared_targets = _normalise_targets(targets)
    prepared_geometry = _normalise_geometry(geometry)
    prepared_schur = _normalise_schur_rows(schur_rows, prepared_geometry)
    prepared_moats = _normalise_moat_rows(moat_rows)
    _validate_target_alignment(prepared_targets, prepared_moats)
    return _PreparedInputs(
        targets=prepared_targets,
        geometry=prepared_geometry,
        schur_rows=prepared_schur,
        moat_rows=prepared_moats,
    )


def _build_certification_audit(
    *,
    map_label: str,
    capability: CertificationCapability,
    target_count: int,
    geometry: dict[str, Any],
    schur_rows: pd.DataFrame,
    moat_rows: pd.DataFrame,
) -> pd.DataFrame:
    '''Explanation: The numerical argument has logically different levels: sampled geometry, finite calculations, certified bounds, and finally rank transfer. This audit keeps those levels separate so attractive diagnostics cannot masquerade as theorem hypotheses.
Functionality: Build the retained diagnostic ladder without an external helper.'''

    finite_counts = pd.to_numeric(
        moat_rows["finite_eigenvalue_count"], errors="coerce"
    )
    expected_counts = pd.to_numeric(moat_rows["multiplicity"], errors="coerce")
    finite_counts_available = bool(
        finite_counts.notna().all() and expected_counts.notna().all()
    )
    finite_counts_ok = finite_counts_available and bool(
        (finite_counts == expected_counts).all()
    )
    finite_counts_certified = bool(
        len(moat_rows)
        and "finite_count_certified" in moat_rows
        and moat_rows["finite_count_certified"].fillna(False).astype(bool).all()
    )
    moat_values = pd.to_numeric(moat_rows["s_gamma_H"], errors="coerce")
    sampled_moats_positive = bool(len(moat_values) and (moat_values > 0).all())
    small_gain = pd.to_numeric(moat_rows["epsilon_m_gamma"], errors="coerce")
    small_gain_available = bool(
        len(small_gain)
        and small_gain.notna().all()
        and np.isfinite(small_gain).all()
    )
    sampled_small_gain = small_gain_available and bool((small_gain < 1).all())

    geometry_available = bool(geometry) and math.isfinite(
        float(geometry.get("r_tau", np.nan))
    )
    schur_available = bool(
        len(schur_rows) and "epsilon_schur_diagnostic" in schur_rows
    )
    rigorous_operator_inputs = all(
        status == "theorem_certified"
        for status in (
            capability.geometry_status,
            capability.deterministic_status,
            capability.contour_status,
        )
    )
    finite_count_status = (
        "theorem_certified"
        if finite_counts_ok and finite_counts_certified
        else "sampled_pass"
        if finite_counts_ok
        else "sampled_fail"
        if finite_counts_available
        else "sampled_unresolved"
    )
    small_gain_status = (
        "theorem_certified"
        if sampled_small_gain and rigorous_operator_inputs
        else "sampled_pass"
        if sampled_small_gain
        else "sampled_fail"
        if small_gain_available
        else "sampled_unresolved"
    )
    target_role = (
        "exact-model contour centres"
        if capability.target_status == "theorem_certified"
        else "reference-derived contour proposals; exact target identity is not assumed"
    )
    rows = [
        {
            "audit_item": "target source and contour role",
            "status": capability.target_status,
            "evidence": (
                f"{target_count} nontrivial packets in {capability.target_mode} "
                f"mode; {target_role}"
            ),
        },
        {
            "audit_item": "Bernstein branch geometry",
            "status": capability.geometry_status if geometry_available else "missing",
            "evidence": (
                f"rho={geometry.get('rho')}, r_tau={geometry.get('r_tau')}, "
                f"r={geometry.get('r')}"
            ),
        },
        {
            "audit_item": "single-space Schur envelope",
            "status": capability.deterministic_status if schur_available else "missing",
            "evidence": f"{len(schur_rows)} computed rows",
        },
        {
            "audit_item": "finite packet counts",
            "status": finite_count_status,
            "evidence": (
                f"matching counts on the first {min(target_count, len(moat_rows))} "
                f"contour packets; certified_count={finite_counts_certified}"
                if finite_counts_ok
                else "finite counts disagree with expected packet ranks"
                if finite_counts_available
                else "no complete finite-count table"
            ),
        },
        {
            "audit_item": "finite-section contour moats",
            "status": (
                capability.contour_status
                if sampled_moats_positive
                else "sampled_unresolved"
            ),
            "evidence": (
                "positive sampled minima"
                if sampled_moats_positive
                else "no positive sampled moat table"
            ),
        },
        {
            "audit_item": "sampled small-gain test",
            "status": small_gain_status,
            "evidence": (
                "epsilon times the resolvent factor is below one on every packet"
                if sampled_small_gain
                else "epsilon times the resolvent factor is at least one on one or more packets"
                if small_gain_available
                else "small-gain values are incomplete"
            ),
        },
    ]
    theorem_complete = bool(
        target_count > 0
        and rigorous_operator_inputs
        and finite_counts_ok
        and finite_counts_certified
        and sampled_moats_positive
        and sampled_small_gain
    )
    missing_gates: list[str] = []
    if capability.geometry_status != "theorem_certified":
        missing_gates.append("certified branch geometry")
    if capability.deterministic_status != "theorem_certified":
        missing_gates.append("certified X-to-X perturbation bound")
    if capability.contour_status != "theorem_certified":
        missing_gates.append("validated contour-moat lower bound")
    if not (finite_counts_ok and finite_counts_certified):
        missing_gates.append("certified finite-section packet count")
    if not sampled_small_gain:
        missing_gates.append("strict small-gain inequality")
    rows.append(
        {
            "audit_item": "overall spectral certification",
            "status": (
                "theorem_certified"
                if theorem_complete
                else "diagnostic_not_theorem_certified"
            ),
            "evidence": (
                "The transfer operator and finite block have equal Riesz-projector "
                "rank inside every tested contour."
                if theorem_complete
                else "Missing theorem gates: " + "; ".join(missing_gates)
            ),
        }
    )
    audit = pd.DataFrame(rows)
    audit.insert(0, "map_label", str(map_label))
    return audit


def _build_universal_audit(inputs: _PreparedInputs) -> pd.DataFrame:
    '''Explanation: A universal sampled envelope is useful for seeing whether the proposed asymptotic mechanism is plausible across targets. Because sampling cannot prove a whole-boundary maximum or moat, the thesis records it only as diagnostic evidence.
Functionality: Build the locked diagnostic-only universal audit while preventing sampled evidence from being promoted to a theorem certificate.'''
    audit = _build_certification_audit(
        map_label=MAP_LABEL,
        capability=_DIAGNOSTIC_CAPABILITY,
        target_count=len(inputs.targets),
        geometry=inputs.geometry,
        schur_rows=inputs.schur_rows,
        moat_rows=inputs.moat_rows,
    )
    _require_columns(audit, UNIVERSAL_AUDIT_COLUMNS, "universal audit output")
    audit = audit.loc[:, UNIVERSAL_AUDIT_COLUMNS].copy()
    theorem_rows = audit.loc[audit["status"] == "theorem_certified", "audit_item"]
    if tuple(theorem_rows) != ("target source and contour role",):
        raise RuntimeError(
            "diagnostic universal audit attempted to promote sampled evidence"
        )
    overall = audit.loc[audit["audit_item"] == "overall spectral certification"]
    if len(overall) != 1 or overall.iloc[0]["status"] != (
        "diagnostic_not_theorem_certified"
    ):
        raise RuntimeError("universal audit did not remain diagnostic-only")
    return audit


def _build_first14_audit(inputs: _PreparedInputs) -> pd.DataFrame:
    '''Explanation: The first fourteen packets provide a focused stress test of counts, distances, and small-gain margins. Combining them in one table reveals agreement patterns while retaining their non-theorem status when any ingredient is merely sampled.
Functionality: Merge the first-fourteen targets with sampled moat and count diagnostics and label every resulting row as non-theorem evidence.'''
    targets = inputs.targets.copy(deep=True).rename(columns={"centre": "packet_centre"})
    moat_rows = inputs.moat_rows.copy(deep=True)
    packet_table = targets.merge(
        moat_rows,
        left_on="name",
        right_on="target",
        how="left",
        suffixes=("_packet", "_moat"),
        validate="one_to_one",
    )
    if len(packet_table) != len(_EXPECTED_TARGETS):
        raise RuntimeError("first-fourteen merge changed the target cardinality")

    epsilon_x = float(packet_table["epsilon_X"].iloc[0])
    packet_table["epsilon_X"] = epsilon_x
    packet_table["epsilon_m_gamma"] = (
        pd.to_numeric(packet_table["m_gamma_X"], errors="coerce") * epsilon_x
    )
    packet_table["pass_sampled_small_gain"] = packet_table["epsilon_m_gamma"] < 1.0
    packet_table["contour_interval_certified"] = False
    packet_table["validation_status"] = "sampled_not_interval_certified"
    packet_table["target_identity_status"] = _DIAGNOSTIC_CAPABILITY.target_status
    packet_table["finite_section_status"] = _DIAGNOSTIC_CAPABILITY.contour_status
    packet_table["contour_role"] = "exact-model contour centre"
    packet_table["finite_count_certified"] = False

    finite_count = pd.to_numeric(
        packet_table["finite_eigenvalue_count"], errors="coerce"
    )
    expected_count = pd.to_numeric(packet_table["multiplicity_packet"], errors="coerce")
    packet_table["finite_count_matches"] = (
        finite_count.notna() & expected_count.notna() & (finite_count == expected_count)
    )

    sampled_values = pd.to_numeric(packet_table["epsilon_m_gamma"], errors="coerce")
    sampled_available = sampled_values.notna() & np.isfinite(sampled_values)
    sampled_pass = sampled_available & (sampled_values < 1.0)
    packet_table["sampled_small_gain_pass"] = sampled_pass
    packet_table["small_gain_pass"] = sampled_pass
    packet_table["certified_small_gain_pass"] = False

    def packet_riesz_rank_status(row: pd.Series) -> str:
        '''Explanation: A packet's observed count and sampled small-gain outcome describe what the finite experiment suggests. The status rule deliberately withholds Riesz-rank equality until the corresponding complete-circle and perturbation hypotheses are certified.
Functionality: Classify one packet from its sampled count and small-gain outcomes without asserting certified Riesz-rank equality.'''
        if not bool(row["finite_count_matches"]):
            return "sampled_count_fail"
        if bool(sampled_available.loc[row.name]) and not bool(
            sampled_pass.loc[row.name]
        ):
            return "sampled_small_gain_fail"
        if bool(sampled_pass.loc[row.name]):
            return "sampled_pass_not_theorem_certified"
        return "diagnostic_not_theorem_certified"

    packet_table["riesz_rank_status"] = packet_table.apply(
        packet_riesz_rank_status, axis=1
    )
    _require_columns(packet_table, FIRST14_AUDIT_COLUMNS, "first14 audit output")
    packet_table = packet_table.loc[:, FIRST14_AUDIT_COLUMNS].copy()
    if (
        packet_table["contour_interval_certified"].any()
        or packet_table["finite_count_certified"].any()
        or packet_table["certified_small_gain_pass"].any()
        or packet_table["riesz_rank_status"]
        .str.contains("theorem_certified_equal", regex=False)
        .any()
    ):
        raise RuntimeError("first14 audit attempted to promote sampled evidence")
    return packet_table


def generate_diagnostic_audit_frames(
    *,
    map_label: str,
    targets: pd.DataFrame,
    geometry: pd.DataFrame,
    schur_rows: pd.DataFrame,
    moat_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Explanation: Placing the diagnostic ladders side by side makes the logical provenance of every reported spectral claim visible. This supports thesis interpretation without adding any new mathematical assumption.
Functionality: Purely generate both validated diagnostic frames from DataFrames.'''

    inputs = _prepare_inputs(
        map_label=map_label,
        targets=targets,
        geometry=geometry,
        schur_rows=schur_rows,
        moat_rows=moat_rows,
    )
    return _build_universal_audit(inputs), _build_first14_audit(inputs)


def generate_universal_certification_audit(
    *,
    map_label: str,
    targets: pd.DataFrame,
    geometry: pd.DataFrame,
    schur_rows: pd.DataFrame,
    moat_rows: pd.DataFrame,
) -> pd.DataFrame:
    '''Explanation: This table asks how the same sampled envelope behaves across the full target family. It is an exploratory consistency check, not a substitute for the target-by-target interval certificates used in the theorem.
Functionality: Purely generate the source-Cell-100 universal diagnostic audit.'''

    universal, _ = generate_diagnostic_audit_frames(
        map_label=map_label,
        targets=targets,
        geometry=geometry,
        schur_rows=schur_rows,
        moat_rows=moat_rows,
    )
    return universal


def generate_first14_packet_audit(
    *,
    map_label: str,
    targets: pd.DataFrame,
    geometry: pd.DataFrame,
    schur_rows: pd.DataFrame,
    moat_rows: pd.DataFrame,
) -> pd.DataFrame:
    '''Explanation: This focused packet table compares finite spectral geometry with proposed moat margins for the first fourteen targets. Its purpose is to expose discrepancies early while leaving the certified 24-contour argument logically independent.
Functionality: Purely generate the source-Cell-102 first-fourteen packet audit.'''

    _, first14 = generate_diagnostic_audit_frames(
        map_label=map_label,
        targets=targets,
        geometry=geometry,
        schur_rows=schur_rows,
        moat_rows=moat_rows,
    )
    return first14


def _load_frame(value: FrameInput, label: str) -> _LoadedFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.copy(deep=True)
        return _LoadedFrame(
            frame=frame,
            source={
                "kind": "explicit_generated_dataframe",
                "rows": len(frame),
                "columns": [str(column) for column in frame.columns],
                "sha256": _sha256_bytes(_csv_bytes(frame)),
            },
        )

    path = Path(value)
    if path.name in _LEGACY_AUDIT_FILENAMES:
        raise DiagnosticAuditValidationError(
            f"{label} cannot use retained audit output {path.name!r} as a seed"
        )
    if not path.is_file():
        raise FileNotFoundError(f"{label} path is not a file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, float_precision="round_trip")
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise DiagnosticAuditValidationError(
            f"{label} path must be CSV or Parquet: {path}"
        )
    return _LoadedFrame(
        frame=frame,
        source={
            "kind": "explicit_generated_path",
            "path": _portable_path(path),
            "sha256": _sha256(path),
            "rows": len(frame),
            "columns": [str(column) for column in frame.columns],
        },
    )


def _notebook_source_record() -> dict[str, Any]:
    notebook_path = _canonical_numerics_file(
        "blaschke_deformation_certifier.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells: dict[str, dict[str, str]] = {}
    for label in ("Cell 100", "Cell 102"):
        matches: list[tuple[dict[str, Any], str]] = []
        marker = f"# {label}"
        for cell in notebook.get("cells", []):
            source_value = cell.get("source", "")
            source = (
                "".join(source_value)
                if isinstance(source_value, list)
                else str(source_value)
            )
            first = next(
                (line.strip() for line in source.splitlines() if line.strip()), ""
            )
            if cell.get("cell_type") == "code" and first == marker:
                matches.append((cell, source))
        if len(matches) != 1:
            raise RuntimeError(
                f"source notebook requires exactly one code cell labelled {label}"
            )
        cell, source = matches[0]
        cells[label] = {
            "cell_id": str(cell.get("id", "")),
            "source_sha256": _sha256_bytes(source.encode("utf-8")),
        }
    return {**_source_record(notebook_path), "cells": cells}


def rebuild_diagnostic_audits(
    *,
    map_label: str,
    targets: FrameInput,
    geometry: FrameInput,
    schur_rows: FrameInput,
    moat_rows: FrameInput,
    data_dir: Path,
    report_path: Path,
) -> DiagnosticAuditResult:
    """Validate explicit sources, atomically write both CSVs, then the report."""

    _require_map_lock(map_label)
    loaded = {
        "targets": _load_frame(targets, "targets"),
        "geometry": _load_frame(geometry, "geometry"),
        "schur_rows": _load_frame(schur_rows, "schur_rows"),
        "moat_rows": _load_frame(moat_rows, "moat_rows"),
    }
    universal, first14 = generate_diagnostic_audit_frames(
        map_label=map_label,
        targets=loaded["targets"].frame,
        geometry=loaded["geometry"].frame,
        schur_rows=loaded["schur_rows"].frame,
        moat_rows=loaded["moat_rows"].frame,
    )

    data_dir = Path(data_dir)
    report_path = Path(report_path)
    universal_path = data_dir / UNIVERSAL_AUDIT_FILENAME
    first14_path = data_dir / FIRST14_AUDIT_FILENAME
    destinations = {
        universal_path.resolve(),
        first14_path.resolve(),
        report_path.resolve(),
    }
    if len(destinations) != 3:
        raise DiagnosticAuditValidationError(
            "the two CSV outputs and JSON report require distinct paths"
        )

    universal_payload = _csv_bytes(universal)
    first14_payload = _csv_bytes(first14)
    output_records = {
        UNIVERSAL_AUDIT_FILENAME: {
            "path": _portable_path(universal_path),
            "rows": len(universal),
            "columns": list(UNIVERSAL_AUDIT_COLUMNS),
            "sha256": _sha256_bytes(universal_payload),
            "diagnostic_only": True,
        },
        FIRST14_AUDIT_FILENAME: {
            "path": _portable_path(first14_path),
            "rows": len(first14),
            "columns": list(FIRST14_AUDIT_COLUMNS),
            "sha256": _sha256_bytes(first14_payload),
            "diagnostic_only": True,
        },
    }
    report = {
        "producer_schema": PRODUCER_SCHEMA,
        "map_label": MAP_LABEL,
        "diagnostic_only": True,
        "diagnostic_status": (
            "retained sampled comparison audits; not inputs to theorem gates"
        ),
        "legacy_seed_dependency": False,
        "source_extraction": {
            "notebook": _notebook_source_record(),
            "producer": _source_record(
                _canonical_numerics_file(
                    "blaschke_deformation_diagnostic_audits.py"
                )
            ),
        },
        "inputs": {name: item.source for name, item in loaded.items()},
        "checks": {
            "map_lock": True,
            "schema_alignment": True,
            "target_alignment": True,
            "sampled_claims_remain_diagnostic": True,
            "target_count": len(_EXPECTED_TARGETS),
            "target_names": [row[0] for row in _EXPECTED_TARGETS],
            "sampled_schur_N": list(_EXPECTED_SCHUR_N),
        },
        "outputs": output_records,
    }
    report_payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )

    _atomic_bytes(universal_path, universal_payload)
    _atomic_bytes(first14_path, first14_payload)
    _atomic_bytes(report_path, report_payload)
    return DiagnosticAuditResult(
        universal_audit=universal,
        first14_audit=first14,
        universal_csv_path=universal_path,
        first14_csv_path=first14_path,
        report_path=report_path,
        report_digest=_sha256_bytes(report_payload),
    )


__all__ = [
    "MAP_LABEL",
    "PRODUCER_SCHEMA",
    "UNIVERSAL_AUDIT_FILENAME",
    "FIRST14_AUDIT_FILENAME",
    "REPORT_FILENAME",
    "UNIVERSAL_AUDIT_COLUMNS",
    "FIRST14_AUDIT_COLUMNS",
    "DiagnosticAuditValidationError",
    "DiagnosticAuditResult",
    "CertificationCapability",
    "generate_diagnostic_audit_frames",
    "generate_universal_certification_audit",
    "generate_first14_packet_audit",
    "rebuild_diagnostic_audits",
]


def main() -> None:
    output_root = (
        _deployment_root()
        / "Numerics"
        / "outputs"
        / "blaschke_deformation_certifier"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-label", default=MAP_LABEL)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--schur-rows", required=True, type=Path)
    parser.add_argument("--moat-rows", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=output_root / "data")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=output_root / "reports" / REPORT_FILENAME,
    )
    arguments = parser.parse_args()
    result = rebuild_diagnostic_audits(
        map_label=arguments.map_label,
        targets=arguments.targets,
        geometry=arguments.geometry,
        schur_rows=arguments.schur_rows,
        moat_rows=arguments.moat_rows,
        data_dir=arguments.data_dir,
        report_path=arguments.report_path,
    )
    print(
        json.dumps(
            {
                "universal_csv_path": str(result.universal_csv_path),
                "first14_csv_path": str(result.first14_csv_path),
                "report_path": str(result.report_path),
                "report_sha256": result.report_digest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
