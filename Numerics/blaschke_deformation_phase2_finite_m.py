"""Standalone safe finite-order Gauss--Legendre completion for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import flint
from flint import arb
import pandas as pd

try:
    from .blaschke_deformation_phase2_geometry import (
        CANONICAL_SELECTED_HARDY_RADIUS_TEXT,
        exact_q_gap_contract,
        require_canonical_selected_hardy_radius,
    )
except ImportError:
    from blaschke_deformation_phase2_geometry import (
        CANONICAL_SELECTED_HARDY_RADIUS_TEXT,
        exact_q_gap_contract,
        require_canonical_selected_hardy_radius,
    )


MAP_LABEL = "blaschke_mu_0p3"
PRODUCER_SCHEMA = "phase2-finite-m-completion-v1"


@dataclass(frozen=True)
class Phase2FiniteMConfig:
    N: int = 600
    M: int = 610
    rho: str = "2.725"
    map_label: str = MAP_LABEL
    precision_bits: int = 192

    @classmethod
    def production_n600_m610(cls) -> "Phase2FiniteMConfig":
        return cls()


@dataclass(frozen=True)
class Phase2FiniteMResult:
    certificate: dict[str, Any]
    safe_row: dict[str, Any]
    certificate_csv_path: Path
    safe_row_csv_path: Path
    report_json_path: Path
    report_markdown_path: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_one(path: Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one row in {path}; found {len(rows)}.")
    return rows[0]


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _upper_decimal(value: arb) -> Decimal:
    mid, rad, exponent = value.upper().mid_rad_10exp()
    return (Decimal(int(mid)) + abs(Decimal(int(rad)))) * (
        Decimal(10) ** int(exponent)
    )


def _lower_decimal(value: arb) -> Decimal:
    mid, rad, exponent = value.lower().mid_rad_10exp()
    return (Decimal(int(mid)) - abs(Decimal(int(rad)))) * (
        Decimal(10) ** int(exponent)
    )


def _upper_float(value: arb) -> float:
    return math.nextafter(float(_upper_decimal(value)), math.inf)


def _lower_float(value: arb) -> float:
    return math.nextafter(float(_lower_decimal(value)), -math.inf)


def _interval_text(value: arb, digits: int = 70) -> str:
    return value.str(int(digits), radius=True)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def certify_finite_m_completion(
    config: Phase2FiniteMConfig,
    *,
    output_dir: Path,
) -> Phase2FiniteMResult:
    '''Explanation: Gauss--Legendre assembly uses finitely many nodes, so its quadrature remainder needs a safe prefactor before it can enter the matrix defect. This stage certifies that finite-M amplification rather than assuming asymptotic exactness.
Functionality: Certify the safe finite-M prefactor from the raw balanced matrix row.'''

    if config.map_label != MAP_LABEL:
        raise ValueError(f"This producer is restricted to {MAP_LABEL}.")
    if config.N < 1 or config.M < config.N:
        raise ValueError("The finite-M stage requires M at least N at least one.")

    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    report_dir = output_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    flint.ctx.prec = int(config.precision_bits)
    getcontext().prec = 100

    input_path = data_dir / (
        f"branch_image_balanced_candidate_single_space_row_N{config.N}_M{config.M}.csv"
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing raw balanced Phase 2 row: {input_path}")
    input_row = _read_one(input_path)
    if int(input_row["N"]) != config.N or int(input_row["M"]) != config.M:
        raise RuntimeError("The raw balanced row has the wrong finite dimensions.")
    if Decimal(input_row["rho"]) != Decimal(config.rho):
        raise RuntimeError("The raw balanced row uses a different response radius.")
    production_contract = bool(config.N == 600 and config.M == 610)
    q_gap_contract: dict[str, Any] = {}
    if production_contract:
        require_canonical_selected_hardy_radius(
            input_row["r"], label="finite-M input Hardy radius"
        )
        q_gap_contract = exact_q_gap_contract(
            r_tau_upper=input_row["r_tau_interval_u"],
            hardy_radius=input_row["r"],
            q_gap_target=input_row["q_gap"],
        )
    for flag in (
        "analytic_branch_sufficient",
        "branch_image_in_Er_sufficient",
        "admissible_chain_sufficient",
        "transport_certified",
        "matrix_certificate_arb",
        "tail_components_interval",
    ):
        if not _is_true(input_row.get(flag, False)):
            raise RuntimeError(f"The upstream balanced row failed {flag}.")

    rho = arb(config.rho)
    if rho <= 1:
        raise ArithmeticError("The safe Gauss--Legendre prefactor requires rho > 1.")
    c_safe_interval = 8 * rho / (rho - 1)
    c_core_interval = (
        arb.pi()
        * (1 + rho ** -2).sqrt()
        * (1 + 1 / (2 * (rho ** 2 - 1)))
    )
    if c_core_interval.lower() <= 0:
        raise ArithmeticError("The reference Gauss--Legendre interval is not positive.")
    c_safe = c_safe_interval.upper()
    d_safe_arb = (c_safe_interval / c_core_interval).upper()
    d_safe_float = math.nextafter(
        _upper_float(c_safe_interval) / _lower_float(c_core_interval), math.inf
    )
    d_safe = arb(str(max(d_safe_float, _upper_float(d_safe_arb)))).upper()

    bmat_star_d1 = arb(input_row["B_mat_star_arb_u"]).upper()
    bmat_unstar_d1 = arb(input_row["B_mat_unstar_arb_u"]).upper()
    kappa_hat = arb(input_row["kappa_hat_transport_cert"]).upper()
    bmat_star_safe = (d_safe * bmat_star_d1).upper()
    bmat_unstar_safe = (d_safe * bmat_unstar_d1).upper()
    kappa_bmat_star_safe = (kappa_hat * bmat_star_safe).upper()
    kappa_bmat_unstar_safe = (kappa_hat * bmat_unstar_safe).upper()

    b_out = (arb(input_row["two_B_out_star_interval_u"]) / 2).upper()
    b_in = arb(input_row["B_in_branch_image_interval_u"]).upper()
    noncollocation_rss = (b_out**2 + b_in**2).sqrt().upper()
    epsilon_star_safe = (noncollocation_rss + kappa_bmat_star_safe).upper()
    epsilon_unstar_safe = (noncollocation_rss + kappa_bmat_unstar_safe).upper()

    certificate = {
        "map_label": config.map_label,
        "N": config.N,
        "M": config.M,
        "rho": float(config.rho),
        "precision_bits": int(config.precision_bits),
        "C_GL_safe_u": _upper_float(c_safe),
        "C_GL_core_D_M_1_l": _lower_float(c_core_interval),
        "C_GL_core_D_M_1_u": _upper_float(c_core_interval),
        "D_safe_u": _upper_float(d_safe),
        "output_schema_version": 2,
        "output_leakage_count": 1,
        "B_out_interval_u": _upper_float(b_out),
        "B_mat_star_D_M_1_diagnostic_u": _upper_float(bmat_star_d1),
        "B_mat_star_safe_u": _upper_float(bmat_star_safe),
        "kappa_Bmat_star_D_M_1_diagnostic_u": _upper_float(
            arb(input_row["kappa_Bmat_star"])
        ),
        "kappa_Bmat_star_safe_u": _upper_float(kappa_bmat_star_safe),
        "epsilon_safe_u": _upper_float(epsilon_star_safe),
        "finite_M_prefactor_certified": True,
        "status": (
            "unconditional Chebyshev-truncation Gauss--Legendre "
            "prefactor certificate"
        ),
    }
    safe_row = dict(input_row)
    safe_row.update(
        {
            "label": (
                "branch-image balanced row with unconditional finite-M "
                f"prefactor N={config.N}"
            ),
            "rho": config.rho,
            "r": (
                CANONICAL_SELECTED_HARDY_RADIUS_TEXT
                if production_contract
                else str(input_row["r"])
            ),
            **q_gap_contract,
            "B_mat_star_D_M_1_diagnostic_u": _upper_float(bmat_star_d1),
            "B_mat_unstar_D_M_1_diagnostic_u": _upper_float(bmat_unstar_d1),
            "kappa_Bmat_star_D_M_1_diagnostic_u": _upper_float(
                arb(input_row["kappa_Bmat_star"])
            ),
            "kappa_Bmat_unstar_D_M_1_diagnostic_u": _upper_float(
                arb(input_row["kappa_Bmat_unstar"])
            ),
            "C_GL_safe_u": _upper_float(c_safe),
            "C_GL_core_D_M_1_u": _upper_float(c_core_interval),
            "D_safe_u": _upper_float(d_safe),
            "output_schema_version": 2,
            "output_leakage_count": 1,
            "B_out_interval_u": _upper_float(b_out),
            "B_mat_star_arb_u": _upper_float(bmat_star_safe),
            "B_mat_unstar_arb_u": _upper_float(bmat_unstar_safe),
            "kappa_Bmat_star": _upper_float(kappa_bmat_star_safe),
            "kappa_Bmat_unstar": _upper_float(kappa_bmat_unstar_safe),
            "noncollocation_rss_interval_u": _upper_float(noncollocation_rss),
            "tail_floor_interval_u": _upper_float(noncollocation_rss),
            "epsilon_star_candidate": _upper_float(epsilon_star_safe),
            "epsilon_with_unstar_matrix_candidate": _upper_float(
                epsilon_unstar_safe
            ),
            "quadrature_prefactor_route": (
                "Chebyshev truncation through degree 2M-1"
            ),
            "finite_M_prefactor_certified": True,
            "total_certified": True,
            "status": (
                "balanced branch-image row with unconditional safe "
                "finite-M certificate"
            ),
        }
    )
    safe_row.pop("two_B_out_star_interval_u", None)

    certificate_csv_path = data_dir / (
        f"finite_M_safe_GL_prefactor_certificate_N{config.N}_M{config.M}.csv"
    )
    safe_row_csv_path = data_dir / (
        "branch_image_balanced_candidate_single_space_row_"
        f"N{config.N}_M{config.M}_safe_GL.csv"
    )
    report_json_path = report_dir / (
        f"finite_M_safe_GL_prefactor_certificate_N{config.N}_M{config.M}.json"
    )
    report_markdown_path = report_dir / (
        f"finite_M_safe_GL_prefactor_certificate_N{config.N}_M{config.M}.md"
    )
    _atomic_dataframe(certificate_csv_path, pd.DataFrame([certificate]))
    _atomic_dataframe(safe_row_csv_path, pd.DataFrame([safe_row]))

    provenance = {
        "producer_helper": Path(__file__).name,
        "producer_schema": PRODUCER_SCHEMA,
        "method": (
            "Chebyshev truncation plus positivity and mass of "
            "Gauss--Legendre weights"
        ),
        "inputs": {str(input_path): _sha256(input_path)},
        "source_sha256": _sha256(Path(__file__)),
        "intervals": {
            "C_GL_safe": _interval_text(c_safe),
            "C_GL_core_D_M_1": _interval_text(c_core_interval),
            "D_safe": _interval_text(d_safe),
            "B_mat_star_safe": _interval_text(bmat_star_safe),
            "kappa_Bmat_star_safe": _interval_text(kappa_bmat_star_safe),
        },
        "certificate": certificate,
    }
    _atomic_text(report_json_path, json.dumps(provenance, indent=2) + "\n")
    _atomic_text(
        report_markdown_path,
        "# Unconditional finite-M Gauss--Legendre prefactor\n\n"
        + f"N={config.N}, M={config.M}, rho={config.rho}.\n\n"
        + f"C_GL_safe <= {_upper_float(c_safe):.17e}.\n\n"
        + f"D_safe <= {_upper_float(d_safe)!r}.\n\n"
        + (
            "Transported matrix contribution <= "
            f"{_upper_float(kappa_bmat_star_safe):.17e}.\n\n"
        )
        + (
            "The finite-M quadrature prefactor is theorem-certified by the "
            "elementary Chebyshev-truncation fallback.\n"
        ),
    )
    return Phase2FiniteMResult(
        certificate=certificate,
        safe_row=safe_row,
        certificate_csv_path=certificate_csv_path,
        safe_row_csv_path=safe_row_csv_path,
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
    )


__all__ = [
    "MAP_LABEL",
    "PRODUCER_SCHEMA",
    "Phase2FiniteMConfig",
    "Phase2FiniteMResult",
    "certify_finite_m_completion",
]
