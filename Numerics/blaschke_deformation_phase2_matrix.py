"""Pure scaled finite-matrix certificate for the Phase 2 balanced row."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import flint
from flint import arb

try:
    from .blaschke_deformation_phase2_geometry import upper_text
except ImportError:
    from blaschke_deformation_phase2_geometry import upper_text


MAP_LABEL = "blaschke_mu_0p3"
PRODUCER_SCHEMA = "phase2-matrix-v1"


@dataclass(frozen=True)
class Phase2MatrixConfig:
    N: int
    M: int
    precision_bits: int = 384


@dataclass(frozen=True)
class Phase2MatrixResult:
    record: dict[str, Any]
    csv_path: Path
    report_path: Path


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


def _atomic_csv(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record))
            writer.writeheader()
            writer.writerow(record)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _theta_sum(N: int, value: arb) -> arb:
    '''Explanation: Theta_N is the finite squared size of the first N scaled basis modes at a geometric factor. It converts a scalar quadrature remainder into a norm bound for the entire N-by-N matrix block.
Functionality: Evaluate the finite weighted sum Theta_N(value) with outward-rounded Arb arithmetic.'''
    square = (value * value).upper()
    power = arb(1)
    total = arb(0)
    for index in range(int(N)):
        total += arb(2 * index + 1) * power
        power *= square
    return total.upper()


def _c_gl_core(rho: arb) -> arb:
    '''Explanation: Gauss--Legendre quadrature is exponentially accurate for functions analytic on an ellipse. This core constant records the non-exponential part of that rigorous remainder estimate.
Functionality: Evaluate the certified Gauss-Legendre remainder prefactor core at rho.'''
    return (
        arb.pi()
        * (1 + rho ** (-2)).sqrt()
        * (1 + 1 / (2 * (rho * rho - 1)))
    ).upper()


def _matrix_bound(
    N: int,
    M: int,
    r: arb,
    rho: arb,
    q_gap: arb,
    phi: arb,
) -> arb:
    '''Explanation: The matrix defect is obtained by multiplying quadrature error, branch-profile size, analytic decay, and finite basis-growth factors. Their product bounds the pure scaled finite block before changing to the thesis packet gauge.
Functionality: Assemble the certified pure-scaled matrix-defect bound from the Gauss-Legendre, profile, decay, and Theta factors.'''
    return (
        arb("0.5")
        * _c_gl_core(rho)
        * phi
        * rho ** (-2 * int(M))
        * _theta_sum(N, r * rho).sqrt()
        * _theta_sum(N, q_gap).sqrt()
    ).upper()


def _configuration_digest(
    config: Phase2MatrixConfig,
    geometry: dict[str, Any],
    transport: dict[str, Any],
) -> str:
    payload = {
        "map_label": MAP_LABEL,
        "producer_schema": PRODUCER_SCHEMA,
        "N": config.N,
        "M": config.M,
        "rho": str(geometry["rho"]),
        "r": str(geometry["r_candidate"]),
        "r_tau": str(geometry["r_tau_interval_u"]),
        "q_gap": str(geometry["q_gap_target"]),
        "geometry_configuration_digest": str(geometry["configuration_digest"]),
        "transport_configuration_digest": str(transport["configuration_digest"]),
        "GL_convention": (
            "raw D_M equals one diagnostic; safe finite-M factor applied by "
            "blaschke_deformation_phase2_finite_m.py through Cell 24A"
        ),
        "coordinate_orientation": "pure r-scaled matrix D_r,N Lhat D_r,N inverse",
    }
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def certify_matrix_row(
    config: Phase2MatrixConfig,
    *,
    selected_geometry: dict[str, Any],
    transport_record: dict[str, Any],
    data_dir: Path,
    report_dir: Path,
) -> Phase2MatrixResult:
    '''Explanation: Ordinary and starred estimates are two certified ways to control the same finite matrix error. Computing both exposes which scaling is sharper and supplies the matrix component later transported by the connection condition number.
Functionality: Certify the starred and ordinary raw matrix factors in Arb.'''

    if config.N < 1 or config.M < config.N:
        raise ValueError("The matrix producer requires M at least N at least one.")
    if int(transport_record["N"]) != int(config.N):
        raise RuntimeError("The transport and matrix dimensions do not agree.")
    if str(transport_record["geometry_configuration_digest"]) != str(
        selected_geometry["configuration_digest"]
    ):
        raise RuntimeError("The transport and geometry configuration digests differ.")
    if not _is_true(transport_record.get("transport_certified", False)):
        raise RuntimeError("The matrix row cannot use an uncertified transport factor.")

    flint.ctx.prec = int(config.precision_bits)
    rho = arb(str(selected_geometry["rho"]))
    r = arb(str(selected_geometry["r_candidate"]))
    r_tau = arb(str(selected_geometry["r_tau_interval_u"])).upper()
    q_gap = arb(str(selected_geometry["q_gap_target"]))
    q_out = arb(str(selected_geometry["q_out"])).upper()
    q_star = arb(str(selected_geometry["q_star"])).upper()
    phi_star = arb(str(selected_geometry["phi_star_interval_u"])).upper()
    phi = (
        arb(str(selected_geometry["branch1_phi_u"]))
        + arb(str(selected_geometry["branch2_phi_u"]))
    ).upper()
    kappa = arb(str(transport_record["kappa_hat"])).upper()

    starred = _matrix_bound(config.N, config.M, r, rho, q_gap, phi_star)
    ordinary = _matrix_bound(config.N, config.M, r, rho, q_gap, phi)
    if not (starred <= ordinary):
        raise ArithmeticError("The starred matrix factor exceeds the ordinary factor.")
    transported_starred = (kappa * starred).upper()
    transported_ordinary = (kappa * ordinary).upper()
    two_output = arb(str(selected_geometry["two_B_out_u"])).upper()
    input_tail = arb(str(selected_geometry["branch_image_B_in_u"])).upper()
    selection_score = arb(str(selected_geometry["tail_floor_u"])).upper()
    epsilon_starred = (selection_score + transported_starred).upper()
    epsilon_ordinary = (selection_score + transported_ordinary).upper()
    configuration_digest = _configuration_digest(
        config, selected_geometry, transport_record
    )

    record = {
        "label": "branch-image balanced high-cell candidate N=600",
        "N": int(config.N),
        "M": int(config.M),
        "m": int(config.M - config.N),
        "rho": str(selected_geometry["rho"]),
        "r": str(selected_geometry["r_candidate"]),
        "r_tau_interval_u": upper_text(r_tau),
        "q_out": upper_text(q_out),
        "q_gap": str(selected_geometry["q_gap_target"]),
        "q_star": upper_text(q_star),
        "two_B_out_star_interval_u": upper_text(two_output),
        "B_in_branch_image_interval_u": upper_text(input_tail),
        "tail_floor_interval_u": upper_text(selection_score),
        "input_over_output": str(selected_geometry["input_over_output"]),
        "B_mat_star_arb_u": upper_text(starred),
        "B_mat_unstar_arb_u": upper_text(ordinary),
        "kappa_hat_transport_cert": upper_text(kappa),
        "kappa_diag": str(transport_record["kappa_diag"]),
        "kappa_Bmat_star": upper_text(transported_starred),
        "kappa_Bmat_unstar": upper_text(transported_ordinary),
        "epsilon_star_candidate": upper_text(epsilon_starred),
        "epsilon_with_unstar_matrix_candidate": upper_text(epsilon_ordinary),
        "lambda_min_cert": str(transport_record["lambda_min_cert"]),
        "lambda_max_cert": str(transport_record["lambda_max_cert"]),
        "residual_delta_cert": str(transport_record["residual_delta_cert"]),
        "norm_T_frob_cert": str(transport_record["norm_T_frob_cert"]),
        "norm_B0_frob_cert": str(transport_record["norm_B0_frob_cert"]),
        "max_abs_mu_cos_u": str(selected_geometry["max_abs_mu_cos_u"]),
        "acos_unit_disk_margin_l": str(selected_geometry["acos_unit_disk_margin_l"]),
        "min_abs_1_minus_y2_l": str(selected_geometry["min_abs_1_minus_y2_l"]),
        "basic_singularity_margin": str(selected_geometry["basic_singularity_margin"]),
        "analytic_branch_sufficient": _is_true(
            selected_geometry["analytic_branch_sufficient"]
        ),
        "branch_image_in_Er_sufficient": _is_true(
            selected_geometry["branch_image_in_Er_sufficient"]
        ),
        "admissible_chain_sufficient": _is_true(
            selected_geometry["admissible_chain_sufficient"]
        ),
        "transport_certified": True,
        "matrix_certificate_arb": True,
        "tail_components_interval": True,
        "status": "balanced candidate theorem row after branch-image refinement",
        "configuration_digest": configuration_digest,
        "geometry_configuration_digest": str(
            selected_geometry["configuration_digest"]
        ),
        "transport_configuration_digest": str(
            transport_record["configuration_digest"]
        ),
        "producer_schema": PRODUCER_SCHEMA,
        "finite_M_certified": False,
        "D_M_route": (
            "raw D_M equals one diagnostic; "
            "blaschke_deformation_phase2_finite_m.py applies the safe "
            "finite-M prefactor through Cell 24A"
        ),
    }

    data_dir = Path(data_dir)
    report_dir = Path(report_dir)
    csv_path = data_dir / (
        f"branch_image_balanced_candidate_single_space_row_N{config.N}_M{config.M}.csv"
    )
    report_path = report_dir / (
        f"branch_image_balanced_candidate_single_space_row_N{config.N}_M{config.M}.json"
    )
    _atomic_csv(csv_path, record)
    report = {
        "map_label": MAP_LABEL,
        "producer_schema": PRODUCER_SCHEMA,
        "configuration_digest": configuration_digest,
        "geometry_configuration_digest": record["geometry_configuration_digest"],
        "transport_configuration_digest": record["transport_configuration_digest"],
        "certificate": record,
        "csv_path": str(csv_path),
        "csv_sha256": _sha256(csv_path),
        "theorem_status": {
            "raw_D_M_equals_one": "diagnostic only",
            "safe_finite_M_prefactor": (
                "applied downstream by blaschke_deformation_phase2_finite_m.py "
                "through Cell 24A"
            ),
        },
    }
    _atomic_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return Phase2MatrixResult(record=record, csv_path=csv_path, report_path=report_path)


__all__ = [
    "MAP_LABEL",
    "PRODUCER_SCHEMA",
    "Phase2MatrixConfig",
    "Phase2MatrixResult",
    "certify_matrix_row",
]
