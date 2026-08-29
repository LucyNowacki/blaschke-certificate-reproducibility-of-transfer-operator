"""Standalone complete-boundary resolved-response completion for Phase 2."""

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
import time
from typing import Any, Callable

import flint
from flint import acb, arb
import pandas as pd

try:
    from .blaschke_deformation_certification import (
        ResolvedResponseCertificateConfig,
        certify_resolved_response_rows,
    )
except ImportError:
    from blaschke_deformation_certification import (
        ResolvedResponseCertificateConfig,
        certify_resolved_response_rows,
    )


MAP_LABEL = "blaschke_mu_0p3"
PRODUCER_SCHEMA = "phase2-resolved-response-completion-v1"


@dataclass(frozen=True)
class Phase2ResolvedResponseConfig:
    N: int = 600
    M: int = 610
    rho: str = "2.725"
    r: str = "2.473669807791324"
    mu: str = "0.3"
    cells: int = 65536
    precision_bits: int = 192
    prefix_terms: int = 24
    map_label: str = MAP_LABEL

    @classmethod
    def production_n600_m610(cls) -> "Phase2ResolvedResponseConfig":
        return cls()


@dataclass(frozen=True)
class Phase2ResolvedResponseResult:
    certificate: dict[str, Any]
    response_summary: dict[str, Any]
    response_profile: list[dict[str, Any]]
    coherent_summary: dict[str, Any]
    candidate_row: dict[str, Any]
    principal_csv_path: Path
    profile_csv_path: Path
    coherent_csv_path: Path
    candidate_csv_path: Path
    report_json_path: Path
    report_markdown_path: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_one(path: Path) -> dict[str, str]:
    rows = _read_rows(path)
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


def _interval_text(value: arb, digits: int = 60) -> str:
    return value.str(int(digits), radius=True)


def _close_decimal(left: Any, right: Any, tolerance: str = "1e-12") -> bool:
    return abs(Decimal(str(left)) - Decimal(str(right))) <= Decimal(tolerance)


def _tau_phi_extended(branch: int, omega: acb, mu: arb, pi: arb):
    '''Explanation: The response bound needs branch geometry on a slightly enlarged certified cover, not only at isolated boundary points. Enclosing the same weighted pullback there closes the continuum gap in the resolved-response estimate.
Functionality: Evaluate the inverse branches and transfer weights on the extended boundary cover used for response completion.'''
    sign = -1 if int(branch) == 1 else 1
    cosine = (pi * omega / 2).cos()
    sine = (pi * omega / 2).sin()
    argument = mu * cosine
    denominator = (1 - argument * argument).sqrt()
    tau = omega / 2 + sign * argument.acos() / pi
    phi = arb("0.5") + sign * mu / 2 * sine / denominator
    return tau, phi, argument, denominator


def _bernstein_radius_upper(point: acb) -> arb:
    '''Explanation: On the extended cover, the branch image radius still controls packet growth. An outward upper enclosure guarantees that the local geometric remainder is valid for every point represented by the cell.
Functionality: Return an outward-rounded Bernstein radius upper bound for an extended-cover branch image.'''
    semimajor = (abs(point - 1).upper() + abs(point + 1).upper()) / 2
    if semimajor < 1:
        semimajor = arb(1)
    return (semimajor + (semimajor * semimajor - 1).sqrt()).upper()


def _legendre_kernel(N: int, branch_radius_upper: arb, hardy_radius: arb):
    '''Explanation: The finite Christoffel kernel is the squared norm of point evaluation on the first N Legendre modes. It turns evaluation of a resolved polynomial into a sharp finite-dimensional L2 bound.
Functionality: Evaluate the finite orthonormal Legendre Christoffel kernel at the supplied complex point.'''
    q = (branch_radius_upper / hardy_radius).upper()
    if not (q < 1):
        raise ArithmeticError(f"Branch-image radius is not below r: q={q}")
    x = q * q
    finite_sum = (
        1
        + x
        - (2 * int(N) + 1) * x ** int(N)
        + (2 * int(N) - 1) * x ** (int(N) + 1)
    ) / (1 - x) ** 2
    return (finite_sum / 2).sqrt().upper(), q


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


def certify_resolved_response_completion(
    config: Phase2ResolvedResponseConfig,
    *,
    output_dir: Path,
    progress: Callable[[str, int, int], None] | None = None,
) -> Phase2ResolvedResponseResult:
    '''Explanation: This completion bounds how the resolved input block responds under both weighted inverse branches over the whole boundary. It supplies the missing finite response factor required to combine the input tail with the transfer action.
Functionality: Certify and persist the complete resolved-response completion stage.'''

    if config.map_label != MAP_LABEL or config.mu != "0.3":
        raise ValueError(f"This producer is restricted to {MAP_LABEL}.")
    if config.N < 1 or config.M < config.N or config.cells < 4:
        raise ValueError("Invalid resolved-response dimensions or boundary cover.")

    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    report_dir = output_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    flint.ctx.prec = int(config.precision_bits)
    getcontext().prec = 100

    balanced_path = data_dir / (
        "branch_image_balanced_candidate_single_space_row_"
        f"N{config.N}_M{config.M}_safe_GL.csv"
    )
    transport_path = data_dir / (
        f"branch_image_balanced_candidate_transport_cert_N{config.N}.csv"
    )
    tail_path = data_dir / "branch_image_radius_reoptimisation_balanced_highcell_scan.csv"
    for path in (balanced_path, transport_path, tail_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing upstream Phase 2 certificate: {path}")

    balanced = _read_one(balanced_path)
    if int(balanced["N"]) != config.N or int(balanced["M"]) != config.M:
        raise RuntimeError("The safe balanced row has the wrong dimensions.")
    if not _close_decimal(balanced["rho"], config.rho):
        raise RuntimeError("The safe balanced row has the wrong response radius.")
    if not _close_decimal(balanced["r"], config.r):
        raise RuntimeError("The safe balanced row has the wrong Hardy radius.")
    for flag in (
        "matrix_certificate_arb",
        "tail_components_interval",
        "finite_M_prefactor_certified",
    ):
        if not _is_true(balanced.get(flag, False)):
            raise RuntimeError(f"The safe balanced row failed {flag}.")

    transport_candidates = [
        row for row in _read_rows(transport_path) if int(row["N"]) == config.N
    ]
    if len(transport_candidates) != 1:
        raise RuntimeError("Expected one matching certified transport row.")
    transport = transport_candidates[0]
    if not _is_true(transport.get("transport_certified", False)):
        raise RuntimeError("The finite transport inverse is not certified.")

    tail_candidates = [
        row
        for row in _read_rows(tail_path)
        if int(row["cells"]) >= config.cells
        and _close_decimal(row["rho"], config.rho)
        and _close_decimal(row["r_candidate"], config.r)
        and str(row.get("status", "")).strip().lower() == "ok"
    ]
    if not tail_candidates:
        raise RuntimeError("No matching certified output-tail row was found.")
    tail = min(tail_candidates, key=lambda row: int(row["cells"]))
    for flag in (
        "analytic_branch_sufficient",
        "branch_image_in_Er_sufficient",
        "admissible_chain_sufficient",
    ):
        if not _is_true(tail.get(flag, False)):
            raise RuntimeError(f"The branch-image row failed {flag}.")

    rho = arb(config.rho)
    hardy_radius = arb(config.r)
    q_gap = arb(balanced["q_gap"])
    r_tau_upper = arb(tail["r_tau_interval_u"]).upper()
    phi_star_upper = arb(tail["phi_star_interval_u"]).upper()
    pi = arb.pi()
    imaginary_unit = acb(0, 1)
    two_pi = 2 * pi

    lambda_min_lower = arb(transport["lambda_min_cert"])
    if lambda_min_lower <= 0:
        raise ArithmeticError("The lower transport singular-value bound is not positive.")
    t_inverse_upper = (1 / lambda_min_lower.sqrt()).upper()
    z_tail_upper = arb(tail["Z_tail_out_u"]).upper()

    resolved_config = ResolvedResponseCertificateConfig(
        N=config.N,
        rho=config.rho,
        r=config.r,
        r_tau=tail["r_tau_interval_u"],
        phi_star_upper=tail["phi_star_interval_u"],
        mu=config.mu,
        cells=config.cells,
        precision_bits=config.precision_bits,
        prefix_terms=config.prefix_terms,
    )

    def coherent_progress(done: int, total: int) -> None:
        if progress is not None:
            progress("coherent-response", done, total)

    certificate = certify_resolved_response_rows(
        resolved_config, progress=coherent_progress
    )
    coherent_summary = dict(certificate["summary"])
    coherent_by_cell = {
        int(row["cell"]): row for row in certificate["profile"]
    }
    if len(coherent_by_cell) != config.cells:
        raise AssertionError("The coherent response profile does not cover every cell.")

    whole_fallback = arb(
        str(coherent_summary["C_resp_whole_ellipse_fallback_u"])
    ).upper()
    coherent_global = arb(
        str(coherent_summary["C_resp_coherent_packet_cert_u"])
    ).upper()
    profile: list[dict[str, Any]] = []
    scaled_branch_row_upper = arb(0)
    scaled_response_upper = arb(0)
    selected_response_upper = arb(0)
    maximum_q = arb(0)
    maximum_phi_sum = arb(0)
    maximum_argument_abs = arb(0)
    minimum_denominator_abs = None
    scaled_max_cell = None
    selected_max_cell = None
    selection_counts = {
        "coherent_packet": 0,
        "scaled_legendre": 0,
        "whole_ellipse_fallback": 0,
    }
    start_time = time.time()

    for cell in range(config.cells):
        midpoint = two_pi * (2 * cell + 1) / (2 * config.cells)
        radius = pi / config.cells
        theta = arb(midpoint, radius)
        zeta = rho * (imaginary_unit * theta).exp()
        omega = (zeta + 1 / zeta) / 2
        scaled_branch_local = arb(0)
        phi_sum = arb(0)
        record: dict[str, Any] = {
            "cell": cell,
            "theta_mid": float(2 * math.pi * (cell + 0.5) / config.cells),
        }

        for branch in (1, 2):
            tau, phi, argument, denominator = _tau_phi_extended(
                branch, omega, arb(config.mu), pi
            )
            phi_upper = abs(phi).upper()
            radius_upper = _bernstein_radius_upper(tau)
            kernel, q = _legendre_kernel(config.N, radius_upper, hardy_radius)
            scaled_branch_local += phi_upper * kernel
            phi_sum += phi_upper
            argument_abs = abs(argument).upper()
            denominator_abs = abs(denominator).lower()
            maximum_argument_abs = max(maximum_argument_abs, argument_abs)
            if minimum_denominator_abs is None or denominator_abs < minimum_denominator_abs:
                minimum_denominator_abs = denominator_abs
            maximum_q = max(maximum_q, q)
            record[f"phi_u_b{branch}"] = _upper_float(phi_upper)
            record[f"s_u_b{branch}"] = _upper_float(radius_upper)
            record[f"q_u_b{branch}"] = _upper_float(q)
            record[f"scaled_legendre_kernel_u_b{branch}"] = _upper_float(kernel)

        scaled_branch_local = scaled_branch_local.upper()
        scaled_local = (scaled_branch_local * t_inverse_upper).upper()
        coherent_record = coherent_by_cell[cell]
        coherent_local = arb(str(coherent_record["coherent_packet_row_u"])).upper()
        candidates = {
            "coherent_packet": coherent_local,
            "scaled_legendre": scaled_local,
            "whole_ellipse_fallback": whole_fallback,
        }
        selected_route, selected_local = min(candidates.items(), key=lambda item: item[1])
        selected_local = selected_local.upper()
        selection_counts[selected_route] += 1
        if scaled_branch_local > scaled_branch_row_upper:
            scaled_branch_row_upper = scaled_branch_local
            scaled_max_cell = cell
        scaled_response_upper = max(scaled_response_upper, scaled_local)
        if selected_local > selected_response_upper:
            selected_response_upper = selected_local
            selected_max_cell = cell
        maximum_phi_sum = max(maximum_phi_sum, phi_sum.upper())
        record.update(coherent_record)
        record.update(
            {
                "phi_sum_u": _upper_float(phi_sum),
                "scaled_legendre_branch_row_u": _upper_float(scaled_branch_local),
                "scaled_legendre_response_u": _upper_float(scaled_local),
                "whole_ellipse_fallback_u": _upper_float(whole_fallback),
                "selected_response_u": _upper_float(selected_local),
                "selected_response_route": selected_route,
            }
        )
        profile.append(record)
        if progress is not None:
            progress("scaled-response", cell + 1, config.cells)

    seconds = time.time() - start_time + float(coherent_summary["seconds"])
    if selected_response_upper > min(
        coherent_global, scaled_response_upper, whole_fallback
    ):
        raise AssertionError("The selected response exceeds an independent certificate.")
    if minimum_denominator_abs is None:
        raise AssertionError("No branch denominator was evaluated.")

    b_out = (selected_response_upper * z_tail_upper).upper()
    c2 = ((hardy_radius**2 + hardy_radius**-2) / (
        hardy_radius**2 - hardy_radius**-2
    )).sqrt()
    s_inf = (1 + q_gap) / (1 - q_gap) ** 2
    old_prefactor = (phi_star_upper * c2 * s_inf).upper()
    old_b_out = (old_prefactor * z_tail_upper).upper()
    improvement = (old_b_out / b_out).upper()

    selection_text = (
        "cellwise minimum of coherent packet, scaled Legendre, and "
        "whole-ellipse fallback certificates"
    )
    response_summary = {
        "map_label": config.map_label,
        "N": config.N,
        "rho": float(config.rho),
        "r": float(config.r),
        "cells": config.cells,
        "precision_bits": config.precision_bits,
        "output_schema_version": 3,
        "output_leakage_count": 1,
        "C_resp_coherent_packet_cert_u": _upper_float(coherent_global),
        "C_resp_scaled_legendre_cert_u": _upper_float(scaled_response_upper),
        "C_resp_whole_ellipse_fallback_u": _upper_float(whole_fallback),
        "C_resp_selected_cert_u": _upper_float(selected_response_upper),
        "response_prefactor_selection": selection_text,
        "response_prefix_terms": int(coherent_summary["K"]),
        "response_boundary_cells": config.cells,
        "response_selection_coherent_cells": selection_counts["coherent_packet"],
        "response_selection_scaled_cells": selection_counts["scaled_legendre"],
        "response_selection_fallback_cells": selection_counts[
            "whole_ellipse_fallback"
        ],
        "scaled_legendre_branch_row_cert_u": _upper_float(scaled_branch_row_upper),
        "branch_image_response_prefactor_interval_u": _upper_float(
            scaled_branch_row_upper
        ),
        "branch_image_response_prefactor_interval_role": (
            "compatibility alias for the untransported scaled-Legendre branch row"
        ),
        "max_branch_q_interval_u": _upper_float(maximum_q),
        "max_phi_sum_interval_u": _upper_float(maximum_phi_sum),
        "max_abs_mu_cos_interval_u": _upper_float(maximum_argument_abs),
        "min_abs_denominator_interval_l": _lower_float(minimum_denominator_abs),
        "scaled_legendre_max_cell": int(scaled_max_cell),
        "selected_max_cell": int(selected_max_cell),
        "lambda_min_cert": float(transport["lambda_min_cert"]),
        "T_inverse_norm_cert": _upper_float(t_inverse_upper),
        "response_prefactor_with_Tinv_cert": _upper_float(scaled_response_upper),
        "response_prefactor_with_Tinv_role": (
            "compatibility alias for the complete scaled-Legendre candidate"
        ),
        "Z_tail_N_inferred": _upper_float(z_tail_upper),
        "B_out_response_prefactor_cert": _upper_float(b_out),
        "old_prefactor": _upper_float(old_prefactor),
        "old_B_out_comparator": _upper_float(old_b_out),
        "prefactor_improvement_vs_old": _upper_float(improvement),
        "B_out_improvement_vs_old": _upper_float(improvement),
        "seconds": seconds,
        "response_boundary_cover_certified": True,
        "response_coherent_prefix_certified": True,
        "response_remainder_certified": True,
        "response_scaled_legendre_certified": True,
        "response_whole_ellipse_fallback_certified": True,
        "response_prefactor_certified": True,
        "finite_M_prefactor_certified": True,
        "status": (
            "interval-certified coherent resolved response with independent "
            "scaled-Legendre and whole-ellipse fallbacks"
        ),
    }

    old_b_out_comparator = arb(balanced["B_out_interval_u"]).upper()
    b_in = arb(balanced["B_in_branch_image_interval_u"]).upper()
    collocation = arb(balanced["kappa_Bmat_star"]).upper()
    old_noncollocation = (old_b_out_comparator**2 + b_in**2).sqrt().upper()
    new_noncollocation = (b_out**2 + b_in**2).sqrt().upper()
    old_epsilon = (old_noncollocation + collocation).upper()
    new_epsilon = (new_noncollocation + collocation).upper()
    candidate = {
        "label": (
            "balanced branch-image plus selected certified resolved-response "
            f"prefactor N={config.N}"
        ),
        "N": config.N,
        "M": config.M,
        "rho": float(config.rho),
        "r": float(config.r),
        "q_gap": float(balanced["q_gap"]),
        "q_out": float(balanced["q_out"]),
        "q_star": float(balanced["q_star"]),
        "output_schema_version": 3,
        "output_leakage_count": 1,
        "old_B_out_comparator": _upper_float(old_b_out_comparator),
        "B_out_response_prefactor_cert": _upper_float(b_out),
        "C_resp_coherent_packet_cert_u": _upper_float(coherent_global),
        "C_resp_scaled_legendre_cert_u": _upper_float(scaled_response_upper),
        "C_resp_whole_ellipse_fallback_u": _upper_float(whole_fallback),
        "C_resp_selected_cert_u": _upper_float(selected_response_upper),
        "response_prefactor_selection": selection_text,
        "response_prefix_terms": int(coherent_summary["K"]),
        "response_boundary_cells": config.cells,
        "response_boundary_cover_certified": True,
        "response_coherent_prefix_certified": True,
        "response_remainder_certified": True,
        "B_in_branch_image_interval_u": _upper_float(b_in),
        "kappa_Bmat_star": _upper_float(collocation),
        "old_epsilon_star_candidate": _upper_float(old_epsilon),
        "new_epsilon_response_prefactor_candidate": _upper_float(new_epsilon),
        "new_over_old_balanced_candidate": _upper_float(new_epsilon / old_epsilon),
        "output_improvement_factor": _upper_float(improvement),
        "noncollocation_rss_without_collocation": _upper_float(new_noncollocation),
        "tail_floor_without_collocation": _upper_float(new_noncollocation),
        "response_prefactor_certified": True,
        "finite_M_prefactor_certified": True,
        "total_certified": True,
        "phase2_aggregation_status": "provisional before Cell 24C input refresh",
        "status": (
            "unconditional selected resolved-response single-space row with "
            "safe finite-M certificate"
        ),
    }

    principal_csv_path = data_dir / (
        f"output_response_branch_image_prefactor_interval_cert_balanced_N{config.N}.csv"
    )
    profile_csv_path = data_dir / (
        f"output_response_branch_image_prefactor_interval_profile_balanced_N{config.N}.csv"
    )
    coherent_csv_path = data_dir / (
        f"output_response_coherent_packet_interval_cert_balanced_N{config.N}.csv"
    )
    candidate_csv_path = data_dir / (
        "branch_image_balanced_response_prefactor_candidate_row_"
        f"N{config.N}_M{config.M}.csv"
    )
    report_json_path = report_dir / (
        f"output_response_branch_image_prefactor_interval_cert_balanced_N{config.N}.json"
    )
    report_markdown_path = report_dir / (
        f"output_response_branch_image_prefactor_interval_cert_balanced_N{config.N}.md"
    )
    _atomic_dataframe(principal_csv_path, pd.DataFrame([response_summary]))
    _atomic_dataframe(profile_csv_path, pd.DataFrame(profile))
    _atomic_dataframe(coherent_csv_path, pd.DataFrame([coherent_summary]))
    _atomic_dataframe(candidate_csv_path, pd.DataFrame([candidate]))

    certification_source = Path(certify_resolved_response_rows.__code__.co_filename)
    provenance = {
        "producer_helper": Path(__file__).name,
        "producer_schema": PRODUCER_SCHEMA,
        "method": (
            "complete Arb boundary cover; coherent Chebyshev prefix plus "
            "certified local remainder; independent scaled-Legendre and "
            "whole-ellipse candidates; cellwise certified minimum"
        ),
        "output_schema": {"version": 3, "output_leakage_count": 1},
        "inputs": {
            str(balanced_path): _sha256(balanced_path),
            str(transport_path): _sha256(transport_path),
            str(tail_path): _sha256(tail_path),
            certification_source.name: _sha256(certification_source),
        },
        "source_sha256": _sha256(Path(__file__)),
        "summary": response_summary,
        "interval_enclosures": {
            "coherent_response_interval": certificate["intervals"][
                "coherent_packet_response"
            ],
            "scaled_legendre_response_interval": _interval_text(
                scaled_response_upper
            ),
            "whole_ellipse_fallback_interval": _interval_text(whole_fallback),
            "selected_response_interval": _interval_text(selected_response_upper),
            "B_out_interval": _interval_text(b_out),
        },
        "profile_path": str(profile_csv_path),
        "coherent_summary_path": str(coherent_csv_path),
    }
    _atomic_text(report_json_path, json.dumps(provenance, indent=2) + "\n")
    report_lines = [
        "# Certified resolved-response prefactor",
        "",
        f"Producer: {Path(__file__).name}; map: {config.map_label}.",
        f"N={config.N}, M={config.M}, rho={config.rho}, r={config.r}.",
        (
            f"Arb precision: {config.precision_bits} bits; boundary cells: "
            f"{config.cells}; coherent prefix: {int(coherent_summary['K'])} modes."
        ),
        "",
        f"Coherent Chebyshev-packet response upper: {_upper_float(coherent_global):.17e}.",
        f"Scaled-Legendre response upper: {_upper_float(scaled_response_upper):.17e}.",
        f"Whole-ellipse finite-restriction fallback: {_upper_float(whole_fallback):.17e}.",
        f"Selected cellwise response upper: {_upper_float(selected_response_upper):.17e}.",
        f"Selection policy: {selection_text}.",
        f"Selection counts: {json.dumps(selection_counts, sort_keys=True)}.",
        f"Certified output-tail contribution B_out: {_upper_float(b_out):.17e}.",
        f"Provisional deterministic epsilon before Cell 24C: {_upper_float(new_epsilon):.17e}.",
        "",
        (
            "Every selected value is the minimum of complete independent "
            "certificates for the same response row on the same Arb cell."
        ),
        "",
        "Upstream SHA-256 hashes:",
    ]
    for name, digest in provenance["inputs"].items():
        report_lines.append(f"- `{Path(name).name}`: `{digest}`")
    _atomic_text(report_markdown_path, "\n".join(report_lines) + "\n")

    return Phase2ResolvedResponseResult(
        certificate=certificate,
        response_summary=response_summary,
        response_profile=profile,
        coherent_summary=coherent_summary,
        candidate_row=candidate,
        principal_csv_path=principal_csv_path,
        profile_csv_path=profile_csv_path,
        coherent_csv_path=coherent_csv_path,
        candidate_csv_path=candidate_csv_path,
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
    )


__all__ = [
    "MAP_LABEL",
    "PRODUCER_SCHEMA",
    "Phase2ResolvedResponseConfig",
    "Phase2ResolvedResponseResult",
    "certify_resolved_response_completion",
]

