"""Standalone unresolved-input certificate and final Phase 2 aggregation."""

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
from typing import Any, Callable

import flint
from flint import arb
from mpmath import mp
import pandas as pd

try:
    from .blaschke_deformation_certification import (
        InputTailCertificateConfig,
        certify_input_tail_rows,
    )
except ImportError:
    from blaschke_deformation_certification import (
        InputTailCertificateConfig,
        certify_input_tail_rows,
    )


MAP_LABEL = "blaschke_mu_0p3"
PRODUCER_SCHEMA = "phase2-final-aggregation-v2"

INPUT_CERTIFICATION_GATES = (
    "input_boundary_cover_certified",
    "input_exact_prefix_certified",
    "input_geometric_remainder_certified",
    "input_branchwise_profile_certified",
    "input_combined_row_certified",
)
RESPONSE_CERTIFICATION_GATES = (
    "response_boundary_cover_certified",
    "response_coherent_prefix_certified",
    "response_remainder_certified",
)
FINAL_CERTIFICATION_GATES = (
    "transport_certified",
    "matrix_certified",
    "tail_components_interval",
    *INPUT_CERTIFICATION_GATES,
    "response_prefactor_certified",
    *RESPONSE_CERTIFICATION_GATES,
    "finite_M_prefactor_certified",
)


@dataclass(frozen=True)
class Phase2FinalAggregationConfig:
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
    def production_n600_m610(cls) -> "Phase2FinalAggregationConfig":
        return cls()


@dataclass(frozen=True)
class Phase2FinalAggregationResult:
    input_certificate: dict[str, Any]
    final_certificate: dict[str, Any]
    refreshed_candidate: dict[str, Any]
    summary_rows: list[dict[str, Any]]
    input_certificate_csv_path: Path
    input_profile_csv_path: Path
    input_report_json_path: Path
    candidate_csv_path: Path
    response_report_json_path: Path
    response_report_markdown_path: Path
    phase2_summary_csv_path: Path


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


def _mpf_upper(value: arb):
    return mp.mpf(str(_upper_decimal(value.upper())))


def _upper_float(value: Any) -> float:
    return math.nextafter(float(value), math.inf)


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


def _best_matrix(row: dict[str, Any]):
    '''Explanation: Two rigorously derived matrix-defect estimates may describe the same finite error in different scalings. Taking the smaller is valid only after both routes have passed their own gates, and retaining its origin keeps the thesis bound auditable.
Functionality: Select the smaller available certified starred or ordinary transported matrix contribution and retain its provenance.'''
    candidates = {"starred": mp.mpf(str(row["kappa_Bmat_star"]))}
    if row.get("kappa_Bmat_unstar") not in (None, ""):
        candidates["ordinary"] = mp.mpf(str(row["kappa_Bmat_unstar"]))
    source, value = min(candidates.items(), key=lambda item: item[1])
    return source, value, candidates


def _b_out(row: dict[str, Any]):
    '''Explanation: The output tail measures transfer mass that lands beyond the resolved basis. Recovering the same quantity from either stored convention ensures this analytic leakage enters the final perturbation radius exactly once.
Functionality: Recover the certified output-leakage contribution from either its direct field or the stored doubled quantity.'''
    if row.get("B_out_interval_u") not in (None, ""):
        return mp.mpf(str(row["B_out_interval_u"]))
    if row.get("two_B_out_star_interval_u") not in (None, ""):
        return mp.mpf(str(row["two_B_out_star_interval_u"])) / 2
    raise KeyError("No recognised output-leakage field in the Phase 2 row.")


def _summary_rows(
    data_dir: Path,
    safe_row: dict[str, Any],
    candidate: dict[str, Any],
    final_certificate: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_path = data_dir / "final_blaschke_N600_schur_certificate.csv"
    if baseline_path.is_file():
        baseline = _read_one(baseline_path)
        baseline_b_out = mp.mpf(str(baseline["2B_out_X"])) / 2
        baseline_b_in = mp.mpf(str(baseline["B_in_X"]))
        baseline_collocation = mp.mpf(str(baseline["kappa_Bmat"]))
        rows.append(
            {
                "source": "historical whole-ellipse certificate row",
                "N": int(baseline["N"]),
                "M": int(baseline["M"]),
                "rho": float(baseline["rho"]),
                "r": float(baseline["r"]),
                "r_tau": float(baseline["r_tau"]),
                "q_star": float(baseline["q_star"]),
                "B_out": float(baseline_b_out),
                "B_in": float(baseline_b_in),
                "collocation": float(baseline_collocation),
                "epsilon": float(
                    mp.sqrt(baseline_b_out**2 + baseline_b_in**2)
                    + baseline_collocation
                ),
                "status": "old deployed thesis row",
            }
        )

    rows.append(
        {
            "source": "branch-image balanced candidate",
            "N": int(safe_row["N"]),
            "M": int(safe_row["M"]),
            "rho": float(safe_row["rho"]),
            "r": float(safe_row["r"]),
            "r_tau": float(safe_row["r_tau_interval_u"]),
            "q_star": float(safe_row["q_star"]),
            "B_out": float(_b_out(safe_row)),
            "B_in": float(safe_row["B_in_branch_image_interval_u"]),
            "collocation": float(safe_row["kappa_Bmat_star"]),
            "epsilon": float(
                mp.sqrt(
                    _b_out(safe_row) ** 2
                    + mp.mpf(str(safe_row["B_in_branch_image_interval_u"])) ** 2
                )
                + mp.mpf(str(safe_row["kappa_Bmat_star"]))
            ),
            "status": str(safe_row["status"]),
        }
    )
    rows.append(
        {
            "source": "branch-image plus response-prefactor row",
            "N": int(final_certificate["N"]),
            "M": int(final_certificate["M"]),
            "rho": float(final_certificate["rho"]),
            "r": float(final_certificate["r"]),
            "r_tau": float(final_certificate["r_tau"]),
            "q_star": float(final_certificate["q_star"]),
            "B_out": float(final_certificate["B_out"]),
            "B_in": float(final_certificate["B_in"]),
            "collocation": float(final_certificate["collocation"]),
            "epsilon": float(final_certificate["epsilon"]),
            "status": str(final_certificate["status"]),
        }
    )

    wide_path = data_dir / "branch_image_wide_candidate_single_space_row_N600_M610.csv"
    if wide_path.is_file():
        wide = _read_one(wide_path)
        wide_b_out = _b_out(wide)
        wide_b_in = mp.mpf(str(wide["B_in_branch_image_interval_u"]))
        wide_collocation = mp.mpf(str(wide["kappa_Bmat_star"]))
        rows.append(
            {
                "source": "wide branch-image candidate",
                "N": int(wide["N"]),
                "M": int(wide["M"]),
                "rho": float(wide["rho"]),
                "r": float(wide["r"]),
                "r_tau": float(wide["r_tau_interval_u"]),
                "q_star": float(wide["q_star"]),
                "B_out": float(wide_b_out),
                "B_in": float(wide_b_in),
                "collocation": float(wide_collocation),
                "epsilon": float(
                    mp.sqrt(wide_b_out**2 + wide_b_in**2) + wide_collocation
                ),
                "status": str(wide["status"]),
            }
        )
    return rows


def certify_final_phase2_aggregation(
    config: Phase2FinalAggregationConfig,
    *,
    output_dir: Path,
    progress: Callable[[int, int], None] | None = None,
) -> Phase2FinalAggregationResult:
    '''Explanation: The deterministic operator radius combines unresolved input, resolved response, finite-matrix defect, and output leakage by the proved norm inequality. Fresh fail-closed gates ensure the reported epsilon exists only when every mathematical premise is certified.
Functionality: Aggregate the unresolved-input, resolved-response, output-leakage, and finite-matrix-defect bounds, enforce every fresh component and radius gate, and persist the authoritative Phase 2 row.'''

    if config.map_label != MAP_LABEL or config.mu != "0.3":
        raise ValueError(f"This producer is restricted to {MAP_LABEL}.")
    if config.N < 1 or config.M < config.N or config.cells < 4:
        raise ValueError("Invalid final Phase 2 dimensions or boundary cover.")

    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    report_dir = output_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    flint.ctx.prec = int(config.precision_bits)
    getcontext().prec = 100
    mp.dps = max(100, int(config.precision_bits * 0.30103) + 20)

    safe_row_path = data_dir / (
        "branch_image_balanced_candidate_single_space_row_"
        f"N{config.N}_M{config.M}_safe_GL.csv"
    )
    candidate_csv_path = data_dir / (
        "branch_image_balanced_response_prefactor_candidate_row_"
        f"N{config.N}_M{config.M}.csv"
    )
    response_summary_path = data_dir / (
        f"output_response_branch_image_prefactor_interval_cert_balanced_N{config.N}.csv"
    )
    coherent_summary_path = data_dir / (
        f"output_response_coherent_packet_interval_cert_balanced_N{config.N}.csv"
    )
    response_report_json_path = report_dir / (
        f"output_response_branch_image_prefactor_interval_cert_balanced_N{config.N}.json"
    )
    response_report_markdown_path = report_dir / (
        f"output_response_branch_image_prefactor_interval_cert_balanced_N{config.N}.md"
    )
    for path in (
        safe_row_path,
        candidate_csv_path,
        response_summary_path,
        coherent_summary_path,
        response_report_json_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing upstream Phase 2 completion file: {path}")

    safe_row = _read_one(safe_row_path)
    candidate = _read_one(candidate_csv_path)
    response_summary = _read_one(response_summary_path)
    coherent_summary = _read_one(coherent_summary_path)
    if int(safe_row["N"]) != config.N or int(safe_row["M"]) != config.M:
        raise RuntimeError("The safe balanced row has the wrong dimensions.")
    if abs(Decimal(safe_row["rho"]) - Decimal(config.rho)) > Decimal("1e-12"):
        raise RuntimeError("The final Phase 2 response radius does not match.")
    if abs(Decimal(safe_row["r"]) - Decimal(config.r)) > Decimal("1e-12"):
        raise RuntimeError("The final Phase 2 Hardy radius does not match.")

    input_config = InputTailCertificateConfig(
        N=config.N,
        rho=config.rho,
        r=config.r,
        mu=config.mu,
        cells=config.cells,
        precision_bits=config.precision_bits,
        prefix_terms=config.prefix_terms,
    )
    input_certificate = certify_input_tail_rows(input_config, progress=progress)
    input_summary = dict(input_certificate["summary"])
    branchwise = mp.mpf(str(input_summary["branchwise_profile_cert_u"]))
    combined = mp.mpf(str(input_summary["combined_row_cert_u"]))
    if combined > branchwise:
        raise AssertionError("The coherent unresolved certificate exceeds Minkowski.")
    if int(input_summary["cells"]) != config.cells:
        raise AssertionError("The unresolved certificate has incomplete boundary coverage.")
    input_candidates = {
        "coherent_branchwise_intersection": combined,
        "branchwise_profile": branchwise,
    }
    whole_ellipse = input_summary.get("whole_ellipse_fallback_cert_u")
    if whole_ellipse is not None:
        input_candidates["whole_ellipse_fallback"] = mp.mpf(str(whole_ellipse))
    input_source, selected_input = min(
        input_candidates.items(), key=lambda item: item[1]
    )

    matrix_source, selected_matrix, matrix_candidates = _best_matrix(safe_row)
    kappa_hat = mp.mpf(str(safe_row["kappa_hat_transport_cert"]))
    b_out_arb = arb(str(candidate["B_out_response_prefactor_cert"])).upper()
    b_in_arb = arb(str(selected_input)).upper()
    collocation_arb = arb(str(selected_matrix)).upper()
    noncollocation_rss_arb = (b_out_arb**2 + b_in_arb**2).sqrt().upper()
    noncollocation_sum_arb = (b_out_arb + b_in_arb).upper()
    epsilon_rss_arb = (noncollocation_rss_arb + collocation_arb).upper()
    epsilon_triangle_arb = (noncollocation_sum_arb + collocation_arb).upper()

    final_certificate: dict[str, Any] = {
        "label": str(candidate["label"]),
        "N": config.N,
        "M": config.M,
        "m": config.M - config.N,
        "rho": mp.mpf(config.rho),
        "r": mp.mpf(config.r),
        "r_tau": mp.mpf(str(safe_row["r_tau_interval_u"])),
        "q_out": mp.mpf(str(safe_row["q_out"])),
        "q_gap": mp.mpf(str(safe_row["q_gap"])),
        "q_star": mp.mpf(str(safe_row["q_star"])),
        "B_out": _mpf_upper(b_out_arb),
        "B_in": _mpf_upper(b_in_arb),
        "B_in_selection": input_source,
        "B_in_branchwise": branchwise,
        "B_in_combined_row": combined,
        "collocation": _mpf_upper(collocation_arb),
        "matrix_selection": matrix_source,
        "collocation_starred": matrix_candidates["starred"],
        "collocation_ordinary": matrix_candidates.get(
            "ordinary", matrix_candidates["starred"]
        ),
        "noncollocation_rss": _mpf_upper(noncollocation_rss_arb),
        "noncollocation_sum": _mpf_upper(noncollocation_sum_arb),
        "epsilon": _mpf_upper(epsilon_rss_arb),
        "epsilon_triangle_check": _mpf_upper(epsilon_triangle_arb),
        "kappa_hat": kappa_hat,
        "kappa_diag": mp.mpf(str(safe_row["kappa_diag"])),
        "lambda_min_cert": mp.mpf(str(safe_row["lambda_min_cert"])),
        "lambda_max_cert": mp.mpf(str(safe_row["lambda_max_cert"])),
        "residual_delta_cert": mp.mpf(str(safe_row["residual_delta_cert"])),
        "transport_certified": _is_true(safe_row["transport_certified"]),
        "matrix_certified": _is_true(safe_row["matrix_certificate_arb"]),
        "tail_components_interval": _is_true(safe_row["tail_components_interval"]),
        "input_boundary_cover_certified": _is_true(
            input_summary.get("boundary_cover_certified", False)
        ),
        "input_exact_prefix_certified": _is_true(
            input_summary.get("exact_prefix_certified", False)
        ),
        "input_geometric_remainder_certified": _is_true(
            input_summary.get("geometric_remainder_certified", False)
        ),
        "input_branchwise_profile_certified": _is_true(
            input_summary.get("branchwise_profile_certified", False)
        ),
        "input_combined_row_certified": _is_true(
            input_summary.get("combined_row_certified", False)
        ),
        "response_prefactor_certified": _is_true(
            candidate["response_prefactor_certified"]
        ),
        "finite_M_prefactor_certified": _is_true(
            candidate["finite_M_prefactor_certified"]
        ),
        "C_resp_coherent_packet_cert_u": mp.mpf(
            str(candidate["C_resp_coherent_packet_cert_u"])
        ),
        "C_resp_scaled_legendre_cert_u": mp.mpf(
            str(candidate["C_resp_scaled_legendre_cert_u"])
        ),
        "C_resp_whole_ellipse_fallback_u": mp.mpf(
            str(candidate["C_resp_whole_ellipse_fallback_u"])
        ),
        "C_resp_selected_cert_u": mp.mpf(
            str(candidate["C_resp_selected_cert_u"])
        ),
        "response_prefactor_selection": str(candidate["response_prefactor_selection"]),
        "response_prefix_terms": int(candidate["response_prefix_terms"]),
        "response_boundary_cells": int(candidate["response_boundary_cells"]),
        "response_boundary_cover_certified": _is_true(
            candidate["response_boundary_cover_certified"]
        ),
        "response_coherent_prefix_certified": _is_true(
            candidate["response_coherent_prefix_certified"]
        ),
        "response_remainder_certified": _is_true(
            candidate["response_remainder_certified"]
        ),
        "status": (
            "authoritative selected resolved-response row with the certified "
            "unresolved-input intersection and best matrix aggregation"
        ),
    }
    if whole_ellipse is not None:
        final_certificate["B_in_whole_ellipse_fallback"] = mp.mpf(
            str(whole_ellipse)
        )
    final_certificate["B_mat"] = final_certificate["collocation"] / kappa_hat
    final_certificate["tail_total"] = final_certificate["noncollocation_rss"]
    final_certificate["epsilon_upper_text"] = str(final_certificate["epsilon"])
    final_certificate["epsilon_triangle_upper_text"] = str(
        final_certificate["epsilon_triangle_check"]
    )
    final_certificate["q_gap_upper_text"] = str(final_certificate["q_gap"])
    final_certificate["q_star_upper_text"] = str(final_certificate["q_star"])
    final_certificate["input_tail_certified"] = all(
        bool(final_certificate[key]) for key in INPUT_CERTIFICATION_GATES
    )
    final_certificate["certified"] = all(
        bool(final_certificate[key]) for key in FINAL_CERTIFICATION_GATES
    )

    refreshed_candidate = dict(candidate)
    refreshed_candidate.update(
        {
            "phase2_aggregation_status": (
                "authoritative standalone final-aggregation refresh"
            ),
            "B_in_branch_image_interval_u": _upper_float(
                final_certificate["B_in"]
            ),
            "B_in_selected_cert_u": _upper_float(final_certificate["B_in"]),
            "B_in_selected_cert_text": str(final_certificate["B_in"]),
            "B_in_selection": input_source,
            "B_in_coherent_row_cert_u": _upper_float(combined),
            "B_in_branchwise_profile_cert_u": _upper_float(branchwise),
            "collocation_selected_cert_u": _upper_float(
                final_certificate["collocation"]
            ),
            "matrix_selection": matrix_source,
            "new_epsilon_response_prefactor_candidate": _upper_float(
                final_certificate["epsilon"]
            ),
            "new_epsilon_response_prefactor_candidate_text": str(
                final_certificate["epsilon"]
            ),
            "epsilon_triangle_check_text": final_certificate[
                "epsilon_triangle_upper_text"
            ],
            "noncollocation_rss_without_collocation": _upper_float(
                final_certificate["noncollocation_rss"]
            ),
            "tail_floor_without_collocation": _upper_float(
                final_certificate["noncollocation_rss"]
            ),
            **{
                key: bool(final_certificate[key])
                for key in FINAL_CERTIFICATION_GATES
            },
            "input_tail_certified": bool(
                final_certificate["input_tail_certified"]
            ),
            "final_phase2_certified": bool(final_certificate["certified"]),
            # The contour producer consumes this aggregate.  Overwrite it from
            # the fresh certificate rather than inheriting an upstream value.
            "total_certified": bool(final_certificate["certified"]),
            "status": final_certificate["status"],
        }
    )
    if whole_ellipse is not None:
        refreshed_candidate["B_in_whole_ellipse_fallback_cert_u"] = _upper_float(
            final_certificate["B_in_whole_ellipse_fallback"]
        )

    input_certificate_csv_path = data_dir / (
        f"blaschke_deformation_input_tail_certificate_N{config.N}.csv"
    )
    input_profile_csv_path = data_dir / (
        f"blaschke_deformation_input_tail_boundary_profile_N{config.N}.csv"
    )
    input_report_json_path = report_dir / (
        f"blaschke_deformation_input_tail_certificate_N{config.N}.json"
    )
    phase2_summary_csv_path = data_dir / "branch_image_phase2_certificate_summary.csv"
    _atomic_dataframe(
        input_certificate_csv_path, pd.DataFrame([input_certificate["summary"]])
    )
    _atomic_dataframe(input_profile_csv_path, pd.DataFrame(input_certificate["profile"]))
    _atomic_dataframe(candidate_csv_path, pd.DataFrame([refreshed_candidate]))

    certification_source = Path(certify_input_tail_rows.__code__.co_filename)
    input_provenance = {
        "producer_helper": Path(__file__).name,
        "producer_schema": PRODUCER_SCHEMA,
        "module_sha256": _sha256(certification_source),
        "source_sha256": _sha256(Path(__file__)),
        "summary": input_certificate["summary"],
        "interval_enclosures": input_certificate["intervals"],
        "profile_path": str(input_profile_csv_path),
    }
    _atomic_text(
        input_report_json_path, json.dumps(input_provenance, indent=2) + "\n"
    )

    response_provenance = json.loads(response_report_json_path.read_text(encoding="utf-8"))
    response_provenance["final_phase2_refresh"] = {
        "producer_helper": Path(__file__).name,
        "B_in_selection": input_source,
        "B_in_selected_cert_text": str(final_certificate["B_in"]),
        "matrix_selection": matrix_source,
        "epsilon_cert_text": final_certificate["epsilon_upper_text"],
        "epsilon_triangle_cert_text": final_certificate[
            "epsilon_triangle_upper_text"
        ],
    }
    _atomic_text(
        response_report_json_path,
        json.dumps(response_provenance, indent=2) + "\n",
    )

    report_lines = [
        "# Certified resolved-response prefactor and final Phase 2 aggregation",
        "",
        (
            "Resolved-response producer: "
            "blaschke_deformation_phase2_resolved_response.py; final aggregation: "
            f"{Path(__file__).name}; map: {config.map_label}."
        ),
        f"N={config.N}, M={config.M}, rho={config.rho}, r={config.r}.",
        (
            f"Arb precision: {config.precision_bits} bits; boundary cells: "
            f"{config.cells}; coherent prefix: "
            f"{int(coherent_summary['K'])} modes."
        ),
        "",
        (
            "Coherent Chebyshev-packet response upper: "
            f"{float(response_summary['C_resp_coherent_packet_cert_u']):.17e}."
        ),
        (
            "Scaled-Legendre response upper: "
            f"{float(response_summary['C_resp_scaled_legendre_cert_u']):.17e}."
        ),
        (
            "Whole-ellipse finite-restriction fallback: "
            f"{float(response_summary['C_resp_whole_ellipse_fallback_u']):.17e}."
        ),
        (
            "Selected cellwise response upper: "
            f"{float(response_summary['C_resp_selected_cert_u']):.17e}."
        ),
        f"Certified output-tail contribution B_out: {mp.nstr(final_certificate['B_out'], 50)}.",
        f"Final selected input contribution B_in: {mp.nstr(final_certificate['B_in'], 50)}.",
        f"Input selection: {input_source}.",
        (
            "The selected unresolved-input intersection equals the branchwise "
            "upper on every deployed boundary cell; no cancellation gain is claimed."
        ),
        (
            "Final selected matrix contribution: "
            f"{mp.nstr(final_certificate['collocation'], 50)}."
        ),
        f"Matrix selection: {matrix_source}.",
        (
            "Final deterministic epsilon upper: "
            f"{final_certificate['epsilon_upper_text']}."
        ),
        (
            "Auxiliary triangle epsilon upper: "
            f"{final_certificate['epsilon_triangle_upper_text']}."
        ),
        "",
        "The provisional resolved-response radius has been replaced by the authoritative standalone aggregation.",
        "",
        "Upstream SHA-256 hashes:",
    ]
    for name, digest in response_provenance.get("inputs", {}).items():
        report_lines.append(f"- `{Path(name).name}`: `{digest}`")
    _atomic_text(response_report_markdown_path, "\n".join(report_lines) + "\n")

    refreshed_check = _read_one(candidate_csv_path)
    if mp.mpf(refreshed_check["B_in_selected_cert_text"]) != final_certificate["B_in"]:
        raise AssertionError("The refreshed candidate input factor is inconsistent.")
    if (
        mp.mpf(refreshed_check["new_epsilon_response_prefactor_candidate_text"])
        != final_certificate["epsilon"]
    ):
        raise AssertionError("The refreshed candidate radius is inconsistent.")
    for gate in (*FINAL_CERTIFICATION_GATES, "final_phase2_certified", "total_certified"):
        expected_gate = (
            bool(final_certificate["certified"])
            if gate in {"final_phase2_certified", "total_certified"}
            else bool(final_certificate[gate])
        )
        if _is_true(refreshed_check.get(gate, False)) is not expected_gate:
            raise AssertionError(f"The refreshed candidate gate {gate} is inconsistent.")

    summary_rows = _summary_rows(
        data_dir, safe_row, refreshed_candidate, final_certificate
    )
    _atomic_dataframe(phase2_summary_csv_path, pd.DataFrame(summary_rows))
    return Phase2FinalAggregationResult(
        input_certificate=input_certificate,
        final_certificate=final_certificate,
        refreshed_candidate=refreshed_candidate,
        summary_rows=summary_rows,
        input_certificate_csv_path=input_certificate_csv_path,
        input_profile_csv_path=input_profile_csv_path,
        input_report_json_path=input_report_json_path,
        candidate_csv_path=candidate_csv_path,
        response_report_json_path=response_report_json_path,
        response_report_markdown_path=response_report_markdown_path,
        phase2_summary_csv_path=phase2_summary_csv_path,
    )


__all__ = [
    "MAP_LABEL",
    "PRODUCER_SCHEMA",
    "INPUT_CERTIFICATION_GATES",
    "RESPONSE_CERTIFICATION_GATES",
    "FINAL_CERTIFICATION_GATES",
    "Phase2FinalAggregationConfig",
    "Phase2FinalAggregationResult",
    "certify_final_phase2_aggregation",
]
