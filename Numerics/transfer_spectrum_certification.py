"""Map-independent certification utilities for the transfer-spectrum lab.

The module separates numerical evidence from theorem-level certification.  It
contains no concrete map formulae: notebook-level map specifications supply all
inverse branches, weights, exact targets, and capability metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CertificationCapability:
    """Describe which claims are justified for one notebook-supplied map."""

    target_mode: str
    target_status: str
    geometry_status: str
    deterministic_status: str
    contour_status: str
    exact_model_adapter: str | None = None
    notes: str = ""

    def as_record(self, map_label: str) -> dict[str, Any]:
        """Return a flat row suitable for a notebook status table."""
        return {"map_label": str(map_label), **asdict(self)}


def balanced_single_space_geometry(rho: float, r_tau: float) -> dict[str, float]:
    """Balance the sampled output and branch-gap exponential bases.

    The Hardy radius must exceed one.  When a sampled branch radius is below
    one, the effective lower constraint is therefore one rather than r_tau.
    """
    rho = float(rho)
    r_tau = float(r_tau)
    if not (math.isfinite(rho) and math.isfinite(r_tau) and rho > 1.0 and r_tau < rho):
        raise ValueError("an admissible geometry requires rho > 1 and r_tau < rho")
    effective_inner = max(1.0, r_tau)
    r = math.sqrt(rho * effective_inner)
    if not (1.0 < r < rho):
        r = 1.0 + 0.5 * (rho - 1.0)
    q_out = r / rho
    q_gap = r_tau / r
    return {
        "rho": rho,
        "r_tau": r_tau,
        "r": r,
        "q_out": q_out,
        "q_gap": q_gap,
        "q_star": max(q_out, q_gap),
    }


def select_sampled_geometry(
    evaluator: Callable[[float, int], Mapping[str, Any]],
    rho_candidates: Iterable[float],
    coarse_samples: int = 256,
    final_samples: int = 1024,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select the best admissible sampled geometry from candidate ellipses.

    This routine is deliberately diagnostic.  Boundary sampling does not
    replace an interval enclosure of branch images or transfer weights.
    """
    rows: list[dict[str, Any]] = []
    for rho in rho_candidates:
        row: dict[str, Any] = {"rho": float(rho), "boundary_samples": int(coarse_samples)}
        try:
            geometry = dict(evaluator(float(rho), int(coarse_samples)))
            balanced = balanced_single_space_geometry(float(rho), float(geometry["r_tau"]))
            row.update(
                {
                    "r_tau_sampled": float(geometry["r_tau"]),
                    "Phi_sampled": float(geometry.get("Phi", np.nan)),
                    "Phi_star_sampled": float(geometry.get("Phi_star", np.nan)),
                    "r_balanced": balanced["r"],
                    "q_out": balanced["q_out"],
                    "q_gap": balanced["q_gap"],
                    "q_star": balanced["q_star"],
                    "admissible_sampled": True,
                    "error": "",
                }
            )
        except Exception as exc:  # diagnostic scan records rejected candidates
            row.update(
                {
                    "r_tau_sampled": np.nan,
                    "Phi_sampled": np.nan,
                    "Phi_star_sampled": np.nan,
                    "r_balanced": np.nan,
                    "q_out": np.nan,
                    "q_gap": np.nan,
                    "q_star": np.nan,
                    "admissible_sampled": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        rows.append(row)

    scan = pd.DataFrame(rows)
    admissible = scan.loc[scan["admissible_sampled"]].sort_values("q_star")
    if admissible.empty:
        errors = "; ".join(scan["error"].dropna().astype(str).head(3))
        raise RuntimeError(f"no sampled admissible Bernstein geometry was found: {errors}")

    rho_best = float(admissible.iloc[0]["rho"])
    final_geometry = dict(evaluator(rho_best, int(final_samples)))
    balanced = balanced_single_space_geometry(rho_best, float(final_geometry["r_tau"]))
    best = {
        **balanced,
        "Phi": float(final_geometry.get("Phi", np.nan)),
        "Phi_star": float(final_geometry.get("Phi_star", np.nan)),
        "boundary_samples": int(final_samples),
        "status": "sampled_geometry_not_interval_certified",
        "method": str(final_geometry.get("method", "sampled boundary evaluation")),
    }
    return best, scan


def _theta_log(N: int, t: float) -> float:
    values = [math.log(2 * j + 1) + 2 * j * math.log(float(t)) for j in range(int(N))]
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _tail_T(N: int, q: float) -> float:
    return q**N * ((2 * N + 1) / (1 - q) + 2 * q / (1 - q) ** 2)


def _S_infty(q: float) -> float:
    return (1 + q) / (1 - q) ** 2


def _C2(radius: float) -> float:
    return math.sqrt((radius**2 + radius**-2) / (radius**2 - radius**-2))


def _Cinf(radius: float) -> float:
    return (radius + radius**-1) / (radius - radius**-1)


def _CGL_core(rho: float) -> float:
    return math.pi * math.sqrt(1 + rho**-2) * (1 + 1 / (2 * (rho**2 - 1)))


def _input_tail(N: int, r: float, r_tau: float) -> float:
    q = r_tau / r
    return (1 + r_tau ** (-2 * N)) / math.sqrt(1 - q * q) * q**N


def sampled_schur_envelope(
    N_values: Iterable[int],
    oversampling: int,
    geometry: Mapping[str, float],
    kappa: Callable[[int, float], float] | None = None,
) -> pd.DataFrame:
    """Evaluate the closed Schur fallback using sampled geometry.

    The auxiliary ZWX13 factor D_M is set to one.  Consequently these rows are
    diagnostics unless every supplied constant has a separate certified upper
    enclosure and the transport factor is certified.
    """
    rho = float(geometry["rho"])
    r_tau = float(geometry["r_tau"])
    r = float(geometry["r"])
    Phi = float(geometry["Phi"])
    q_out = r / rho
    q_gap = r_tau / r
    q_star = max(q_out, q_gap)
    response = Phi * _C2(r) * _S_infty(q_gap)
    collocation_core = 0.5 * _CGL_core(rho) * Phi
    rows: list[dict[str, Any]] = []
    for N in map(int, N_values):
        M = N + int(oversampling)
        B_out = response * _Cinf(rho) * _tail_T(N, q_out)
        B_in = Phi * _input_tail(N, r, r_tau)
        log_Bmat = (
            math.log(collocation_core)
            - 2 * M * math.log(rho)
            + 0.5 * _theta_log(N, r * rho)
            + 0.5 * _theta_log(N, q_gap)
        )
        Bmat = math.exp(log_Bmat)
        kappa_value = float(kappa(N, r)) if kappa is not None else np.nan
        transported = kappa_value * Bmat if math.isfinite(kappa_value) else Bmat
        epsilon = B_out + B_in + transported
        rows.append(
            {
                "N": N,
                "M": M,
                "rho": rho,
                "r_tau_sampled": r_tau,
                "r": r,
                "q_out": q_out,
                "q_gap": q_gap,
                "q_star": q_star,
                "Phi_sampled": Phi,
                "B_out_diagnostic": B_out,
                "B_in_diagnostic": B_in,
                "Bmat_D_M_set_to_1": Bmat,
                "kappa_T_diagnostic": kappa_value,
                "transported_Bmat_diagnostic": transported,
                "epsilon_schur_diagnostic": epsilon,
                "status": "sampled_schur_diagnostic_not_theorem_certified",
            }
        )
    return pd.DataFrame(rows)


def build_certification_audit(
    map_label: str,
    capability: CertificationCapability,
    target_count: int,
    geometry: Mapping[str, Any] | None,
    schur_rows: pd.DataFrame | None,
    moat_rows: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build a truth-preserving Riesz-rank certification ladder.

    Exact eigenvalue formulae are useful for naming contours, but they are not
    a hypothesis of the perturbative Riesz-projector argument.  For an
    unknown-spectrum map, high-resolution reference points may therefore be
    used only to propose contours.  A theorem-level conclusion still requires
    certified branch geometry, a certified operator perturbation radius, a
    validated contour-moat lower bound, a certified finite-section count, and
    a strict small-gain inequality.
    """
    geometry = dict(geometry or {})
    schur_rows = pd.DataFrame() if schur_rows is None else schur_rows
    moat_rows = pd.DataFrame() if moat_rows is None else moat_rows
    finite_counts_available = False
    finite_counts_ok = False
    finite_counts_certified = False
    sampled_moats_positive = False
    small_gain_available = False
    sampled_small_gain = False
    if not moat_rows.empty:
        if {"finite_eigenvalue_count", "multiplicity"}.issubset(moat_rows.columns):
            finite_counts = pd.to_numeric(moat_rows["finite_eigenvalue_count"], errors="coerce")
            expected_counts = pd.to_numeric(moat_rows["multiplicity"], errors="coerce")
            finite_counts_available = bool(finite_counts.notna().all() and expected_counts.notna().all())
            finite_counts_ok = finite_counts_available and bool((finite_counts == expected_counts).all())
        if "finite_count_certified" in moat_rows:
            certified_flags = moat_rows["finite_count_certified"].fillna(False).astype(bool)
            finite_counts_certified = bool(len(certified_flags) and certified_flags.all())
        moat_column = "s_gamma_H" if "s_gamma_H" in moat_rows else "s_min_gamma"
        if moat_column in moat_rows:
            sampled_moats_positive = bool((pd.to_numeric(moat_rows[moat_column], errors="coerce") > 0).all())
        small_gain_column = next(
            (column for column in ("epsilon_m_gamma", "small_gain_diagnostic") if column in moat_rows),
            None,
        )
        if small_gain_column is not None:
            small_gain = pd.to_numeric(moat_rows[small_gain_column], errors="coerce")
            small_gain_available = bool(len(small_gain) and small_gain.notna().all() and np.isfinite(small_gain).all())
            sampled_small_gain = small_gain_available and bool((small_gain < 1).all())

    geometry_available = bool(geometry) and math.isfinite(float(geometry.get("r_tau", np.nan)))
    schur_available = not schur_rows.empty and "epsilon_schur_diagnostic" in schur_rows
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
            "evidence": f"{target_count} nontrivial packets in {capability.target_mode} mode; {target_role}",
        },
        {
            "audit_item": "Bernstein branch geometry",
            "status": capability.geometry_status if geometry_available else "missing",
            "evidence": f"rho={geometry.get('rho')}, r_tau={geometry.get('r_tau')}, r={geometry.get('r')}",
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
                f"matching counts on the first {min(target_count, len(moat_rows))} contour packets; "
                f"certified_count={finite_counts_certified}"
                if finite_counts_ok
                else "finite counts disagree with expected packet ranks"
                if finite_counts_available
                else "no complete finite-count table"
            ),
        },
        {
            "audit_item": "finite-section contour moats",
            "status": capability.contour_status if sampled_moats_positive else "sampled_unresolved",
            "evidence": "positive sampled minima" if sampled_moats_positive else "no positive sampled moat table",
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
            "status": "theorem_certified" if theorem_complete else "diagnostic_not_theorem_certified",
            "evidence": (
                "The transfer operator and finite block have equal Riesz-projector rank inside every tested contour."
                if theorem_complete
                else "Missing theorem gates: " + "; ".join(missing_gates)
            ),
        }
    )
    out = pd.DataFrame(rows)
    out.insert(0, "map_label", str(map_label))
    return out
