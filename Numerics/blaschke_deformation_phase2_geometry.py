"""Clean-room Phase 2 geometry producer for ``blaschke_mu_0p3``.

The producer certifies the complete response boundary with Arb balls, evaluates
the fixed 36-point radius grid, and retains the historical analytic-tail score
only as the configuration-selection criterion.  Every angular cell is built
from ``arb.pi()``; binary floats are used only as outward compatibility fields
after the certified endpoint has been obtained.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
import csv
from fractions import Fraction
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import flint
from flint import arb, acb


MAP_LABEL = "blaschke_mu_0p3"
PRODUCER_SCHEMA = "phase2-geometry-v3"
RHO_CANDIDATES = ("2.700", "2.7125", "2.725", "2.735", "2.740", "2.745")
Q_GAP_CANDIDATES = ("0.925", "0.926", "0.927", "0.928", "0.929", "0.930")
CANONICAL_SELECTED_HARDY_RADIUS_TEXT = "2.473669807791324109321273260"
CANONICAL_SELECTED_Q_GAP_TARGET_TEXT = "0.927"


def _finite_decimal(value: Any, *, label: str) -> Decimal:
    """Parse one finite decimal without passing through binary floating point."""

    parsed = Decimal(str(value))
    if not parsed.is_finite():
        raise ValueError(f"{label} must be a finite decimal, got {value!r}.")
    return parsed


def require_canonical_selected_hardy_radius(
    value: Any,
    *,
    label: str = "Hardy radius",
) -> Decimal:
    """Require exact decimal identity with the selected Phase 2 radius.

    Equivalent decimal spellings (including the geometry producer's scientific
    notation and trailing zeros) are accepted.  A shortened decimal is not.
    """

    observed = _finite_decimal(value, label=label)
    expected = Decimal(CANONICAL_SELECTED_HARDY_RADIUS_TEXT)
    if observed != expected:
        raise ValueError(
            f"{label} does not equal the canonical selected Hardy radius: "
            f"expected {CANONICAL_SELECTED_HARDY_RADIUS_TEXT}, got {value!r}."
        )
    return observed


def exact_q_gap_contract(
    *,
    r_tau_upper: Any,
    hardy_radius: Any,
    q_gap_target: Any,
    require_canonical_radius: bool = True,
) -> dict[str, Any]:
    """Recompute ``r_tau_upper / r`` as an exact rational and gate it.

    The theorem uses the conservative target ``q_gap_target``.  It does not
    identify that target with the generally non-terminating derived quotient.
    The comparison is performed by exact integer cross multiplication, while a
    high-precision decimal rendering is retained only for inspection.
    """

    r_tau_decimal = _finite_decimal(r_tau_upper, label="r_tau upper endpoint")
    radius_decimal = (
        require_canonical_selected_hardy_radius(hardy_radius)
        if require_canonical_radius
        else _finite_decimal(hardy_radius, label="Hardy radius")
    )
    target_decimal = _finite_decimal(q_gap_target, label="q_gap target")
    if r_tau_decimal <= 0 or radius_decimal <= 0 or target_decimal <= 0:
        raise ValueError("The q_gap contract requires three positive decimals.")

    derived = Fraction(r_tau_decimal) / Fraction(radius_decimal)
    target = Fraction(target_decimal)
    if derived > target:
        raise ArithmeticError(
            "The exact derived branch-gap ratio exceeds the conservative "
            f"target: r_tau/r={derived.numerator}/{derived.denominator}, "
            f"target={q_gap_target}."
        )
    with localcontext() as context:
        context.prec = 120
        derived_decimal = Decimal(derived.numerator) / Decimal(derived.denominator)
    return {
        "q_gap_target_text": str(q_gap_target),
        "q_gap_derived_decimal_text": format(derived_decimal, ".110E"),
        "q_gap_derived_exact_numerator": str(derived.numerator),
        "q_gap_derived_exact_denominator": str(derived.denominator),
        "q_gap_derived_le_target": True,
    }


@dataclass(frozen=True)
class Phase2GeometryConfig:
    N: int = 600
    M: int = 610
    mu: str = "0.3"
    cells: int = 65536
    precision_bits: int = 192
    workers: int = 24
    rho_candidates: tuple[str, ...] = RHO_CANDIDATES
    q_gap_candidates: tuple[str, ...] = Q_GAP_CANDIDATES


@dataclass(frozen=True)
class Phase2GeometryResult:
    rows: tuple[dict[str, Any], ...]
    selected_geometry: dict[str, Any]
    csv_path: Path
    report_path: Path
    scan_digest: str


def _endpoint_decimal(value: arb, *, upper: bool) -> Decimal:
    endpoint = value.upper() if upper else value.lower()
    midpoint, radius, exponent = endpoint.mid_rad_10exp()
    midpoint_decimal = Decimal(int(midpoint))
    radius_decimal = abs(Decimal(int(radius)))
    integer = midpoint_decimal + radius_decimal if upper else midpoint_decimal - radius_decimal
    with localcontext() as context:
        context.prec = 120
        return integer * (Decimal(10) ** int(exponent))


def upper_text(value: arb) -> str:
    return format(_endpoint_decimal(value, upper=True), ".80E")


def lower_text(value: arb) -> str:
    return format(_endpoint_decimal(value, upper=False), ".80E")


def upper_float(value: arb) -> float:
    return math.nextafter(float(_endpoint_decimal(value, upper=True)), math.inf)


def lower_float(value: arb) -> float:
    return math.nextafter(float(_endpoint_decimal(value, upper=False)), -math.inf)


def _arb_from_outward_float(value: float) -> arb:
    return arb(repr(float(value))).upper()


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


def _atomic_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
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


def _tau_phi(branch: int, omega: acb, mu: arb, pi: arb):
    '''Explanation: Each Blaschke inverse branch supplies where the test function is evaluated and how strongly that value is weighted. Rigorous branch and weight enclosures are the geometric input to every Phase 2 operator bound.
Functionality: Evaluate both inverse branches of the symmetric Blaschke map and their transfer weights in Arb arithmetic.'''
    sign = -1 if int(branch) == 1 else 1
    cosine = (pi * omega / 2).cos()
    sine = (pi * omega / 2).sin()
    argument = mu * cosine
    denominator_squared = 1 - argument * argument
    denominator = denominator_squared.sqrt()
    tau = omega / 2 + sign * argument.acos() / pi
    phi = arb("0.5") + sign * mu / 2 * sine / denominator
    return tau, phi, argument, denominator_squared


def _bernstein_radius_upper(point: acb) -> arb:
    '''Explanation: The Bernstein radius of a branch image translates complex geometry into a geometric coefficient-growth factor. Bounding it from above makes every later decay ratio pessimistic in the safe direction.
Functionality: Return an outward-rounded Bernstein radius upper bound for a complex branch-image point.'''
    root = (point * point - 1).sqrt()
    positive_sheet = abs(point + root).upper()
    reciprocal_sheet = abs(point - root).upper()
    return max(positive_sheet, reciprocal_sheet).upper()


def _geometry_block(arguments: tuple[str, str, int, int, int, int]):
    '''Explanation: A complete ellipse boundary is divided into cells, and both inverse branches are enclosed on each cell. This turns a continuum maximum problem into finitely many interval statements without leaving gaps between sample points.
Functionality: Enclose both inverse-branch images and weighted branch profiles over one block of boundary cells.'''
    rho_text, mu_text, cells, precision_bits, start, stop = arguments
    flint.ctx.prec = int(precision_bits)
    rho = arb(rho_text)
    mu = arb(mu_text)
    pi = arb.pi()
    imaginary_unit = acb(0, 1)

    records: list[tuple[float, float, float, float]] = []
    max_phi_sum = arb(0)
    max_argument = arb(0)
    min_denominator_squared = None
    branch_radius = {1: arb(0), 2: arb(0)}
    branch_phi = {1: arb(0), 2: arb(0)}

    for cell in range(int(start), int(stop)):
        midpoint = pi * (2 * cell + 1) / cells
        radius = pi / cells
        theta = arb(midpoint, radius)
        zeta = rho * (imaginary_unit * theta).exp()
        omega = (zeta + 1 / zeta) / 2

        local: dict[int, tuple[arb, arb]] = {}
        phi_sum = arb(0)
        for branch in (1, 2):
            tau, phi, argument, denominator_squared = _tau_phi(
                branch, omega, mu, pi
            )
            phi_upper = abs(phi).upper()
            radius_upper = _bernstein_radius_upper(tau)
            argument_upper = abs(argument).upper()
            denominator_lower = abs(denominator_squared).lower()
            local[branch] = (phi_upper, radius_upper)
            phi_sum += phi_upper
            if phi_upper > branch_phi[branch]:
                branch_phi[branch] = phi_upper.upper()
            if radius_upper > branch_radius[branch]:
                branch_radius[branch] = radius_upper.upper()
            if argument_upper > max_argument:
                max_argument = argument_upper.upper()
            if min_denominator_squared is None or denominator_lower < min_denominator_squared:
                min_denominator_squared = denominator_lower.lower()

        if phi_sum > max_phi_sum:
            max_phi_sum = phi_sum.upper()
        records.append(
            (
                upper_float(local[1][0]),
                upper_float(local[1][1]),
                upper_float(local[2][0]),
                upper_float(local[2][1]),
            )
        )

    return {
        "start": int(start),
        "stop": int(stop),
        "records": records,
        "phi_sum_u": upper_text(max_phi_sum),
        "argument_u": upper_text(max_argument),
        "denominator_squared_l": lower_text(min_denominator_squared),
        "branch1_radius_u": upper_text(branch_radius[1]),
        "branch2_radius_u": upper_text(branch_radius[2]),
        "branch1_phi_u": upper_text(branch_phi[1]),
        "branch2_phi_u": upper_text(branch_phi[2]),
    }


def _partition(cells: int, workers: int) -> list[tuple[int, int]]:
    workers = max(1, min(int(workers), int(cells)))
    base, remainder = divmod(int(cells), workers)
    blocks = []
    start = 0
    for index in range(workers):
        width = base + (1 if index < remainder else 0)
        blocks.append((start, start + width))
        start += width
    if start != cells:
        raise AssertionError("The boundary partition does not cover every cell.")
    return blocks


def _maximum(values: Iterable[str]) -> arb:
    '''Explanation: Global analytic bounds use the worst boundary cell. Taking the largest upper endpoint preserves an enclosure of that continuum maximum after the boundary subdivision.
Functionality: Return an Arb upper enclosure of the maximum over a nonempty collection of interval quantities.'''
    maximum = arb(0)
    for value in values:
        candidate = arb(value).upper()
        if candidate > maximum:
            maximum = candidate
    return maximum.upper()


def _minimum(values: Iterable[str]) -> arb:
    '''Explanation: Admissible decay and separation conditions often depend on the least favourable boundary value. Taking the smallest lower endpoint preserves a rigorous lower bound across the complete cover.
Functionality: Return an Arb lower enclosure of the minimum over a nonempty collection of interval quantities.'''
    iterator = iter(values)
    minimum = arb(next(iterator)).lower()
    for value in iterator:
        candidate = arb(value).lower()
        if candidate < minimum:
            minimum = candidate
    return minimum.lower()


def _u_tail(N: int, hardy_radius: arb, branch_radius: arb) -> arb:
    '''Explanation: Once the branch-to-gauge ratio is below one, all unresolved input modes form a weighted geometric tail. Its closed sum is the analytic mechanism that turns truncation at N into an exponentially small error.
Functionality: Evaluate the closed geometric unresolved-input tail factor beginning at mode N.'''
    q = (branch_radius / hardy_radius).upper()
    if not (q < 1):
        raise ArithmeticError("The certified branch radius is not below the Hardy radius.")
    return (
        (1 + branch_radius ** (-2 * int(N)))
        * q ** int(N)
        / (1 - q * q).sqrt()
    ).upper()


def _c2(hardy_radius: arb) -> arb:
    '''Explanation: Holomorphy on a Bernstein ellipse forces Legendre coefficients to decay geometrically, with an L2 conversion factor depending on the ellipse radius. This factor supplies the norm-compatible constant in that coefficient estimate.
Functionality: Evaluate the ellipse L2 coefficient factor C2 at the supplied certified radius.'''
    return (
        (hardy_radius * hardy_radius + hardy_radius ** (-2))
        / (hardy_radius * hardy_radius - hardy_radius ** (-2))
    ).sqrt().upper()


def _s_infinity(q: arb) -> arb:
    '''Explanation: The complete weighted geometric series represents the accumulated contribution of all polynomial degrees. Evaluating it in closed form avoids replacing an infinite analytic bound by an arbitrary numerical cutoff.
Functionality: Evaluate the closed infinite weighted geometric sum used in the analytic tail bounds.'''
    return ((1 + q) / (1 - q) ** 2).upper()


def _sum_n_qn_from(start: int, q: arb) -> arb:
    '''Explanation: Some Legendre mode bounds carry an additional factor of the degree n. The closed tail of n times q to the n controls all such higher modes at once when q is strictly below one.
Functionality: Evaluate the closed tail sum of n q^n starting at the requested index.'''
    return (
        q ** int(start)
        * (arb(int(start)) / (1 - q) + q / (1 - q) ** 2)
    ).upper()


def _zwx13_coefficient_bound(n: int, rho: arb) -> arb:
    '''Explanation: The ZWX13 estimate converts analyticity on an ellipse into an explicit majorant for one Legendre coefficient. It is the modewise ingredient used to prove that the omitted output coordinates are small.
Functionality: Compute the ZWX13-style coefficient majorant for one Legendre mode and certified ellipse radius.'''
    integer = arb(int(n))
    exponent = arb(8 * int(n) - 1) / arb(12 * int(n) * (2 * int(n) - 1))
    bracket = arb(int(n) + 2) / arb(2 * int(n) + 3) + 1 / (rho * rho - 1)
    return (
        (arb.pi() * integer).sqrt()
        * exponent.exp()
        * rho ** (-int(n))
        * bracket
    ).upper()


def _legendre_h2_mode_bound(n: int, hardy_radius: arb) -> arb:
    '''Explanation: A single orthonormal Legendre polynomial has a computable size in the chosen Hardy/ellipse gauge. Bounding that size lets coefficient decay be summed in the same norm used for the transfer-operator perturbation.
Functionality: Bound one orthonormal Legendre mode in the Hardy/ellipse norm used by the Phase 2 tail estimate.'''
    return (
        (hardy_radius ** (2 * int(n)) + hardy_radius ** (-2 * int(n))) / 2
    ).sqrt().upper()


def _zwx13_tail_after(start: int, hardy_radius: arb, rho: arb) -> arb:
    '''Explanation: After a chosen mode, the ZWX13 coefficient and Hardy-mode estimates reduce to closed geometric tails. This controls the genuinely infinite remainder without evaluating infinitely many polynomials.
Functionality: Bound the ZWX13 output tail strictly after a selected truncation index.'''
    q = (hardy_radius / rho).upper()
    constant = (arb(1).exp() * arb.pi().sqrt() * (1 + 1 / (rho * rho - 1))).upper()
    return (constant * _sum_n_qn_from(int(start), q)).upper()


def _zwx13_tail(
    N: int,
    hardy_radius: arb,
    rho: arb,
    *,
    relative_tolerance: str = "1e-18",
    absolute_tolerance: str = "1e-80",
    minimum_extra: int = 60,
    maximum_extra: int = 900,
) -> tuple[arb, int]:
    '''Explanation: Low omitted modes are bounded individually while the far tail is summed analytically. Joining those two regimes gives a sharp but rigorous output-leakage estimate for the thesis truncation.
Functionality: Combine the retained prefix and closed remainder into the certified ZWX13 output-tail bound.'''
    relative = arb(relative_tolerance)
    absolute = arb(absolute_tolerance)
    finite_sum = arb(0)
    tail = arb("inf")
    used = 0
    for offset in range(int(maximum_extra) + 1):
        n = int(N) + offset
        finite_sum += (
            _zwx13_coefficient_bound(n, rho)
            * _legendre_h2_mode_bound(n, hardy_radius)
        ).upper()
        tail = _zwx13_tail_after(n + 1, hardy_radius, rho)
        used = offset + 1
        if offset < int(minimum_extra):
            continue
        if tail <= absolute or tail <= (relative * finite_sum).upper():
            break
    return (finite_sum + tail).upper(), int(used)


def _singularity_radius_lower(mu: arb) -> arb:
    '''Explanation: The inverse branches cease to be analytic at their nearest complex singularities. A lower bound on that singularity's Bernstein radius certifies that the chosen working ellipse lies safely inside the analytic domain.
Functionality: Certify a lower bound for the nearest branch singularity's Bernstein radius.'''
    height = (2 * (1 / mu).acosh() / arb.pi()).lower()
    return (height + (height * height + 1).sqrt()).lower()


def _configuration_digest(config: Phase2GeometryConfig) -> str:
    payload = {
        "map_label": MAP_LABEL,
        "producer_schema": PRODUCER_SCHEMA,
        **asdict(config),
    }
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _rho_boundary_data(
    config: Phase2GeometryConfig, rho_text: str, workers: int
) -> dict[str, Any]:
    '''Explanation: For one candidate Hardy radius, all branch images, weights, gaps, and tail factors must hold simultaneously over the full boundary. Aggregating the interval cover produces one auditable row of geometric hypotheses and bounds.
Functionality: Evaluate and aggregate the complete certified boundary cover for one candidate rho.'''
    blocks = _partition(config.cells, workers)
    arguments = [
        (
            str(rho_text),
            str(config.mu),
            int(config.cells),
            int(config.precision_bits),
            int(start),
            int(stop),
        )
        for start, stop in blocks
    ]
    if workers == 1:
        results = list(map(_geometry_block, arguments))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            results = list(executor.map(_geometry_block, arguments))
    results.sort(key=lambda result: result["start"])
    records = [record for result in results for record in result["records"]]
    if len(records) != config.cells:
        raise RuntimeError("The process blocks did not return the complete boundary cover.")
    return {
        "records": records,
        "phi_star_u": _maximum(result["phi_sum_u"] for result in results),
        "max_argument_u": _maximum(result["argument_u"] for result in results),
        "min_denominator_squared_l": _minimum(
            result["denominator_squared_l"] for result in results
        ),
        "branch1_radius_u": _maximum(
            result["branch1_radius_u"] for result in results
        ),
        "branch2_radius_u": _maximum(
            result["branch2_radius_u"] for result in results
        ),
        "branch1_phi_u": _maximum(result["branch1_phi_u"] for result in results),
        "branch2_phi_u": _maximum(result["branch2_phi_u"] for result in results),
        "block_count": len(results),
    }


def certify_geometry_scan(
    config: Phase2GeometryConfig,
    *,
    data_dir: Path,
    report_dir: Path,
    process_workers: int | None = None,
) -> Phase2GeometryResult:
    '''Explanation: The radius is selected from a fixed finite design only after each candidate has a complete-boundary certificate. The scan exposes the trade-off between branch-image contraction, analytic room, and tail size used in Phase 2.
Functionality: Certify and persist the fixed 36-point Phase 2 radius scan.'''

    if config.N < 1 or config.M < config.N:
        raise ValueError("The geometry producer requires M at least N at least one.")
    if config.cells < 4:
        raise ValueError("At least four exact Arb boundary cells are required.")
    if len(config.rho_candidates) * len(config.q_gap_candidates) != 36:
        raise ValueError("The theorem-facing configuration scan must contain 36 rows.")

    workers = max(
        1,
        min(
            int(process_workers if process_workers is not None else config.workers),
            24,
            os.cpu_count() or 1,
        ),
    )
    flint.ctx.prec = int(config.precision_bits)
    mu = arb(config.mu)
    singularity_radius_lower = _singularity_radius_lower(mu)
    configuration_digest = _configuration_digest(config)
    rows: list[dict[str, Any]] = []

    for rho_index, rho_text in enumerate(config.rho_candidates):
        boundary = _rho_boundary_data(config, str(rho_text), workers)
        rho = arb(str(rho_text))
        r_tau = max(boundary["branch1_radius_u"], boundary["branch2_radius_u"]).upper()
        phi_star = boundary["phi_star_u"].upper()
        max_argument = boundary["max_argument_u"].upper()
        min_denominator = boundary["min_denominator_squared_l"].lower()
        singularity_margin = (singularity_radius_lower - rho).lower()

        for q_index, q_text in enumerate(config.q_gap_candidates):
            q_target = arb(str(q_text))
            hardy_radius = (r_tau / q_target).upper()
            q_out = (hardy_radius / rho).upper()
            q_gap_cert = (r_tau / hardy_radius).upper()

            input_upper = arb(0)
            input_max_cell = 0
            for cell, record in enumerate(boundary["records"]):
                phi1, radius1, phi2, radius2 = (
                    _arb_from_outward_float(value) for value in record
                )
                local = (
                    phi1 * _u_tail(config.N, hardy_radius, radius1)
                    + phi2 * _u_tail(config.N, hardy_radius, radius2)
                ).upper()
                if local > input_upper:
                    input_upper = local
                    input_max_cell = cell

            z_tail, z_terms = _zwx13_tail(config.N, hardy_radius, rho)
            two_output = (
                2
                * phi_star
                * _c2(hardy_radius)
                * _s_infinity(q_target)
                * z_tail
            ).upper()
            tail_score = (two_output + input_upper).upper()
            q_star = max(q_out, q_target).upper()
            margin = (1 - max_argument).lower()
            analytic_ok = bool(
                max_argument < 1 and min_denominator > 0 and singularity_margin > 0
            )
            branch_image_ok = bool(r_tau < hardy_radius)
            chain_ok = bool(1 < r_tau and r_tau < hardy_radius and hardy_radius < rho)
            status = "ok" if analytic_ok and branch_image_ok and chain_ok else "inadmissible"

            r_candidate_text = upper_text(hardy_radius)
            r_tau_text = upper_text(r_tau)

            row = {
                "rho": str(rho_text),
                "q_gap_target": str(q_text),
                "r_candidate": r_candidate_text,
                "q_out": upper_text(q_out),
                "q_star": upper_text(q_star),
                "r_tau_interval_u": r_tau_text,
                "phi_star_interval_u": upper_text(phi_star),
                "two_B_out_u": upper_text(two_output),
                "branch_image_B_in_u": upper_text(input_upper),
                "tail_floor_u": upper_text(tail_score),
                "input_over_output": upper_text((input_upper / two_output).upper()),
                "max_input_cell": int(input_max_cell),
                "Z_tail_out_u": upper_text(z_tail),
                "Z_terms_used": int(z_terms),
                "cells": int(config.cells),
                "max_abs_mu_cos_u": upper_text(max_argument),
                "acos_unit_disk_margin_l": lower_text(margin),
                "min_abs_1_minus_y2_l": lower_text(min_denominator),
                "branch1_r_tau_u": upper_text(boundary["branch1_radius_u"]),
                "branch2_r_tau_u": upper_text(boundary["branch2_radius_u"]),
                "branch1_phi_u": upper_text(boundary["branch1_phi_u"]),
                "branch2_phi_u": upper_text(boundary["branch2_phi_u"]),
                "status": status,
                "nearest_basic_singularity_radius": lower_text(
                    singularity_radius_lower
                ),
                "basic_singularity_margin": lower_text(singularity_margin),
                "basic_singularity_safe": bool(singularity_margin > 0),
                "analytic_branch_sufficient": analytic_ok,
                "branch_image_in_Er_sufficient": branch_image_ok,
                "admissible_chain_sufficient": chain_ok,
                "q_gap_certified_u": upper_text(q_gap_cert),
                "configuration_digest": configuration_digest,
                "producer_schema": PRODUCER_SCHEMA,
                "boundary_blocks": int(boundary["block_count"]),
                "rho_grid_index": int(rho_index),
                "q_gap_grid_index": int(q_index),
            }
            rows.append(row)

    admissible = [row for row in rows if row["status"] == "ok"]
    if len(admissible) != 36:
        raise RuntimeError(f"Expected 36 admissible rows, obtained {len(admissible)}.")
    admissible.sort(
        key=lambda row: (
            Decimal(row["tail_floor_u"]),
            Decimal(row["q_star"]),
            int(row["rho_grid_index"]),
            int(row["q_gap_grid_index"]),
        )
    )
    selected = dict(admissible[0])
    production_geometry = bool(
        config.N == 600
        and config.M == 610
        and config.cells == 65536
        and config.precision_bits == 192
        and tuple(config.rho_candidates) == RHO_CANDIDATES
        and tuple(config.q_gap_candidates) == Q_GAP_CANDIDATES
    )
    if production_geometry:
        require_canonical_selected_hardy_radius(
            selected["r_candidate"], label="selected Phase 2 Hardy radius"
        )
        if str(selected["q_gap_target"]) != CANONICAL_SELECTED_Q_GAP_TARGET_TEXT:
            raise RuntimeError("The selected Phase 2 q_gap target is not canonical.")
        selected_q_gap_contract = exact_q_gap_contract(
            r_tau_upper=selected["r_tau_interval_u"],
            hardy_radius=selected["r_candidate"],
            q_gap_target=selected["q_gap_target"],
        )
        selected.update(selected_q_gap_contract)
    selected["selected"] = True
    selected["selection_rank"] = 1
    selected["selection_rule"] = (
        "minimum certified historical tail-floor upper; q-star and grid order are tie-breakers"
    )

    ordered_rows = []
    for rank, row in enumerate(admissible, start=1):
        item = dict(row)
        if rank == 1 and production_geometry:
            item.update(selected_q_gap_contract)
        item["selected"] = rank == 1
        item["selection_rank"] = rank
        item["selection_rule"] = selected["selection_rule"]
        ordered_rows.append(item)

    data_dir = Path(data_dir)
    report_dir = Path(report_dir)
    csv_path = data_dir / "branch_image_radius_reoptimisation_balanced_highcell_scan.csv"
    report_path = report_dir / "branch_image_radius_reoptimisation_balanced_highcell_scan.json"
    fieldnames = list(ordered_rows[0])
    _atomic_csv(csv_path, ordered_rows, fieldnames)
    scan_digest = _sha256(csv_path)
    report = {
        "map_label": MAP_LABEL,
        "producer_schema": PRODUCER_SCHEMA,
        "configuration": asdict(config),
        "configuration_digest": configuration_digest,
        "process_workers": workers,
        "complete_boundary_cover": {
            "cells": config.cells,
            "cell_formula": "theta_j = pi*(2*j+1)/C plus or minus pi/C",
            "uses_arb_pi": True,
            "covers_zero_to_two_pi": True,
        },
        "candidate_count": len(ordered_rows),
        "selection_rule": selected["selection_rule"],
        "selected_geometry": selected,
        "csv_path": str(csv_path),
        "csv_sha256": scan_digest,
    }
    _atomic_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return Phase2GeometryResult(
        rows=tuple(ordered_rows),
        selected_geometry=selected,
        csv_path=csv_path,
        report_path=report_path,
        scan_digest=scan_digest,
    )


__all__ = [
    "CANONICAL_SELECTED_HARDY_RADIUS_TEXT",
    "CANONICAL_SELECTED_Q_GAP_TARGET_TEXT",
    "MAP_LABEL",
    "PRODUCER_SCHEMA",
    "Phase2GeometryConfig",
    "Phase2GeometryResult",
    "certify_geometry_scan",
    "exact_q_gap_contract",
    "lower_float",
    "lower_text",
    "require_canonical_selected_hardy_radius",
    "upper_float",
    "upper_text",
]
