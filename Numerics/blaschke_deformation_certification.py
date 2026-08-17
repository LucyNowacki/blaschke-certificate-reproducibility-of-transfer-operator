"""Rigorous Hardy-tail certification for the ASBJ24 Blaschke deformation.

The routines in this module are deliberately map-specific.  They certify the
``blaschke_mu_0p3`` inverse branches on a complete Arb cover of the response
ellipse boundary.  They enclose the coherent resolved-response row as well as
two unresolved-input quantities:

* the exact resolved Chebyshev-packet row through a coherent prefix and a
  certified local high-mode remainder;
* the unresolved branchwise profile used by the deployed single-space theorem;
* the cancellation-preserving unresolved combined coefficient row.

All enclosures use exact Chebyshev-packet values for a finite prefix and the
explicit local Joukowski remainder from the thesis for the remaining modes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, getcontext
import math
import time
from typing import Callable

try:
    import flint
    from flint import arb, acb
except ImportError as exc:  # pragma: no cover - exercised in notebook setup
    raise ImportError(
        "blaschke_deformation_certification requires python-flint"
    ) from exc


MAP_LABEL = "blaschke_mu_0p3"


@dataclass(frozen=True)
class InputTailCertificateConfig:
    """Configuration for a complete boundary-cover certificate."""

    N: int = 600
    rho: str = "2.725"
    r: str = "2.473669807791324"
    mu: str = "0.3"
    cells: int = 65536
    precision_bits: int = 192
    prefix_terms: int = 24

    @property
    def J(self) -> int:
        return self.N + self.prefix_terms - 1


def _upper_decimal(value: arb) -> Decimal:
    mid, rad, exponent = value.upper().mid_rad_10exp()
    return (Decimal(int(mid)) + abs(Decimal(int(rad)))) * (
        Decimal(10) ** int(exponent)
    )


def upper_float(value: arb) -> float:
    """Convert an Arb upper endpoint to an outward-rounded binary float."""

    getcontext().prec = 100
    return math.nextafter(float(_upper_decimal(value)), math.inf)


def interval_text(value: arb, digits: int = 60) -> str:
    return value.str(int(digits), radius=True)


def tau_phi(branch: int, omega: acb, mu: arb, pi: arb):
    """Certified inverse branch and Perron--Frobenius weight."""

    sign = -1 if int(branch) == 1 else 1
    cosine = (pi * omega / 2).cos()
    sine = (pi * omega / 2).sin()
    argument = mu * cosine
    denominator = (1 - argument * argument).sqrt()
    tau = omega / 2 + sign * argument.acos() / pi
    phi = arb("0.5") + sign * mu / 2 * sine / denominator
    return tau, phi


def bernstein_radius_upper(point: acb) -> arb:
    """Branch-free upper bound for the outer Bernstein radius."""

    semimajor = (abs(point - 1).upper() + abs(point + 1).upper()) / 2
    if semimajor < 1:
        semimajor = arb(1)
    return (semimajor + (semimajor * semimajor - 1).sqrt()).upper()


def selected_joukowski_preimage(point: acb) -> acb:
    """Select one reciprocal preimage, favouring the outer enclosure.

    The packet expression is invariant under reciprocal replacement, so this
    pointwise choice does not assert or require a global inverse branch.
    """

    root = (point * point - 1).sqrt()
    first = point + root
    second = point - root
    first_lower = abs(first).lower()
    second_lower = abs(second).lower()
    return first if first_lower >= second_lower else second


def packet_value(m: int, point: acb, hardy_radius: arb) -> acb:
    """Exact Arb enclosure of the normalised Chebyshev packet at a point."""

    if int(m) < 1:
        raise ValueError("The unresolved-tail packet formula requires m at least one")
    denominator = (
        hardy_radius ** (2 * int(m)) + hardy_radius ** (-2 * int(m))
    ).sqrt()
    # Evaluating the polynomial itself avoids choosing a square-root branch on
    # cells meeting the ramification interval.  python-flint evaluates the
    # Chebyshev polynomial directly with complex-ball arithmetic.
    return 2 * point.chebyshev_t(int(m)) / denominator


def packet_value_from_preimage(
    m: int, preimage: acb, hardy_radius: arb
) -> acb:
    """Evaluate a packet through a certified pointwise Joukowski preimage."""

    denominator = (
        hardy_radius ** (2 * int(m)) + hardy_radius ** (-2 * int(m))
    ).sqrt()
    return (preimage**int(m) + preimage ** (-int(m))) / denominator


def local_tail_l2_upper(start: int, radius_upper: arb, hardy_radius: arb) -> arb:
    """Certified local geometric envelope for packets from ``start`` onward."""

    start = int(start)
    q = (radius_upper / hardy_radius).upper()
    if not (q < 1):
        raise ArithmeticError(
            f"Branch-image radius is not strictly below the Hardy radius: q={q}"
        )
    return (
        (1 + radius_upper ** (-2 * start))
        * q**start
        / (1 - q * q).sqrt()
    ).upper()


def certify_input_tail_rows(
    config: InputTailCertificateConfig,
    progress: Callable[[int, int], None] | None = None,
):
    """Certify branchwise and combined unresolved-input rows on the boundary.

    For every boundary cell, modes ``N`` through ``J`` are evaluated with Arb.
    The remaining modes are bounded by ``local_tail_l2_upper``.  Orthogonality
    of the finite prefix and the remainder gives a square-sum assembly.  The
    combined row retains inter-branch cancellation throughout the exact prefix.
    """

    if config.N < 1:
        raise ValueError("N must be at least one")
    if config.prefix_terms < 1:
        raise ValueError("prefix_terms must be positive")
    if config.cells < 4:
        raise ValueError("At least four boundary cells are required")

    flint.ctx.prec = int(config.precision_bits)
    getcontext().prec = 100

    N = int(config.N)
    J = int(config.J)
    cells = int(config.cells)
    rho = arb(config.rho)
    hardy_radius = arb(config.r)
    mu = arb(config.mu)
    pi = arb.pi()
    imaginary_unit = acb(0, 1)
    two_pi = 2 * pi

    branchwise_max = arb(0)
    combined_max = arb(0)
    whole_profile = []
    branchwise_max_cell = None
    combined_max_cell = None
    maximum_branch_radius = arb(0)
    start_time = time.time()

    for cell in range(cells):
        midpoint = two_pi * (2 * cell + 1) / (2 * cells)
        radius = pi / cells
        theta = arb(midpoint, radius)
        zeta = rho * (imaginary_unit * theta).exp()
        omega = (zeta + 1 / zeta) / 2

        branch_data = []
        combined_prefix = arb(0)
        for branch in (1, 2):
            tau, phi = tau_phi(branch, omega, mu, pi)
            phi_abs = abs(phi).upper()
            local_radius = bernstein_radius_upper(tau)
            preimage = selected_joukowski_preimage(tau)
            stable_preimage = abs(preimage).lower() > 1
            if local_radius > maximum_branch_radius:
                maximum_branch_radius = local_radius.upper()
            branch_data.append(
                {
                    "branch": branch,
                    "tau": tau,
                    "phi": phi,
                    "phi_abs": phi_abs,
                    "radius": local_radius,
                    "preimage": preimage,
                    "stable_preimage": stable_preimage,
                    "prefix": arb(0),
                }
            )

        for mode in range(N, J + 1):
            combined_packet = acb(0)
            for data in branch_data:
                if data["stable_preimage"]:
                    packet = packet_value_from_preimage(
                        mode, data["preimage"], hardy_radius
                    )
                else:
                    packet = packet_value(mode, data["tau"], hardy_radius)
                data["prefix"] += abs(packet).upper() ** 2
                combined_packet += data["phi"] * packet
            combined_prefix += abs(combined_packet).upper() ** 2

        branchwise_local = arb(0)
        combined_remainder = arb(0)
        record = {
            "cell": cell,
            "theta_mid": float(2 * math.pi * (cell + 0.5) / cells),
        }
        for data in branch_data:
            remainder = local_tail_l2_upper(
                J + 1, data["radius"], hardy_radius
            )
            prefix_kernel = (
                data["prefix"] + remainder * remainder
            ).sqrt().upper()
            full_envelope = local_tail_l2_upper(
                N, data["radius"], hardy_radius
            )
            # The direct local envelope and the finite-prefix construction are
            # independent upper certificates for the same exact kernel.
            kernel = (
                prefix_kernel
                if prefix_kernel <= full_envelope
                else full_envelope
            )
            branchwise_local += data["phi_abs"] * kernel
            combined_remainder += data["phi_abs"] * remainder

            branch = data["branch"]
            record[f"phi_u_b{branch}"] = upper_float(data["phi_abs"])
            record[f"s_u_b{branch}"] = upper_float(data["radius"])
            record[f"K_N_u_b{branch}"] = upper_float(kernel)
            record[f"K_N_prefix_u_b{branch}"] = upper_float(prefix_kernel)
            record[f"K_N_envelope_u_b{branch}"] = upper_float(full_envelope)
            record[f"K_remainder_u_b{branch}"] = upper_float(remainder)
            record[f"outer_preimage_stable_b{branch}"] = bool(
                data["stable_preimage"]
            )

        combined_local_raw = (
            combined_prefix + combined_remainder * combined_remainder
        ).sqrt().upper()
        branchwise_local = branchwise_local.upper()
        # Both quantities enclose the same combined row, while the branchwise
        # value follows independently from Minkowski.  Intersect the two upper
        # certificates so interval dependency in the finite combined prefix
        # can never make the reported result worse than the safe profile.
        combined_local = (
            combined_local_raw
            if combined_local_raw <= branchwise_local
            else branchwise_local
        )

        if branchwise_local > branchwise_max:
            branchwise_max = branchwise_local
            branchwise_max_cell = cell
        if combined_local > combined_max:
            combined_max = combined_local
            combined_max_cell = cell

        record["branchwise_row_u"] = upper_float(branchwise_local)
        record["combined_row_u"] = upper_float(combined_local)
        record["combined_row_raw_u"] = upper_float(combined_local_raw)
        record["combined_prefix_l2_u"] = upper_float(combined_prefix.sqrt())
        record["combined_remainder_l2_u"] = upper_float(combined_remainder)
        whole_profile.append(record)

        if progress is not None:
            progress(cell + 1, cells)

    elapsed = time.time() - start_time
    summary = {
        "map_label": MAP_LABEL,
        **asdict(config),
        "J": J,
        "branchwise_profile_cert_u": upper_float(branchwise_max),
        "combined_row_cert_u": upper_float(combined_max),
        "combined_over_branchwise_u": upper_float(
            (combined_max / branchwise_max).upper()
        ),
        "maximum_branch_radius_u": upper_float(maximum_branch_radius),
        "branchwise_max_cell": int(branchwise_max_cell),
        "combined_max_cell": int(combined_max_cell),
        "seconds": elapsed,
        "boundary_cover_certified": True,
        "exact_prefix_certified": True,
        "geometric_remainder_certified": True,
        "branchwise_profile_certified": True,
        "combined_row_certified": True,
        "status": (
            "complete Arb boundary cover with exact Chebyshev prefix and "
            "certified geometric remainder"
        ),
    }
    intervals = {
        "branchwise_profile": interval_text(branchwise_max),
        "combined_row": interval_text(combined_max),
        "maximum_branch_radius": interval_text(maximum_branch_radius),
    }
    return {
        "summary": summary,
        "intervals": intervals,
        "profile": whole_profile,
    }
@dataclass(frozen=True)
class ResolvedResponseCertificateConfig:
    """Configuration for the resolved coherent packet-row certificate."""

    N: int = 600
    rho: str = "2.725"
    r: str = "2.473669807791324"
    r_tau: str = "2.2930919118225574"
    phi_star_upper: str = "4.728315062820911"
    mu: str = "0.3"
    cells: int = 65536
    precision_bits: int = 192
    prefix_terms: int = 24

    @property
    def K(self) -> int:
        return min(int(self.N), int(self.prefix_terms))


def resolved_restriction_l2_upper(
    N: int, inner_radius: arb, hardy_radius: arb
) -> arb:
    """Exact finite Chebyshev restriction norm, enclosed from above."""

    N = int(N)
    if N < 1:
        raise ValueError("N must be at least one")
    if inner_radius <= 1 or inner_radius >= hardy_radius:
        raise ArithmeticError("The restriction radii must satisfy 1 < s < r")

    total = arb(1)  # The constant packet Psi_0 is identically one.
    for mode in range(1, N):
        numerator = inner_radius**mode + inner_radius**(-mode)
        denominator_sq = hardy_radius ** (2 * mode) + hardy_radius ** (-2 * mode)
        total += numerator * numerator / denominator_sq
    return total.sqrt().upper()


def certify_resolved_response_rows(
    config: ResolvedResponseCertificateConfig,
    progress: Callable[[int, int], None] | None = None,
):
    """Certify the coherent resolved Chebyshev-packet row on the boundary.

    Modes zero through K minus one are combined branchwise before their
    squared moduli are accumulated.  The remaining resolved coordinates are
    bounded by the infinite local packet tail beginning at K.  This is a safe
    majorant because the finite range K through N minus one is contained in
    that infinite tail.
    """

    if config.N < 1:
        raise ValueError("N must be at least one")
    if config.prefix_terms < 1:
        raise ValueError("prefix_terms must be positive")
    if config.cells < 4:
        raise ValueError("At least four boundary cells are required")

    flint.ctx.prec = int(config.precision_bits)
    getcontext().prec = 100

    N = int(config.N)
    K = int(config.K)
    cells = int(config.cells)
    rho = arb(config.rho)
    hardy_radius = arb(config.r)
    r_tau = arb(config.r_tau).upper()
    phi_star_upper = arb(config.phi_star_upper).upper()
    mu = arb(config.mu)
    pi = arb.pi()
    imaginary_unit = acb(0, 1)
    two_pi = 2 * pi

    finite_restriction = resolved_restriction_l2_upper(
        N, r_tau, hardy_radius
    )
    whole_ellipse_fallback = (
        phi_star_upper * finite_restriction
    ).upper()

    coherent_max = arb(0)
    coherent_max_cell = None
    maximum_branch_radius = arb(0)
    whole_profile = []
    start_time = time.time()

    for cell in range(cells):
        midpoint = two_pi * (2 * cell + 1) / (2 * cells)
        radius = pi / cells
        theta = arb(midpoint, radius)
        zeta = rho * (imaginary_unit * theta).exp()
        omega = (zeta + 1 / zeta) / 2

        branch_data = []
        for branch in (1, 2):
            tau, phi = tau_phi(branch, omega, mu, pi)
            local_radius = bernstein_radius_upper(tau)
            preimage = selected_joukowski_preimage(tau)
            stable_preimage = abs(preimage).lower() > 1
            if local_radius > maximum_branch_radius:
                maximum_branch_radius = local_radius.upper()
            branch_data.append(
                {
                    "branch": branch,
                    "tau": tau,
                    "phi": phi,
                    "phi_abs": abs(phi).upper(),
                    "radius": local_radius,
                    "preimage": preimage,
                    "stable_preimage": stable_preimage,
                }
            )

        coherent_prefix_sq = arb(0)
        for mode in range(K):
            combined_packet = acb(0)
            for data in branch_data:
                if mode == 0:
                    packet = acb(1)
                elif data["stable_preimage"]:
                    packet = packet_value_from_preimage(
                        mode, data["preimage"], hardy_radius
                    )
                else:
                    packet = packet_value(mode, data["tau"], hardy_radius)
                combined_packet += data["phi"] * packet
            coherent_prefix_sq += abs(combined_packet).upper() ** 2

        coherent_remainder = arb(0)
        if K < N:
            for data in branch_data:
                coherent_remainder += data["phi_abs"] * local_tail_l2_upper(
                    K, data["radius"], hardy_radius
                )
        coherent_local = (
            coherent_prefix_sq + coherent_remainder * coherent_remainder
        ).sqrt().upper()

        if coherent_local > coherent_max:
            coherent_max = coherent_local
            coherent_max_cell = cell

        record = {
            "cell": cell,
            "theta_mid": float(2 * math.pi * (cell + 0.5) / cells),
            "coherent_packet_row_u": upper_float(coherent_local),
            "coherent_prefix_l2_u": upper_float(coherent_prefix_sq.sqrt()),
            "coherent_remainder_l2_u": upper_float(coherent_remainder),
        }
        for data in branch_data:
            branch = data["branch"]
            record[f"coherent_phi_u_b{branch}"] = upper_float(data["phi_abs"])
            record[f"coherent_s_u_b{branch}"] = upper_float(data["radius"])
            record[f"coherent_outer_preimage_stable_b{branch}"] = bool(
                data["stable_preimage"]
            )
        whole_profile.append(record)

        if progress is not None:
            progress(cell + 1, cells)

    elapsed = time.time() - start_time
    summary = {
        "map_label": MAP_LABEL,
        **asdict(config),
        "K": K,
        "C_resp_coherent_packet_cert_u": upper_float(coherent_max),
        "C_resp_whole_ellipse_fallback_u": upper_float(
            whole_ellipse_fallback
        ),
        "resolved_restriction_norm_u": upper_float(finite_restriction),
        "coherent_max_cell": int(coherent_max_cell),
        "maximum_branch_radius_u": upper_float(maximum_branch_radius),
        "seconds": elapsed,
        "response_boundary_cover_certified": True,
        "response_coherent_prefix_certified": True,
        "response_remainder_certified": True,
        "response_whole_ellipse_fallback_certified": True,
        "status": (
            "complete Arb boundary cover with coherent resolved packet "
            "prefix and certified local remainder"
        ),
    }
    intervals = {
        "coherent_packet_response": interval_text(coherent_max),
        "whole_ellipse_fallback": interval_text(whole_ellipse_fallback),
        "resolved_restriction_norm": interval_text(finite_restriction),
        "maximum_branch_radius": interval_text(maximum_branch_radius),
    }
    return {
        "summary": summary,
        "intervals": intervals,
        "profile": whole_profile,
    }
