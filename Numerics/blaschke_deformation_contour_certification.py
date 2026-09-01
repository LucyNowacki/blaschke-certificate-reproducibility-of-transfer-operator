"""Validated contour certification for the ASBJ24 Blaschke deformation.

The producer starts from the exact-dyadic Hardy-gauge midpoint certified by
``blaschke_deformation_spectral_certification``.  All finite algebraic counts
are obtained from strict inclusion tests on the diagonal of the exact-binary
upper-triangular Schur matrix.  Seventeen complete-circle moats are obtained
from the triangular Schur resolvent recurrence; the seven deeper moats are
obtained from validated Laurent approximate inverses in Schur coordinates.
Sampled inverses only propose Laurent coefficients; all theorem gates use Arb
residuals against the exact-dyadic matrix.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
from decimal import Decimal
from fractions import Fraction
import gc
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import platform
import sys
import time
from typing import Iterable

import numpy as np
import scipy
from scipy import linalg
from threadpoolctl import threadpool_info, threadpool_limits

try:
    import flint
    from flint import acb, acb_mat, arb
except ImportError as exc:  # pragma: no cover - checked by the notebook
    raise ImportError(
        "blaschke_deformation_contour_certification requires python-flint"
    ) from exc

from blaschke_deformation_spectral_certification import (
    CANONICAL_SELECTED_HARDY_RADIUS_TEXT,
    logical_source_hashes,
    load_exact_dyadic_midpoint,
    normalise_source_hashes,
    require_canonical_selected_hardy_radius,
    sha256_file,
)
try:
    from .blaschke_deformation_phase2_geometry import (
        CANONICAL_SELECTED_Q_GAP_TARGET_TEXT,
        exact_q_gap_contract,
    )
except ImportError:
    from blaschke_deformation_phase2_geometry import (
        CANONICAL_SELECTED_Q_GAP_TARGET_TEXT,
        exact_q_gap_contract,
    )


MAP_LABEL = "blaschke_mu_0p3"
SCHEMA = "blaschke-deformation-24-contour-hybrid-v4"
SCHUR_SCHEMA = "blaschke-deformation-exact-dyadic-schur-v2"

COUNT_METHOD_SCHUR_DIAGONAL = "certified Schur-diagonal algebraic count"
MOAT_METHOD_SCHUR_TRIANGULAR = (
    "uniform complete-circle triangular Schur resolvent"
)
MOAT_METHOD_LAURENT = (
    "complete-circle Laurent approximate inverse in Schur coordinates"
)
CERTIFICATE_ROUTE_SCHUR = "Schur-count/Schur-moat"
CERTIFICATE_ROUTE_LAURENT = "Schur-count/Laurent-moat"
COUNT_REFERENCE_MATRIX = "exact-binary upper-triangular Schur matrix T"
LEGACY_5C0_COMPATIBILITY_PRECISION_BITS = 53

# A deterministic epsilon is theorem-eligible only when every fresh Phase 2
# component gate is present and true. Missing columns fail closed, preventing
# an inherited aggregate flag from authorising contour reuse.
REQUIRED_EPSILON_CERTIFICATION_GATES = (
    "transport_certified",
    "matrix_certified",
    "tail_components_interval",
    "q_gap_derived_le_target",
    "input_boundary_cover_certified",
    "input_exact_prefix_certified",
    "input_geometric_remainder_certified",
    "input_branchwise_profile_certified",
    "input_combined_row_certified",
    "input_tail_certified",
    "response_prefactor_certified",
    "response_boundary_cover_certified",
    "response_coherent_prefix_certified",
    "response_remainder_certified",
    "finite_M_prefactor_certified",
    "final_phase2_certified",
    "total_certified",
)


@dataclass(frozen=True)
class ContourCertificateConfig:
    """Arithmetic and contour settings for the fixed production run."""

    N: int = 600
    M: int = 610
    rho: str = "2.725"
    r: str = CANONICAL_SELECTED_HARDY_RADIUS_TEXT
    mu_numerator: int = 3
    mu_denominator: int = 10
    alpha_numerator: int = 13
    alpha_denominator: int = 20
    alpha_power_count: int = 18
    mu_power_count: int = 6
    default_radius_numerator: int = 1
    default_radius_denominator: int = 5
    precision_bits: int = 256
    flint_threads: int = 24


@dataclass(frozen=True)
class TargetContour:
    """One exact rational target circle."""

    rank: int
    name: str
    family: str
    power: int
    expected_multiplicity: int
    centre: Fraction
    nearest_target_separation: Fraction
    radius_fraction: Fraction
    radius: Fraction
    laurent_sample_count: int | None


# These are proposal parameters only.  Every selected Laurent polynomial is
# subsequently validated against the exact-dyadic matrix in Arb.
_LAURENT_PROPOSALS: dict[str, tuple[Fraction, int]] = {
    "mu^5": (Fraction(1, 4), 44),
    "alpha^14": (Fraction(9, 50), 36),
    "alpha^15": (Fraction(1, 5), 48),
    "alpha^16": (Fraction(1, 5), 48),
    "mu^6": (Fraction(1, 4), 36),
    "alpha^17": (Fraction(1, 4), 40),
    "alpha^18": (Fraction(1, 5), 48),
}

# Historical full-tensor digests are provenance checks only.  They do not enter
# any Schur count, Laurent residual, moat, homotopy or small-gain theorem gate.
_LAURENT_REFERENCE_DIGESTS: dict[str, str] = {
    "mu^5": "5ff9d98a62ed213189da64d7ace168a555816a6b32b8d750fbce0d196a9cd982",
    "alpha^14": "00a85f99c6369d6a2364b9e9626b4216d64da9646fda9599e56dd21c9f32f3ab",
    "alpha^15": "4da5d5813112f4e74550c35bd8e72bd00fee1d5d6978098a93a1dbeb313fe02a",
    "alpha^16": "9ada35489c32a58968c11f5bba84929caed6470247568e865d350701bb3f7d23",
    "mu^6": "9a6809ba73c82b5b64cfba6a22a35465dade23f5327f47189e8fc83e2a1c7324",
    "alpha^17": "cf57966aca93efcc91ecc727e82199f08a3098a404b9c616cdaaac78f12ec42c",
    "alpha^18": "e774344bd462abfca87b819e3de75acbe3e10c3e7cea2015814a44a95c6c6a17",
}


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _fraction_arb(value: Fraction) -> arb:
    return arb(value.numerator) / arb(value.denominator)


def _fraction_acb(value: Fraction) -> acb:
    return acb(_fraction_arb(value))


def _exact_float(value: float) -> arb:
    numerator, denominator = float(value).as_integer_ratio()
    return arb(numerator) / arb(denominator)


def _exact_complex128(value: complex) -> acb:
    value = complex(value)
    return acb(_exact_float(value.real), _exact_float(value.imag))


def _numpy_to_acb_exact(matrix: np.ndarray) -> acb_mat:
    '''Explanation: Every binary64 number is an exact dyadic rational. Embedding those dyadics exactly in ACB prevents an unrecorded conversion error from entering the finite matrix used in the Riesz-projector proof.
Functionality: Convert every binary64 complex matrix entry exactly to an Arb/ACB ball and return the resulting matrix.'''
    matrix = np.asarray(matrix, dtype=np.complex128)
    rows, columns = matrix.shape
    return acb_mat(
        rows,
        columns,
        (
            _exact_complex128(matrix[row, column])
            for row in range(rows)
            for column in range(columns)
        ),
    )


def _frobenius_upper(matrix: acb_mat) -> arb:
    '''Explanation: The Frobenius norm dominates the operator norm. It therefore turns entrywise ball enclosures into a single safe perturbation size for the matrix homotopies in the contour argument.
Functionality: Sum outward-rounded squared entry magnitudes and return a certified upper bound for the Frobenius norm.'''
    total = arb(0)
    for row in range(matrix.nrows()):
        for column in range(matrix.ncols()):
            magnitude = abs(matrix[row, column]).upper()
            total += magnitude * magnitude
    return total.sqrt().upper()


def _upper_float(value: arb) -> float:
    return float(np.nextafter(float(value.upper()), np.inf))


def _lower_float(value: arb) -> float:
    return float(np.nextafter(float(value.lower()), -np.inf))


def _upper_text(value: arb, digits: int = 80) -> str:
    return value.upper().str(int(digits), radius=False, more=True)


def _lower_text(value: arb, digits: int = 80) -> str:
    return value.lower().str(int(digits), radius=False, more=True)


@contextmanager
def _fixed_flint_precision(precision_bits: int):
    """Run one bounded calculation at an explicit, restored FLINT precision."""

    precision_bits = int(precision_bits)
    if precision_bits < 2:
        raise ValueError("FLINT precision must be at least two bits.")
    previous = flint.ctx.prec
    flint.ctx.prec = precision_bits
    try:
        if int(flint.ctx.prec) != precision_bits:
            raise RuntimeError("FLINT did not enter the requested precision scope.")
        yield
    finally:
        flint.ctx.prec = previous


def _precision_separated_small_gain(
    epsilon: arb,
    lifted_moat: arb,
    *,
    epsilon_text: str,
    theorem_precision_bits: int,
    legacy_serialized_moat_lower_text: str | None = None,
) -> dict[str, object]:
    """Return canonical theorem values and a non-gating 5c0 projection.

    The canonical quotient is recomputed at the configured theorem precision.
    The legacy projection deliberately repeats the old 53-bit parse and
    reaggregation, but starts from an upward epsilon enclosure and a downward
    serialized-moat enclosure.  Its quotient is therefore still a rigorous
    upper bound.  It is retained only to explain the authenticated 5c0 bytes;
    no Boolean theorem decision is read from it.
    """

    theorem_precision_bits = int(theorem_precision_bits)
    with _fixed_flint_precision(theorem_precision_bits):
        theorem_epsilon = epsilon.upper()
        theorem_moat = lifted_moat.lower()
        if not theorem_moat > 0:
            raise ArithmeticError("The theorem small-gain moat is not positive.")
        theorem_product = (theorem_epsilon / theorem_moat).upper()
        theorem_epsilon_float = _upper_float(theorem_epsilon)
        theorem_product_float = _upper_float(theorem_product)

    # A cached certificate already stores the outward-rounded binary64 moat.
    # Reuse that text verbatim for the legacy projection: applying
    # ``_lower_float`` to its parsed enclosure would move one more binary64
    # step downward and would no longer reproduce the authenticated 5c0
    # diagnostic.  A fresh calculation has no stored text yet, so it performs
    # the single serialization step here.
    serialized_moat_lower = (
        str(legacy_serialized_moat_lower_text)
        if legacy_serialized_moat_lower_text is not None
        else repr(_lower_float(lifted_moat))
    )
    with _fixed_flint_precision(LEGACY_5C0_COMPATIBILITY_PRECISION_BITS):
        legacy_epsilon = arb(str(epsilon_text)).upper()
        legacy_moat = arb(serialized_moat_lower).lower()
        if not legacy_moat > 0:
            raise ArithmeticError("The legacy compatibility moat is not positive.")
        legacy_product = (legacy_epsilon / legacy_moat).upper()
        legacy_epsilon_float = _upper_float(legacy_epsilon)
        legacy_product_float = _upper_float(legacy_product)

    if (
        legacy_epsilon_float < theorem_epsilon_float
        or legacy_product_float < theorem_product_float
    ):
        raise ArithmeticError(
            "The fixed-53-bit compatibility projection is not conservative "
            "relative to the configured theorem projection."
        )
    return {
        "theorem_precision_bits": theorem_precision_bits,
        "theorem_epsilon_upper": theorem_epsilon_float,
        "theorem_small_gain_product_upper": theorem_product_float,
        "theorem_rounding_role": (
            "configured-precision interval upper bounds used by theorem gates"
        ),
        "legacy_5c0_compatibility_precision_bits": (
            LEGACY_5C0_COMPATIBILITY_PRECISION_BITS
        ),
        "legacy_5c0_compatibility_epsilon_upper": legacy_epsilon_float,
        "legacy_5c0_compatibility_small_gain_product_upper": (
            legacy_product_float
        ),
        "legacy_5c0_compatibility_serialized_moat_lower": float(
            serialized_moat_lower
        ),
        "legacy_5c0_compatibility_rounding_role": (
            "fixed-53-bit upward epsilon over downward serialized moat; "
            "diagnostic compatibility only"
        ),
        "legacy_5c0_compatibility_is_conservative_upper": True,
        "legacy_5c0_compatibility_theorem_gate": False,
    }


def _precision_role_report(
    rows: Iterable[dict[str, object]],
    *,
    theorem_precision_bits: int,
) -> dict[str, object]:
    """Summarise the disjoint theorem and legacy compatibility roles."""

    rows = list(rows)
    if not rows:
        raise ValueError("The precision-role report requires certificate rows.")
    canonical_fields_are_theorem_values = all(
        float(row["epsilon_upper"])
        == float(row["theorem_epsilon_upper"])
        and float(row["certified_small_gain_product_upper"])
        == float(row["theorem_small_gain_product_upper"])
        for row in rows
    )
    legacy_is_conservative = all(
        _csv_bool(row["legacy_5c0_compatibility_is_conservative_upper"])
        and float(row["legacy_5c0_compatibility_epsilon_upper"])
        >= float(row["theorem_epsilon_upper"])
        and float(row["legacy_5c0_compatibility_small_gain_product_upper"])
        >= float(row["theorem_small_gain_product_upper"])
        and not _csv_bool(row["legacy_5c0_compatibility_theorem_gate"])
        for row in rows
    )
    if not canonical_fields_are_theorem_values:
        raise ArithmeticError(
            "A legacy compatibility value entered a canonical theorem field."
        )
    if not legacy_is_conservative:
        raise ArithmeticError(
            "The legacy compatibility projection is not conservative and non-gating."
        )
    return {
        "theorem_projection": {
            "precision_bits": int(theorem_precision_bits),
            "row_count": len(rows),
            "rounding_role": (
                "configured-precision interval upper bounds used by theorem gates"
            ),
            "canonical_fields_are_theorem_values": True,
            "theorem_gate": True,
        },
        "legacy_5c0_compatibility_projection": {
            "precision_bits": LEGACY_5C0_COMPATIBILITY_PRECISION_BITS,
            "row_count": len(rows),
            "rounding_role": (
                "fixed-53-bit upward epsilon over downward serialized moat; "
                "diagnostic compatibility only"
            ),
            "is_conservative_upper": True,
            "theorem_gate": False,
        },
    }


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write an empty certificate table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _is_sha256_text(value: object) -> bool:
    text = str(value)
    if len(text) != 64:
        return False
    try:
        int(text, 16)
    except ValueError:
        return False
    return True


def _is_finite_nonnegative_number(value: object) -> bool:
    """Reject malformed persisted numeric fields without raising on cache reuse."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(math.isfinite(numeric) and numeric >= 0.0)


def _runtime_fingerprint() -> dict[str, object]:
    """Record the binary64 toolchain that determines proposal tensor bytes."""

    blas_records = []
    for record in threadpool_info():
        blas_records.append(
            {
                key: record.get(key)
                for key in (
                    "user_api",
                    "internal_api",
                    "version",
                    "threading_layer",
                    "architecture",
                    "num_threads",
                )
            }
        )
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python_flint": getattr(flint, "__version__", "unknown"),
        "blas": blas_records,
    }


def _target_contours(config: ContourCertificateConfig) -> list[TargetContour]:
    '''Explanation: A Riesz projector has constant rank while its contour stays in the resolvent set. The 24 separated circles isolate the alpha and mu spectral packets whose algebraic multiplicities the thesis certifies, while deliberately excluding zero.
Functionality: Construct and order the 24 exact rational alpha- and mu-family contours with separated radii, multiplicities, and Laurent fallback settings.'''
    alpha = Fraction(config.alpha_numerator, config.alpha_denominator)
    mu = Fraction(config.mu_numerator, config.mu_denominator)
    raw: list[tuple[Fraction, str, str, int, int]] = []
    for power in range(1, int(config.alpha_power_count) + 1):
        raw.append((alpha**power, f"alpha^{power}", "alpha", power, 1))
    for power in range(1, int(config.mu_power_count) + 1):
        raw.append((mu**power, f"mu^{power}", "mu", power, 2))
    raw.sort(key=lambda item: item[0], reverse=True)

    contours: list[TargetContour] = []
    default_fraction = Fraction(
        config.default_radius_numerator, config.default_radius_denominator
    )
    for index, (centre, name, family, power, multiplicity) in enumerate(raw):
        separation = min(
            abs(centre - other_centre)
            for other_centre, *_ in raw
            if other_centre != centre
        )
        radius_fraction, samples = _LAURENT_PROPOSALS.get(
            name, (default_fraction, None)
        )
        radius = radius_fraction * min(centre, separation)
        contours.append(
            TargetContour(
                rank=index + 1,
                name=name,
                family=family,
                power=power,
                expected_multiplicity=multiplicity,
                centre=centre,
                nearest_target_separation=separation,
                radius_fraction=radius_fraction,
                radius=radius,
                laurent_sample_count=samples,
            )
        )
    return contours


def _contour_plan_rows(contours: Iterable[TargetContour]) -> list[dict[str, object]]:
    rows = []
    for contour in contours:
        rows.append(
            {
                "rank": contour.rank,
                "name": contour.name,
                "family": contour.family,
                "power": contour.power,
                "expected_multiplicity": contour.expected_multiplicity,
                "centre_exact": _fraction_text(contour.centre),
                "centre_float": float(contour.centre),
                "nearest_target_separation_exact": _fraction_text(
                    contour.nearest_target_separation
                ),
                "nearest_target_separation_float": float(
                    contour.nearest_target_separation
                ),
                "radius_fraction_exact": _fraction_text(contour.radius_fraction),
                "radius_exact": _fraction_text(contour.radius),
                "radius_float": float(contour.radius),
                "distance_to_zero_exact": _fraction_text(
                    contour.centre - contour.radius
                ),
                "zero_outside_enclosed_region": contour.radius < contour.centre,
                "count_method": COUNT_METHOD_SCHUR_DIAGONAL,
                "moat_method": (
                    MOAT_METHOD_LAURENT
                    if contour.laurent_sample_count is not None
                    else MOAT_METHOD_SCHUR_TRIANGULAR
                ),
                "planned_certificate_route": (
                    CERTIFICATE_ROUTE_LAURENT
                    if contour.laurent_sample_count is not None
                    else CERTIFICATE_ROUTE_SCHUR
                ),
                "count_reference_matrix": COUNT_REFERENCE_MATRIX,
                "laurent_sample_count": contour.laurent_sample_count or "",
            }
        )
    return rows


def _load_epsilon(
    certificate_path: Path,
    config: ContourCertificateConfig,
) -> tuple[arb, str]:
    with Path(certificate_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError("The deterministic epsilon certificate must contain one row.")
    row = rows[0]
    expected = {
        "N": str(config.N),
        "M": str(config.M),
        "rho": str(config.rho),
        "phase2_aggregation_status": (
            "authoritative standalone final-aggregation refresh"
        ),
    }
    for key, value in expected.items():
        if str(row.get(key, "")) != value:
            raise ValueError(f"Deterministic epsilon geometry mismatch for {key}.")
    require_canonical_selected_hardy_radius(
        config.r, label="contour configuration Hardy radius"
    )
    require_canonical_selected_hardy_radius(
        row.get("r", ""), label="deterministic epsilon Hardy radius"
    )
    if Decimal(str(row.get("r", ""))) != Decimal(config.r):
        raise ValueError("Deterministic epsilon geometry mismatch for r.")
    if Decimal(str(row.get("q_gap", ""))) != Decimal(
        CANONICAL_SELECTED_Q_GAP_TARGET_TEXT
    ):
        raise ValueError("Deterministic epsilon geometry mismatch for q_gap.")
    q_gap_contract = exact_q_gap_contract(
        r_tau_upper=row.get("r_tau_interval_u", ""),
        hardy_radius=row.get("r", ""),
        q_gap_target=row.get("q_gap", ""),
    )
    for key, expected_value in q_gap_contract.items():
        if str(row.get(key, "")) != str(expected_value):
            raise ValueError(
                f"Deterministic epsilon q_gap contract mismatch for {key}."
            )
    failed_gates = tuple(
        gate
        for gate in REQUIRED_EPSILON_CERTIFICATION_GATES
        if not _csv_bool(row.get(gate, False))
    )
    if failed_gates:
        raise ArithmeticError(
            "The deterministic epsilon row is not certified; missing or false gates: "
            + ", ".join(failed_gates)
        )
    value = str(row.get("new_epsilon_response_prefactor_candidate_text", ""))
    if not value:
        raise ValueError("The deterministic epsilon certificate has no exact text value.")
    with _fixed_flint_precision(config.precision_bits):
        epsilon = arb(value).upper()
        if epsilon.lower() <= 0:
            raise ArithmeticError(
                "The deterministic perturbation radius is not positive."
            )
    return epsilon, value


def _load_matrix_report(
    report_path: Path,
    payload_path: Path,
    midpoint_path: Path,
    config: ContourCertificateConfig,
) -> dict[str, object]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    expected = {
        "N": int(config.N),
        "M": int(config.M),
        "rho": str(config.rho),
    }
    for key, value in expected.items():
        if str(report.get(key)) != str(value):
            raise ValueError(f"Hardy matrix geometry mismatch for {key}.")
    require_canonical_selected_hardy_radius(
        report.get("r", ""), label="Hardy matrix report radius"
    )
    if Decimal(str(report.get("r", ""))) != Decimal(config.r):
        raise ValueError("Hardy matrix geometry mismatch for r.")
    if not bool(report.get("matrix_enclosure_certified")):
        raise ArithmeticError("The mathematical Hardy matrix is not certified.")
    if not bool(report.get("reference_is_exact_dyadic")):
        raise ArithmeticError("The Hardy midpoint is not exact dyadic.")
    if sha256_file(payload_path) != report.get("payload_sha256"):
        raise ArithmeticError("The exact-dyadic Hardy payload digest failed.")
    midpoint = np.load(midpoint_path, allow_pickle=False)
    if str(midpoint["midpoint_sha256"]) != str(report.get("midpoint_sha256")):
        raise ArithmeticError("The binary64 midpoint provenance digest failed.")
    return report


def _load_exact_payload_metadata(payload_path: Path) -> dict[str, object]:
    with gzip.open(payload_path, "rb") as handle:
        payload = pickle.load(handle)
    return payload


def _exact_dyadic_schur_upper_triangle_audit(
    triangular: np.ndarray,
) -> dict[str, object]:
    """Prove that every exact-binary entry strictly below Schur ``T`` is zero."""

    triangular = np.asarray(triangular)
    if triangular.dtype != np.dtype(np.complex128):
        raise TypeError(
            "The exact-dyadic Schur T array must already have complex128 dtype; "
            f"observed {triangular.dtype}."
        )
    if triangular.ndim != 2 or triangular.shape[0] != triangular.shape[1]:
        raise ValueError("The Schur factor must be a square complex128 matrix.")
    if not np.isfinite(triangular).all():
        raise ArithmeticError("The Schur factor contains a non-finite exact dyadic.")

    order = int(triangular.shape[0])
    entry_count = order * (order - 1) // 2
    zero_count = 0
    first_nonzero: tuple[int, int, complex] | None = None
    for row in range(1, order):
        for column in range(row):
            value = complex(triangular[row, column])
            if value == 0j:
                zero_count += 1
            elif first_nonzero is None:
                first_nonzero = (row, column, value)
    if zero_count != entry_count:
        assert first_nonzero is not None
        row, column, value = first_nonzero
        raise ArithmeticError(
            "The exact-dyadic Schur factor is not upper triangular: "
            f"T[{row},{column}]={value!r}; verified {zero_count} of "
            f"{entry_count} below-diagonal entries as exact zero."
        )
    return {
        "exact_dyadic_schur_below_diagonal_entry_count": entry_count,
        "exact_dyadic_schur_below_diagonal_zero_count": zero_count,
        "exact_dyadic_schur_below_diagonal_all_zero": True,
        "exact_dyadic_schur_upper_triangular_certified": True,
    }


def _load_and_audit_exact_dyadic_schur_cache(
    cache_path: Path,
    *,
    expected_sha256: object,
) -> dict[str, object]:
    """Bind a cached Schur factor to its bytes, dtype, and exact zero pattern."""

    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("The cached Schur report has no valid cache_sha256 field.")
    try:
        int(expected_sha256, 16)
    except ValueError as exc:
        raise ValueError("The cached Schur report has an invalid cache_sha256 field.") from exc

    cache_bytes = Path(cache_path).read_bytes()
    observed_sha256 = hashlib.sha256(cache_bytes).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ArithmeticError("The cached Schur NPZ digest does not match its report.")
    with np.load(io.BytesIO(cache_bytes), allow_pickle=False) as payload:
        triangular = payload["T"]
        if triangular.dtype != np.dtype(np.complex128):
            raise TypeError(
                "The cached Schur T array must already have exact complex128 dtype; "
                f"observed {triangular.dtype}."
            )
        if triangular.ndim != 2 or triangular.shape[0] != triangular.shape[1]:
            raise ValueError("The cached Schur T array must be square.")
        return _exact_dyadic_schur_upper_triangle_audit(triangular)


def _validate_schur_similarity(
    A_float: np.ndarray,
    A_exact: acb_mat,
    config: ContourCertificateConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    '''Explanation: A Schur similarity preserves eigenvalues and algebraic multiplicity. Certifying the intertwining relation makes the triangular diagonal a legitimate source of exact finite-dimensional contour counts rather than a floating-point heuristic.
Functionality: Validate an exact-binary Schur intertwining against exact A_N^circ.'''

    started = time.time()
    with threadpool_limits(limits=int(config.flint_threads), user_api="blas"):
        triangular, transform = linalg.schur(
            np.asarray(A_float, dtype=np.complex128),
            output="complex",
            check_finite=True,
        )
    triangle_audit = _exact_dyadic_schur_upper_triangle_audit(triangular)
    transform_exact = _numpy_to_acb_exact(transform)
    triangular_exact = _numpy_to_acb_exact(triangular)
    identity = acb_mat(int(config.N), int(config.N), 1)
    gram_defect = (
        transform_exact.transpose().conjugate() * transform_exact - identity
    )
    gram_delta = _frobenius_upper(gram_defect)
    if not gram_delta < 1:
        raise ArithmeticError("The exact-binary Schur basis is not certified invertible.")
    transform_norm = (1 + gram_delta).sqrt().upper()
    transform_inverse_norm = (1 / (1 - gram_delta)).sqrt().upper()
    transform_condition = (transform_norm * transform_inverse_norm).upper()

    intertwining = A_exact * transform_exact - transform_exact * triangular_exact
    intertwining_norm = _frobenius_upper(intertwining)
    eta_schur = (intertwining_norm * transform_inverse_norm).upper()

    absolute_upper = np.zeros(triangular.shape, dtype=np.float64)
    for row in range(int(config.N)):
        for column in range(row, int(config.N)):
            absolute_upper[row, column] = _upper_float(
                abs(triangular_exact[row, column])
            )
    report = {
        "certificate_schema": SCHUR_SCHEMA,
        "N": int(config.N),
        "M": int(config.M),
        "precision_bits": int(config.precision_bits),
        "flint_threads": int(config.flint_threads),
        "gram_defect_frobenius_upper": _upper_float(gram_delta),
        "gram_defect_frobenius_upper_text": _upper_text(gram_delta),
        "Q_norm_upper": _upper_float(transform_norm),
        "Q_inverse_norm_upper": _upper_float(transform_inverse_norm),
        "Q_condition_upper": _upper_float(transform_condition),
        "Q_condition_upper_text": _upper_text(transform_condition),
        "intertwining_frobenius_upper": _upper_float(intertwining_norm),
        "eta_schur_upper": _upper_float(eta_schur),
        "eta_schur_upper_text": _upper_text(eta_schur),
        "schur_similarity_certified": True,
        **triangle_audit,
        "similarity_identity": "A_N_circ Q = Q T + R",
        "elapsed_seconds": time.time() - started,
        "status": "interval-certified exact-binary Schur similarity",
    }
    del (
        transform_exact,
        triangular_exact,
        identity,
        gram_defect,
        intertwining,
    )
    gc.collect()
    return triangular, transform, absolute_upper, report


def _certified_schur_diagonal_geometry(
    triangular: np.ndarray,
    contour: TargetContour,
) -> dict[str, object]:
    '''Explanation: The eigenvalues of an upper-triangular matrix are its diagonal entries with multiplicity. Certified inside/outside distance tests therefore count the finite eigenvalues enclosed by a circle without relying on a sampled winding number.
Functionality: Certify Schur-diagonal membership and distance from one contour. This is not a numerical argument-principle contour integral. It is the certified algebraic count obtained from the diagonal of the exact-binary upper-triangular Schur matrix.'''

    triangular = np.asarray(triangular, dtype=np.complex128)
    if triangular.ndim != 2 or triangular.shape[0] != triangular.shape[1]:
        raise ValueError("The Schur matrix must be square.")
    centre = _fraction_acb(contour.centre)
    radius = _fraction_arb(contour.radius)
    diagonal_boundary_distances: list[arb] = []
    inside_flags: list[bool] = []
    for index in range(triangular.shape[0]):
        diagonal = _exact_complex128(triangular[index, index])
        centre_distance = abs(diagonal - centre)
        if centre_distance.upper() < radius.lower():
            inside_flags.append(True)
            boundary_distance = (radius - centre_distance).lower()
        elif centre_distance.lower() > radius.upper():
            inside_flags.append(False)
            boundary_distance = (centre_distance - radius).lower()
        else:
            raise ArithmeticError(
                f"A Schur diagonal enclosure meets contour {contour.name}, "
                "so strict membership is ambiguous."
            )
        if boundary_distance <= 0:
            raise ArithmeticError(
                f"A Schur diagonal enclosure meets contour {contour.name}."
            )
        diagonal_boundary_distances.append(arb(boundary_distance))

    return {
        "schur_diagonal_algebraic_count": int(sum(inside_flags)),
        "inside_flags": tuple(inside_flags),
        "diagonal_boundary_distances": tuple(diagonal_boundary_distances),
        "minimum_diagonal_boundary_distance": min(
            diagonal_boundary_distances
        ),
        "schur_diagonal_membership_certified": True,
    }


def _uniform_triangular_inverse_bound(
    triangular: np.ndarray,
    absolute_upper: np.ndarray,
    contour: TargetContour,
    diagonal_geometry: dict[str, object] | None = None,
) -> dict[str, object]:
    '''Explanation: On a circle separated from the Schur diagonal, the strictly upper-triangular part is nilpotent. A finite Neumann expansion then bounds the inverse uniformly around the complete circle and supplies a resolvent moat.
Functionality: Bound the triangular resolvent over the complete rational circle.'''

    N = triangular.shape[0]
    geometry = (
        _certified_schur_diagonal_geometry(triangular, contour)
        if diagonal_geometry is None
        else diagonal_geometry
    )
    diagonal_lower = list(geometry["diagonal_boundary_distances"])

    row_bounds = [arb(0) for _ in range(N)]
    row_bounds[N - 1] = 1 / diagonal_lower[N - 1]
    for row in range(N - 2, -1, -1):
        numerator = arb(1)
        for column in range(row + 1, N):
            numerator += (
                _exact_float(absolute_upper[row, column]) * row_bounds[column]
            )
        row_bounds[row] = numerator / diagonal_lower[row]

    column_bounds = [arb(0) for _ in range(N)]
    column_bounds[0] = 1 / diagonal_lower[0]
    for column in range(1, N):
        numerator = arb(1)
        for row in range(column):
            numerator += (
                _exact_float(absolute_upper[row, column]) * column_bounds[row]
            )
        column_bounds[column] = numerator / diagonal_lower[column]

    bound = (max(row_bounds) * max(column_bounds)).sqrt().upper()
    return {
        "triangular_inverse_bound": bound,
        **geometry,
    }


def _schur_contour_attempt(
    contour: TargetContour,
    triangular: np.ndarray,
    absolute_upper: np.ndarray,
    schur_report: dict[str, object],
    eta_A: arb,
    epsilon: arb,
    *,
    epsilon_text: str | None = None,
    precision_bits: int = 256,
) -> dict[str, object]:
    '''Explanation: This is the finite-to-infinite bridge: the Schur diagonal gives the finite count, the triangular inverse gives the moat, and a strict small-gain inequality prevents the exact operator homotopy from crossing the contour.
Functionality: Derive one Schur-diagonal count, triangular complete-circle moat, perturbation transports, and finite-to-exact small-gain verdict.'''
    if not bool(
        schur_report.get("exact_dyadic_schur_upper_triangular_certified")
    ):
        raise ArithmeticError(
            "The exact-dyadic Schur lower-triangle zero gate is absent or false."
        )
    diagonal_geometry = _certified_schur_diagonal_geometry(
        triangular, contour
    )
    triangular_data = _uniform_triangular_inverse_bound(
        triangular,
        absolute_upper,
        contour,
        diagonal_geometry=diagonal_geometry,
    )
    q_condition = arb(str(schur_report["Q_condition_upper_text"]))
    eta_schur = arb(str(schur_report["eta_schur_upper_text"]))
    triangular_bound = triangular_data["triangular_inverse_bound"]
    surrogate_resolvent = (q_condition * triangular_bound).upper()
    surrogate_moat = (1 / surrogate_resolvent).lower()
    midpoint_moat = (surrogate_moat - eta_schur).lower()
    mathematical_moat = (midpoint_moat - eta_A).lower()
    zero_distance = _fraction_arb(contour.centre - contour.radius).lower()
    lifted_moat = (
        arb(min(mathematical_moat, zero_distance)).lower()
        if mathematical_moat > 0 and zero_distance > 0
        else arb(0)
    )
    small_gain_projection = (
        _precision_separated_small_gain(
            epsilon,
            lifted_moat,
            epsilon_text=(
                str(epsilon_text)
                if epsilon_text is not None
                else _upper_text(epsilon, 90)
            ),
            theorem_precision_bits=precision_bits,
        )
        if lifted_moat > 0
        else None
    )
    schur_count = int(
        diagonal_geometry["schur_diagonal_algebraic_count"]
    )
    count_matches = bool(schur_count == contour.expected_multiplicity)
    midpoint_count_transport = bool(eta_schur < surrogate_moat)
    mathematical_count_transport = bool(
        midpoint_count_transport and eta_A < midpoint_moat
    )
    finite_count_certified = bool(
        diagonal_geometry["schur_diagonal_membership_certified"]
        and mathematical_count_transport
    )
    gain_pass = bool(
        small_gain_projection is not None
        and float(small_gain_projection["theorem_small_gain_product_upper"]) < 1
    )
    finite_to_exact_rank_transfer = bool(finite_count_certified and gain_pass)
    theorem = bool(finite_to_exact_rank_transfer and count_matches)
    if theorem:
        status = "theorem_certified"
    elif not count_matches:
        status = "benchmark_multiplicity_mismatch"
    else:
        status = "requires_laurent_fallback"
    return {
        **_base_result_row(contour),
        "count_method": COUNT_METHOD_SCHUR_DIAGONAL,
        "moat_method": MOAT_METHOD_SCHUR_TRIANGULAR,
        "certificate_route": CERTIFICATE_ROUTE_SCHUR,
        "count_reference_matrix": COUNT_REFERENCE_MATRIX,
        "schur_diagonal_algebraic_count": schur_count,
        "schur_diagonal_membership_certified": bool(
            diagonal_geometry["schur_diagonal_membership_certified"]
        ),
        "exact_dyadic_schur_upper_triangular_certified": True,
        "minimum_schur_diagonal_boundary_distance_lower": _lower_float(
            triangular_data["minimum_diagonal_boundary_distance"]
        ),
        "triangular_inverse_bound_upper": _upper_float(triangular_bound),
        "Q_condition_upper": schur_report["Q_condition_upper"],
        "surrogate_resolvent_bound_upper": _upper_float(surrogate_resolvent),
        "surrogate_moat_lower": _lower_float(surrogate_moat),
        "eta_schur_upper": schur_report["eta_schur_upper"],
        "A_N_circ_moat_lower": _lower_float(midpoint_moat),
        "eta_A_upper": _upper_float(eta_A),
        "mathematical_finite_matrix_moat_lower": _lower_float(mathematical_moat),
        "distance_to_zero_lower": _lower_float(zero_distance),
        "lifted_finite_section_moat_lower": _lower_float(lifted_moat),
        "epsilon_upper": (
            small_gain_projection["theorem_epsilon_upper"]
            if small_gain_projection is not None
            else math.inf
        ),
        "certified_small_gain_product_upper": (
            small_gain_projection["theorem_small_gain_product_upper"]
            if small_gain_projection is not None
            else math.inf
        ),
        **(small_gain_projection or {}),
        "finite_count_matches_expected": count_matches,
        "A_N_circ_count_transport_certified": midpoint_count_transport,
        "mathematical_finite_count_transport_certified": (
            mathematical_count_transport
        ),
        "finite_count_certified": finite_count_certified,
        "certified_small_gain_pass": gain_pass,
        "finite_to_exact_rank_transfer_certified": (
            finite_to_exact_rank_transfer
        ),
        "complete_circle_covered": True,
        "sampled_values_used_in_theorem_gate": False,
        "laurent_used_for_count": False,
        "laurent_used_for_moat": False,
        "theorem_certified": theorem,
        "status": status,
    }


def _base_result_row(contour: TargetContour) -> dict[str, object]:
    return {
        "rank": contour.rank,
        "name": contour.name,
        "family": contour.family,
        "power": contour.power,
        "expected_multiplicity": contour.expected_multiplicity,
        "centre_exact": _fraction_text(contour.centre),
        "centre_float": float(contour.centre),
        "radius_fraction_exact": _fraction_text(contour.radius_fraction),
        "radius_exact": _fraction_text(contour.radius),
        "radius_float": float(contour.radius),
        "zero_outside_enclosed_region": contour.radius < contour.centre,
    }


def _laurent_coefficients(
    reference_float: np.ndarray,
    contour: TargetContour,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    '''Explanation: A matrix resolvent on a circle has a Fourier--Laurent description. Discrete samples propose coefficients for an approximate inverse, but at this stage they are only a candidate for the later whole-circle interval proof.
Functionality: Sample the resolvent on an equispaced circle and Fourier-transform the samples into proposed Laurent inverse coefficients and residual diagnostics.'''
    sample_count = int(contour.laurent_sample_count or 0)
    if sample_count < 4:
        raise ValueError(f"No Laurent proposal is configured for {contour.name}.")
    N = reference_float.shape[0]
    identity = np.eye(N, dtype=np.complex128)
    centre, radius = float(contour.centre), float(contour.radius)
    samples = np.empty((sample_count, N, N), dtype=np.complex128)
    residuals = np.empty(sample_count, dtype=np.float64)
    started = time.time()
    with threadpool_limits(limits=flint.ctx.threads, user_api="blas"):
        for index in range(sample_count):
            wave = np.exp(2j * np.pi * index / sample_count)
            pencil = (centre + radius * wave) * identity - reference_float
            inverse = linalg.inv(pencil, check_finite=False)
            samples[index] = inverse
            residuals[index] = linalg.norm(
                identity - inverse @ pencil, ord="fro", check_finite=False
            )
    raw = np.fft.fft(samples, axis=0) / sample_count
    del samples
    indices = np.arange(sample_count)
    modes = np.where(
        indices <= sample_count // 2,
        indices,
        indices - sample_count,
    )
    order = np.argsort(modes)
    coefficients = np.ascontiguousarray(raw[order])
    modes = modes[order].astype(int)
    digest = hashlib.sha256(coefficients.view(np.uint8)).hexdigest()
    return modes, coefficients, residuals, digest, time.time() - started


def _laurent_contour_certificate(
    contour: TargetContour,
    triangular: np.ndarray,
    triangular_exact: acb_mat,
    schur_report: dict[str, object],
    eta_A: arb,
    epsilon: arb,
    *,
    epsilon_text: str | None = None,
    precision_bits: int = 256,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    '''Explanation: If a Laurent matrix polynomial has residual norm below one everywhere on the circle, a Neumann argument proves the true resolvent exists there. This supplies the rigorous moat when the triangular Schur bound is too pessimistic.
Functionality: Validate a complete-circle Laurent approximate inverse in Arb. The Laurent approximate inverse certifies the complete-circle resolvent moat only. The finite algebraic count is obtained from the validated Schur diagonal and transported through the matrix homotopies.'''

    if not bool(
        schur_report.get("exact_dyadic_schur_upper_triangular_certified")
    ):
        raise ArithmeticError(
            "The exact-dyadic Schur lower-triangle zero gate is absent or false."
        )
    started = time.time()
    diagonal_geometry = _certified_schur_diagonal_geometry(
        triangular, contour
    )
    modes, coefficients, sample_residuals, digest, inverse_seconds = (
        _laurent_coefficients(triangular, contour)
    )
    N = triangular.shape[0]
    centre = _fraction_acb(contour.centre)
    radius = _fraction_arb(contour.radius)
    base = -triangular_exact
    for index in range(N):
        base[index, index] += centre
    identity = acb_mat(N, N, 1)
    coefficient_sum = arb(0)
    residual_sum = arb(0)
    mode_index = {int(mode): index for index, mode in enumerate(modes)}
    previous: acb_mat | None = None
    mode_rows: list[dict[str, object]] = []
    for mode in range(int(modes.min()), int(modes.max()) + 2):
        if mode in mode_index:
            current = _numpy_to_acb_exact(coefficients[mode_index[mode]])
            coefficient_norm = _frobenius_upper(current)
            coefficient_sum += coefficient_norm
            residual = -(current * base)
        else:
            current = None
            coefficient_norm = arb(0)
            residual = acb_mat(N, N)
        if previous is not None:
            residual -= radius * previous
        if mode == 0:
            residual += identity
        residual_norm = _frobenius_upper(residual)
        residual_sum += residual_norm
        mode_rows.append(
            {
                "name": contour.name,
                "mode": mode,
                "coefficient_present": current is not None,
                "coefficient_frobenius_upper": _upper_float(coefficient_norm),
                "residual_frobenius_upper": _upper_float(residual_norm),
                "coefficient_sha256": digest,
                "precision_bits": flint.ctx.prec,
            }
        )
        previous = current
        del residual
        gc.collect()

    triangular_residual = residual_sum.upper()
    triangular_moat = (
        ((1 - triangular_residual) / coefficient_sum).lower()
        if triangular_residual < 1
        else arb(0)
    )
    eta_schur = arb(str(schur_report["eta_schur_upper_text"]))
    q_condition = arb(str(schur_report["Q_condition_upper_text"]))
    surrogate_moat = (
        (triangular_moat / q_condition).lower()
        if triangular_moat > 0
        else arb(0)
    )
    midpoint_moat = (surrogate_moat - eta_schur).lower()
    schur_count = int(
        diagonal_geometry["schur_diagonal_algebraic_count"]
    )
    count_matches = bool(schur_count == contour.expected_multiplicity)
    midpoint_count_transport = bool(eta_schur < surrogate_moat)
    mathematical_moat = (midpoint_moat - eta_A).lower()
    mathematical_count_transport = bool(
        midpoint_count_transport and eta_A < midpoint_moat
    )
    finite_count_certified = bool(
        diagonal_geometry["schur_diagonal_membership_certified"]
        and mathematical_count_transport
    )
    zero_distance = _fraction_arb(contour.centre - contour.radius).lower()
    lifted_moat = (
        arb(min(mathematical_moat, zero_distance)).lower()
        if mathematical_moat > 0 and zero_distance > 0
        else arb(0)
    )
    small_gain_projection = (
        _precision_separated_small_gain(
            epsilon,
            lifted_moat,
            epsilon_text=(
                str(epsilon_text)
                if epsilon_text is not None
                else _upper_text(epsilon, 90)
            ),
            theorem_precision_bits=precision_bits,
        )
        if lifted_moat > 0
        else None
    )
    gain_pass = bool(
        small_gain_projection is not None
        and float(small_gain_projection["theorem_small_gain_product_upper"]) < 1
    )
    finite_to_exact_rank_transfer = bool(finite_count_certified and gain_pass)
    theorem = bool(
        triangular_residual < 1
        and finite_to_exact_rank_transfer
        and count_matches
    )
    if theorem:
        status = "theorem_certified"
    elif not count_matches:
        status = "benchmark_multiplicity_mismatch"
    else:
        status = "certificate_failed"
    reference_digest = _LAURENT_REFERENCE_DIGESTS.get(contour.name, "")
    row = {
        **_base_result_row(contour),
        "count_method": COUNT_METHOD_SCHUR_DIAGONAL,
        "moat_method": MOAT_METHOD_LAURENT,
        "certificate_route": CERTIFICATE_ROUTE_LAURENT,
        "count_reference_matrix": COUNT_REFERENCE_MATRIX,
        "schur_diagonal_algebraic_count": schur_count,
        "schur_diagonal_membership_certified": bool(
            diagonal_geometry["schur_diagonal_membership_certified"]
        ),
        "exact_dyadic_schur_upper_triangular_certified": True,
        "minimum_schur_diagonal_boundary_distance_lower": _lower_float(
            diagonal_geometry["minimum_diagonal_boundary_distance"]
        ),
        "laurent_sample_count": int(contour.laurent_sample_count or 0),
        "minimum_laurent_mode": int(modes.min()),
        "maximum_laurent_mode": int(modes.max()),
        "coefficient_sha256": digest,
        "reference_coefficient_sha256": reference_digest,
        "coefficient_digest_matches_recorded_reference": bool(
            reference_digest and digest == reference_digest
        ),
        "coefficient_digest_used_in_theorem_gate": False,
        "candidate_coefficient_matrix_count": int(len(modes)),
        "candidate_coefficient_matrix_rows": int(N),
        "candidate_coefficient_matrix_columns": int(N),
        "candidate_coefficient_dtype": "complex128",
        "candidate_coefficient_layout": "C-contiguous signed-mode order",
        "candidate_generation_method": (
            "FFT of complete-circle binary64 Schur-resolvent samples"
        ),
        "candidate_generated_in_recorded_run": True,
        "maximum_sample_inverse_residual_diagnostic": float(
            np.max(sample_residuals)
        ),
        "coefficient_frobenius_sum_upper": _upper_float(coefficient_sum),
        "exact_dyadic_residual_sum_upper": _upper_float(triangular_residual),
        "neumann_margin_lower": _lower_float(1 - triangular_residual),
        "triangular_laurent_moat_lower": _lower_float(triangular_moat),
        "Q_condition_upper": schur_report["Q_condition_upper"],
        "surrogate_moat_lower": _lower_float(surrogate_moat),
        "A_N_circ_moat_lower": _lower_float(midpoint_moat),
        "eta_schur_upper": schur_report["eta_schur_upper"],
        "schur_homotopy_product_upper": (
            _upper_float(eta_schur / surrogate_moat)
            if surrogate_moat > 0
            else math.inf
        ),
        "A_N_circ_count_transport_certified": midpoint_count_transport,
        "eta_A_upper": _upper_float(eta_A),
        "mathematical_finite_count_transport_certified": (
            mathematical_count_transport
        ),
        "mathematical_finite_matrix_moat_lower": _lower_float(mathematical_moat),
        "distance_to_zero_lower": _lower_float(zero_distance),
        "lifted_finite_section_moat_lower": _lower_float(lifted_moat),
        "epsilon_upper": (
            small_gain_projection["theorem_epsilon_upper"]
            if small_gain_projection is not None
            else math.inf
        ),
        "certified_small_gain_product_upper": (
            small_gain_projection["theorem_small_gain_product_upper"]
            if small_gain_projection is not None
            else math.inf
        ),
        **(small_gain_projection or {}),
        "finite_count_matches_expected": count_matches,
        "finite_count_certified": finite_count_certified,
        "certified_small_gain_pass": gain_pass,
        "finite_to_exact_rank_transfer_certified": (
            finite_to_exact_rank_transfer
        ),
        "complete_circle_covered": True,
        "candidate_coefficients_validated_exact_dyadic": True,
        "laurent_reference_matrix": "exact-binary Schur triangular matrix",
        "laurent_used_for_count": False,
        "laurent_used_for_moat": True,
        "sampled_values_used_in_theorem_gate": False,
        "inverse_sampling_seconds": inverse_seconds,
        "elapsed_seconds": time.time() - started,
        "theorem_certified": theorem,
        "status": status,
    }
    del coefficients, base, identity, previous
    gc.collect()
    return row, mode_rows


def _certificate_paths(output_dir: Path, config: ContourCertificateConfig) -> dict[str, Path]:
    stem = f"blaschke_deformation_24_target_N{config.N}_M{config.M}"
    data = output_dir / "data"
    reports = output_dir / "reports"
    return {
        "plan": data / f"{stem}_contour_plan.csv",
        "schur_cache": data / f"{stem}_validated_schur.npz",
        "schur_report": reports / f"{stem}_validated_schur.json",
        "schur_attempts": data / f"{stem}_schur_attempts.csv",
        "laurent_modes": data / f"{stem}_laurent_mode_bounds.csv",
        "laurent_witnesses": data / f"{stem}_laurent_witness_reconstruction.csv",
        "certificate": data / f"{stem}_spectral_certificate.csv",
        "report": reports / f"{stem}_spectral_certificate.json",
    }


def _laurent_witness_rows(
    final_rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Extract one provenance record for each generated Laurent tensor."""

    rows = []
    for result in final_rows:
        if result.get("moat_method") != MOAT_METHOD_LAURENT:
            continue
        rows.append(
            {
                "rank": result["rank"],
                "name": result["name"],
                "laurent_sample_count": result["laurent_sample_count"],
                "minimum_laurent_mode": result["minimum_laurent_mode"],
                "maximum_laurent_mode": result["maximum_laurent_mode"],
                "coefficient_matrix_count": result[
                    "candidate_coefficient_matrix_count"
                ],
                "coefficient_matrix_rows": result[
                    "candidate_coefficient_matrix_rows"
                ],
                "coefficient_matrix_columns": result[
                    "candidate_coefficient_matrix_columns"
                ],
                "coefficient_dtype": result["candidate_coefficient_dtype"],
                "coefficient_layout": result["candidate_coefficient_layout"],
                "generation_method": result["candidate_generation_method"],
                "coefficient_sha256": result["coefficient_sha256"],
                "reference_coefficient_sha256": result[
                    "reference_coefficient_sha256"
                ],
                "digest_matches_recorded_reference": result[
                    "coefficient_digest_matches_recorded_reference"
                ],
                "digest_used_in_theorem_gate": result[
                    "coefficient_digest_used_in_theorem_gate"
                ],
                "generated_in_recorded_run": result[
                    "candidate_generated_in_recorded_run"
                ],
                "candidate_coefficients_validated_exact_dyadic": result[
                    "candidate_coefficients_validated_exact_dyadic"
                ],
                "exact_dyadic_residual_sum_upper": result[
                    "exact_dyadic_residual_sum_upper"
                ],
                "theorem_certified": result["theorem_certified"],
            }
        )
    rows.sort(key=lambda row: int(row["rank"]))
    return rows


def _laurent_witness_records_are_reusable(
    witness_rows: list[dict[str, str]],
    certificate_rows: list[dict[str, str]],
    mode_rows: list[dict[str, str]],
    *,
    expected_precision_bits: int = 256,
) -> bool:
    """Bind all persisted Laurent tables without requiring historical byte parity."""

    expected_mode_row_count = sum(
        sample_count + 1 for _, sample_count in _LAURENT_PROPOSALS.values()
    )
    if (
        len(witness_rows) != len(_LAURENT_PROPOSALS)
        or len(mode_rows) != expected_mode_row_count
        or {row.get("name", "") for row in witness_rows}
        != set(_LAURENT_PROPOSALS)
    ):
        return False
    laurent_certificate_rows = [
        row
        for row in certificate_rows
        if row.get("moat_method") == MOAT_METHOD_LAURENT
    ]
    if len(laurent_certificate_rows) != len(_LAURENT_PROPOSALS):
        return False
    certificate_by_name = {
        row.get("name", ""): row
        for row in laurent_certificate_rows
    }
    if set(certificate_by_name) != set(_LAURENT_PROPOSALS):
        return False
    modes_by_name: dict[str, list[dict[str, str]]] = {
        name: [] for name in _LAURENT_PROPOSALS
    }
    for mode_row in mode_rows:
        name = mode_row.get("name", "")
        if name not in modes_by_name:
            return False
        modes_by_name[name].append(mode_row)
    expected_rank_by_name = {
        name: rank for rank, name in enumerate(_LAURENT_PROPOSALS, 18)
    }
    for witness in witness_rows:
        name = witness.get("name", "")
        certificate = certificate_by_name.get(name)
        if certificate is None:
            return False
        try:
            proposal_sample_count = int(_LAURENT_PROPOSALS[name][1])
            expected_modes = tuple(
                sorted(
                    index
                    if index <= proposal_sample_count // 2
                    else index - proposal_sample_count
                    for index in range(proposal_sample_count)
                )
            )
            expected_mode_support = set(expected_modes) | {max(expected_modes) + 1}
            target_mode_rows = modes_by_name[name]
            mode_by_index = {
                int(row.get("mode", "")): row for row in target_mode_rows
            }
            witness_residual = Decimal(
                str(witness.get("exact_dyadic_residual_sum_upper", ""))
            )
            certificate_residual = Decimal(
                str(certificate.get("exact_dyadic_residual_sum_upper", ""))
            )
            theorem_precision_bits = int(
                certificate.get("theorem_precision_bits", 0)
            )
            expected_digest = _LAURENT_REFERENCE_DIGESTS.get(name)
            observed_digest = witness.get("coefficient_sha256")
            reference_digest = witness.get("reference_coefficient_sha256")
            reference_matches = bool(
                expected_digest and observed_digest == expected_digest
            )
        except (ArithmeticError, KeyError, TypeError, ValueError):
            return False
        if not (
            _is_sha256_text(expected_digest)
            and _is_sha256_text(observed_digest)
            and int(witness.get("rank", 0)) == expected_rank_by_name[name]
            and int(certificate.get("rank", -1)) == expected_rank_by_name[name]
            and int(witness.get("laurent_sample_count", 0))
            == proposal_sample_count
            and int(certificate.get("laurent_sample_count", 0))
            == proposal_sample_count
            and int(witness.get("minimum_laurent_mode", 0))
            == min(expected_modes)
            and int(certificate.get("minimum_laurent_mode", 0))
            == min(expected_modes)
            and int(witness.get("maximum_laurent_mode", 0))
            == max(expected_modes)
            and int(certificate.get("maximum_laurent_mode", 0))
            == max(expected_modes)
            and len(mode_by_index) == proposal_sample_count + 1
            and set(mode_by_index) == expected_mode_support
            and all(
                _csv_bool(row.get("coefficient_present"))
                == (mode in expected_modes)
                and row.get("coefficient_sha256") == observed_digest
                and int(row.get("precision_bits", 0)) == theorem_precision_bits
                and _is_finite_nonnegative_number(
                    row.get("coefficient_frobenius_upper")
                )
                and _is_finite_nonnegative_number(
                    row.get("residual_frobenius_upper")
                )
                for mode, row in mode_by_index.items()
            )
            and observed_digest == certificate.get("coefficient_sha256")
            and reference_digest == expected_digest
            and certificate.get("reference_coefficient_sha256")
            == expected_digest
            and _csv_bool(witness.get("digest_matches_recorded_reference"))
            == reference_matches
            and _csv_bool(
                certificate.get("coefficient_digest_matches_recorded_reference")
            )
            == reference_matches
            and not _csv_bool(witness.get("digest_used_in_theorem_gate"))
            and not _csv_bool(
                certificate.get("coefficient_digest_used_in_theorem_gate")
            )
            and _csv_bool(witness.get("generated_in_recorded_run"))
            and _csv_bool(certificate.get("candidate_generated_in_recorded_run"))
            and _csv_bool(
                witness.get("candidate_coefficients_validated_exact_dyadic")
            )
            and _csv_bool(
                certificate.get("candidate_coefficients_validated_exact_dyadic")
            )
            and _csv_bool(witness.get("theorem_certified"))
            and _csv_bool(certificate.get("theorem_certified"))
            and int(witness.get("coefficient_matrix_count", 0))
            == proposal_sample_count
            and int(certificate.get("candidate_coefficient_matrix_count", 0))
            == proposal_sample_count
            and int(witness.get("coefficient_matrix_rows", 0)) == 600
            and int(witness.get("coefficient_matrix_columns", 0)) == 600
            and int(certificate.get("candidate_coefficient_matrix_rows", 0)) == 600
            and int(certificate.get("candidate_coefficient_matrix_columns", 0))
            == 600
            and witness.get("coefficient_dtype") == "complex128"
            and witness.get("coefficient_dtype")
            == certificate.get("candidate_coefficient_dtype")
            and witness.get("coefficient_layout")
            == certificate.get("candidate_coefficient_layout")
            == "C-contiguous signed-mode order"
            and witness.get("generation_method")
            == certificate.get("candidate_generation_method")
            == "FFT of complete-circle binary64 Schur-resolvent samples"
            and theorem_precision_bits == int(expected_precision_bits)
            and witness_residual == certificate_residual
            and Decimal(0) <= witness_residual < Decimal(1)
        ):
            return False
    return True


def _validate_target_contour_plan(
    contours: Iterable[TargetContour],
    *,
    expected_target_count: int | None = None,
    expected_total_multiplicity: int | None = None,
) -> list[TargetContour]:
    '''Explanation: Disjoint contours prevent one spectral packet from being counted twice, and strict separation from zero prevents the silent zero cluster from contaminating the nonzero multiplicity total. These are structural hypotheses of the thesis count.
Functionality: Validate disjoint target circles and strict silent-zero exclusion.'''

    contours = list(contours)
    if expected_target_count is not None and len(contours) != int(
        expected_target_count
    ):
        raise AssertionError(
            f"The target plan does not contain {expected_target_count} contours."
        )
    total_expected = sum(row.expected_multiplicity for row in contours)
    if (
        expected_total_multiplicity is not None
        and total_expected != int(expected_total_multiplicity)
    ):
        raise AssertionError(
            "The target plan has the wrong expected total multiplicity."
        )
    if any(row.radius >= row.centre for row in contours):
        raise AssertionError("A target contour encloses the silent zero block.")
    ordered = sorted(contours, key=lambda row: row.centre, reverse=True)
    for left, right in zip(ordered, ordered[1:]):
        if left.centre - left.radius <= right.centre + right.radius:
            raise AssertionError("Two target contours overlap.")
    return contours


def _existing_geometry_is_reusable(
    *,
    existing: dict[str, object],
    rows: list[dict[str, str]],
    witness_rows: list[dict[str, str]],
    mode_rows: list[dict[str, str]],
    schur_cache_path: Path,
    source_hashes: dict[str, str],
    geometry_input_hashes: dict[str, str],
    expected_precision_bits: int = 256,
) -> bool:
    '''Explanation: Finite Schur counts and finite-matrix moats do not depend on a later refinement of the exact-operator perturbation radius. Hash and geometry checks justify retaining those proved facts while recomputing only the changed small-gain step.
Functionality: Validate cached counts and moats independently of the current epsilon.'''

    if existing.get("certificate_schema") != SCHEMA or len(rows) != 24:
        return False
    try:
        decoded_triangle_audit = _load_and_audit_exact_dyadic_schur_cache(
            schur_cache_path,
            expected_sha256=existing.get("cache_sha256"),
        )
    except (ArithmeticError, KeyError, OSError, TypeError, ValueError):
        return False
    if any(
        existing.get(key) != value
        for key, value in decoded_triangle_audit.items()
    ):
        return False
    precision_roles = existing.get("small_gain_precision_roles")
    if not isinstance(precision_roles, dict):
        return False
    theorem_role = precision_roles.get("theorem_projection")
    legacy_role = precision_roles.get("legacy_5c0_compatibility_projection")
    if not isinstance(theorem_role, dict) or not isinstance(legacy_role, dict):
        return False
    if not (
        int(theorem_role.get("precision_bits", 0))
        == int(existing.get("precision_bits", 0))
        and int(theorem_role.get("row_count", 0)) == 24
        and bool(theorem_role.get("canonical_fields_are_theorem_values"))
        and bool(theorem_role.get("theorem_gate"))
        and int(legacy_role.get("precision_bits", 0))
        == LEGACY_5C0_COMPATIBILITY_PRECISION_BITS
        and int(legacy_role.get("row_count", 0)) == 24
        and bool(legacy_role.get("is_conservative_upper"))
        and not bool(legacy_role.get("theorem_gate"))
    ):
        return False
    try:
        stored_sources = normalise_source_hashes(
            dict(existing.get("source_hashes", {}))
        )
    except (TypeError, ValueError):
        return False
    if stored_sources != source_hashes:
        return False
    stored_inputs = dict(existing.get("input_hashes", {}))
    if any(stored_inputs.get(key) != digest for key, digest in geometry_input_hashes.items()):
        return False
    if not (
        bool(existing.get("all_finite_counts_schur_derived"))
        and bool(existing.get("q_gap_derived_le_target"))
        and bool(existing.get("exact_dyadic_schur_below_diagonal_all_zero"))
        and bool(existing.get("exact_dyadic_schur_upper_triangular_certified"))
        and bool(existing.get("all_schur_diagonal_memberships_certified"))
        and bool(existing.get("all_finite_count_transports_certified"))
        and bool(existing.get("all_finite_counts_certified"))
        and bool(existing.get("all_finite_counts_match_expected"))
        and bool(existing.get("all_complete_circles_covered"))
        and bool(existing.get("all_complete_circle_moats_positive"))
        and bool(existing.get("zero_excluded_from_every_contour"))
        and not bool(existing.get("sampled_values_used_in_any_theorem_gate"))
    ):
        return False
    if not all(
        row.get("count_method") == COUNT_METHOD_SCHUR_DIAGONAL
        and _csv_bool(
            row.get("exact_dyadic_schur_upper_triangular_certified")
        )
        and _csv_bool(row.get("finite_count_certified"))
        and _csv_bool(row.get("finite_count_matches_expected"))
        and _csv_bool(row.get("complete_circle_covered"))
        and _csv_bool(row.get("zero_outside_enclosed_region"))
        and not _csv_bool(row.get("sampled_values_used_in_theorem_gate"))
        and float(row.get("lifted_finite_section_moat_lower", 0.0)) > 0.0
        and int(row.get("theorem_precision_bits", 0))
        == int(existing.get("precision_bits", 0))
        and float(row.get("epsilon_upper", math.inf))
        == float(row.get("theorem_epsilon_upper", math.nan))
        and float(row.get("certified_small_gain_product_upper", math.inf))
        == float(row.get("theorem_small_gain_product_upper", math.nan))
        and int(row.get("legacy_5c0_compatibility_precision_bits", 0))
        == LEGACY_5C0_COMPATIBILITY_PRECISION_BITS
        and _csv_bool(
            row.get("legacy_5c0_compatibility_is_conservative_upper")
        )
        and not _csv_bool(row.get("legacy_5c0_compatibility_theorem_gate"))
        for row in rows
    ):
        return False
    if not _laurent_witness_records_are_reusable(
        witness_rows,
        rows,
        mode_rows,
        expected_precision_bits=expected_precision_bits,
    ):
        return False
    if not (
        int(existing.get("laurent_witness_count", 0)) == 7
        and int(existing.get("laurent_coefficient_matrix_count", 0)) == 300
        and bool(existing.get("all_laurent_witnesses_reconstructed_in_recorded_run"))
        and bool(existing.get("laurent_internal_digest_bindings_certified"))
        and bool(existing.get("all_laurent_digests_match_recorded_reference"))
        == all(
            _csv_bool(row.get("digest_matches_recorded_reference"))
            for row in witness_rows
        )
        and not bool(existing.get("laurent_digests_used_in_any_theorem_gate"))
    ):
        return False
    schur_rows = sum(
        row.get("moat_method") == MOAT_METHOD_SCHUR_TRIANGULAR for row in rows
    )
    laurent_rows = sum(
        row.get("moat_method") == MOAT_METHOD_LAURENT for row in rows
    )
    return schur_rows == 17 and laurent_rows == 7


def _reaggregate_small_gain_rows(
    rows: list[dict[str, str]],
    epsilon: arb,
    *,
    epsilon_text: str | None = None,
    precision_bits: int = 256,
) -> list[dict[str, object]]:
    '''Explanation: The final rank transfer depends on the product of the updated perturbation radius and each already-certified resolvent bound. Re-evaluating that inequality is enough to decide whether every finite count still transfers to the exact operator.
    Functionality: Reuse certified finite moats and recompute only epsilon-dependent gates.'''

    refreshed: list[dict[str, object]] = []
    for stored in rows:
        row: dict[str, object] = dict(stored)
        with _fixed_flint_precision(precision_bits):
            moat = arb(
                str(stored["lifted_finite_section_moat_lower"])
            ).lower()
        projection = _precision_separated_small_gain(
            epsilon,
            moat,
            epsilon_text=(
                str(epsilon_text)
                if epsilon_text is not None
                else _upper_text(epsilon, 90)
            ),
            theorem_precision_bits=precision_bits,
            legacy_serialized_moat_lower_text=str(
                stored["lifted_finite_section_moat_lower"]
            ),
        )
        gain_pass = bool(
            float(projection["theorem_small_gain_product_upper"]) < 1
        )
        finite_count = _csv_bool(stored.get("finite_count_certified"))
        count_matches = _csv_bool(stored.get("finite_count_matches_expected"))
        complete_circle = _csv_bool(stored.get("complete_circle_covered"))
        zero_excluded = _csv_bool(stored.get("zero_outside_enclosed_region"))
        no_sampled_gate = not _csv_bool(
            stored.get("sampled_values_used_in_theorem_gate")
        )
        schur_triangle_verified = _csv_bool(
            stored.get("exact_dyadic_schur_upper_triangular_certified")
        )
        laurent_gate = True
        if stored.get("moat_method") == MOAT_METHOD_LAURENT:
            laurent_gate = bool(
                float(stored.get("exact_dyadic_residual_sum_upper", math.inf)) < 1.0
                and _csv_bool(
                    stored.get("candidate_coefficients_validated_exact_dyadic")
                )
                and not _csv_bool(stored.get("laurent_used_for_count"))
                and _csv_bool(stored.get("laurent_used_for_moat"))
            )
        finite_to_exact = bool(finite_count and gain_pass)
        theorem = bool(
            finite_to_exact
            and count_matches
            and complete_circle
            and zero_excluded
            and no_sampled_gate
            and schur_triangle_verified
            and laurent_gate
        )
        row.update(
            {
                "epsilon_upper": projection["theorem_epsilon_upper"],
                "certified_small_gain_product_upper": projection[
                    "theorem_small_gain_product_upper"
                ],
                **projection,
                "certified_small_gain_pass": gain_pass,
                "finite_to_exact_rank_transfer_certified": finite_to_exact,
                "theorem_certified": theorem,
                "status": (
                    "theorem_certified"
                    if theorem
                    else "small_gain_reaggregation_failed"
                ),
            }
        )
        refreshed.append(row)
    refreshed.sort(key=lambda item: int(item["rank"]))
    return refreshed


def _reaggregate_existing_certificate(
    *,
    existing: dict[str, object],
    rows: list[dict[str, str]],
    paths: dict[str, Path],
    epsilon: arb,
    epsilon_text: str,
    precision_bits: int,
    source_hashes: dict[str, str],
    input_hashes: dict[str, str],
) -> dict[str, object]:
    '''Explanation: A certificate is a chain of claims, not just a table of numbers. Reaggregation preserves the independently validated finite geometry and rebuilds all epsilon-dependent conclusions so stale truth flags cannot survive a changed perturbation bound.
Functionality: Refresh a final package after an epsilon-only provenance change.'''

    started = time.time()
    refreshed = _reaggregate_small_gain_rows(
        rows,
        epsilon,
        epsilon_text=epsilon_text,
        precision_bits=precision_bits,
    )
    precision_roles = _precision_role_report(
        refreshed,
        theorem_precision_bits=precision_bits,
    )
    all_gain = all(bool(row["certified_small_gain_pass"]) for row in refreshed)
    all_rank_transfers = all(
        bool(row["finite_to_exact_rank_transfer_certified"])
        for row in refreshed
    )
    all_theorem = all(bool(row["theorem_certified"]) for row in refreshed)
    if not (all_gain and all_rank_transfers and all_theorem):
        raise AssertionError("The reaggregated small-gain certificate failed.")

    _write_csv(paths["certificate"], refreshed)
    report = dict(existing)
    report.update(
        {
            "epsilon_upper_text": epsilon_text,
            "all_small_gain_tests_pass": all_gain,
            "all_finite_to_exact_rank_transfers_certified": all_rank_transfers,
            "all_24_targets_theorem_certified": all_theorem,
            "small_gain_precision_roles": precision_roles,
            "source_hashes": dict(source_hashes),
            "geometry_input_hashes": {
                key: value
                for key, value in input_hashes.items()
                if key != "epsilon_certificate"
            },
            "input_hashes": dict(input_hashes),
            "small_gain_reaggregation_seconds": time.time() - started,
            "status": (
                "theorem-certified twenty-four-target Riesz-rank package; "
                "finite moats reused and small-gain products reaggregated"
            ),
        }
    )
    _write_json(paths["report"], report)
    return {**report, "execution_status": "reaggregated_existing_moats"}


def certify_all_target_contours(
    *,
    config: ContourCertificateConfig,
    output_dir: Path,
    matrix_payload_path: Path,
    matrix_midpoint_path: Path,
    matrix_report_path: Path,
    epsilon_certificate_path: Path,
    source_files: Iterable[Path],
    force: bool = False,
    progress: bool = True,
) -> dict[str, object]:
    '''Explanation: The thesis conclusion is obtained only after every target circle has both a certified finite count and a certified moat satisfying small gain. Their 24 transferred ranks then give the complete nonzero algebraic multiplicity total.
Functionality: Build and transactionally promote all twenty-four contour certificates.'''

    require_canonical_selected_hardy_radius(
        config.r, label="contour certificate Hardy radius"
    )
    output_dir = Path(output_dir).resolve()
    paths = _certificate_paths(output_dir, config)
    source_hashes = logical_source_hashes(source_files)
    geometry_input_hashes = {
        "matrix_payload": sha256_file(matrix_payload_path),
        "matrix_midpoint": sha256_file(matrix_midpoint_path),
        "matrix_report": sha256_file(matrix_report_path),
    }
    input_hashes = {
        **geometry_input_hashes,
        "epsilon_certificate": sha256_file(epsilon_certificate_path),
    }
    if (
        not force
        and paths["report"].exists()
        and paths["certificate"].exists()
        and paths["laurent_witnesses"].exists()
        and paths["laurent_modes"].exists()
    ):
        existing = json.loads(paths["report"].read_text(encoding="utf-8"))
        existing_rows = _read_csv(paths["certificate"])
        existing_witness_rows = _read_csv(paths["laurent_witnesses"])
        existing_mode_rows = _read_csv(paths["laurent_modes"])
        if _existing_geometry_is_reusable(
            existing=existing,
            rows=existing_rows,
            witness_rows=existing_witness_rows,
            mode_rows=existing_mode_rows,
            schur_cache_path=paths["schur_cache"],
            source_hashes=source_hashes,
            geometry_input_hashes=geometry_input_hashes,
            expected_precision_bits=config.precision_bits,
        ):
            if (
                existing.get("source_hashes") == source_hashes
                and existing.get("input_hashes") == input_hashes
                and bool(existing.get("all_24_targets_theorem_certified"))
                and bool(existing.get("all_finite_to_exact_rank_transfers_certified"))
            ):
                return {
                    **existing,
                    "execution_status": "reused_validated_checkpoint",
                }
            epsilon, epsilon_text = _load_epsilon(epsilon_certificate_path, config)
            return _reaggregate_existing_certificate(
                existing=existing,
                rows=existing_rows,
                paths=paths,
                epsilon=epsilon,
                epsilon_text=epsilon_text,
                precision_bits=config.precision_bits,
                source_hashes=source_hashes,
                input_hashes=input_hashes,
            )

    matrix_report = _load_matrix_report(
        matrix_report_path,
        matrix_payload_path,
        matrix_midpoint_path,
        config,
    )
    payload_metadata = _load_exact_payload_metadata(matrix_payload_path)
    epsilon, epsilon_text = _load_epsilon(epsilon_certificate_path, config)
    eta_A = arb(str(matrix_report["eta_A_upper_text"])).upper()
    contours = _validate_target_contour_plan(
        _target_contours(config),
        expected_target_count=24,
        expected_total_multiplicity=30,
    )
    _write_csv(paths["plan"], _contour_plan_rows(contours))

    old_precision, old_threads = flint.ctx.prec, flint.ctx.threads
    flint.ctx.prec = int(config.precision_bits)
    flint.ctx.threads = int(config.flint_threads)
    started = time.time()
    try:
        A_real, exact_payload = load_exact_dyadic_midpoint(matrix_payload_path)
        if exact_payload["midpoint_sha256"] != matrix_report["midpoint_sha256"]:
            raise ArithmeticError("The reconstructed exact midpoint digest failed.")
        A_exact = acb_mat(
            int(config.N),
            int(config.N),
            (
                A_real[row, column]
                for row in range(int(config.N))
                for column in range(int(config.N))
            ),
        )
        del A_real
        midpoint_payload = np.load(matrix_midpoint_path, allow_pickle=False)
        A_float = np.asarray(
            midpoint_payload["A_N_circ_binary64_diagnostic"],
            dtype=np.complex128,
        )
        if A_float.shape != (int(config.N), int(config.N)):
            raise ValueError("The diagnostic midpoint has the wrong shape.")
        if not np.isfinite(A_float).all():
            raise ArithmeticError("The diagnostic midpoint contains non-finite values.")

        if progress:
            print("validating exact-dyadic Schur similarity", flush=True)
        triangular, transform, absolute_upper, schur_report = (
            _validate_schur_similarity(A_float, A_exact, config)
        )
        triangular_exact = _numpy_to_acb_exact(triangular)
        schur_report.update(
            {
                "map_label": MAP_LABEL,
                "rho": config.rho,
                "r": config.r,
                "midpoint_sha256": matrix_report["midpoint_sha256"],
                "matrix_payload_sha256": input_hashes["matrix_payload"],
                "source_hashes": source_hashes,
            }
        )
        temporary_cache = paths["schur_cache"].with_suffix(".npz.tmp")
        temporary_cache.parent.mkdir(parents=True, exist_ok=True)
        with temporary_cache.open("wb") as handle:
            np.savez_compressed(
                handle,
                T=triangular,
                Q=transform,
                absolute_T_upper=absolute_upper,
                midpoint_sha256=np.asarray(matrix_report["midpoint_sha256"]),
                eta_schur_upper=np.asarray(schur_report["eta_schur_upper"]),
                Q_condition_upper=np.asarray(schur_report["Q_condition_upper"]),
            )
        temporary_cache.replace(paths["schur_cache"])
        schur_report["cache_path"] = str(paths["schur_cache"])
        schur_report["cache_sha256"] = sha256_file(paths["schur_cache"])
        _write_json(paths["schur_report"], schur_report)

        schur_attempts = []
        final_rows = []
        mode_rows: list[dict[str, object]] = []
        for index, contour in enumerate(contours, 1):
            if progress:
                print(
                    f"Schur count/moat attempt {index}/24: {contour.name}",
                    flush=True,
                )
            attempt = _schur_contour_attempt(
                contour,
                triangular,
                absolute_upper,
                schur_report,
                eta_A,
                epsilon,
                epsilon_text=epsilon_text,
                precision_bits=config.precision_bits,
            )
            schur_attempts.append(attempt)
            if contour.laurent_sample_count is None:
                if not bool(attempt["theorem_certified"]):
                    raise AssertionError(
                        f"Unexpected Schur failure before the Laurent frontier: {contour.name}"
                    )
                final_rows.append(attempt)
        _write_csv(paths["schur_attempts"], schur_attempts)

        laurent_contours = [
            contour for contour in contours if contour.laurent_sample_count is not None
        ]
        for index, contour in enumerate(laurent_contours, 1):
            if progress:
                print(
                    f"Laurent moat {index}/{len(laurent_contours)}: "
                    f"{contour.name} with {contour.laurent_sample_count} samples",
                    flush=True,
                )
            row, rows = _laurent_contour_certificate(
                contour,
                triangular,
                triangular_exact,
                schur_report,
                eta_A,
                epsilon,
                epsilon_text=epsilon_text,
                precision_bits=config.precision_bits,
            )
            if not bool(row["theorem_certified"]):
                _write_csv(paths["laurent_modes"], mode_rows + rows)
                raise AssertionError(
                    f"Laurent theorem gate failed for {contour.name}: {row}"
                )
            final_rows.append(row)
            mode_rows.extend(rows)
            _write_csv(paths["laurent_modes"], mode_rows)
            _write_csv(
                paths["certificate"],
                sorted(final_rows, key=lambda item: int(item["rank"])),
            )

        final_rows.sort(key=lambda item: int(item["rank"]))
        if len(final_rows) != 24:
            raise AssertionError("The promoted certificate does not contain all targets.")
        if not all(
            row["count_method"] == COUNT_METHOD_SCHUR_DIAGONAL
            for row in final_rows
        ):
            raise AssertionError("A finite count is not Schur-diagonal derived.")
        schur_moat_count = sum(
            row["moat_method"] == MOAT_METHOD_SCHUR_TRIANGULAR
            for row in final_rows
        )
        laurent_moat_count = sum(
            row["moat_method"] == MOAT_METHOD_LAURENT
            for row in final_rows
        )
        if schur_moat_count != 17 or laurent_moat_count != 7:
            raise AssertionError("The certified moat-route split is not 17 plus 7.")
        for row in final_rows:
            if row["moat_method"] == MOAT_METHOD_LAURENT:
                if not (
                    row["certificate_route"] == CERTIFICATE_ROUTE_LAURENT
                    and row["count_method"] == COUNT_METHOD_SCHUR_DIAGONAL
                    and row["laurent_used_for_count"] is False
                    and row["laurent_used_for_moat"] is True
                ):
                    raise AssertionError("A Laurent moat row has incorrect provenance.")
            elif row["certificate_route"] != CERTIFICATE_ROUTE_SCHUR:
                raise AssertionError("A Schur moat row has incorrect provenance.")
        all_theorem = all(bool(row["theorem_certified"]) for row in final_rows)
        all_memberships = all(
            bool(row["schur_diagonal_membership_certified"])
            for row in final_rows
        )
        all_exact_dyadic_upper_triangular = all(
            bool(row["exact_dyadic_schur_upper_triangular_certified"])
            for row in final_rows
        )
        all_count_transports = all(
            bool(row["mathematical_finite_count_transport_certified"])
            for row in final_rows
        )
        all_counts = all(bool(row["finite_count_certified"]) for row in final_rows)
        all_matches = all(
            bool(row["finite_count_matches_expected"])
            for row in final_rows
        )
        all_rank_transfers = all(
            bool(row["finite_to_exact_rank_transfer_certified"])
            for row in final_rows
        )
        all_gain = all(bool(row["certified_small_gain_pass"]) for row in final_rows)
        all_complete = all(bool(row["complete_circle_covered"]) for row in final_rows)
        all_positive_moats = all(
            float(row["lifted_finite_section_moat_lower"]) > 0
            for row in final_rows
        )
        all_zero_excluded = all(
            bool(row["zero_outside_enclosed_region"])
            for row in final_rows
        )
        no_sampled_gate = all(
            not bool(row["sampled_values_used_in_theorem_gate"])
            for row in final_rows
        )
        total_computed_multiplicity = sum(
            int(row["schur_diagonal_algebraic_count"])
            for row in final_rows
        )
        total_expected_multiplicity = sum(
            int(row["expected_multiplicity"]) for row in final_rows
        )
        if not (
            all_theorem
            and all_exact_dyadic_upper_triangular
            and all_memberships
            and all_count_transports
            and all_counts
            and all_matches
            and all_rank_transfers
            and all_gain
            and all_complete
            and all_positive_moats
            and all_zero_excluded
            and no_sampled_gate
            and total_computed_multiplicity == 30
            and total_expected_multiplicity == 30
        ):
            raise AssertionError("The terminal twenty-four-target audit failed.")
        precision_roles = _precision_role_report(
            final_rows,
            theorem_precision_bits=config.precision_bits,
        )
        _write_csv(paths["certificate"], final_rows)
        witness_rows = _laurent_witness_rows(final_rows)
        _write_csv(paths["laurent_witnesses"], witness_rows)
        persisted_witness_rows = _read_csv(paths["laurent_witnesses"])
        persisted_certificate_rows = _read_csv(paths["certificate"])
        persisted_mode_rows = _read_csv(paths["laurent_modes"])
        if not _laurent_witness_records_are_reusable(
            persisted_witness_rows,
            persisted_certificate_rows,
            persisted_mode_rows,
            expected_precision_bits=config.precision_bits,
        ):
            raise AssertionError(
                "The persisted Laurent reconstruction manifest failed validation."
            )
        coefficient_matrix_count = sum(
            int(row["coefficient_matrix_count"]) for row in witness_rows
        )
        all_digest_matches = all(
            bool(row["digest_matches_recorded_reference"])
            for row in witness_rows
        )
        if coefficient_matrix_count != 300:
            raise AssertionError(
                "The clean-room Laurent reconstruction has the wrong tensor count."
            )
        report = {
            "certificate_schema": SCHEMA,
            "map_label": MAP_LABEL,
            **asdict(config),
            "epsilon_upper_text": epsilon_text,
            "eta_A_upper_text": matrix_report["eta_A_upper_text"],
            "eta_schur_upper_text": schur_report["eta_schur_upper_text"],
            "target_count": 24,
            "total_certified_algebraic_multiplicity": (
                total_computed_multiplicity
            ),
            "total_expected_algebraic_multiplicity": (
                total_expected_multiplicity
            ),
            "schur_count_target_count": 24,
            "schur_triangular_moat_target_count": schur_moat_count,
            "laurent_moat_target_count": laurent_moat_count,
            "laurent_witness_count": len(witness_rows),
            "laurent_coefficient_matrix_count": coefficient_matrix_count,
            "all_laurent_witnesses_reconstructed_in_recorded_run": True,
            "laurent_internal_digest_bindings_certified": True,
            "all_laurent_digests_match_recorded_reference": all_digest_matches,
            "laurent_digests_used_in_any_theorem_gate": False,
            "laurent_reference_digest_policy": (
                "diagnostic historical byte parity only; generated coefficient "
                "bytes are bound internally to the persisted certificate and the "
                "theorem uses the exact-dyadic residual gate"
            ),
            "laurent_candidate_storage_policy": (
                "transient coefficient tensors; persisted full-tensor digests "
                "and exact-dyadic modewise validation bounds"
            ),
            "finite_geometry_generation_status": (
                "forced_source_reconstruction"
                if force
                else "source_reconstruction_after_cache_miss"
            ),
            "binary64_candidate_runtime": _runtime_fingerprint(),
            "all_finite_counts_schur_derived": True,
            "q_gap_derived_le_target": True,
            "exact_dyadic_schur_below_diagonal_entry_count": schur_report[
                "exact_dyadic_schur_below_diagonal_entry_count"
            ],
            "exact_dyadic_schur_below_diagonal_zero_count": schur_report[
                "exact_dyadic_schur_below_diagonal_zero_count"
            ],
            "exact_dyadic_schur_below_diagonal_all_zero": schur_report[
                "exact_dyadic_schur_below_diagonal_all_zero"
            ],
            "exact_dyadic_schur_upper_triangular_certified": schur_report[
                "exact_dyadic_schur_upper_triangular_certified"
            ],
            "all_24_targets_theorem_certified": all_theorem,
            "all_schur_diagonal_memberships_certified": all_memberships,
            "all_finite_count_transports_certified": all_count_transports,
            "all_finite_counts_certified": all_counts,
            "all_finite_counts_match_expected": all_matches,
            "all_finite_to_exact_rank_transfers_certified": all_rank_transfers,
            "all_small_gain_tests_pass": all_gain,
            "small_gain_precision_roles": precision_roles,
            "all_complete_circles_covered": all_complete,
            "all_complete_circle_moats_positive": all_positive_moats,
            "zero_excluded_from_every_contour": all_zero_excluded,
            "sampled_values_used_in_any_theorem_gate": not no_sampled_gate,
            "cache_sha256": schur_report["cache_sha256"],
            "matrix_midpoint_sha256": matrix_report["midpoint_sha256"],
            "matrix_payload_schema": payload_metadata.get("schema"),
            "matrix_precision_bits": matrix_report["precision_bits"],
            "contour_precision_bits": int(config.precision_bits),
            "flint_threads": int(config.flint_threads),
            "source_hashes": source_hashes,
            "geometry_input_hashes": geometry_input_hashes,
            "input_hashes": input_hashes,
            "artifacts": {key: str(value) for key, value in paths.items()},
            "laurent_witness_reconstruction_sha256": sha256_file(
                paths["laurent_witnesses"]
            ),
            "elapsed_seconds": time.time() - started,
            "status": "theorem-certified twenty-four-target Riesz-rank package",
        }
        _write_json(paths["report"], report)
        return {**report, "execution_status": "computed_and_certified"}
    finally:
        flint.ctx.prec, flint.ctx.threads = old_precision, old_threads


def _default_paths(root: Path) -> dict[str, Path]:
    output = root / "Numerics" / "outputs" / "blaschke_deformation_certifier"
    stem = "blaschke_deformation_balanced_hardy_reference_N600_M610_bits2048"
    return {
        "output": output,
        "payload": output / "data" / f"{stem}.pkl.gz",
        "midpoint": output / "data" / f"{stem}_diagnostic_midpoint.npz",
        "matrix_report": output / "reports" / f"{stem}_certificate.json",
        "epsilon_certificate": (
            output
            / "data"
            / "branch_image_balanced_response_prefactor_candidate_row_N600_M610.csv"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    paths = _default_paths(root)
    report = certify_all_target_contours(
        config=ContourCertificateConfig(),
        output_dir=paths["output"],
        matrix_payload_path=paths["payload"],
        matrix_midpoint_path=paths["midpoint"],
        matrix_report_path=paths["matrix_report"],
        epsilon_certificate_path=paths["epsilon_certificate"],
        source_files=(
            Path(__file__).resolve(),
            root / "Numerics" / "blaschke_deformation_spectral_certification.py",
            root / "Numerics" / "blaschke_deformation_certification.py",
        ),
        force=bool(args.force),
        progress=not bool(args.quiet),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
