"""Validated finite Hardy matrix for the ASBJ24 Blaschke deformation.

This module certifies the mathematical finite block at one fixed geometry.  It
uses Arb for the Gauss--Legendre rule, inverse branches, transfer weights,
Legendre recurrences, the scaled Legendre--Chebyshev connection and every
matrix product.  The resulting interval matrix is replaced by an exact-dyadic
midpoint only after the induced two-norm replacement error has been enclosed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, getcontext
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import time
from typing import Callable, Iterable

import numpy as np

try:
    import flint
    from flint import arb, arb_mat
except ImportError as exc:  # pragma: no cover - checked by the notebook
    raise ImportError(
        "blaschke_deformation_spectral_certification requires python-flint"
    ) from exc


MAP_LABEL = "blaschke_mu_0p3"
SCHEMA = "blaschke-deformation-exact-dyadic-hardy-reference-v1"


@dataclass(frozen=True)
class HardyMatrixCertificateConfig:
    """Fixed finite-section geometry and arithmetic settings."""

    N: int = 600
    M: int = 610
    rho: str = "2.725"
    r: str = "2.473669807791324"
    mu: str = "0.3"
    precision_bits: int = 2048
    flint_threads: int = 24
    guard_digits: int = 30
    diagnostic_resolution: str = "1e-20"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_source_hashes(source_files: Iterable[Path]) -> dict[str, str]:
    """Hash source files under stable, location-independent logical names.

    The thesis-mathematics notebook materialises inline helpers in a fresh
    temporary directory on every run.  Resolved absolute paths therefore do
    not identify source content and must not participate in checkpoint keys.
    Helper filenames are unique in this deployment and form stable logical
    names both for standalone and inline execution.
    """

    result: dict[str, str] = {}
    for source in source_files:
        path = Path(source)
        key = path.name
        if key in result:
            raise ValueError(f"Duplicate logical source key: {key}")
        result[key] = sha256_file(path)
    return result


def normalise_source_hashes(source_hashes: dict[str, str]) -> dict[str, str]:
    """Normalise legacy absolute-path provenance to logical filenames."""

    result: dict[str, str] = {}
    for raw_key, digest in source_hashes.items():
        key = Path(raw_key).name
        if key in result and result[key] != digest:
            raise ValueError(f"Conflicting source hashes for logical key: {key}")
        result[key] = str(digest)
    return result


def _upper_decimal(value: arb) -> Decimal:
    midpoint, radius, exponent = value.upper().mid_rad_10exp()
    return (Decimal(int(midpoint)) + abs(Decimal(int(radius)))) * (
        Decimal(10) ** int(exponent)
    )


def upper_float(value: arb) -> float:
    """Convert an Arb upper endpoint to an outward-rounded binary float."""

    getcontext().prec = 120
    return math.nextafter(float(_upper_decimal(value)), math.inf)


def upper_text(value: arb, digits: int = 120) -> str:
    return value.upper().str(int(digits), radius=False, more=True)


def _write_csv_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow(row)
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _orthonormal_legendre_values(count: int, x: arb) -> list[arb]:
    '''Explanation: The first N orthonormal Legendre polynomials are the trial and test coordinates of the finite transfer section. Their recurrence evaluates the whole basis while preserving the L2 normalisation used in the thesis.
Functionality: Evaluate the first ``count`` orthonormal Legendre modes by recurrence.'''

    count = int(count)
    if count <= 0:
        return []
    values = [arb(0) for _ in range(count)]
    values[0] = arb(1)
    if count >= 2:
        values[1] = x
        for degree in range(1, count - 1):
            values[degree + 1] = (
                (2 * degree + 1) * x * values[degree]
                - degree * values[degree - 1]
            ) / (degree + 1)
    return [
        (arb(2 * degree + 1) / 2).sqrt() * values[degree]
        for degree in range(count)
    ]


def _gauss_legendre_rule(order: int) -> tuple[list[arb], list[arb]]:
    '''Explanation: Gauss--Legendre nodes are the roots of a Legendre polynomial and integrate low-degree polynomials exactly. Isolating every root and weight in Arb makes the finite assembly a certified quadrature object.
Functionality: Certify the complete Gauss--Legendre rule by Arb root isolation.'''

    order = int(order)
    nodes: list[arb] = []
    weights: list[arb] = []
    for node_index in range(order):
        root_index = order - 1 - node_index
        node, weight = arb.legendre_p_root(order, root_index, weight=True)
        if weight.lower() <= 0:
            raise ArithmeticError("A certified Gauss weight is not positive.")
        nodes.append(node)
        weights.append(weight)
    if not all(
        nodes[index].upper() < nodes[index + 1].lower()
        for index in range(order - 1)
    ):
        raise ArithmeticError("Certified Gauss-node intervals overlap.")
    if not sum(weights, arb(0)).contains(2):
        raise ArithmeticError("Certified Gauss weights do not enclose mass two.")
    return nodes, weights


def _real_branch_values(x: arb, mu: arb, pi: arb) -> tuple[tuple[arb, arb], ...]:
    '''Explanation: At each real quadrature node, both inverse images and transfer weights determine the two weighted pullbacks. Rigorous real enclosures ensure the assembled Galerkin entries contain their exact mathematical values.
Functionality: Certify both real inverse branches and transfer weights at one node.'''

    cosine = (pi * x / 2).cos()
    sine = (pi * x / 2).sin()
    argument = mu * cosine
    denominator_square = 1 - argument * argument
    if denominator_square.lower() <= 0:
        raise ArithmeticError("The Blaschke branch denominator is not positive.")
    denominator = denominator_square.sqrt()
    angle = argument.acos() / pi
    branches = (
        (x / 2 - angle, arb("0.5") - mu * sine / (2 * denominator)),
        (x / 2 + angle, arb("0.5") + mu * sine / (2 * denominator)),
    )
    for tau, phi in branches:
        if tau.lower() < -1 or tau.upper() > 1:
            raise ArithmeticError("A certified inverse branch left the real interval.")
        if phi.lower() <= 0:
            raise ArithmeticError("A certified transfer weight is not positive.")
    if not branches[0][0].upper() < branches[1][0].lower():
        raise ArithmeticError("Certified inverse branches overlap.")
    return branches


def _connection_matrix(count: int, hardy_radius: arb) -> arb_mat:
    '''Explanation: The finite Legendre matrix and the Hardy-gauge matrix represent the same operator section in two bases. A rigorous connection matrix links them without changing the enclosed eigenvalue problem.
Functionality: Construct the scaled Legendre--Chebyshev connection in Arb.'''

    count = int(count)
    coefficients: list[list[arb]] = []
    if count:
        coefficients.append([arb(1)])
    if count >= 2:
        coefficients.append([arb(0), arb(1)])
    for degree in range(1, count - 1):
        multiplied = [arb(0)] * (len(coefficients[degree]) + 1)
        for mode, value in enumerate(coefficients[degree]):
            if mode == 0:
                multiplied[1] += value
            else:
                multiplied[mode + 1] += value / 2
                multiplied[mode - 1] += value / 2
        new = [arb(0)] * (degree + 2)
        for mode, value in enumerate(multiplied):
            new[mode] += (2 * degree + 1) * value / (degree + 1)
        for mode, value in enumerate(coefficients[degree - 1]):
            new[mode] -= degree * value / (degree + 1)
        coefficients.append(new)

    column_scales = [
        (arb(2 * column + 1) / 2).sqrt() * hardy_radius ** (-column)
        for column in range(count)
    ]
    row_scales = [arb(1)] + [
        (
            hardy_radius ** (2 * row) + hardy_radius ** (-2 * row)
        ).sqrt()
        / 2
        for row in range(1, count)
    ]
    return arb_mat(
        [
            [
                row_scales[row]
                * column_scales[column]
                * coefficients[column][row]
                if row < len(coefficients[column])
                else arb(0)
                for column in range(count)
            ]
            for row in range(count)
        ]
    )


def assemble_interval_hardy_matrix(
    config: HardyMatrixCertificateConfig,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[arb_mat, dict[str, object]]:
    '''Explanation: Integrating weighted branch pullbacks against Legendre test modes gives the finite Galerkin operator. Conjugating by the certified basis connection places that entire interval matrix in the packet Hardy gauge used by the contour proof.
Functionality: Assemble the complete mathematical Hardy-gauge block in Arb.'''

    N, M = int(config.N), int(config.M)
    if N < 1 or M < N:
        raise ValueError("The matrix certificate requires M at least N at positive order.")
    old_precision, old_threads = flint.ctx.prec, flint.ctx.threads
    flint.ctx.prec = int(config.precision_bits)
    flint.ctx.threads = int(config.flint_threads)
    started = time.time()
    try:
        mu, pi = arb(config.mu), arb.pi()
        nodes, weights = _gauss_legendre_rule(M)
        testing = arb_mat(M, N)
        response = arb_mat(M, N)
        maximum_node_radius = arb(0)
        maximum_branch_radius = arb(0)
        minimum_weight = None
        for node_index, (node, weight) in enumerate(zip(nodes, weights)):
            maximum_node_radius = max(maximum_node_radius, node.rad())
            minimum_weight = weight if minimum_weight is None else min(minimum_weight, weight)
            p_node = _orthonormal_legendre_values(N, node)
            response_values = [arb(0) for _ in range(N)]
            for tau, phi in _real_branch_values(node, mu, pi):
                maximum_branch_radius = max(maximum_branch_radius, tau.rad())
                p_tau = _orthonormal_legendre_values(N, tau)
                for degree in range(N):
                    response_values[degree] += phi * p_tau[degree]
            for degree in range(N):
                testing[node_index, degree] = p_node[degree]
                response[node_index, degree] = weight * response_values[degree]
            if progress is not None:
                progress(node_index + 1, M)

        transfer = testing.transpose() * response
        del testing, response

        hardy_radius = arb(config.r)
        powers = [hardy_radius**degree for degree in range(N)]
        pure_scaled = arb_mat(N, N)
        for row in range(N):
            for column in range(N):
                pure_scaled[row, column] = (
                    powers[row] * transfer[row, column] / powers[column]
                )
        del transfer, powers

        connection = _connection_matrix(N, hardy_radius)
        connection_inverse = connection.inv()
        hardy = connection * pure_scaled * connection_inverse
        if not all(
            hardy[row, column].is_finite
            for row in range(N)
            for column in range(N)
        ):
            raise ArithmeticError("The validated Hardy block contains a non-finite ball.")

        return hardy, {
            "gauss_nodes_disjoint": True,
            "gauss_weight_sum_contains_two": True,
            "branch_values_certified": True,
            "branch_weights_positive": True,
            "branch_ordering_certified": True,
            "legendre_recurrence_certified": True,
            "connection_inverse_certified": True,
            "all_matrix_products_certified": True,
            "maximum_gauss_node_radius_upper": upper_float(maximum_node_radius),
            "maximum_branch_radius_upper": upper_float(maximum_branch_radius),
            "minimum_gauss_weight_lower": float(minimum_weight.lower()),
            "assembly_seconds": time.time() - started,
        }
    finally:
        flint.ctx.prec, flint.ctx.threads = old_precision, old_threads


def _dyadic_midpoint_payload(
    interval_matrix: arb_mat,
    config: HardyMatrixCertificateConfig,
    source_hashes: dict[str, str],
) -> tuple[dict[str, object], dict[str, object], np.ndarray]:
    '''Explanation: The deployed finite matrix is stored as exact dyadic midpoints for deterministic replay. The discarded interval radii are not forgotten: their norm becomes an explicit replacement error in the total perturbation budget.
Functionality: Extract an exact-dyadic midpoint and certify its replacement error.'''

    N = interval_matrix.nrows()
    if interval_matrix.ncols() != N:
        raise ValueError("The Hardy matrix must be square.")
    old_precision = flint.ctx.prec
    flint.ctx.prec = int(config.precision_bits)
    try:
        row_sums = [arb(0) for _ in range(N)]
        column_sums = [arb(0) for _ in range(N)]
        frobenius_square = arb(0)
        mantissas: list[int] = []
        exponents: list[int] = []
        midpoint_float = np.empty((N, N), dtype=np.float64)
        digest = hashlib.sha256()

        for row in range(N):
            for column in range(N):
                midpoint = interval_matrix[row, column].mid()
                mantissa, exponent = midpoint.man_exp()
                mantissa_int, exponent_int = int(mantissa), int(exponent)
                reference = arb(mantissa_int) * arb(2) ** exponent_int
                error = abs(interval_matrix[row, column] - reference).upper()
                row_sums[row] += error
                column_sums[column] += error
                frobenius_square += error * error
                mantissas.append(mantissa_int)
                exponents.append(exponent_int)
                midpoint_float[row, column] = float(reference)
                digest.update(f"{mantissa_int}:{exponent_int};".encode("ascii"))

        norm_one = max(column_sums)
        norm_infinity = max(row_sums)
        eta = (norm_one * norm_infinity).sqrt().upper()
        frobenius = frobenius_square.sqrt().upper()
    finally:
        flint.ctx.prec = old_precision
    if not np.isfinite(midpoint_float).all():
        raise ArithmeticError("The diagnostic binary64 midpoint contains a non-finite value.")

    payload = {
        "schema": SCHEMA,
        "map_label": MAP_LABEL,
        "config": asdict(config),
        "shape": (N, N),
        "mantissas": mantissas,
        "exponents": exponents,
        "eta_A_upper_text": upper_text(eta),
        "norm_one_upper_text": upper_text(norm_one),
        "norm_infinity_upper_text": upper_text(norm_infinity),
        "frobenius_upper_text": upper_text(frobenius),
        "midpoint_sha256": digest.hexdigest(),
        "source_hashes": dict(source_hashes),
    }
    bounds = {
        "eta_A_upper": upper_float(eta),
        "eta_A_upper_text": payload["eta_A_upper_text"],
        "norm_one_upper": upper_float(norm_one),
        "norm_infinity_upper": upper_float(norm_infinity),
        "frobenius_upper": upper_float(frobenius),
        "midpoint_sha256": payload["midpoint_sha256"],
    }
    return payload, bounds, midpoint_float


def _precision_audit(config: HardyMatrixCertificateConfig) -> dict[str, object]:
    '''Explanation: Large gauge scalings can consume many binary digits before the desired matrix accuracy is reached. The precision audit checks that Arb has enough resolution and guard digits for those losses, preventing a vacuous enclosure.
Functionality: Compare the available Arb precision with twice the transport scaling span, requested resolution, and guard-digit budget.'''
    available_digits = float(config.precision_bits) * math.log10(2)
    scale_span = (int(config.N) - 1) * math.log10(float(config.r))
    resolution_digits = -math.log10(float(config.diagnostic_resolution))
    # The explicit route forms T B T^{-1}.  Both dense products can expose the
    # full scaled-coordinate dynamic range before cancellation, so a safe
    # arithmetic budget pays the scaling span twice.
    cancellation_span_count = 2
    required_digits = (
        cancellation_span_count * scale_span
        + resolution_digits
        + int(config.guard_digits)
    )
    return {
        "available_decimal_digits": available_digits,
        "pure_scaling_decimal_span": scale_span,
        "transport_cancellation_span_count": cancellation_span_count,
        "transport_cancellation_decimal_budget": cancellation_span_count * scale_span,
        "diagnostic_resolution": str(config.diagnostic_resolution),
        "diagnostic_resolution_digits": resolution_digits,
        "guard_digits": int(config.guard_digits),
        "required_decimal_digits": required_digits,
        "precision_margin_digits": available_digits - required_digits,
        "precision_budget_pass": available_digits > required_digits,
    }


def _payload_is_valid(
    payload_path: Path,
    summary: dict[str, object],
    config: HardyMatrixCertificateConfig,
    source_hashes: dict[str, str],
) -> bool:
    try:
        with gzip.open(payload_path, "rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return False
    if payload.get("schema") != SCHEMA:
        return False
    if payload.get("config") != asdict(config):
        return False
    try:
        stored_source_hashes = normalise_source_hashes(
            dict(payload.get("source_hashes", {}))
        )
    except (TypeError, ValueError):
        return False
    if stored_source_hashes != source_hashes:
        return False
    if payload.get("midpoint_sha256") != summary.get("midpoint_sha256"):
        return False
    expected_entries = int(config.N) * int(config.N)
    return (
        len(payload.get("mantissas", ())) == expected_entries
        and len(payload.get("exponents", ())) == expected_entries
        and bool(summary.get("mathematical_block_enclosed"))
        and bool(summary.get("reference_is_exact_dyadic"))
    )


def _migrate_cached_source_provenance(
    *,
    payload_path: Path,
    summary_path: Path,
    report_path: Path,
    summary: dict[str, object],
    source_hashes: dict[str, str],
) -> dict[str, object]:
    """Rewrite a valid legacy checkpoint with stable logical source keys."""

    with gzip.open(payload_path, "rb") as handle:
        payload = pickle.load(handle)
    payload_hashes = dict(payload.get("source_hashes", {}))
    summary_hashes = dict(summary.get("source_hashes", {}))
    if payload_hashes == source_hashes and summary_hashes == source_hashes:
        return summary

    payload["source_hashes"] = dict(source_hashes)
    temporary_payload = payload_path.with_suffix(payload_path.suffix + ".tmp")
    with gzip.open(temporary_payload, "wb", compresslevel=3) as handle:
        pickle.dump(payload, handle, protocol=5)
    temporary_payload.replace(payload_path)

    migrated = dict(summary)
    migrated["source_hashes"] = dict(source_hashes)
    migrated["payload_sha256"] = sha256_file(payload_path)
    _write_csv_row(
        summary_path,
        {
            key: json.dumps(value, sort_keys=True)
            if isinstance(value, dict)
            else value
            for key, value in migrated.items()
        },
    )
    _write_json(report_path, migrated)
    return migrated


def build_or_load_hardy_matrix_certificate(
    *,
    config: HardyMatrixCertificateConfig,
    output_dir: Path,
    source_files: Iterable[Path],
    force: bool = False,
    require_precision_budget: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    '''Explanation: The contour theorem needs one authoritative finite Hardy matrix together with a proof of how accurately it represents the interval assembly. Transactional construction or hash-checked loading keeps that mathematical object reproducible.
Functionality: Transactionally build or validate the fixed Hardy-matrix certificate.'''

    output_dir = Path(output_dir).resolve()
    data_dir, report_dir = output_dir / "data", output_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"blaschke_deformation_balanced_hardy_reference_"
        f"N{int(config.N)}_M{int(config.M)}_bits{int(config.precision_bits)}"
    )
    payload_path = data_dir / f"{stem}.pkl.gz"
    midpoint_path = data_dir / f"{stem}_diagnostic_midpoint.npz"
    summary_path = data_dir / f"{stem}_certificate.csv"
    report_path = report_dir / f"{stem}_certificate.json"
    source_hashes = logical_source_hashes(source_files)
    precision = _precision_audit(config)
    if require_precision_budget and not precision["precision_budget_pass"]:
        raise ArithmeticError("The requested Arb precision fails the explicit budget audit.")

    if (
        not force
        and payload_path.exists()
        and midpoint_path.exists()
        and report_path.exists()
    ):
        summary = json.loads(report_path.read_text(encoding="utf-8"))
        if _payload_is_valid(payload_path, summary, config, source_hashes):
            summary = _migrate_cached_source_provenance(
                payload_path=payload_path,
                summary_path=summary_path,
                report_path=report_path,
                summary=summary,
                source_hashes=source_hashes,
            )
            return {
                **summary,
                "execution_status": "reused_validated_checkpoint",
                "payload_path": str(payload_path),
                "midpoint_path": str(midpoint_path),
                "summary_path": str(summary_path),
                "report_path": str(report_path),
            }

    started = time.time()
    interval_matrix, assembly_audit = assemble_interval_hardy_matrix(
        config, progress=progress
    )
    payload, bounds, midpoint_float = _dyadic_midpoint_payload(
        interval_matrix, config, source_hashes
    )
    del interval_matrix

    temporary_payload = payload_path.with_suffix(payload_path.suffix + ".tmp")
    with gzip.open(temporary_payload, "wb", compresslevel=3) as handle:
        pickle.dump(payload, handle, protocol=5)
    temporary_payload.replace(payload_path)

    temporary_midpoint = midpoint_path.with_suffix(midpoint_path.suffix + ".tmp")
    with temporary_midpoint.open("wb") as handle:
        np.savez_compressed(
            handle,
            A_N_circ_binary64_diagnostic=midpoint_float,
            N=np.asarray(config.N),
            M=np.asarray(config.M),
            rho=np.asarray(config.rho),
            r=np.asarray(config.r),
            precision_bits=np.asarray(config.precision_bits),
            midpoint_sha256=np.asarray(payload["midpoint_sha256"]),
            authoritative_exact_dyadic_payload=np.asarray(str(payload_path)),
        )
    temporary_midpoint.replace(midpoint_path)

    summary = {
        "schema": SCHEMA,
        "map_label": MAP_LABEL,
        **asdict(config),
        **precision,
        **assembly_audit,
        **bounds,
        "entry_count": int(config.N) * int(config.N),
        "reference_is_exact_dyadic": True,
        "binary64_midpoint_is_diagnostic_only": True,
        "mathematical_block_enclosed": True,
        "matrix_enclosure_certified": True,
        "precision_budget_enforced": bool(require_precision_budget),
        "spectral_use_ready": (
            float(bounds["eta_A_upper"]) < float(config.diagnostic_resolution)
        ),
        "source_hashes": source_hashes,
        "payload_sha256": sha256_file(payload_path),
        "payload_path": str(payload_path),
        "midpoint_path": str(midpoint_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "elapsed_seconds": time.time() - started,
        "status": "interval-certified mathematical Hardy block with exact-dyadic midpoint",
    }
    _write_csv_row(
        summary_path,
        {
            key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
            for key, value in summary.items()
        },
    )
    _write_json(report_path, summary)
    return {**summary, "execution_status": "computed_and_certified"}


def load_exact_dyadic_midpoint(payload_path: Path) -> tuple[arb_mat, dict[str, object]]:
    '''Explanation: Reconstructing the stored dyadic entries exactly gives every auditor the identical finite matrix, independent of decimal parsing or platform rounding. That exact matrix is the anchor for the finite Schur and Laurent certificates.
Functionality: Reconstruct the authoritative exact-dyadic matrix in Arb arithmetic.'''

    with gzip.open(Path(payload_path), "rb") as handle:
        payload = pickle.load(handle)
    if payload.get("schema") != SCHEMA:
        raise ValueError("Unrecognised exact-dyadic Hardy-reference schema.")
    rows, columns = (int(value) for value in payload["shape"])
    mantissas = payload["mantissas"]
    exponents = payload["exponents"]
    if len(mantissas) != rows * columns or len(exponents) != rows * columns:
        raise ValueError("The exact-dyadic Hardy-reference payload is incomplete.")
    old_precision = flint.ctx.prec
    flint.ctx.prec = int(payload["config"]["precision_bits"])
    try:
        matrix = arb_mat(rows, columns)
        digest = hashlib.sha256()
        for index, (mantissa, exponent) in enumerate(zip(mantissas, exponents)):
            row, column = divmod(index, columns)
            matrix[row, column] = arb(int(mantissa)) * arb(2) ** int(exponent)
            digest.update(f"{int(mantissa)}:{int(exponent)};".encode("ascii"))
    finally:
        flint.ctx.prec = old_precision
    if digest.hexdigest() != payload["midpoint_sha256"]:
        raise ArithmeticError("The exact-dyadic Hardy-reference digest failed.")
    return matrix, payload
