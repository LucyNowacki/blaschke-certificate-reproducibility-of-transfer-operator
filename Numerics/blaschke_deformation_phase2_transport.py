"""Certified finite Legendre--Chebyshev transport for Phase 2.

The midpoint connection matrix is inverted numerically, but the resulting
inverse is only a witness.  Arb recomputes the residual ``I - T B0`` and all
theorem-facing endpoints remain directed until they are serialised.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
from typing import Any

import flint
from flint import arb, arb_mat
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

try:
    from .blaschke_deformation_phase2_geometry import lower_text, upper_text
except ImportError:
    from blaschke_deformation_phase2_geometry import lower_text, upper_text


MAP_LABEL = "blaschke_mu_0p3"
PRODUCER_SCHEMA = "phase2-transport-v2"
INVERSE_BLAS_THREADS = 24
LOCKED_OPENBLAS_RUNTIME = {
    "user_api": "blas",
    "internal_api": "openblas",
    "version": "0.3.30",
    "threading_layer": "pthreads",
    "architecture": "Haswell",
}


@dataclass(frozen=True)
class Phase2TransportConfig:
    N: int
    r: str
    rho: str
    r_tau: str
    geometry_configuration_digest: str
    precision_bits: int = 384
    flint_threads: int = 24


@dataclass(frozen=True)
class Phase2TransportResult:
    record: dict[str, Any]
    csv_path: Path
    report_path: Path
    witness_path: Path
    witness_digest: str


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


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary_name, **arrays)
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


def _verified_openblas_runtime(
    expected_threads: int,
    *,
    stage: str,
) -> dict[str, Any]:
    """Return portable evidence for the exact locked OpenBLAS runtime.

    The replay keeps every process and worker at one BLAS thread.  The two
    explicitly authorised binary64 compatibility kernels temporarily use 24
    threads, and only those kernels may do so.  Checking the live library here
    prevents an ABI-compatible but numerically different BLAS implementation,
    microarchitecture dispatch, or thread count from silently changing the
    retained witness bytes.
    """

    if int(expected_threads) < 1:
        raise ValueError("The expected BLAS thread count must be positive.")
    records = [
        record
        for record in threadpool_info()
        if record.get("user_api") == "blas"
    ]
    invalid = [
        record
        for record in records
        if any(
            str(record.get(key)) != value
            for key, value in LOCKED_OPENBLAS_RUNTIME.items()
        )
        or int(record.get("num_threads") or 0) != int(expected_threads)
    ]
    if len(records) != 1 or invalid:
        raise RuntimeError(
            "The finite transport inverse requires exactly one live OpenBLAS "
            "0.3.30 pthreads Haswell runtime at "
            f"{int(expected_threads)} threads during {stage}; observed "
            f"{records!r}."
        )
    portable = {
        key: records[0].get(key)
        for key in (
            "user_api",
            "internal_api",
            "version",
            "threading_layer",
            "architecture",
            "num_threads",
        )
    }
    return {
        "stage": str(stage),
        "expected_threads": int(expected_threads),
        "runtime": portable,
    }


def _invert_midpoint_with_locked_blas(
    midpoint: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Invert one midpoint under the exact scoped compatibility runtime."""

    outer_before = _verified_openblas_runtime(1, stage="outer-before-inverse")
    with threadpool_limits(limits=INVERSE_BLAS_THREADS, user_api="blas"):
        inside = _verified_openblas_runtime(
            INVERSE_BLAS_THREADS,
            stage="scoped-midpoint-inverse",
        )
        inverse = np.linalg.inv(midpoint)
    outer_after = _verified_openblas_runtime(1, stage="outer-after-inverse")
    evidence = {
        "schema": "numerics1-scoped-openblas-runtime-v1",
        "operation": "numpy.linalg.inv(midpoint)",
        "scope_threads": INVERSE_BLAS_THREADS,
        "outer_threads": 1,
        "outer_before": outer_before,
        "inside": inside,
        "outer_after": outer_after,
        "portable_fields_only": True,
    }
    return inverse, evidence


def _alpha_values(N: int) -> np.ndarray:
    '''Explanation: Central-binomial coefficients encode the Legendre-to-Chebyshev packet change of basis. The floating recurrence gives a fast candidate matrix whose conditioning can be explored before rigorous certification.
Functionality: Generate binary64 central-binomial coefficients alpha_j by recurrence for the diagnostic connection matrix.'''
    values = np.empty(int(N), dtype=np.float64)
    values[0] = 1.0
    for index in range(int(N) - 1):
        values[index + 1] = values[index] * (2 * index + 1) / (2 * index + 2)
    return values


def connection_matrix_float(N: int, r_text: str) -> np.ndarray:
    '''Explanation: This matrix expresses finite scaled Legendre coordinates in the packet gauge used by the Hardy-space proof. Its numerical version proposes the inverse and condition scale that Arb later verifies.
Functionality: Return the finite scaled-Legendre to packet connection matrix.'''

    N = int(N)
    r = float(r_text)
    alpha = _alpha_values(N)
    matrix = np.zeros((N, N), dtype=np.float64)
    for n in range(N):
        scale = math.sqrt((2 * n + 1) / 2)
        if n % 2 == 0:
            half = n // 2
            matrix[0, n] = scale * alpha[half] * alpha[half] * r ** (-n)
        for m in range(1, n + 1):
            if (n - m) % 2:
                continue
            left = (n - m) // 2
            right = (n + m) // 2
            packet_scale = r ** (m - n) * math.sqrt(1 + r ** (-4 * m))
            matrix[m, n] = scale * alpha[left] * alpha[right] * packet_scale
    return matrix


def _alpha_values_arb(N: int) -> list[arb]:
    '''Explanation: The same central-binomial recurrence can be evaluated with interval arithmetic, enclosing every basis coefficient. Those enclosures remove coefficient-rounding ambiguity from the transport proof.
Functionality: Generate rigorously enclosed central-binomial coefficients alpha_j by exact Arb recurrence.'''
    values = [arb(1)]
    for index in range(int(N) - 1):
        values.append(
            values[-1] * arb(2 * index + 1) / arb(2 * index + 2)
        )
    return values


def connection_matrix_arb(N: int, r_text: str) -> arb_mat:
    '''Explanation: A rigorous change-of-basis matrix is needed because matrix errors are first bounded in scaled Legendre coordinates but the spectral moat lives in packet coordinates. This object performs that finite transport with certified entries.
Functionality: Construct the rigorous finite scaled-Legendre to Chebyshev-packet connection matrix in Arb arithmetic.'''
    N = int(N)
    r = arb(str(r_text))
    alpha = _alpha_values_arb(N)
    matrix = arb_mat(N, N)
    for n in range(N):
        scale = (arb(2 * n + 1) / 2).sqrt()
        if n % 2 == 0:
            half = n // 2
            matrix[0, n] = (
                scale * alpha[half] * alpha[half] * r ** (-n)
            )
        for m in range(1, n + 1):
            if (n - m) % 2:
                continue
            left = (n - m) // 2
            right = (n + m) // 2
            packet_scale = r ** (m - n) * (1 + r ** (-4 * m)).sqrt()
            matrix[m, n] = (
                scale * alpha[left] * alpha[right] * packet_scale
            )
    return matrix


def _connection_entry_arb(
    row: int,
    column: int,
    *,
    r: arb,
    alpha: list[arb],
) -> arb:
    '''Explanation: Each nonzero connection coefficient combines exact combinatorial factors and gauge scalings. Certifying it individually is the atomic step from which the full transport matrix is assembled.
Functionality: Return one rigorous nonzero connection entry.'''

    if row > column or (column - row) % 2:
        return arb(0)
    scale = (arb(2 * column + 1) / 2).sqrt()
    if row == 0:
        half = column // 2
        return scale * alpha[half] * alpha[half] * r ** (-column)
    left = (column - row) // 2
    right = (column + row) // 2
    packet_scale = r ** (row - column) * (1 + r ** (-4 * row)).sqrt()
    return scale * alpha[left] * alpha[right] * packet_scale


def _partition_rows(N: int, workers: int) -> list[tuple[int, int]]:
    workers = max(1, min(int(workers), int(N)))
    base, remainder = divmod(int(N), workers)
    blocks = []
    start = 0
    for worker in range(workers):
        width = base + (1 if worker < remainder else 0)
        blocks.append((start, start + width))
        start += width
    if start != int(N):
        raise AssertionError("The transport row partition is incomplete.")
    return blocks


def _transport_row_block(
    arguments: tuple[str, np.ndarray, int, int, int, int]
) -> dict[str, str | int]:
    '''Explanation: Frobenius norms are sums of squared entries, so the transport and inverse candidates can be certified independently by row blocks. This decomposition preserves the global norm while making the large finite proof reproducible.
Functionality: Evaluate rigorous Frobenius-square contributions for a row block.'''

    r_text, inverse, N, precision_bits, start, stop = arguments
    flint.ctx.prec = int(precision_bits)
    if hasattr(flint.ctx, "threads"):
        flint.ctx.threads = 1
    r = arb(str(r_text))
    alpha = _alpha_values_arb(int(N))
    norm_connection_squared = arb(0)
    norm_inverse_squared = arb(0)
    residual_squared = arb(0)

    for row in range(int(start), int(stop)):
        for column in range(row, int(N), 2):
            connection_entry = _connection_entry_arb(
                row,
                column,
                r=r,
                alpha=alpha,
            )
            connection_magnitude = abs(connection_entry).upper()
            norm_connection_squared += (
                connection_magnitude * connection_magnitude
            ).upper()

            inverse_entry = arb(repr(float(inverse[row, column])))
            inverse_magnitude = abs(inverse_entry).upper()
            norm_inverse_squared += (
                inverse_magnitude * inverse_magnitude
            ).upper()

            product = arb(0)
            for inner in range(row, column + 1, 2):
                product += _connection_entry_arb(
                    row,
                    inner,
                    r=r,
                    alpha=alpha,
                ) * arb(repr(float(inverse[inner, column])))
            residual = (arb(1) if row == column else arb(0)) - product
            residual_magnitude = abs(residual).upper()
            residual_squared += (
                residual_magnitude * residual_magnitude
            ).upper()

    return {
        "start": int(start),
        "stop": int(stop),
        "norm_connection_squared_u": upper_text(norm_connection_squared),
        "norm_inverse_squared_u": upper_text(norm_inverse_squared),
        "residual_squared_u": upper_text(residual_squared),
    }


def _certify_transport_by_row_blocks(
    *,
    N: int,
    r_text: str,
    inverse: np.ndarray,
    precision_bits: int,
    workers: int,
) -> tuple[arb, arb, arb, int]:
    '''Explanation: The basis transport cost is governed by the norms of the connection and its inverse. Summing rigorous row-block contributions proves their product without trusting a black-box floating condition number.
Functionality: Return rigorous Frobenius bounds using parity-aware process blocks.'''

    blocks = _partition_rows(int(N), int(workers))
    arguments = [
        (
            str(r_text),
            inverse,
            int(N),
            int(precision_bits),
            int(start),
            int(stop),
        )
        for start, stop in blocks
    ]
    if len(blocks) == 1:
        results = list(map(_transport_row_block, arguments))
    else:
        with ProcessPoolExecutor(
            max_workers=len(blocks),
            mp_context=mp.get_context("spawn"),
        ) as executor:
            results = list(executor.map(_transport_row_block, arguments))
    results.sort(key=lambda result: int(result["start"]))

    flint.ctx.prec = int(precision_bits)
    connection_squared = arb(0)
    inverse_squared = arb(0)
    residual_squared = arb(0)
    for result in results:
        connection_squared += arb(str(result["norm_connection_squared_u"]))
        inverse_squared += arb(str(result["norm_inverse_squared_u"]))
        residual_squared += arb(str(result["residual_squared_u"]))
    return (
        connection_squared.sqrt().upper(),
        inverse_squared.sqrt().upper(),
        residual_squared.sqrt().upper(),
        len(blocks),
    )


def _float_matrix_as_arb(matrix: np.ndarray) -> arb_mat:
    '''Explanation: A floating inverse is only a proposal, but each binary64 entry has an exact dyadic value. Embedding it exactly lets Arb measure its residual against the rigorous connection matrix with no hidden conversion error.
Functionality: Embed a binary64 matrix entrywise as exact Arb values for residual certification.'''
    rows, columns = matrix.shape
    result = arb_mat(int(rows), int(columns))
    for row in range(int(rows)):
        for column in range(row, int(columns)):
            value = float(matrix[row, column])
            if value:
                result[row, column] = arb(repr(value))
    return result


def _identity(N: int) -> arb_mat:
    '''Explanation: Inverse certification compares the product of the connection and candidate inverse with the identity operator. Constructing the identity exactly fixes the reference object in that residual inequality.
Functionality: Construct the exact Arb identity matrix of the requested dimension.'''
    result = arb_mat(int(N), int(N))
    for index in range(int(N)):
        result[index, index] = 1
    return result


def _frobenius_upper(matrix: arb_mat) -> arb:
    '''Explanation: A Frobenius upper bound controls the operator norm of a matrix residual. If that residual is below one, the Neumann argument proves invertibility and bounds the true inverse.
Functionality: Return an outward-rounded Frobenius-norm upper bound for an Arb matrix.'''
    total = arb(0)
    rows = matrix.nrows()
    columns = matrix.ncols()
    for row in range(rows):
        for column in range(columns):
            magnitude = abs(matrix[row, column]).upper()
            total += magnitude * magnitude
    return total.sqrt().upper()


def _configuration_digest(config: Phase2TransportConfig) -> str:
    payload = {
        "map_label": MAP_LABEL,
        "producer_schema": PRODUCER_SCHEMA,
        **asdict(config),
        "binary64_inverse_policy": {
            "operation": "numpy.linalg.inv(midpoint)",
            "scope_threads": INVERSE_BLAS_THREADS,
            "outer_threads": 1,
            **LOCKED_OPENBLAS_RUNTIME,
        },
        "basis_normalisation": "X-orthonormal Chebyshev packets",
        "coordinate_orientation": "c to y=D_r,N c to d=T_N(r)y",
    }
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def certify_transport(
    config: Phase2TransportConfig,
    *,
    data_dir: Path,
    report_dir: Path,
) -> Phase2TransportResult:
    '''Explanation: The pure matrix defect must be multiplied by the conditioning of the Legendre-to-packet map. Residual-based inverse certification proves this finite condition factor and therefore makes the transported thesis defect rigorous.
Functionality: Certify the finite connection condition number by inverse residual.'''

    if config.N < 1:
        raise ValueError("N must be positive.")
    flint.ctx.prec = int(config.precision_bits)
    if hasattr(flint.ctx, "threads"):
        flint.ctx.threads = max(
            1,
            min(int(config.flint_threads), 24, os.cpu_count() or 1),
        )

    midpoint = connection_matrix_float(config.N, config.r)
    if not np.all(np.isfinite(midpoint)):
        raise ArithmeticError("The midpoint connection matrix is not finite.")
    inverse, inverse_runtime_evidence = _invert_midpoint_with_locked_blas(
        midpoint
    )
    inverse[np.tril_indices(config.N, -1)] = 0.0
    row_indices, column_indices = np.indices(inverse.shape)
    inverse[(column_indices - row_indices) % 2 != 0] = 0.0
    if not np.all(np.isfinite(inverse)):
        raise ArithmeticError("The numerical inverse witness is not finite.")

    diagonal = np.abs(np.diag(midpoint))
    if np.any(diagonal <= 0):
        raise ArithmeticError("The finite connection matrix has a zero diagonal entry.")
    kappa_diagonal = float(diagonal.max() / diagonal.min())

    process_workers = max(
        1,
        min(int(config.flint_threads), 24, os.cpu_count() or 1),
    )
    (
        norm_connection,
        norm_inverse,
        residual_delta,
        transport_blocks,
    ) = _certify_transport_by_row_blocks(
        N=config.N,
        r_text=config.r,
        inverse=inverse,
        precision_bits=config.precision_bits,
        workers=process_workers,
    )
    if not (residual_delta < 1):
        raise ArithmeticError(
            "The inverse-residual transport certificate failed delta less than one."
        )

    one_minus_delta = (1 - residual_delta).lower()
    lambda_maximum = (norm_connection * norm_connection).upper()
    lambda_minimum = ((one_minus_delta / norm_inverse) ** 2).lower()
    if not (lambda_minimum > 0):
        raise ArithmeticError("The certified minimum Gram eigenvalue is not positive.")
    kappa_upper = (
        norm_connection * norm_inverse / one_minus_delta
    ).upper()
    configuration_digest = _configuration_digest(config)
    inverse_runtime_json = json.dumps(
        inverse_runtime_evidence,
        sort_keys=True,
        separators=(",", ":"),
    )

    data_dir = Path(data_dir)
    report_dir = Path(report_dir)
    witness_path = data_dir / (
        f"branch_image_balanced_candidate_transport_inverse_witness_N{config.N}.npz"
    )
    _atomic_npz(
        witness_path,
        inverse_witness=inverse,
        midpoint_connection=midpoint,
        N=np.array([config.N], dtype=np.int64),
        r=np.array([config.r]),
        configuration_digest=np.array([configuration_digest]),
        inverse_blas_runtime_evidence=np.array(inverse_runtime_json),
    )
    witness_digest = _sha256(witness_path)

    record = {
        "N": int(config.N),
        "r": str(config.r),
        "rho": str(config.rho),
        "r_tau": str(config.r_tau),
        "kappa_hat": upper_text(kappa_upper),
        "kappa_diag": repr(kappa_diagonal),
        "lambda_max_cert": upper_text(lambda_maximum),
        "lambda_min_cert": lower_text(lambda_minimum),
        "norm_T_frob_cert": upper_text(norm_connection),
        "norm_B0_frob_cert": upper_text(norm_inverse),
        "residual_delta_cert": upper_text(residual_delta),
        "transport_certified": True,
        "transport_method": (
            "process-block upper-triangular Arb inverse-residual "
            "Frobenius certificate"
        ),
        "transport_process_workers": int(process_workers),
        "transport_row_blocks": int(transport_blocks),
        "arb_dps": int(math.ceil(config.precision_bits / math.log2(10))),
        "inverse_backend": "numpy binary64 witness with Arb residual",
        "inverse_blas_runtime_evidence": inverse_runtime_json,
        "configuration_digest": configuration_digest,
        "geometry_configuration_digest": config.geometry_configuration_digest,
        "producer_schema": PRODUCER_SCHEMA,
        "inverse_witness_sha256": witness_digest,
        "inverse_witness_path": str(witness_path),
        "coordinate_orientation": "c to y=D_r,N c to d=T_N(r)y",
    }
    csv_path = data_dir / (
        f"branch_image_balanced_candidate_transport_cert_N{config.N}.csv"
    )
    report_path = report_dir / (
        f"branch_image_balanced_candidate_transport_cert_N{config.N}.json"
    )
    _atomic_csv(csv_path, record)
    report = {
        "map_label": MAP_LABEL,
        "producer_schema": PRODUCER_SCHEMA,
        "configuration": asdict(config),
        "configuration_digest": configuration_digest,
        "certificate": record,
        "csv_path": str(csv_path),
        "csv_sha256": _sha256(csv_path),
        "inverse_witness_path": str(witness_path),
        "inverse_witness_sha256": witness_digest,
        "binary64_inverse_runtime": inverse_runtime_evidence,
    }
    _atomic_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return Phase2TransportResult(
        record=record,
        csv_path=csv_path,
        report_path=report_path,
        witness_path=witness_path,
        witness_digest=witness_digest,
    )


__all__ = [
    "MAP_LABEL",
    "PRODUCER_SCHEMA",
    "Phase2TransportConfig",
    "Phase2TransportResult",
    "certify_transport",
    "connection_matrix_arb",
    "connection_matrix_float",
]
