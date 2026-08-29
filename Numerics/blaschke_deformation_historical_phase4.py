"""Rebuild the retained wide-radius Phase 4 diagnostics from source.

This producer is deliberately separate from the theorem-facing Phase 4
certificate.  It rederives the historical ``rho=2.45`` exploratory row's
pure-scaled finite section, Hardy-gauge similarity, sampled contours, and moat
surfaces.  Every emitted record is diagnostic-only and is ineligible for a
theorem gate.

No retained CSV or NPZ file is a seed.  Existing artefacts are reusable only
when they carry this producer's complete configuration, source, runtime, and
matrix hashes.  Unkeyed or stale files are ignored and rebuilt atomically.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import mpmath as mp
import numpy as np
import scipy
import scipy.linalg as spla


MAP_LABEL = "blaschke_mu_0p3"
PRODUCER_SCHEMA = "historical-wide-phase4-source-rebuild-v2"
DIAGNOSTIC_STATUS = "diagnostic_only_not_theorem_gate"
DIAGNOSTIC_DESCRIPTION = (
    "retained historical wide-radius sampled finite-section diagnostic; "
    "not interval-certified and not eligible for a theorem gate"
)

_PRODUCTION_TARGETS = (
    "alpha^1",
    "alpha^2",
    "mu^1",
    "alpha^3",
    "alpha^4",
    "alpha^5",
    "mu^2",
    "alpha^6",
    "alpha^7",
    "alpha^8",
    "mu^3",
    "alpha^9",
    "alpha^10",
    "alpha^11",
    "mu^4",
)
_FRAGILE_TARGETS = ("alpha^11", "mu^4")
_CACHE_METADATA_KEYS = (
    "producer_schema",
    "configuration_digest",
    "source_digest",
    "cache_key",
    "diagnostic_status",
)
_OUTPUT_METADATA_FIELDS = (
    "diagnostic_status",
    "configuration_digest",
    "source_digest",
    "cache_key",
)


@dataclass(frozen=True)
class HistoricalPhase4Config:
    """Numerical and artifact configuration for the historical diagnostics."""

    mode: str = "production"
    N: int = 600
    M: int = 610
    mu: str = "0.3"
    rho: str = "2.45"
    r: str = "2.225974769705636"
    epsilon_x: str = "1.1641169413997097e-17"
    dps: int = 200
    target_count: int = 15
    maximum_target_power: int = 28
    contour_samples: int = 128
    fragile_samples: tuple[int, int] = (128, 256)
    local_surface_grid: int = 25
    global_surface_grid: int = 45
    zoom_surface_grid: int = 161
    deep_zoom_surface_grid: int = 201
    surface_row_block_size: int = 2
    surface_log_floor: str = "1e-18"
    assembly_row_block_size: int | None = 32

    @classmethod
    def production_n600_m610(cls) -> "HistoricalPhase4Config":
        return cls()

    @classmethod
    def small_test(
        cls,
        *,
        N: int = 6,
        M: int = 8,
        dps: int = 45,
        contour_samples: int = 8,
        fragile_samples: tuple[int, int] = (8, 12),
        local_surface_grid: int = 5,
        global_surface_grid: int = 5,
        zoom_surface_grid: int = 7,
        deep_zoom_surface_grid: int = 7,
    ) -> "HistoricalPhase4Config":
        return cls(
            mode="test",
            N=N,
            M=M,
            dps=dps,
            contour_samples=contour_samples,
            fragile_samples=fragile_samples,
            local_surface_grid=local_surface_grid,
            global_surface_grid=global_surface_grid,
            zoom_surface_grid=zoom_surface_grid,
            deep_zoom_surface_grid=deep_zoom_surface_grid,
            surface_row_block_size=2,
            assembly_row_block_size=2,
        )

    def validate(self) -> None:
        if self.mode not in {"production", "test"}:
            raise ValueError("mode must be either 'production' or 'test'.")
        if self.N < 2 or self.M < self.N:
            raise ValueError("The producer requires M at least N at least two.")
        if self.dps < 30:
            raise ValueError("At least 30 decimal digits are required.")
        if self.target_count < 15:
            raise ValueError("The two fragile targets require the first 15 packets.")
        if self.maximum_target_power < 12:
            raise ValueError("maximum_target_power must include alpha^12.")
        if self.contour_samples < 4:
            raise ValueError("At least four contour samples are required.")
        if len(self.fragile_samples) != 2:
            raise ValueError("Exactly two fragile angular resolutions are required.")
        if self.contour_samples not in self.fragile_samples:
            raise ValueError("fragile_samples must include contour_samples.")
        if any(int(value) < 4 for value in self.fragile_samples):
            raise ValueError("Each fragile angular resolution must be at least four.")
        grids = (
            self.local_surface_grid,
            self.global_surface_grid,
            self.zoom_surface_grid,
            self.deep_zoom_surface_grid,
        )
        if any(int(value) < 3 or int(value) % 2 == 0 for value in grids):
            raise ValueError("Surface grids must be odd integers of at least three.")
        if self.surface_row_block_size < 1:
            raise ValueError("surface_row_block_size must be positive.")
        if self.assembly_row_block_size is not None and self.assembly_row_block_size < 1:
            raise ValueError("assembly_row_block_size must be positive when supplied.")
        mu = float(self.mu)
        rho = float(self.rho)
        radius = float(self.r)
        epsilon = float(self.epsilon_x)
        log_floor = float(self.surface_log_floor)
        if not (0.0 < mu < 1.0):
            raise ValueError("mu must lie strictly between zero and one.")
        if not (1.0 < radius < rho):
            raise ValueError("The historical radii must satisfy 1 < r < rho.")
        if not (math.isfinite(epsilon) and epsilon > 0.0):
            raise ValueError("epsilon_x must be a positive finite number.")
        if not (math.isfinite(log_floor) and log_floor > 0.0):
            raise ValueError("surface_log_floor must be positive and finite.")

        if self.mode == "production":
            locked = {
                "N": 600,
                "M": 610,
                "mu": "0.3",
                "rho": "2.45",
                "r": "2.225974769705636",
                "epsilon_x": "1.1641169413997097e-17",
                "dps": 200,
                "assembly_row_block_size": 32,
                "target_count": 15,
                "contour_samples": 128,
                "fragile_samples": (128, 256),
                "local_surface_grid": 25,
                "global_surface_grid": 45,
                "zoom_surface_grid": 161,
                "deep_zoom_surface_grid": 201,
            }
            for name, expected in locked.items():
                if getattr(self, name) != expected:
                    raise ValueError(
                        f"Production configuration is locked: {name} must be {expected!r}."
                    )
            if self.dps < 200:
                raise ValueError("Production assembly requires at least 200 decimal digits.")


@dataclass(frozen=True)
class TargetContour:
    rank: int
    name: str
    family: str
    power: int
    centre: float
    expected_multiplicity: int
    nearest_target_separation: float
    radius: float


@dataclass(frozen=True)
class CacheIdentity:
    configuration: dict[str, Any]
    configuration_digest: str
    source_records: dict[str, dict[str, str]]
    runtime_versions: dict[str, str]
    source_digest: str
    cache_key: str


@dataclass(frozen=True)
class HistoricalPhase4Result:
    artifact_paths: dict[str, Path]
    report_path: Path
    report_digest: str
    cache_hits: dict[str, bool]
    cache_key: str


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_digest(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(contiguous.shape)).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _atomic_text(path: Path, payload: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_csv(
    path: Path,
    records: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    rows = tuple(dict(record) for record in records)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV artifact: {path}")
    expected = set(fieldnames)
    for row in rows:
        if set(row) != expected:
            missing = sorted(expected - set(row))
            extra = sorted(set(row) - expected)
            raise ValueError(
                f"CSV schema mismatch for {path.name}: missing={missing}, extra={extra}"
            )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
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


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _module_paths() -> dict[str, Path]:
    producer = Path(__file__).resolve()
    candidates = (
        Path.cwd() / "Numerics",
        Path.cwd() if Path.cwd().name == "Numerics" else None,
        producer.parent,
    )
    numerics_dir = next(
        (
            candidate.resolve()
            for candidate in candidates
            if candidate is not None
            and (candidate / "blaschke_deformation_historical_phase4.py").is_file()
            and (candidate / "mpmath_pf_raw.py").is_file()
            and (candidate / "hardy_moat_surface_worker.py").is_file()
        ),
        None,
    )
    if numerics_dir is None:
        raise FileNotFoundError(
            "Cannot locate the self-contained Final Deployment Numerics sources."
        )
    return {
        "historical_phase4_producer": (
            numerics_dir / "blaschke_deformation_historical_phase4.py"
        ),
        "mpmath_row_worker": numerics_dir / "mpmath_pf_raw.py",
        "hardy_surface_worker": numerics_dir / "hardy_moat_surface_worker.py",
    }


def _source_record(path: Path, working_root: Path) -> dict[str, str]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required producer source is missing: {path}")
    try:
        display_path = str(path.relative_to(working_root))
    except ValueError:
        display_path = path.name
    return {"path": display_path, "sha256": _sha256(path)}


def blaschke_formula_map_spec(mu: str, dps: int, max_power: int = 28) -> dict[str, Any]:
    '''Explanation: The symmetric Blaschke map determines both inverse branches and the closed-form alpha and mu target families. Keeping those formulas together fixes the mathematical model used to name every historical spectral packet.
Functionality: Return the formula map and exact target families used by row workers.'''

    with mp.workdps(int(dps)):
        mu_mp = mp.mpf(str(mu))
        alpha = (1 + mu_mp) / 2
        exact_clusters = [
            {
                "name": "1",
                "value": "1",
                "multiplicity": 1,
                "family": "trivial",
                "power": 0,
            }
        ]
        for power in range(1, int(max_power) + 1):
            exact_clusters.append(
                {
                    "name": f"alpha^{power}",
                    "value": f"alpha**{power}",
                    "multiplicity": 1,
                    "family": "alpha",
                    "power": power,
                }
            )
        for power in range(1, int(max_power) + 1):
            exact_clusters.append(
                {
                    "name": f"mu^{power}",
                    "value": f"mu**{power}",
                    "multiplicity": 2,
                    "family": "mu",
                    "power": power,
                }
            )
        return {
            "name": "formula_inverse_branches",
            "params": {
                "label": MAP_LABEL,
                "parameters": {
                    "mu": mp.nstr(mu_mp, int(dps)),
                    "alpha": mp.nstr(alpha, int(dps)),
                },
                "branches": [
                    {
                        "label": 1,
                        "tau": "x/2 - acos(mu*cos(pi*x/2))/pi",
                        "phi": (
                            "0.5 - mu/2*sin(pi*x/2)/"
                            "sqrt(1 - mu**2*cos(pi*x/2)**2)"
                        ),
                    },
                    {
                        "label": 2,
                        "tau": "x/2 + acos(mu*cos(pi*x/2))/pi",
                        "phi": (
                            "0.5 + mu/2*sin(pi*x/2)/"
                            "sqrt(1 - mu**2*cos(pi*x/2)**2)"
                        ),
                    },
                ],
                "exact_clusters": exact_clusters,
            },
        }


def _build_identity(config: HistoricalPhase4Config) -> tuple[CacheIdentity, dict[str, Any]]:
    config.validate()
    paths = _module_paths()
    working_root = paths["historical_phase4_producer"].resolve().parents[1]
    source_records = {
        name: _source_record(path, working_root) for name, path in paths.items()
    }
    runtime_versions = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mpmath": mp.__version__,
        "byteorder": sys.byteorder,
    }
    map_spec = blaschke_formula_map_spec(
        config.mu, config.dps, config.maximum_target_power
    )
    configuration = {"config": asdict(config), "map_spec": map_spec}
    configuration_digest = _json_digest(configuration)
    source_digest = _json_digest(
        {"producer_sources": source_records, "runtime_versions": runtime_versions}
    )
    cache_key = _json_digest(
        {
            "producer_schema": PRODUCER_SCHEMA,
            "configuration_digest": configuration_digest,
            "source_digest": source_digest,
        }
    )
    return (
        CacheIdentity(
            configuration=configuration,
            configuration_digest=configuration_digest,
            source_records=source_records,
            runtime_versions=runtime_versions,
            source_digest=source_digest,
            cache_key=cache_key,
        ),
        map_spec,
    )


def _load_module(module_name: str, path: Path):
    path = Path(path).resolve()
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    existing = sys.modules.get(module_name)
    if existing is not None and Path(existing.__file__).resolve() == path:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load required helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise ImportError(f"Loaded helper path does not match {path}")
    return module


def _load_helpers():
    paths = _module_paths()
    mpmath_worker = _load_module("mpmath_pf_raw", paths["mpmath_row_worker"])
    surface_worker = _load_module(
        "hardy_moat_surface_worker", paths["hardy_surface_worker"]
    )
    required_mpmath = (
        "assemble_pure_scaled_transfer_block",
        "exact_clusters_for_map",
        "serialisable_map_spec",
    )
    required_surface = ("initialise_surface_worker", "sample_surface_row_block")
    if any(not hasattr(mpmath_worker, name) for name in required_mpmath):
        raise RuntimeError("mpmath_pf_raw.py does not expose the required worker API.")
    if any(not hasattr(surface_worker, name) for name in required_surface):
        raise RuntimeError(
            "hardy_moat_surface_worker.py does not expose the required worker API."
        )
    return mpmath_worker, surface_worker


def _metadata_arrays(identity: CacheIdentity) -> dict[str, np.ndarray]:
    return {
        "producer_schema": np.asarray(PRODUCER_SCHEMA),
        "configuration_digest": np.asarray(identity.configuration_digest),
        "source_digest": np.asarray(identity.source_digest),
        "cache_key": np.asarray(identity.cache_key),
        "diagnostic_status": np.asarray(DIAGNOSTIC_STATUS),
    }


def _output_metadata(identity: CacheIdentity) -> dict[str, str]:
    return {
        "diagnostic_status": DIAGNOSTIC_STATUS,
        "configuration_digest": identity.configuration_digest,
        "source_digest": identity.source_digest,
        "cache_key": identity.cache_key,
    }


def _scalar_text(payload: Any, key: str) -> str | None:
    if key not in payload.files:
        return None
    value = np.asarray(payload[key])
    if value.shape != ():
        return None
    return str(value.item())


def _metadata_matches(payload: Any, identity: CacheIdentity) -> bool:
    expected = {
        "producer_schema": PRODUCER_SCHEMA,
        "configuration_digest": identity.configuration_digest,
        "source_digest": identity.source_digest,
        "cache_key": identity.cache_key,
        "diagnostic_status": DIAGNOSTIC_STATUS,
    }
    return all(_scalar_text(payload, key) == value for key, value in expected.items())


def _alpha_array(nmax: int) -> np.ndarray:
    '''Explanation: Central-binomial coefficients describe the change between Legendre and symmetric Chebyshev/Laurent coordinates. Their recurrence builds this basis bridge stably without repeatedly evaluating large factorials.
Functionality: Generate the central-binomial coefficients alpha_j by their stable first-order recurrence.'''
    alpha = np.empty(int(nmax) + 1, dtype=np.float64)
    alpha[0] = 1.0
    for index in range(int(nmax)):
        alpha[index + 1] = alpha[index] * (2 * index + 1) / (2 * index + 2)
    return alpha


def connection_T_matrix(N: int, r: float) -> np.ndarray:
    '''Explanation: The connection matrix expresses the same finite polynomial in the scaled Legendre basis and in Chebyshev packets. It is the similarity transform that places the finite transfer matrix in the Hardy gauge used for moat diagnostics.
Functionality: Build the historical Hardy connection matrix from the source routine.'''

    N = int(N)
    r = float(r)
    if N < 1 or not (math.isfinite(r) and r > 1.0):
        raise ValueError("connection_T_matrix requires N >= 1 and finite r > 1.")
    alpha = _alpha_array(N)
    matrix = np.zeros((N, N), dtype=np.float64)
    for degree in range(N):
        scale = math.sqrt((2 * degree + 1) / 2.0)
        for mode in range(degree + 1):
            if (degree - mode) % 2:
                continue
            if mode == 0:
                coefficient = alpha[degree // 2] ** 2
                matrix[mode, degree] = r ** (-degree) * scale * coefficient
            else:
                coefficient = (
                    2.0
                    * alpha[(degree - mode) // 2]
                    * alpha[(degree + mode) // 2]
                )
                normalised = (
                    0.5
                    * r ** (mode - degree)
                    * math.sqrt(1.0 + r ** (-4 * mode))
                )
                matrix[mode, degree] = scale * coefficient * normalised
    if not np.all(np.diag(matrix) > 0.0) or not np.isfinite(matrix).all():
        raise ArithmeticError("The Hardy connection matrix is invalid.")
    return matrix


def _mp_matrix_to_complex128(matrix: Any) -> np.ndarray:
    rows = int(matrix.rows)
    columns = int(matrix.cols)
    if rows != columns:
        raise ValueError("The pure-scaled transfer block must be square.")
    result = np.empty((rows, columns), dtype=np.complex128)
    for row in range(rows):
        for column in range(columns):
            result[row, column] = complex(matrix[row, column])
    if not np.isfinite(result).all():
        raise ArithmeticError("The pure-scaled transfer block contains non-finite data.")
    return result


def build_hardy_matrix(
    scaled_block: np.ndarray, r: float
) -> tuple[np.ndarray, np.ndarray]:
    '''Explanation: Rescaling by the basis connection changes coordinates, not eigenvalues. The resulting Hardy-gauge matrix makes coefficient decay and resolvent geometry visible in the coordinates used by the thesis perturbation argument.
Functionality: Apply the historical binary64 Hardy similarity after mpmath scaling.'''

    scaled_block = np.ascontiguousarray(scaled_block, dtype=np.complex128)
    if scaled_block.ndim != 2 or scaled_block.shape[0] != scaled_block.shape[1]:
        raise ValueError("scaled_block must be a square matrix.")
    connection = connection_T_matrix(scaled_block.shape[0], float(r))
    left_product = connection @ scaled_block
    hardy_matrix = np.linalg.solve(connection.T, left_product.T).T
    hardy_matrix = np.ascontiguousarray(hardy_matrix, dtype=np.complex128)
    if not np.isfinite(hardy_matrix).all():
        raise ArithmeticError("The Hardy-gauge matrix contains non-finite data.")
    return hardy_matrix, connection


def _artifact_paths(
    config: HistoricalPhase4Config, data_dir: Path, report_dir: Path
) -> dict[str, Path]:
    N, M, J = int(config.N), int(config.M), int(config.contour_samples)
    r_tag = str(config.r).replace("-", "m").replace(".", "p")
    return {
        "scaled_block": data_dir
        / f"branch_image_wide_candidate_B_scaled_mp_N{N}_M{M}_r{r_tag}.npz",
        "hardy_matrix": data_dir
        / f"branch_image_wide_candidate_A_X_from_mp_scaled_N{N}_M{M}_r{r_tag}.npz",
        "eigenvalues": data_dir / f"phase4_hp_hardy_gauge_eigenvalues_N{N}_M{M}.csv",
        "packets": data_dir
        / f"branch_image_wide_candidate_first15_sampled_spectral_packets_N{N}_M{M}.csv",
        "moats": data_dir
        / f"branch_image_wide_candidate_first15_contour_moats_N{N}_M{M}_J{J}.csv",
        "profile_summary": data_dir
        / f"branch_image_wide_candidate_first15_contour_profile_summary_N{N}_M{M}_J{J}.csv",
        "profiles": data_dir
        / f"branch_image_wide_candidate_first15_contour_profiles_N{N}_M{M}_J{J}.csv",
        "plot_table": data_dir
        / f"branch_image_wide_candidate_first15_contour_moat_plot_table_N{N}_M{M}_J{J}.csv",
        "minima": data_dir
        / f"branch_image_wide_candidate_first15_contour_profile_minima_table_N{N}_M{M}_J{J}.csv",
        "fragile": data_dir
        / f"branch_image_wide_candidate_first15_fragile_robustness_N{N}_M{M}.csv",
        "final_packets": data_dir
        / f"final_blaschke_branch-image_first15_sampled_spectral_packets_N{N}_M{M}.csv",
        "final_moats": data_dir
        / f"final_blaschke_branch-image_first15_contour_moats_N{N}_M{M}.csv",
        "final_fragile": data_dir
        / f"final_blaschke_branch-image_first15_fragile_robustness_N{N}_M{M}.csv",
        "compatibility_summary": data_dir
        / "final_blaschke_branch-image_first15_phase4_summary.json",
        "local_surface": data_dir
        / f"branch_image_wide_candidate_mu2_hardy_moat_surface_N{N}_M{M}_grid{config.local_surface_grid}.npz",
        "local_surface_csv": data_dir
        / f"branch_image_wide_candidate_mu2_hardy_moat_surface_N{N}_M{M}.csv",
        "global_surface": data_dir
        / f"branch_image_wide_candidate_first15_global_hardy_moat_surface_N{N}_M{M}_grid{config.global_surface_grid}.npz",
        "global_surface_csv": data_dir
        / f"branch_image_wide_candidate_first15_global_hardy_moat_surface_N{N}_M{M}.csv",
        "zoom_surface": data_dir
        / f"branch_image_wide_candidate_first15_zoom_hardy_moat_surface_N{N}_M{M}_grid{config.zoom_surface_grid}.npz",
        "deep_zoom_surface": data_dir
        / f"branch_image_wide_candidate_first15_deep_zoom_hardy_moat_surface_N{N}_M{M}_grid{config.deep_zoom_surface_grid}.npz",
        "rebuild_report": report_dir
        / f"blaschke_deformation_historical_phase4_rebuild_N{N}_M{M}.json",
    }


def _load_valid_scaled_block(
    path: Path, identity: CacheIdentity, shape: tuple[int, int]
) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            if not _metadata_matches(payload, identity) or "B_scaled" not in payload.files:
                return None
            block = np.asarray(payload["B_scaled"])
            if block.shape != shape or block.dtype != np.dtype(np.complex128):
                return None
            if not np.isfinite(block).all():
                return None
            if _scalar_text(payload, "B_scaled_sha256") != _array_sha256(block):
                return None
            return np.ascontiguousarray(block)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _load_valid_hardy_matrix(
    path: Path,
    identity: CacheIdentity,
    scaled_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    expected_shape = scaled_block.shape
    scaled_hash = _array_sha256(scaled_block)
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {"A_X", "B_scaled", "T", "eigenvalues"}
            if not _metadata_matches(payload, identity) or not required.issubset(
                payload.files
            ):
                return None
            hardy = np.asarray(payload["A_X"])
            cached_scaled = np.asarray(payload["B_scaled"])
            connection = np.asarray(payload["T"])
            eigenvalues = np.asarray(payload["eigenvalues"])
            if (
                hardy.shape != expected_shape
                or hardy.dtype != np.dtype(np.complex128)
                or cached_scaled.shape != expected_shape
                or cached_scaled.dtype != np.dtype(np.complex128)
                or connection.shape != expected_shape
                or connection.dtype != np.dtype(np.float64)
                or eigenvalues.shape != (expected_shape[0],)
                or eigenvalues.dtype != np.dtype(np.complex128)
            ):
                return None
            if not all(
                np.isfinite(array).all()
                for array in (hardy, cached_scaled, connection, eigenvalues)
            ):
                return None
            if not np.array_equal(cached_scaled, scaled_block):
                return None
            if _scalar_text(payload, "B_scaled_sha256") != scaled_hash:
                return None
            if _scalar_text(payload, "A_X_sha256") != _array_sha256(hardy):
                return None
            if _scalar_text(payload, "T_sha256") != _array_sha256(connection):
                return None
            if _scalar_text(payload, "eigenvalues_sha256") != _array_sha256(
                eigenvalues
            ):
                return None
            return (
                np.ascontiguousarray(hardy),
                np.ascontiguousarray(connection),
                np.ascontiguousarray(eigenvalues),
            )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _build_or_load_matrices(
    config: HistoricalPhase4Config,
    identity: CacheIdentity,
    map_spec: dict[str, Any],
    paths: Mapping[str, Path],
    mpmath_worker: Any,
    *,
    assembly_workers: int,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, bool]]:
    shape = (int(config.N), int(config.N))
    scaled = None if force else _load_valid_scaled_block(
        paths["scaled_block"], identity, shape
    )
    scaled_hit = scaled is not None
    if scaled is None:
        serial_map = mpmath_worker.serialisable_map_spec(map_spec=map_spec)
        block_mp = mpmath_worker.assemble_pure_scaled_transfer_block(
            int(config.N),
            int(config.M),
            str(config.r),
            map_spec=serial_map,
            assembly_workers=int(assembly_workers),
            row_block_size=config.assembly_row_block_size,
            dps=int(config.dps),
        )
        scaled = _mp_matrix_to_complex128(block_mp)
        scaled_hash = _array_sha256(scaled)
        _atomic_npz(
            paths["scaled_block"],
            B_scaled=scaled,
            B_scaled_sha256=np.asarray(scaled_hash),
            **_metadata_arrays(identity),
        )

    matrix_payload = None if force else _load_valid_hardy_matrix(
        paths["hardy_matrix"], identity, scaled
    )
    matrix_hit = matrix_payload is not None
    if matrix_payload is None:
        hardy, connection = build_hardy_matrix(scaled, float(config.r))
        eigenvalues = np.ascontiguousarray(
            np.linalg.eigvals(hardy), dtype=np.complex128
        )
        if not np.isfinite(eigenvalues).all():
            raise ArithmeticError("Hardy eigenvalue computation returned non-finite data.")
        _atomic_npz(
            paths["hardy_matrix"],
            A_X=hardy,
            B_scaled=scaled,
            T=connection,
            eigenvalues=eigenvalues,
            B_scaled_sha256=np.asarray(_array_sha256(scaled)),
            A_X_sha256=np.asarray(_array_sha256(hardy)),
            T_sha256=np.asarray(_array_sha256(connection)),
            eigenvalues_sha256=np.asarray(_array_sha256(eigenvalues)),
            **_metadata_arrays(identity),
        )
    else:
        hardy, connection, eigenvalues = matrix_payload
    return scaled, hardy, connection, eigenvalues, {
        "scaled_block": scaled_hit,
        "hardy_matrix": matrix_hit,
    }


_EIGENVALUE_FIELDS = (
    "real",
    "imag",
    "abs",
    "diagnostic_status",
    "configuration_digest",
    "source_digest",
    "cache_key",
    "hardy_matrix_sha256",
)


def _load_valid_eigenvalue_csv(
    path: Path,
    identity: CacheIdentity,
    matrix_hash: str,
    expected: np.ndarray,
) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != _EIGENVALUE_FIELDS:
                return False
            rows = list(reader)
        if len(rows) != len(expected):
            return False
        recovered = []
        for row in rows:
            if (
                row["diagnostic_status"] != DIAGNOSTIC_STATUS
                or row["configuration_digest"] != identity.configuration_digest
                or row["source_digest"] != identity.source_digest
                or row["cache_key"] != identity.cache_key
                or row["hardy_matrix_sha256"] != matrix_hash
            ):
                return False
            value = complex(float(row["real"]), float(row["imag"]))
            if not math.isclose(float(row["abs"]), abs(value), rel_tol=1e-14, abs_tol=0.0):
                return False
            recovered.append(value)
        return np.array_equal(np.asarray(recovered, dtype=np.complex128), expected)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _write_eigenvalues(
    path: Path,
    eigenvalues: np.ndarray,
    identity: CacheIdentity,
    matrix_hash: str,
) -> None:
    metadata = _output_metadata(identity)
    records = []
    for value in eigenvalues:
        z = complex(value)
        records.append(
            {
                "real": float(z.real),
                "imag": float(z.imag),
                "abs": float(abs(z)),
                **metadata,
                "hardy_matrix_sha256": matrix_hash,
            }
        )
    _atomic_csv(path, records, _EIGENVALUE_FIELDS)


def _build_target_contours(
    config: HistoricalPhase4Config,
    map_spec: dict[str, Any],
    mpmath_worker: Any,
    eigenvalues: np.ndarray,
) -> list[TargetContour]:
    '''Explanation: Historical contours were proposed by placing circles between the expected packet and neighbouring finite eigenvalues. This is useful contour discovery, while the final thesis proof replaces the sampled proposal by exact rational contour data and validation.
Functionality: Derive the retained target contours from exact map clusters and finite-eigenvalue mid-gaps while excluding zero.'''
    with mp.workdps(int(config.dps)):
        clusters = mpmath_worker.exact_clusters_for_map(
            map_spec,
            max_power=int(config.maximum_target_power),
            include_trivial=True,
            full_multiplicity=True,
        )
        if not clusters or str(clusters[0]["name"]) != "1":
            raise RuntimeError("The formula map does not expose the trivial target first.")
        real_clusters = []
        for cluster in clusters:
            value = mp.mpc(cluster["value"])
            if abs(mp.im(value)) > mp.mpf(10) ** (-(int(config.dps) // 2)):
                raise RuntimeError("Historical Phase 4 expects real exact targets.")
            real_clusters.append((cluster, mp.re(value)))
        selected = [item for item in real_clusters if str(item[0]["name"]) != "1"]
        selected = selected[: int(config.target_count)]
        if len(selected) != int(config.target_count):
            raise RuntimeError("The map specification does not contain enough targets.")
        all_values = [value for _, value in real_clusters]
        contours = []
        for rank, (cluster, value) in enumerate(selected, 1):
            separations = [abs(value - other) for other in all_values if other != value]
            if not separations:
                raise RuntimeError("A target has no distinct neighbour for a contour.")
            nearest = min(separations)
            centre = float(value)
            multiplicity = int(cluster.get("multiplicity", 1))
            distances = np.abs(np.asarray(eigenvalues, dtype=np.complex128) - centre)
            order = np.argsort(distances)
            selected_indices = order[:multiplicity]
            outside_indices = order[multiplicity:]
            cluster_radius = (
                float(np.max(distances[selected_indices]))
                if len(selected_indices)
                else math.nan
            )
            nearest_outside = (
                float(np.min(distances[outside_indices]))
                if len(outside_indices)
                else math.inf
            )
            outer_limit = min(nearest_outside, abs(centre))
            if not (math.isfinite(outer_limit) and outer_limit > 0.0):
                raise RuntimeError("A historical target has no positive outer gap.")
            if not math.isfinite(cluster_radius):
                raise RuntimeError("A historical target has no finite cluster radius.")
            radius = (
                0.45 * outer_limit
                if cluster_radius >= outer_limit
                else 0.5 * (cluster_radius + outer_limit)
            )
            family_name = str(cluster.get("family", "target"))
            family = (
                "alpha packet"
                if family_name == "alpha"
                else "mu packet"
                if family_name == "mu"
                else "fixed point"
            )
            contours.append(
                TargetContour(
                    rank=rank,
                    name=str(cluster["name"]),
                    family=family,
                    power=int(cluster.get("power", 0)),
                    centre=centre,
                    expected_multiplicity=multiplicity,
                    nearest_target_separation=float(nearest),
                    radius=float(radius),
                )
            )
    if config.mode == "production" and tuple(row.name for row in contours) != _PRODUCTION_TARGETS:
        raise RuntimeError("The production first-fifteen target order has changed.")
    if any(row.centre <= row.radius for row in contours):
        raise RuntimeError("A source-derived contour intersects the zero complement.")
    return contours


def _target_plan_digest(targets: Sequence[TargetContour]) -> str:
    return _json_digest([asdict(target) for target in targets])


def _format_selected_eigenvalues(values: Sequence[complex]) -> str:
    return "; ".join(
        f"{float(value.real):.16g}{float(value.imag):+.2e}i" for value in values
    )


def _packet_matches(
    targets: Sequence[TargetContour], eigenvalues: np.ndarray
) -> dict[str, dict[str, Any]]:
    '''Explanation: Matching finite eigenvalues to the formula packets measures how the truncation approximates each expected cluster. It is a convergence diagnostic and does not itself prove that an exact-operator Riesz projector has the same rank.
Functionality: Assign finite eigenvalues to target packets by multiplicity and record each packet's largest sampled displacement.'''
    pool = [complex(value) for value in eigenvalues]
    matches: dict[str, dict[str, Any]] = {}
    for target in targets:
        multiplicity = int(target.expected_multiplicity)
        if len(pool) < multiplicity:
            selected: list[complex] = []
            max_error = math.nan
        else:
            indices = sorted(
                range(len(pool)), key=lambda index: abs(pool[index] - target.centre)
            )[:multiplicity]
            selected = [pool[index] for index in indices]
            max_error = max(abs(value - target.centre) for value in selected)
            for index in sorted(indices, reverse=True):
                pool.pop(index)
        selected_count = len(selected)
        matches[target.name] = {
            "selected_count": selected_count,
            "selected_eigenvalues": _format_selected_eigenvalues(selected),
            "max_numeric_error": float(max_error),
            "pass_sampled_packet": bool(
                selected_count == multiplicity
                and math.isfinite(max_error)
                and max_error < target.radius
            ),
        }
    return matches


def _smallest_singular_value(
    hardy_matrix: np.ndarray, identity_matrix: np.ndarray, zeta: complex
) -> float:
    '''Explanation: At a fixed point, the smallest singular value of zeta I minus A is the reciprocal of the finite resolvent norm. It measures local spectral separation, but one point cannot certify an entire contour.
Functionality: Compute the smallest singular value of the shifted Hardy matrix zeta I minus A for one sampled point.'''
    value = float(
        spla.svdvals(zeta * identity_matrix - hardy_matrix, check_finite=False)[-1]
    )
    if not (math.isfinite(value) and value >= 0.0):
        raise ArithmeticError("Surface or contour sampling returned an invalid singular value.")
    return value


def _sample_contour(
    hardy_matrix: np.ndarray,
    identity_matrix: np.ndarray,
    target: TargetContour,
    samples: int,
) -> np.ndarray:
    '''Explanation: Sampling singular values around a circle visualises where the finite resolvent is weakest. The sampled minimum lies above the unknown whole-circle infimum, so the thesis treats it as diagnostic rather than a certified moat.
Functionality: Evaluate the smallest singular value on an equispaced grid around one retained circular contour.'''
    theta = 2.0 * np.pi * np.arange(int(samples), dtype=np.float64) / int(samples)
    values = np.empty(int(samples), dtype=np.float64)
    for index, angle in enumerate(theta):
        zeta = target.centre + target.radius * np.exp(1j * angle)
        values[index] = _smallest_singular_value(hardy_matrix, identity_matrix, zeta)
    return values


def _load_valid_contour_cache(
    path: Path,
    config: HistoricalPhase4Config,
    identity: CacheIdentity,
    target_digest: str,
    matrix_hash: str,
) -> tuple[np.ndarray, dict[int, np.ndarray]] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            if not _metadata_matches(payload, identity):
                return None
            if (
                _scalar_text(payload, "target_plan_digest") != target_digest
                or _scalar_text(payload, "A_X_sha256") != matrix_hash
            ):
                return None
            profiles = np.asarray(payload["profile_smins"])
            expected_shape = (int(config.target_count), int(config.contour_samples))
            if profiles.shape != expected_shape or profiles.dtype != np.dtype(np.float64):
                return None
            if not np.isfinite(profiles).all() or np.any(profiles < 0.0):
                return None
            if _scalar_text(payload, "profile_smins_sha256") != _array_sha256(
                profiles
            ):
                return None
            fragile: dict[int, np.ndarray] = {}
            for samples in config.fragile_samples:
                key = f"fragile_smins_J{int(samples)}"
                values = np.asarray(payload[key])
                if values.shape != (2, int(samples)) or values.dtype != np.dtype(np.float64):
                    return None
                if not np.isfinite(values).all() or np.any(values < 0.0):
                    return None
                if _scalar_text(payload, f"{key}_sha256") != _array_sha256(values):
                    return None
                fragile[int(samples)] = np.ascontiguousarray(values)
            return np.ascontiguousarray(profiles), fragile
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _build_or_load_contours(
    config: HistoricalPhase4Config,
    identity: CacheIdentity,
    targets: Sequence[TargetContour],
    hardy_matrix: np.ndarray,
    cache_dir: Path,
    *,
    force: bool,
) -> tuple[np.ndarray, dict[int, np.ndarray], bool, Path]:
    target_digest = _target_plan_digest(targets)
    matrix_hash = _array_sha256(hardy_matrix)
    cache_path = cache_dir / identity.cache_key / "contour_samples.npz"
    cached = None if force else _load_valid_contour_cache(
        cache_path, config, identity, target_digest, matrix_hash
    )
    if cached is not None:
        return cached[0], cached[1], True, cache_path

    eye = np.eye(int(config.N), dtype=np.complex128)
    profiles = np.vstack(
        [
            _sample_contour(
                hardy_matrix, eye, target, int(config.contour_samples)
            )
            for target in targets
        ]
    ).astype(np.float64, copy=False)
    by_name = {target.name: index for index, target in enumerate(targets)}
    fragile: dict[int, np.ndarray] = {}
    for samples in config.fragile_samples:
        rows = []
        for name in _FRAGILE_TARGETS:
            index = by_name.get(name)
            if index is None:
                raise RuntimeError(f"The fragile target {name} is missing.")
            if int(samples) == int(config.contour_samples):
                rows.append(profiles[index].copy())
            else:
                rows.append(
                    _sample_contour(hardy_matrix, eye, targets[index], int(samples))
                )
        fragile[int(samples)] = np.vstack(rows).astype(np.float64, copy=False)

    payload: dict[str, Any] = {
        "profile_smins": profiles,
        "profile_smins_sha256": np.asarray(_array_sha256(profiles)),
        "target_plan_digest": np.asarray(target_digest),
        "A_X_sha256": np.asarray(matrix_hash),
        **_metadata_arrays(identity),
    }
    for samples, values in fragile.items():
        key = f"fragile_smins_J{samples}"
        payload[key] = values
        payload[f"{key}_sha256"] = np.asarray(_array_sha256(values))
    _atomic_npz(cache_path, **payload)
    return profiles, fragile, False, cache_path


def _finite_geometry(
    target: TargetContour, eigenvalues: np.ndarray
) -> dict[str, Any]:
    '''Explanation: Counting finite eigenvalues and measuring nearest inside and outside distances shows whether a proposed circle separates the intended packet. These distances guide contour design but need certified arithmetic for theorem use.
Functionality: Count finite eigenvalues inside one contour and return the enclosed cluster radius and nearest exterior distance.'''
    distances = np.abs(eigenvalues - target.centre)
    inside = distances <= target.radius
    finite_count = int(np.count_nonzero(inside))
    cluster_radius = (
        float(np.max(distances[inside])) if finite_count else math.nan
    )
    outside = distances[~inside]
    nearest_outside = (
        float(np.min(outside)) if len(outside) else math.inf
    )
    return {
        "finite_eigenvalue_count": finite_count,
        "cluster_radius": cluster_radius,
        "nearest_outside_distance": nearest_outside,
    }


def _moat_record(
    target: TargetContour,
    singular_values: np.ndarray,
    samples: int,
    epsilon_x: float,
    eigenvalues: np.ndarray,
    packet: Mapping[str, Any],
    identity: CacheIdentity,
) -> dict[str, Any]:
    '''Explanation: A moat compares the finite resolvent separation with the exact-operator perturbation size. Here the separation is sampled, so the resulting small-gain margin explains the historical experiment without asserting the final Riesz-rank theorem.
Functionality: Combine sampled singular minima, zero separation, packet matching, and epsilon into one diagnostic moat and small-gain row.'''
    geometry = _finite_geometry(target, eigenvalues)
    minimum_index = int(np.argmin(singular_values))
    singular_minimum = float(singular_values[minimum_index])
    theta_minimum = 2.0 * math.pi * minimum_index / int(samples)
    distance_zero = abs(target.centre) - target.radius
    if distance_zero <= 0.0:
        raise ArithmeticError("A sampled contour does not exclude zero.")
    inverse_s = math.inf if singular_minimum == 0.0 else 1.0 / singular_minimum
    m_gamma = max(inverse_s, 1.0 / distance_zero)
    epsilon_product = epsilon_x * m_gamma
    count_ok = geometry["finite_eigenvalue_count"] == target.expected_multiplicity
    small_gain = math.isfinite(epsilon_product) and epsilon_product < 1.0
    sampled_pass = bool(count_ok and small_gain and packet["pass_sampled_packet"])
    return {
        "rank": target.rank,
        "target": target.name,
        "centre": target.centre,
        "expected_multiplicity": target.expected_multiplicity,
        "radius": target.radius,
        **geometry,
        "radius_status": "mid_gap",
        "s_min_gamma": singular_minimum,
        "theta_min": theta_minimum,
        "d_Gamma0": distance_zero,
        "m_gamma_X": m_gamma,
        "epsilon_X": epsilon_x,
        "epsilon_m_gamma": epsilon_product,
        "pass_sampled_small_gain": small_gain,
        "selected_count": packet["selected_count"],
        "selected_eigenvalues": packet["selected_eigenvalues"],
        "max_numeric_error": packet["max_numeric_error"],
        "pass_sampled_packet": packet["pass_sampled_packet"],
        "family": target.family,
        "count_ok": count_ok,
        "sampled_validation_pass": sampled_pass,
        "contour_interval_certified": False,
        "validation_status": "sampled_not_interval_certified",
        **_output_metadata(identity),
    }


_PACKET_FIELDS = (
    "rank",
    "target",
    "centre",
    "expected_multiplicity",
    "selected_count",
    "selected_eigenvalues",
    "max_numeric_error",
    "radius",
    "epsilon_m_gamma",
    "pass_sampled_packet",
    "contour_interval_certified",
    "validation_status",
    "pass_sampled_small_gain",
    "sampled_validation_pass",
    *_OUTPUT_METADATA_FIELDS,
)
_MOAT_FIELDS = (
    "rank",
    "target",
    "centre",
    "expected_multiplicity",
    "radius",
    "finite_eigenvalue_count",
    "cluster_radius",
    "nearest_outside_distance",
    "radius_status",
    "s_min_gamma",
    "theta_min",
    "d_Gamma0",
    "m_gamma_X",
    "epsilon_X",
    "epsilon_m_gamma",
    "pass_sampled_small_gain",
    "selected_count",
    "selected_eigenvalues",
    "max_numeric_error",
    "pass_sampled_packet",
    "family",
    "count_ok",
    "sampled_validation_pass",
    "contour_interval_certified",
    "validation_status",
    *_OUTPUT_METADATA_FIELDS,
)
_PROFILE_SUMMARY_FIELDS = (
    "rank",
    "target",
    "family",
    "centre",
    "radius",
    "theta_min",
    "s_min_gamma",
    "d_Gamma0",
    "m_gamma_X",
    "epsilon_X",
    "epsilon_m_gamma",
    "pass_sampled_small_gain",
    "pass_sampled_packet",
    "profile_status",
    *_OUTPUT_METADATA_FIELDS,
)
_PROFILE_FIELDS = (
    "target",
    "theta",
    "z_real",
    "z_imag",
    "smin",
    *_OUTPUT_METADATA_FIELDS,
)
_PLOT_FIELDS = (
    "rank",
    "target",
    "family",
    "expected_multiplicity",
    "finite_eigenvalue_count",
    "radius",
    "s_min_gamma",
    "epsilon_m_gamma",
    "max_numeric_error",
    "sampled_validation_pass",
    *_OUTPUT_METADATA_FIELDS,
)
_MINIMA_FIELDS = (
    "rank",
    "target",
    "family",
    "theta_at_profile_min",
    "profile_s_min",
    "epsilon_m_gamma",
    "pass_sampled_small_gain",
    "pass_sampled_packet",
    *_OUTPUT_METADATA_FIELDS,
)
_FRAGILE_FIELDS = (
    "J",
    "rank",
    "target",
    "centre",
    "expected_multiplicity",
    "radius",
    "finite_eigenvalue_count",
    "cluster_radius",
    "nearest_outside_distance",
    "radius_status",
    "s_min_gamma",
    "theta_min",
    "d_Gamma0",
    "m_gamma_X",
    "epsilon_X",
    "epsilon_m_gamma",
    "pass_sampled_small_gain",
    "count_ok",
    "sampled_validation_pass",
    "contour_interval_certified",
    "validation_status",
    *_OUTPUT_METADATA_FIELDS,
)


def _diagnostic_tables(
    config: HistoricalPhase4Config,
    identity: CacheIdentity,
    targets: Sequence[TargetContour],
    eigenvalues: np.ndarray,
    profiles: np.ndarray,
    fragile_profiles: Mapping[int, np.ndarray],
) -> dict[str, list[dict[str, Any]]]:
    '''Explanation: Contour profiles, packet matches, and small-gain margins are different views of the same historical finite model. Organising them together makes their numerical relationships inspectable while preserving their diagnostic status.
Functionality: Assemble the retained contour-profile, moat, packet, and robustness tables from source-generated samples.'''
    epsilon_x = float(config.epsilon_x)
    packets = _packet_matches(targets, eigenvalues)
    moats = [
        _moat_record(
            target,
            profiles[index],
            int(config.contour_samples),
            epsilon_x,
            eigenvalues,
            packets[target.name],
            identity,
        )
        for index, target in enumerate(targets)
    ]
    moat_by_name = {row["target"]: row for row in moats}
    packet_rows = []
    for target in targets:
        packet = packets[target.name]
        moat = moat_by_name[target.name]
        packet_rows.append(
            {
                "rank": target.rank,
                "target": target.name,
                "centre": target.centre,
                "expected_multiplicity": target.expected_multiplicity,
                "selected_count": packet["selected_count"],
                "selected_eigenvalues": packet["selected_eigenvalues"],
                "max_numeric_error": packet["max_numeric_error"],
                "radius": target.radius,
                "epsilon_m_gamma": moat["epsilon_m_gamma"],
                "pass_sampled_packet": packet["pass_sampled_packet"],
                "contour_interval_certified": False,
                "validation_status": "sampled_not_interval_certified",
                "pass_sampled_small_gain": moat["pass_sampled_small_gain"],
                "sampled_validation_pass": moat["sampled_validation_pass"],
                **_output_metadata(identity),
            }
        )

    profile_rows = []
    minima_rows = []
    for index, target in enumerate(targets):
        singular_values = profiles[index]
        theta = (
            2.0
            * np.pi
            * np.arange(int(config.contour_samples), dtype=np.float64)
            / int(config.contour_samples)
        )
        for angle, singular_value in zip(theta, singular_values):
            zeta = target.centre + target.radius * np.exp(1j * angle)
            profile_rows.append(
                {
                    "target": target.name,
                    "theta": float(angle),
                    "z_real": float(zeta.real),
                    "z_imag": float(zeta.imag),
                    "smin": float(singular_value),
                    **_output_metadata(identity),
                }
            )
        minimum_index = int(np.argmin(singular_values))
        moat = moat_by_name[target.name]
        minima_rows.append(
            {
                "rank": target.rank,
                "target": target.name,
                "family": target.family,
                "theta_at_profile_min": float(theta[minimum_index]),
                "profile_s_min": float(singular_values[minimum_index]),
                "epsilon_m_gamma": moat["epsilon_m_gamma"],
                "pass_sampled_small_gain": moat["pass_sampled_small_gain"],
                "pass_sampled_packet": moat["pass_sampled_packet"],
                **_output_metadata(identity),
            }
        )

    profile_summary = []
    plot_rows = []
    for moat in moats:
        profile_summary.append(
            {
                **{
                    key: moat[key]
                    for key in _PROFILE_SUMMARY_FIELDS
                    if key not in {"profile_status", *_OUTPUT_METADATA_FIELDS}
                },
                "profile_status": (
                    "sampled contour minimum over the source-rebuilt Phase 4 grid"
                ),
                **_output_metadata(identity),
            }
        )
        plot_rows.append(
            {
                **{
                    key: moat[key]
                    for key in _PLOT_FIELDS
                    if key not in _OUTPUT_METADATA_FIELDS
                },
                **_output_metadata(identity),
            }
        )

    targets_by_name = {target.name: target for target in targets}
    fragile_rows = []
    for samples in config.fragile_samples:
        values = fragile_profiles[int(samples)]
        for fragile_index, name in enumerate(_FRAGILE_TARGETS):
            target = targets_by_name[name]
            record = _moat_record(
                target,
                values[fragile_index],
                int(samples),
                epsilon_x,
                eigenvalues,
                packets[name],
                identity,
            )
            fragile_rows.append(
                {
                    "J": int(samples),
                    **{
                        key: record[key]
                        for key in _FRAGILE_FIELDS
                        if key not in {"J", *_OUTPUT_METADATA_FIELDS}
                    },
                    **_output_metadata(identity),
                }
            )
    return {
        "packets": packet_rows,
        "moats": moats,
        "profile_summary": profile_summary,
        "profiles": profile_rows,
        "plot_table": plot_rows,
        "minima": minima_rows,
        "fragile": fragile_rows,
    }


def _write_diagnostic_tables(
    config: HistoricalPhase4Config,
    identity: CacheIdentity,
    paths: Mapping[str, Path],
    tables: Mapping[str, list[dict[str, Any]]],
) -> None:
    schemas = {
        "packets": _PACKET_FIELDS,
        "moats": _MOAT_FIELDS,
        "profile_summary": _PROFILE_SUMMARY_FIELDS,
        "profiles": _PROFILE_FIELDS,
        "plot_table": _PLOT_FIELDS,
        "minima": _MINIMA_FIELDS,
        "fragile": _FRAGILE_FIELDS,
    }
    for name, fields in schemas.items():
        _atomic_csv(paths[name], tables[name], fields)
    _atomic_csv(paths["final_packets"], tables["packets"], _PACKET_FIELDS)
    _atomic_csv(paths["final_moats"], tables["moats"], _MOAT_FIELDS)
    _atomic_csv(paths["final_fragile"], tables["fragile"], _FRAGILE_FIELDS)

    products = [float(row["epsilon_m_gamma"]) for row in tables["moats"]]
    finite_products = [value for value in products if math.isfinite(value)]
    compatibility_summary = {
        "N": int(config.N),
        "M": int(config.M),
        "rho": float(config.rho),
        "r": float(config.r),
        "epsilon_X": float(config.epsilon_x),
        "finite_M_prefactor_certified": False,
        "sampled_validation_passes": int(
            sum(bool(row["sampled_validation_pass"]) for row in tables["moats"])
        ),
        "target_packets": len(tables["moats"]),
        "algebraic_count": int(
            sum(int(row["expected_multiplicity"]) for row in tables["moats"])
        ),
        "worst_epsilon_m_gamma": max(finite_products) if finite_products else None,
        "contour_interval_certified": False,
        "moat_status": "sampled finite-section moat diagnostic only",
        "diagnostic_status": DIAGNOSTIC_STATUS,
        "theorem_gate_eligible": False,
        "legacy_seed_dependency": False,
        "configuration_digest": identity.configuration_digest,
        "source_digest": identity.source_digest,
        "cache_key": identity.cache_key,
    }
    _atomic_text(
        paths["compatibility_summary"],
        json.dumps(
            compatibility_summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def _validate_historical_regression_profile(
    config: HistoricalPhase4Config,
    scaled: np.ndarray,
    hardy: np.ndarray,
    tables: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    '''Explanation: A historical comparison is meaningful only if the rebuilt computation stays in the originally declared parameter regime. The regression guard prevents silent methodological drift from being mistaken for reproducibility.
Functionality: Fail closed if the source replay leaves the retained diagnostic regime.'''

    scaled_maximum = float(np.max(np.abs(scaled)))
    hardy_maximum = float(np.max(np.abs(hardy)))
    sampled_passes = int(
        sum(bool(row["sampled_validation_pass"]) for row in tables["moats"])
    )
    finite_products = [
        float(row["epsilon_m_gamma"])
        for row in tables["moats"]
        if math.isfinite(float(row["epsilon_m_gamma"]))
    ]
    worst_product = max(finite_products) if finite_products else math.inf
    profile = {
        "contract": "source-rederived historical sampled comparison",
        "pure_scaled_binary64_boundary_after_mpmath_dps": int(config.dps),
        "hardy_similarity": "binary64 T B_scaled T_inverse historical path",
        "contour_radius_rule": "finite-eigenvalue cluster-to-nearest-outside mid-gap",
        "scaled_matrix_maximum_modulus": scaled_maximum,
        "hardy_matrix_maximum_modulus": hardy_maximum,
        "sampled_validation_passes": sampled_passes,
        "worst_finite_epsilon_m_gamma": worst_product,
        "theorem_gate_eligible": False,
    }
    if config.mode != "production":
        return profile
    failures = []
    if not 1.9e9 <= scaled_maximum <= 2.1e9:
        failures.append("pure-scaled matrix scale left the historical regime")
    if not 1.7e9 <= hardy_maximum <= 1.9e9:
        failures.append("Hardy matrix scale left the historical regime")
    if sampled_passes != int(config.target_count):
        failures.append("not all historical sampled packets passed")
    if not 0.45 <= worst_product <= 0.55:
        failures.append("worst sampled small-gain product left the historical regime")
    if failures:
        raise RuntimeError(
            "Historical Phase 4 regression gate failed: " + "; ".join(failures)
        )
    return profile


def _surface_windows(
    config: HistoricalPhase4Config, targets: Sequence[TargetContour]
) -> dict[str, dict[str, Any]]:
    '''Explanation: Local and global complex-plane windows show the shape of finite pseudospectral valleys around the target packets. They are chosen for interpretation and visual stress testing, not as domains of a theorem proof.
Functionality: Choose the locked local and global complex-plane windows used for retained Hardy-moat surface sampling.'''
    target_by_name = {target.name: target for target in targets}
    mu2 = target_by_name.get("mu^2")
    if mu2 is None:
        raise RuntimeError("The local surface requires the mu^2 target.")
    local_half_width = max(2.5e-2, 3.5 * mu2.radius)
    local = {
        "xlim": (mu2.centre - local_half_width, mu2.centre + local_half_width),
        "ylim": (-local_half_width, local_half_width),
        "grid": int(config.local_surface_grid),
    }

    x_min = min(target.centre - target.radius for target in targets)
    x_max = max(target.centre + target.radius for target in targets)
    x_pad = max(0.02, 0.08 * max(x_max - x_min, 1.0e-12))
    y_half = max(0.09, 1.35 * max(target.radius for target in targets))
    global_window = {
        "xlim": (x_min - x_pad, x_max + x_pad),
        "ylim": (-y_half, y_half),
        "grid": int(config.global_surface_grid),
    }
    zoom = {
        "xlim": (
            max(global_window["xlim"][0], -0.006),
            min(global_window["xlim"][1], 0.18),
        ),
        "ylim": (
            -min(global_window["ylim"][1], 0.060),
            min(global_window["ylim"][1], 0.060),
        ),
        "grid": int(config.zoom_surface_grid),
    }
    deep_targets = [target for target in targets if target.centre <= 0.055]
    if len(deep_targets) < 3:
        deep_targets = sorted(targets, key=lambda target: target.centre)[:7]
    deep_x_min = min(target.centre - target.radius for target in deep_targets)
    deep_x_max = max(target.centre + target.radius for target in deep_targets)
    deep_x_pad = max(0.0025, 0.14 * max(deep_x_max - deep_x_min, 1.0e-12))
    deep_y_half = max(0.0065, 1.8 * max(target.radius for target in deep_targets))
    deep = {
        "xlim": (
            max(global_window["xlim"][0], deep_x_min - deep_x_pad),
            min(global_window["xlim"][1], deep_x_max + deep_x_pad),
        ),
        "ylim": (-deep_y_half, deep_y_half),
        "grid": int(config.deep_zoom_surface_grid),
    }
    return {
        "local": local,
        "global": global_window,
        "zoom": zoom,
        "deep_zoom": deep,
    }


def _surface_digest(
    identity: CacheIdentity,
    matrix_hash: str,
    tag: str,
    window: Mapping[str, Any],
    row_block_size: int,
) -> str:
    return _json_digest(
        {
            "cache_key": identity.cache_key,
            "A_X_sha256": matrix_hash,
            "tag": tag,
            "xlim": list(window["xlim"]),
            "ylim": list(window["ylim"]),
            "grid": int(window["grid"]),
            "row_block_size": int(row_block_size),
        }
    )


def _load_valid_surface(
    path: Path,
    identity: CacheIdentity,
    matrix_hash: str,
    surface_digest: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            if not _metadata_matches(payload, identity):
                return None
            if (
                _scalar_text(payload, "A_X_sha256") != matrix_hash
                or _scalar_text(payload, "surface_digest") != surface_digest
            ):
                return None
            x_cached = np.asarray(payload["x"])
            y_cached = np.asarray(payload["y"])
            values = np.asarray(payload["s_min"])
            if (
                x_cached.dtype != np.dtype(np.float64)
                or y_cached.dtype != np.dtype(np.float64)
                or values.dtype != np.dtype(np.float64)
                or not np.array_equal(x_cached, x_values)
                or not np.array_equal(y_cached, y_values)
                or values.shape != (len(y_values), len(x_values))
                or not np.isfinite(values).all()
                or np.any(values < 0.0)
            ):
                return None
            if _scalar_text(payload, "s_min_sha256") != _array_sha256(values):
                return None
            return np.ascontiguousarray(values)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _ensure_surface_matrix(
    cache_dir: Path,
    identity: CacheIdentity,
    hardy_matrix: np.ndarray,
    matrix_hash: str,
) -> Path:
    path = cache_dir / identity.cache_key / f"A_X_{matrix_hash}.npy"
    valid = False
    if path.is_file():
        try:
            cached = np.load(path, mmap_mode="r", allow_pickle=False)
            valid = (
                cached.shape == hardy_matrix.shape
                and cached.dtype == np.dtype(np.complex128)
                and _array_sha256(np.asarray(cached)) == matrix_hash
            )
        except (OSError, ValueError, TypeError):
            valid = False
    if not valid:
        _atomic_npy(path, np.ascontiguousarray(hardy_matrix, dtype=np.complex128))
    return path.resolve()


def _load_valid_surface_block(
    path: Path,
    block_id: int,
    y_start: int,
    y_stop: int,
    x_count: int,
) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            if int(np.asarray(payload["block_id"]).item()) != int(block_id):
                return None
            if int(np.asarray(payload["y_start"]).item()) != int(y_start):
                return None
            if int(np.asarray(payload["y_stop"]).item()) != int(y_stop):
                return None
            values = np.asarray(payload["s_block"])
            if (
                values.shape != (y_stop - y_start, x_count)
                or values.dtype != np.dtype(np.float64)
                or not np.isfinite(values).all()
                or np.any(values < 0.0)
            ):
                return None
            return np.ascontiguousarray(values)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _sample_surface(
    *,
    tag: str,
    output_path: Path,
    window: Mapping[str, Any],
    config: HistoricalPhase4Config,
    identity: CacheIdentity,
    hardy_matrix: np.ndarray,
    surface_worker: Any,
    cache_dir: Path,
    surface_workers: int,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, Path]:
    '''Explanation: A grid of smallest singular values depicts how rapidly the finite resolvent grows near spectral clusters. Since gaps between grid points remain unchecked, the surface is an explanatory diagnostic rather than a lower-bound certificate.
Functionality: Evaluate the shifted-matrix smallest singular value over a rectangular complex grid using source-keyed row blocks.'''
    grid = int(window["grid"])
    x_values = np.linspace(*window["xlim"], grid, dtype=np.float64)
    y_values = np.linspace(*window["ylim"], grid, dtype=np.float64)
    matrix_hash = _array_sha256(hardy_matrix)
    digest = _surface_digest(
        identity,
        matrix_hash,
        tag,
        window,
        int(config.surface_row_block_size),
    )
    cached = None if force else _load_valid_surface(
        output_path,
        identity,
        matrix_hash,
        digest,
        x_values,
        y_values,
    )
    if cached is not None:
        return x_values, y_values, cached, True, output_path

    matrix_path = _ensure_surface_matrix(cache_dir, identity, hardy_matrix, matrix_hash)
    block_dir = cache_dir / identity.cache_key / "surfaces" / f"{tag}_{digest}"
    block_dir.mkdir(parents=True, exist_ok=True)
    block_size = int(config.surface_row_block_size)
    block_specs = []
    block_records = []
    for block_id, y_start in enumerate(range(0, len(y_values), block_size)):
        y_stop = min(len(y_values), y_start + block_size)
        block_path = block_dir / f"rows_{y_start:04d}_{y_stop:04d}.npz"
        # Completed artefacts are bypassed in forced mode, but row blocks are
        # source-keyed computational checkpoints.  Reusing a validated block
        # makes an interrupted source rebuild resumable without accepting a
        # retained theorem-facing or diagnostic output as evidence.
        values = _load_valid_surface_block(
            block_path, block_id, y_start, y_stop, len(x_values)
        )
        block_records.append((block_id, y_start, y_stop, block_path, values))
        if values is None:
            block_specs.append(
                (
                    block_id,
                    y_start,
                    y_stop,
                    x_values,
                    y_values[y_start:y_stop],
                    str(block_path),
                )
            )

    if block_specs:
        worker_count = max(1, min(int(surface_workers), len(block_specs)))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=surface_worker.initialise_surface_worker,
            initargs=(str(matrix_path),),
        ) as pool:
            for _ in pool.map(surface_worker.sample_surface_row_block, block_specs):
                pass

    surface = np.empty((len(y_values), len(x_values)), dtype=np.float64)
    for block_id, y_start, y_stop, block_path, prior in block_records:
        values = prior
        if values is None:
            values = _load_valid_surface_block(
                block_path, block_id, y_start, y_stop, len(x_values)
            )
        if values is None:
            raise RuntimeError(f"Incomplete or invalid surface row block: {block_path}")
        surface[y_start:y_stop, :] = values
    if not np.isfinite(surface).all() or np.any(surface < 0.0):
        raise ArithmeticError("The assembled singular-value surface is invalid.")
    _atomic_npz(
        output_path,
        x=x_values,
        y=y_values,
        s_min=surface,
        row_block_dir=np.asarray(str(block_dir)),
        matrix_path=np.asarray(str(matrix_path)),
        A_X_sha256=np.asarray(matrix_hash),
        surface_digest=np.asarray(digest),
        surface_tag=np.asarray(tag),
        s_min_sha256=np.asarray(_array_sha256(surface)),
        **_metadata_arrays(identity),
    )
    return x_values, y_values, surface, False, output_path


_SURFACE_CSV_FIELDS = (
    "z_re",
    "z_im",
    "s_min",
    "log10_s_min",
    *_OUTPUT_METADATA_FIELDS,
)


def _write_surface_csv(
    path: Path,
    x_values: np.ndarray,
    y_values: np.ndarray,
    surface: np.ndarray,
    identity: CacheIdentity,
    log_floor: float,
) -> None:
    rows = []
    metadata = _output_metadata(identity)
    for y_index, y_value in enumerate(y_values):
        for x_index, x_value in enumerate(x_values):
            singular_value = float(surface[y_index, x_index])
            rows.append(
                {
                    "z_re": float(x_value),
                    "z_im": float(y_value),
                    "s_min": singular_value,
                    "log10_s_min": math.log10(max(singular_value, log_floor)),
                    **metadata,
                }
            )
    _atomic_csv(path, rows, _SURFACE_CSV_FIELDS)


def _normalise_workers(requested: int, *, production: bool, label: str) -> int:
    if int(requested) < 1:
        raise ValueError(f"{label} workers must be positive.")
    workers = min(int(requested), os.cpu_count() or 1)
    if production and workers < 2:
        raise RuntimeError(f"Production {label} requires at least two process workers.")
    return workers


def rebuild_historical_phase4(
    config: HistoricalPhase4Config,
    *,
    data_dir: Path,
    report_dir: Path,
    cache_dir: Path | None = None,
    assembly_workers: int = 24,
    surface_workers: int = 6,
    force: bool = False,
) -> HistoricalPhase4Result:
    '''Explanation: Rebuilding the whole historical Hardy-gauge experiment from source preserves a transparent comparison with the final method. The final thesis claims still rely on the later Arb whole-circle certificates, not these sampled reconstructions.
Functionality: Rebuild all retained wide-radius diagnostics from map and source.'''

    config.validate()
    production = config.mode == "production"
    assembly_workers = _normalise_workers(
        assembly_workers, production=production, label="row-block assembly"
    )
    surface_workers = _normalise_workers(
        surface_workers, production=production, label="surface sampling"
    )
    data_dir = Path(data_dir)
    report_dir = Path(report_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else data_dir / ".historical_phase4_cache"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    identity, map_spec = _build_identity(config)
    paths = _artifact_paths(config, data_dir, report_dir)
    mpmath_worker, surface_worker = _load_helpers()

    scaled, hardy, connection, eigenvalues, cache_hits = _build_or_load_matrices(
        config,
        identity,
        map_spec,
        paths,
        mpmath_worker,
        assembly_workers=assembly_workers,
        force=force,
    )
    matrix_hash = _array_sha256(hardy)
    eigenvalue_hit = False if force else _load_valid_eigenvalue_csv(
        paths["eigenvalues"], identity, matrix_hash, eigenvalues
    )
    if not eigenvalue_hit:
        _write_eigenvalues(
            paths["eigenvalues"], eigenvalues, identity, matrix_hash
        )
    cache_hits["eigenvalues"] = eigenvalue_hit

    targets = _build_target_contours(
        config, map_spec, mpmath_worker, eigenvalues
    )
    profiles, fragile_profiles, contour_hit, contour_cache_path = (
        _build_or_load_contours(
            config,
            identity,
            targets,
            hardy,
            cache_dir,
            force=force,
        )
    )
    cache_hits["contours"] = contour_hit
    tables = _diagnostic_tables(
        config,
        identity,
        targets,
        eigenvalues,
        profiles,
        fragile_profiles,
    )
    historical_regression = _validate_historical_regression_profile(
        config, scaled, hardy, tables
    )
    _write_diagnostic_tables(config, identity, paths, tables)

    surface_paths = {
        "local": paths["local_surface"],
        "global": paths["global_surface"],
        "zoom": paths["zoom_surface"],
        "deep_zoom": paths["deep_zoom_surface"],
    }
    windows = _surface_windows(config, targets)
    sampled_surfaces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for tag in ("local", "global", "zoom", "deep_zoom"):
        x_values, y_values, surface, hit, _ = _sample_surface(
            tag=tag,
            output_path=surface_paths[tag],
            window=windows[tag],
            config=config,
            identity=identity,
            hardy_matrix=hardy,
            surface_worker=surface_worker,
            cache_dir=cache_dir,
            surface_workers=surface_workers,
            force=force,
        )
        sampled_surfaces[tag] = (x_values, y_values, surface)
        cache_hits[f"surface_{tag}"] = hit
    _write_surface_csv(
        paths["local_surface_csv"],
        *sampled_surfaces["local"],
        identity,
        float(config.surface_log_floor),
    )
    _write_surface_csv(
        paths["global_surface_csv"],
        *sampled_surfaces["global"],
        identity,
        float(config.surface_log_floor),
    )

    output_keys = tuple(key for key in paths if key != "rebuild_report")
    output_records = {
        key: {
            "path": paths[key].name,
            "sha256": _sha256(paths[key]),
            "diagnostic_status": DIAGNOSTIC_STATUS,
        }
        for key in output_keys
    }
    surface_records = {
        tag: {
            "xlim": list(windows[tag]["xlim"]),
            "ylim": list(windows[tag]["ylim"]),
            "grid": int(windows[tag]["grid"]),
            "minimum_singular_value": float(np.min(sampled_surfaces[tag][2])),
            "cache_hit": bool(cache_hits[f"surface_{tag}"]),
        }
        for tag in windows
    }
    products = [float(row["epsilon_m_gamma"]) for row in tables["moats"]]
    finite_products = [value for value in products if math.isfinite(value)]
    report = {
        "map_label": MAP_LABEL,
        "producer_schema": PRODUCER_SCHEMA,
        "diagnostic_status": DIAGNOSTIC_STATUS,
        "diagnostic_description": DIAGNOSTIC_DESCRIPTION,
        "record_role": "historical_exploratory_diagnostic",
        "authoritative_for_current_thesis": False,
        "theorem_gate_eligible": False,
        "contour_interval_certified": False,
        "legacy_seed_dependency": False,
        "retained_csv_inputs": [],
        "retained_npz_inputs": [],
        "configuration": asdict(config),
        "configuration_digest": identity.configuration_digest,
        "source_digest": identity.source_digest,
        "cache_key": identity.cache_key,
        "producer_sources": identity.source_records,
        "runtime_versions": identity.runtime_versions,
        "process_execution": {
            "row_block_assembly": True,
            "assembly_workers": assembly_workers,
            "assembly_row_block_size": (
                config.assembly_row_block_size
                if config.assembly_row_block_size is not None
                else math.ceil(config.N / assembly_workers)
            ),
            "surface_row_blocks": True,
            "surface_workers": surface_workers,
            "surface_row_block_size": int(config.surface_row_block_size),
        },
        "cache_validation": {
            "keyed_by_configuration_digest": True,
            "keyed_by_source_digest": True,
            "keyed_by_runtime_versions": True,
            "surface_keyed_by_hardy_matrix_hash": True,
            "cache_hits": cache_hits,
            "contour_cache_path": str(contour_cache_path),
        },
        "historical_regression": historical_regression,
        "target_plan": [asdict(target) for target in targets],
        "target_plan_digest": _target_plan_digest(targets),
        "sampled_diagnostics": {
            "target_count": len(targets),
            "algebraic_multiplicity": int(
                sum(target.expected_multiplicity for target in targets)
            ),
            "contour_samples": int(config.contour_samples),
            "profile_rows": len(tables["profiles"]),
            "fragile_rows": len(tables["fragile"]),
            "sampled_validation_passes": int(
                sum(bool(row["sampled_validation_pass"]) for row in tables["moats"])
            ),
            "worst_finite_epsilon_m_gamma": (
                max(finite_products) if finite_products else None
            ),
        },
        "surfaces": surface_records,
        "matrix_hashes": {
            "B_scaled_sha256": _array_sha256(scaled),
            "A_X_sha256": matrix_hash,
            "T_sha256": _array_sha256(connection),
            "eigenvalues_sha256": _array_sha256(eigenvalues),
        },
        "outputs": output_records,
    }
    _atomic_text(
        paths["rebuild_report"],
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return HistoricalPhase4Result(
        artifact_paths={key: paths[key] for key in output_keys},
        report_path=paths["rebuild_report"],
        report_digest=_sha256(paths["rebuild_report"]),
        cache_hits=dict(cache_hits),
        cache_key=identity.cache_key,
    )


def _parse_args() -> argparse.Namespace:
    deployment_root = Path(__file__).resolve().parents[1]
    output_root = (
        deployment_root / "Numerics" / "outputs" / "blaschke_deformation_certifier"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--production",
        action="store_true",
        help="run the locked N=600, M=610 historical diagnostic rebuild",
    )
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="run a reduced-size source and cache smoke rebuild",
    )
    parser.add_argument("--data-dir", type=Path, default=output_root / "data")
    parser.add_argument("--report-dir", type=Path, default=output_root / "reports")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--assembly-workers", type=int, default=24)
    parser.add_argument("--surface-workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    config = (
        HistoricalPhase4Config.production_n600_m610()
        if arguments.production
        else HistoricalPhase4Config.small_test()
    )
    result = rebuild_historical_phase4(
        config,
        data_dir=arguments.data_dir,
        report_dir=arguments.report_dir,
        cache_dir=arguments.cache_dir,
        assembly_workers=arguments.assembly_workers,
        surface_workers=arguments.surface_workers,
        force=arguments.force,
    )
    print(
        json.dumps(
            {
                "diagnostic_status": DIAGNOSTIC_STATUS,
                "cache_key": result.cache_key,
                "cache_hits": result.cache_hits,
                "report_path": str(result.report_path),
                "report_sha256": result.report_digest,
            },
            indent=2,
            sort_keys=True,
        )
    )


__all__ = [
    "DIAGNOSTIC_STATUS",
    "HistoricalPhase4Config",
    "HistoricalPhase4Result",
    "TargetContour",
    "blaschke_formula_map_spec",
    "build_hardy_matrix",
    "connection_T_matrix",
    "rebuild_historical_phase4",
]


if __name__ == "__main__":
    main()
