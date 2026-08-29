"""Source-only reconstruction of the retained Phase 2 comparison rows.

The two CSV files produced here are diagnostic comparisons.  They reproduce
the fixed historical design ``rho=1.5999`` and ``q_gap=0.967`` using the
current exact-Arb-pi complete-boundary convention.  They do not enter the
promoted theorem-facing balanced certificate, whose producer is
``blaschke_deformation_phase2_pipeline``.

No stored CSV is an input.  The old whole-ellipse row, the pointwise
branch-image refinement, the ZWX13 output tail, the finite transport factor
and the pure scaled matrix term are all rebuilt from source.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import flint
from flint import arb

try:
    from . import blaschke_deformation_phase2_geometry as geometry
    from . import blaschke_deformation_phase2_matrix as matrix
    from . import blaschke_deformation_phase2_transport as transport
except ImportError:
    import blaschke_deformation_phase2_geometry as geometry
    import blaschke_deformation_phase2_matrix as matrix
    import blaschke_deformation_phase2_transport as transport


MAP_LABEL = "blaschke_mu_0p3"
PRODUCER_SCHEMA = "phase2-historical-comparisons-v1"


@dataclass(frozen=True)
class HistoricalComparisonConfig:
    N: int = 600
    M: int = 610
    mu: str = "0.3"
    rho: str = "1.5999"
    q_gap: str = "0.967"
    baseline_cells: int = 32768
    branch_image_cells: int = 65536
    geometry_precision_bits: int = 192
    transport_precision_bits: int = 384
    zwx13_maximum_extra: int = 800
    workers: int = 24

    @classmethod
    def production_n600_m610(cls) -> "HistoricalComparisonConfig":
        return cls()


@dataclass(frozen=True)
class HistoricalComparisonResult:
    baseline_record: dict[str, Any]
    effect_record: dict[str, Any]
    baseline_csv_path: Path
    effect_csv_path: Path
    report_path: Path
    report_digest: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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


def _atomic_csv(path: Path, records: Iterable[dict[str, Any]]) -> None:
    records = tuple(records)
    if not records:
        raise ValueError("At least one comparison record is required.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _source_record(path: Path) -> dict[str, str]:
    path = Path(path)
    return {
        "path": f"Numerics/{path.name}",
        "sha256": _sha256(path),
    }


def _fixed_boundary(
    config: HistoricalComparisonConfig,
    *,
    cells: int,
    workers: int,
) -> tuple[geometry.Phase2GeometryConfig, dict[str, Any]]:
    '''Explanation: The historical design chose one ellipse radius and one branch-gap margin. Re-evaluating the complete boundary with intervals shows what that older geometric choice actually guarantees, rather than trusting its formerly sampled profile.
Functionality: Evaluate the complete Arb boundary cover for the fixed historical rho and q-gap design.'''
    boundary_config = geometry.Phase2GeometryConfig(
        N=int(config.N),
        M=int(config.M),
        mu=str(config.mu),
        cells=int(cells),
        precision_bits=int(config.geometry_precision_bits),
        workers=int(workers),
        rho_candidates=(str(config.rho),),
        q_gap_candidates=(str(config.q_gap),),
    )
    boundary = geometry._rho_boundary_data(
        boundary_config,
        str(config.rho),
        int(workers),
    )
    return boundary_config, boundary


def _branch_image_input_upper(
    *,
    N: int,
    hardy_radius: arb,
    records: Iterable[tuple[float, float, float, float]],
) -> tuple[arb, int]:
    '''Explanation: The unresolved input is governed by the worst branch image over the ellipse boundary. Taking the certified maximum identifies the bottleneck cell and gives a fair theorem-aware comparison with the promoted Phase 2 design.
Functionality: Maximise the branch-image unresolved-input tail bound over all supplied boundary cells and return the attaining cell.'''
    maximum = arb(0)
    maximum_cell = 0
    for cell, record in enumerate(records):
        phi1, radius1, phi2, radius2 = (
            geometry._arb_from_outward_float(value) for value in record
        )
        local = (
            phi1 * geometry._u_tail(int(N), hardy_radius, radius1)
            + phi2 * geometry._u_tail(int(N), hardy_radius, radius2)
        ).upper()
        if local > maximum:
            maximum = local
            maximum_cell = int(cell)
    return maximum.upper(), maximum_cell


def rebuild_historical_comparisons(
    config: HistoricalComparisonConfig,
    *,
    data_dir: Path,
    report_dir: Path,
    process_workers: int | None = None,
) -> HistoricalComparisonResult:
    '''Explanation: Reconstructing the earlier designs from the same map and formulas separates mathematical improvement from cache or formatting differences. These rows explain the development of the thesis bound but are not the authoritative final certificate.
Functionality: Rebuild both retained historical-design comparison rows from source.'''

    if config.N < 1 or config.M < config.N:
        raise ValueError("The comparison producer requires M at least N at least one.")
    if config.baseline_cells < 4 or config.branch_image_cells < 4:
        raise ValueError("Each complete-boundary cover requires at least four cells.")
    workers = max(
        1,
        min(
            int(process_workers if process_workers is not None else config.workers),
            24,
            os.cpu_count() or 1,
        ),
    )
    data_dir = Path(data_dir)
    report_dir = Path(report_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    flint.ctx.prec = int(config.geometry_precision_bits)

    baseline_config, baseline_boundary = _fixed_boundary(
        config,
        cells=int(config.baseline_cells),
        workers=workers,
    )
    rho = arb(str(config.rho))
    q_gap = arb(str(config.q_gap))
    r_tau = max(
        baseline_boundary["branch1_radius_u"],
        baseline_boundary["branch2_radius_u"],
    ).upper()
    phi_star = baseline_boundary["phi_star_u"].upper()
    phi = (
        baseline_boundary["branch1_phi_u"]
        + baseline_boundary["branch2_phi_u"]
    ).upper()
    hardy_radius = (r_tau / q_gap).upper()
    q_out = (hardy_radius / rho).upper()
    q_star = max(q_gap, q_out).upper()
    if not (1 < r_tau and r_tau < hardy_radius and hardy_radius < rho):
        raise ArithmeticError("The fixed historical design is not admissible.")

    z_tail, z_terms = geometry._zwx13_tail(
        int(config.N),
        hardy_radius,
        rho,
        maximum_extra=int(config.zwx13_maximum_extra),
    )
    two_output = (
        2
        * phi_star
        * geometry._c2(hardy_radius)
        * geometry._s_infinity(q_gap)
        * z_tail
    ).upper()
    whole_input = (
        phi_star
        * geometry._u_tail(int(config.N), hardy_radius, r_tau)
    ).upper()

    historical_internal_data = data_dir / "historical_comparison_internal"
    historical_internal_reports = report_dir / "historical_comparison_internal"
    geometry_digest = geometry._configuration_digest(baseline_config)
    transport_result = transport.certify_transport(
        transport.Phase2TransportConfig(
            N=int(config.N),
            r=geometry.upper_text(hardy_radius),
            rho=str(config.rho),
            r_tau=geometry.upper_text(r_tau),
            geometry_configuration_digest=geometry_digest,
            precision_bits=int(config.transport_precision_bits),
            flint_threads=workers,
        ),
        data_dir=historical_internal_data,
        report_dir=historical_internal_reports,
    )
    kappa = arb(str(transport_result.record["kappa_hat"])).upper()
    raw_matrix = matrix._matrix_bound(
        int(config.N),
        int(config.M),
        hardy_radius,
        rho,
        q_gap,
        phi_star,
    )
    collocation = (kappa * raw_matrix).upper()
    triangle_epsilon = (two_output + whole_input + collocation).upper()

    baseline_record = {
        "N": int(config.N),
        "M": int(config.M),
        "rho": str(config.rho),
        "r": geometry.upper_text(hardy_radius),
        "r_tau": geometry.upper_text(r_tau),
        "q_out": geometry.upper_text(q_out),
        "q_gap": str(config.q_gap),
        "q_star": geometry.upper_text(q_star),
        "Phi": geometry.upper_text(phi),
        "Phi_star": geometry.upper_text(phi_star),
        "2B_out_X": geometry.upper_text(two_output),
        "B_in_X": geometry.upper_text(whole_input),
        "Bmat": geometry.upper_text(raw_matrix),
        "kappa_hat": geometry.upper_text(kappa),
        "kappa_Bmat": geometry.upper_text(collocation),
        "epsilon_X": geometry.upper_text(triangle_epsilon),
        "branch_data_certified": True,
        "output_tail_certified": True,
        "input_tail_certified": True,
        "transport_certified": True,
        "schur_matrix_certified": True,
        "total_certified": True,
    }

    branch_config, branch_boundary = _fixed_boundary(
        config,
        cells=int(config.branch_image_cells),
        workers=workers,
    )
    branch_input, maximum_cell = _branch_image_input_upper(
        N=int(config.N),
        hardy_radius=hardy_radius,
        records=branch_boundary["records"],
    )
    if not (0 < branch_input < whole_input):
        raise ArithmeticError(
            "The source-rebuilt branch-image input bound did not improve the fallback."
        )
    branch_triangle_epsilon = (two_output + branch_input + collocation).upper()
    effect_record = {
        "N": int(config.N),
        "M": int(config.M),
        "cells": int(config.branch_image_cells),
        "old_two_B_out": geometry.upper_text(two_output),
        "old_B_in": geometry.upper_text(whole_input),
        "new_branch_image_B_in_u": geometry.upper_text(branch_input),
        "old_collocation": geometry.upper_text(collocation),
        "old_epsilon": geometry.upper_text(triangle_epsilon),
        "new_epsilon_candidate": geometry.upper_text(branch_triangle_epsilon),
        "input_reduction_factor": geometry.upper_text(
            (branch_input / whole_input).upper()
        ),
        "epsilon_reduction_factor": geometry.upper_text(
            (branch_triangle_epsilon / triangle_epsilon).upper()
        ),
        "output_to_new_input_ratio": geometry.upper_text(
            (two_output / branch_input).upper()
        ),
    }

    baseline_csv_path = data_dir / "final_blaschke_N600_schur_certificate.csv"
    effect_csv_path = data_dir / "branch_image_input_tail_interval_effect_N600.csv"
    _atomic_csv(baseline_csv_path, (baseline_record,))
    _atomic_csv(effect_csv_path, (effect_record,))

    runtime_sources = {
        "historical_comparisons": Path(__file__),
        "geometry": Path(geometry.__file__),
        "transport": Path(transport.__file__),
        "matrix": Path(matrix.__file__),
    }
    report = {
        "map_label": MAP_LABEL,
        "producer_schema": PRODUCER_SCHEMA,
        "configuration": asdict(config),
        "process_workers": workers,
        "diagnostic_status": (
            "retained source-rebuilt historical-design comparison; not a theorem gate"
        ),
        "historical_design_inputs": {
            "rho": str(config.rho),
            "q_gap": str(config.q_gap),
            "baseline_cells": int(config.baseline_cells),
            "branch_image_cells": int(config.branch_image_cells),
        },
        "complete_boundary_cover": {
            "uses_arb_pi": True,
            "covers_zero_to_two_pi": True,
            "baseline_cells": int(config.baseline_cells),
            "branch_image_cells": int(config.branch_image_cells),
            "branch_image_maximum_cell": int(maximum_cell),
        },
        "aggregation": (
            "historical triangle comparison only: two_B_out plus B_in plus collocation"
        ),
        "current_theorem_aggregation": (
            "root-sum-square of B_out and B_in plus collocation; produced elsewhere"
        ),
        "legacy_seed_dependency": False,
        "zwx13_terms_used": int(z_terms),
        "producer_sources": {
            name: _source_record(path) for name, path in runtime_sources.items()
        },
        "transport": {
            "record": transport_result.record,
            "csv_sha256": _sha256(transport_result.csv_path),
            "report_sha256": _sha256(transport_result.report_path),
            "inverse_witness_sha256": transport_result.witness_digest,
        },
        "outputs": {
            baseline_csv_path.name: _sha256(baseline_csv_path),
            effect_csv_path.name: _sha256(effect_csv_path),
        },
        "baseline_record": baseline_record,
        "effect_record": effect_record,
        "baseline_geometry_configuration_digest": geometry_digest,
        "branch_geometry_configuration_digest": geometry._configuration_digest(
            branch_config
        ),
    }
    report_path = report_dir / "historical_phase2_comparison_rebuild.json"
    _atomic_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return HistoricalComparisonResult(
        baseline_record=baseline_record,
        effect_record=effect_record,
        baseline_csv_path=baseline_csv_path,
        effect_csv_path=effect_csv_path,
        report_path=report_path,
        report_digest=_sha256(report_path),
    )


__all__ = [
    "MAP_LABEL",
    "PRODUCER_SCHEMA",
    "HistoricalComparisonConfig",
    "HistoricalComparisonResult",
    "rebuild_historical_comparisons",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_root = (
        root / "Numerics" / "outputs" / "blaschke_deformation_certifier"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=output_root / "data")
    parser.add_argument("--report-dir", type=Path, default=output_root / "reports")
    parser.add_argument("--workers", type=int, default=24)
    arguments = parser.parse_args()
    result = rebuild_historical_comparisons(
        HistoricalComparisonConfig.production_n600_m610(),
        data_dir=arguments.data_dir,
        report_dir=arguments.report_dir,
        process_workers=arguments.workers,
    )
    print(
        json.dumps(
            {
                "baseline_csv_path": str(result.baseline_csv_path),
                "effect_csv_path": str(result.effect_csv_path),
                "report_path": str(result.report_path),
                "report_sha256": result.report_digest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
