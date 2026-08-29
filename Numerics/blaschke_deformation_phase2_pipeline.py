"""Orchestrate the clean-room Phase 2 producer chain.

The order is geometry, finite transport, and raw matrix assembly.  Each stage
persists its own certificate and the orchestrator records a hash-bound manifest
for the exact balanced configuration consumed by Cells 19 and 24A--24C.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

try:
    from .blaschke_deformation_phase2_geometry import (
        Phase2GeometryConfig,
        Phase2GeometryResult,
        certify_geometry_scan,
    )
    from .blaschke_deformation_phase2_matrix import (
        Phase2MatrixConfig,
        Phase2MatrixResult,
        certify_matrix_row,
    )
    from .blaschke_deformation_phase2_transport import (
        Phase2TransportConfig,
        Phase2TransportResult,
        certify_transport,
    )
except ImportError:
    from blaschke_deformation_phase2_geometry import (
        Phase2GeometryConfig,
        Phase2GeometryResult,
        certify_geometry_scan,
    )
    from blaschke_deformation_phase2_matrix import (
        Phase2MatrixConfig,
        Phase2MatrixResult,
        certify_matrix_row,
    )
    from blaschke_deformation_phase2_transport import (
        Phase2TransportConfig,
        Phase2TransportResult,
        certify_transport,
    )


MAP_LABEL = "blaschke_mu_0p3"
PRODUCER_SCHEMA = "phase2-pipeline-v2"


@dataclass(frozen=True)
class Phase2RebuildConfig:
    N: int = 600
    M: int = 610
    mu: str = "0.3"
    cells: int = 65536
    geometry_precision_bits: int = 192
    transport_precision_bits: int = 384
    matrix_precision_bits: int = 384
    expected_selected_rho: str = "2.725"
    expected_selected_q_gap: str = "0.927"

    @classmethod
    def production_n600_m610(cls) -> "Phase2RebuildConfig":
        return cls()


@dataclass(frozen=True)
class Phase2RebuildResult:
    selected_geometry: dict[str, Any]
    geometry: Phase2GeometryResult
    transport: Phase2TransportResult
    matrix: Phase2MatrixResult
    manifest_path: Path
    manifest_digest: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _module_hash(module_file: str) -> str:
    path = Path(module_file)
    return _sha256(path) if path.is_file() else "inline-module-path-unavailable"


def _producer_source_record(runtime_path: Path) -> dict[str, str]:
    """Return a location-independent identity for a loaded helper source."""

    runtime_path = Path(runtime_path)
    return {
        "path": f"Numerics/{runtime_path.name}",
        "sha256": _module_hash(str(runtime_path)),
    }


def rebuild_phase2_inputs(
    config: Phase2RebuildConfig,
    *,
    data_dir: Path,
    report_dir: Path,
    process_workers: int = 24,
) -> Phase2RebuildResult:
    """Regenerate all three upstream inputs consumed by the Phase 2 notebook."""

    if config.N != 600 or config.M != 610 or config.mu != "0.3":
        raise ValueError("The deployed producer is locked to mu=0.3, N=600, M=610.")
    process_workers = max(
        1, min(int(process_workers), 24, os.cpu_count() or 1)
    )
    data_dir = Path(data_dir)
    report_dir = Path(report_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    geometry_config = Phase2GeometryConfig(
        N=config.N,
        M=config.M,
        mu=config.mu,
        cells=config.cells,
        precision_bits=config.geometry_precision_bits,
        workers=process_workers,
    )
    geometry = certify_geometry_scan(
        geometry_config,
        data_dir=data_dir,
        report_dir=report_dir,
        process_workers=process_workers,
    )
    selected = dict(geometry.selected_geometry)
    if str(selected["rho"]) != config.expected_selected_rho:
        raise RuntimeError(
            "The certified 36-row scan selected an unexpected response radius."
        )
    if str(selected["q_gap_target"]) != config.expected_selected_q_gap:
        raise RuntimeError(
            "The certified 36-row scan selected an unexpected branch-gap target."
        )

    transport_config = Phase2TransportConfig(
        N=config.N,
        r=str(selected["r_candidate"]),
        rho=str(selected["rho"]),
        r_tau=str(selected["r_tau_interval_u"]),
        geometry_configuration_digest=str(selected["configuration_digest"]),
        precision_bits=config.transport_precision_bits,
        flint_threads=process_workers,
    )
    transport = certify_transport(
        transport_config,
        data_dir=data_dir,
        report_dir=report_dir,
    )

    matrix_config = Phase2MatrixConfig(
        N=config.N,
        M=config.M,
        precision_bits=config.matrix_precision_bits,
    )
    matrix = certify_matrix_row(
        matrix_config,
        selected_geometry=selected,
        transport_record=transport.record,
        data_dir=data_dir,
        report_dir=report_dir,
    )

    output_paths = (
        geometry.csv_path,
        geometry.report_path,
        transport.csv_path,
        transport.report_path,
        transport.witness_path,
        matrix.csv_path,
        matrix.report_path,
    )
    runtime_source_paths = {
        "geometry": Path(certify_geometry_scan.__code__.co_filename),
        "transport": Path(certify_transport.__code__.co_filename),
        "matrix": Path(certify_matrix_row.__code__.co_filename),
        "pipeline": Path(__file__),
    }
    manifest = {
        "map_label": MAP_LABEL,
        "producer_schema": PRODUCER_SCHEMA,
        "configuration": asdict(config),
        "process_workers": process_workers,
        "selected_geometry": selected,
        "producer_sources": {
            name: _producer_source_record(path)
            for name, path in runtime_source_paths.items()
        },
        "outputs": {
            str(path): _sha256(path)
            for path in output_paths
        },
        "clean_room_chain": [
            "complete-boundary geometry scan",
            "Arb inverse-residual finite transport",
            "Arb pure-r-scaled raw matrix certificate",
        ],
        "legacy_seed_dependency": False,
        "safe_finite_M_stage": (
            "blaschke_deformation_phase2_finite_m.py, invoked by Cell 24A"
        ),
        "resolved_response_stage": (
            "blaschke_deformation_phase2_resolved_response.py, invoked by Cell 24B"
        ),
        "unresolved_input_and_final_aggregation_stage": (
            "blaschke_deformation_phase2_final_aggregation.py, invoked by Cell 24C"
        ),
    }
    manifest_path = report_dir / "phase2_clean_room_rebuild_manifest.json"
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_digest = _sha256(manifest_path)
    return Phase2RebuildResult(
        selected_geometry=selected,
        geometry=geometry,
        transport=transport,
        matrix=matrix,
        manifest_path=manifest_path,
        manifest_digest=manifest_digest,
    )


__all__ = [
    "MAP_LABEL",
    "PRODUCER_SCHEMA",
    "Phase2RebuildConfig",
    "Phase2RebuildResult",
    "rebuild_phase2_inputs",
]
