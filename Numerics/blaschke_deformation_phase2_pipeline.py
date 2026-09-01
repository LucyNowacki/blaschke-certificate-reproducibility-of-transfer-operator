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
        CANONICAL_SELECTED_HARDY_RADIUS_TEXT,
        CANONICAL_SELECTED_Q_GAP_TARGET_TEXT,
        Phase2GeometryConfig,
        Phase2GeometryResult,
        certify_geometry_scan,
        exact_q_gap_contract,
        require_canonical_selected_hardy_radius,
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
        CANONICAL_SELECTED_HARDY_RADIUS_TEXT,
        CANONICAL_SELECTED_Q_GAP_TARGET_TEXT,
        Phase2GeometryConfig,
        Phase2GeometryResult,
        certify_geometry_scan,
        exact_q_gap_contract,
        require_canonical_selected_hardy_radius,
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
PRODUCER_SCHEMA = "phase2-pipeline-v3"


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
    expected_selected_r: str = CANONICAL_SELECTED_HARDY_RADIUS_TEXT
    expected_selected_q_gap: str = CANONICAL_SELECTED_Q_GAP_TARGET_TEXT

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


def _canonical_phase2_output_member(area: str, filename: str) -> str:
    """Map a staged output basename to its canonical archive member."""

    area = str(area)
    if area not in {"data", "reports"}:
        raise ValueError("A Phase 2 output member must use data or reports.")
    filename_path = Path(str(filename))
    if (
        filename_path.is_absolute()
        or len(filename_path.parts) != 1
        or filename_path.name in {"", ".", ".."}
        or "\\" in filename_path.name
    ):
        raise ValueError(
            "A Phase 2 output member requires one canonical filename, "
            f"not {filename!r}."
        )
    return (
        "Numerics/outputs/blaschke_deformation_certifier/"
        f"{area}/{filename_path.name}"
    )


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
    selected_r = require_canonical_selected_hardy_radius(
        selected["r_candidate"], label="pipeline selected Hardy radius"
    )
    if selected_r != require_canonical_selected_hardy_radius(
        config.expected_selected_r, label="pipeline expected Hardy radius"
    ):
        raise RuntimeError("The certified 36-row scan selected an unexpected Hardy radius.")
    q_gap_contract = exact_q_gap_contract(
        r_tau_upper=selected["r_tau_interval_u"],
        hardy_radius=selected["r_candidate"],
        q_gap_target=selected["q_gap_target"],
    )
    for key, value in q_gap_contract.items():
        if str(selected.get(key)) != str(value):
            raise RuntimeError(f"The selected geometry has an inconsistent {key} field.")

    transport_config = Phase2TransportConfig(
        N=config.N,
        r=CANONICAL_SELECTED_HARDY_RADIUS_TEXT,
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
        ("data", geometry.csv_path),
        ("reports", geometry.report_path),
        ("data", transport.csv_path),
        ("reports", transport.report_path),
        ("data", transport.witness_path),
        ("data", matrix.csv_path),
        ("reports", matrix.report_path),
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
            _canonical_phase2_output_member(area, path.name): _sha256(path)
            for area, path in output_paths
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
