"""Focused temporary-repository tests for fail-closed reproducibility packaging."""

from __future__ import annotations

from contextlib import contextmanager
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile
import zlib


NUMERICS_DIR = Path(__file__).resolve().parent
if str(NUMERICS_DIR) not in sys.path:
    sys.path.insert(0, str(NUMERICS_DIR))

import blaschke_deformation_reproducibility as packager
import blaschke_deformation_diagnostic_audits as audit_producer
import blaschke_deformation_historical_phase4 as phase4_producer
import verify_blaschke_reproducibility as verifier


TEST_CONDA_LOCK = """# platform: linux-64
@EXPLICIT
https://example.invalid/conda/test-package-1.0-0.conda
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", checksum)
    )


def _png_bytes(*, rgba: bytes, note: str) -> bytes:
    if len(rgba) != 4:
        raise ValueError("A one-pixel RGBA payload must contain four bytes.")
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    return b"".join(
        (
            verifier.PNG_SIGNATURE,
            _png_chunk(b"IHDR", ihdr),
            _png_chunk(b"tEXt", b"Comment\x00" + note.encode("ascii")),
            _png_chunk(b"IDAT", zlib.compress(b"\x00" + rgba)),
            _png_chunk(b"IEND", b""),
        )
    )


def _npy_bytes(payload: bytes) -> bytes:
    if not payload:
        raise ValueError("A synthetic NPY array requires at least one byte.")
    header = (
        "{'descr': '|u1', 'fortran_order': False, " f"'shape': ({len(payload)},), }}"
    ).encode("ascii")
    padding = (16 - ((10 + len(header) + 1) % 16)) % 16
    padded_header = header + b" " * padding + b"\n"
    return (
        verifier.NPY_SIGNATURE
        + b"\x01\x00"
        + struct.pack("<H", len(padded_header))
        + padded_header
        + payload
    )


def _npz_bytes(
    members: dict[str, bytes],
    *,
    timestamp: tuple[int, int, int, int, int, int],
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, members[name])
    return buffer.getvalue()


def _read_npz_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path, mode="r") as archive:
        return {name: archive.read(name) for name in archive.namelist()}


class TemporaryDeployment:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.output_dir = root.joinpath(*packager.OUTPUT_RELATIVE.parts)
        self.notebook_path = root.joinpath(*packager.NOTEBOOK_RELATIVE.parts)
        self.plan_path = root.joinpath(*packager.PLAN_RELATIVE.parts)
        self.report_path = root.joinpath(*packager.SPECTRAL_REPORT_RELATIVE.parts)
        self.theorem_input_paths = {
            key: root.joinpath(*relative.parts)
            for key, relative in verifier.THEOREM_INPUT_RELATIVES.items()
        }
        self.geometry_input_paths = {
            key: self.theorem_input_paths[key]
            for key in verifier.HARDY_MATRIX_INPUT_RELATIVES
        }
        self.contour_path = root.joinpath(*verifier.CONTOUR_CERTIFICATE_RELATIVE.parts)
        self.phase2_row_path = root.joinpath(
            *verifier.PHASE2_CERTIFIED_ROW_RELATIVE.parts
        )
        self.historical_report_path = root.joinpath(
            *verifier.HISTORICAL_REPORT_RELATIVE.parts
        )
        self.historical_baseline_path = root.joinpath(
            *verifier.HISTORICAL_BASELINE_RELATIVE.parts
        )
        self.historical_effect_path = root.joinpath(
            *verifier.HISTORICAL_EFFECT_RELATIVE.parts
        )
        self.phase4_report_path = root.joinpath(
            *verifier.HISTORICAL_PHASE4_REPORT_RELATIVE.parts
        )
        self.phase4_data_paths = {
            key: root.joinpath(*relative.parts)
            for key, relative in verifier.HISTORICAL_PHASE4_DATA_RELATIVES.items()
        }
        self.audit_report_path = root.joinpath(
            *verifier.DIAGNOSTIC_AUDIT_REPORT_RELATIVE.parts
        )
        self.audit_data_paths = {
            key: root.joinpath(*relative.parts)
            for key, relative in verifier.DIAGNOSTIC_AUDIT_DATA_RELATIVES.items()
        }

    def _write_text(self, relative: str, payload: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8", newline="")
        return path

    def _write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="",
        )

    def _write_csv(self, path: Path, records: list[dict[str, object]]) -> None:
        if not records:
            raise ValueError("A test CSV requires at least one record.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    @staticmethod
    def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise AssertionError("Test CSV has no header.")
            return list(reader.fieldnames), [dict(row) for row in reader]

    def git(self, *args: str, env: dict[str, str] | None = None) -> str:
        completed = subprocess.run(
            ("git", *args),
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        return completed.stdout.strip()

    @staticmethod
    def source_plan() -> dict[str, object]:
        return {
            "precision_settings": {
                "map_label": "blaschke_mu_0p3",
                "N": 600,
                "M": 610,
                "rho": "2.725",
                "r": "2.473669807791324",
                "hardy_matrix_eta_A_upper": "stale-eta-a",
                "hardy_matrix_midpoint_sha256": "0" * 64,
                "contour_certificate_schema": (
                    "blaschke-deformation-24-contour-hybrid-v2"
                ),
                "contour_schur_count_target_count": 24,
                "contour_schur_triangular_moat_target_count": 17,
                "contour_laurent_moat_target_count": 7,
                "contour_all_finite_counts_schur_derived": True,
                "contour_all_schur_diagonal_memberships_certified": True,
                "contour_all_finite_count_transports_certified": True,
                "contour_all_finite_counts_match_expected": True,
                "contour_all_finite_to_exact_rank_transfers_certified": True,
                "contour_arb_bits": 128,
                "contour_eta_schur_upper": "stale-eta-schur",
                "contour_target_count": 24,
                "contour_total_multiplicity": 30,
            },
            "upstream_artifact_names": ["seed.csv"],
        }

    @staticmethod
    def report() -> dict[str, object]:
        report: dict[str, object] = {
            "certificate_schema": "blaschke-deformation-24-contour-hybrid-v3",
            "map_label": "blaschke_mu_0p3",
            "N": 600,
            "M": 610,
            "rho": "2.725",
            "r": "2.473669807791324",
            "eta_A_upper_text": "1.0e-20",
            "matrix_midpoint_sha256": "a" * 64,
            "contour_precision_bits": 256,
            "eta_schur_upper_text": "2.0e-15",
            "target_count": 24,
            "total_expected_algebraic_multiplicity": 30,
            "total_certified_algebraic_multiplicity": 30,
            "schur_count_target_count": 24,
            "schur_triangular_moat_target_count": 17,
            "laurent_moat_target_count": 7,
            "status": "theorem-certified twenty-four-target Riesz-rank package",
            "binary64_candidate_runtime": {"platform": "archive-host"},
            "artifacts": {"report": "/archive/root/report.json"},
            "elapsed_seconds": 10.0,
        }
        report.update({key: True for key in packager.TRUE_THEOREM_GATES})
        report.update({key: False for key in packager.FALSE_THEOREM_GATES})
        return report

    @staticmethod
    def notebook() -> dict[str, object]:
        cells: list[dict[str, object]] = [
            {
                "cell_type": "markdown",
                "id": "dependency-map-0m",
                "metadata": {"thesis_math_dependency_map": True},
                "source": [
                    "0M\n",
                    "\n",
                    "# Reproducibility dependency map\n",
                    "\n",
                    "Synthetic dependency map used by the archive tests.\n",
                ],
            }
        ]
        for ordinal in range(1, 69):
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": ordinal,
                    "id": f"code-{ordinal:03d}",
                    "metadata": {},
                    "outputs": [],
                    "source": [f"value_{ordinal} = {ordinal}\n"],
                }
            )
        for ordinal in range(1, 72):
            cells.append(
                {
                    "cell_type": "markdown",
                    "id": f"markdown-{ordinal:03d}",
                    "metadata": {},
                    "source": [f"section {ordinal}\n"],
                }
            )
        return {
            "cells": cells,
            "metadata": {"inline_sync": True},
            "nbformat": 4,
            "nbformat_minor": 5,
        }

    def _write_plot_group(
        self, relatives: tuple[PurePosixPath, ...], *, offset: int
    ) -> None:
        for ordinal, relative in enumerate(relatives, start=offset):
            path = self.root.joinpath(*relative.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                _png_bytes(
                    rgba=bytes((ordinal, ordinal + 1, ordinal + 2, 255)),
                    note=f"archive-{ordinal}",
                )
            )

    def _write_historical_phase4_outputs(self) -> None:
        configuration_digest = "4" * 64
        source_digest = "5" * 64
        cache_key = "6" * 64
        metadata = {
            "diagnostic_status": verifier.EXPECTED_HISTORICAL_PHASE4_STATUS,
            "configuration_digest": configuration_digest,
            "source_digest": source_digest,
            "cache_key": cache_key,
        }

        compatibility_sources = {
            "final_packets": "packets",
            "final_moats": "moats",
            "final_fragile": "fragile",
        }
        for key, row_count in verifier.HISTORICAL_PHASE4_CSV_ROW_COUNTS.items():
            if key in compatibility_sources:
                continue
            self._write_csv(
                self.phase4_data_paths[key],
                [
                    {
                        "ordinal": ordinal,
                        **metadata,
                    }
                    for ordinal in range(row_count)
                ],
            )
        for key, source_key in compatibility_sources.items():
            path = self.phase4_data_paths[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.phase4_data_paths[source_key].read_bytes())

        compatibility_summary = {
            "N": 600,
            "M": 610,
            "rho": 2.45,
            "r": 2.225974769705636,
            "epsilon_X": 1.1641169413997097e-17,
            "finite_M_prefactor_certified": False,
            "sampled_validation_passes": 15,
            "target_packets": 15,
            "algebraic_count": 19,
            "worst_epsilon_m_gamma": 0.5,
            "contour_interval_certified": False,
            "moat_status": "sampled finite-section moat diagnostic only",
            "diagnostic_status": verifier.EXPECTED_HISTORICAL_PHASE4_STATUS,
            "theorem_gate_eligible": False,
            "legacy_seed_dependency": False,
            "configuration_digest": configuration_digest,
            "source_digest": source_digest,
            "cache_key": cache_key,
        }
        self._write_json(
            self.phase4_data_paths["compatibility_summary"],
            compatibility_summary,
        )

        for key, member_names in verifier.HISTORICAL_PHASE4_NPZ_MEMBER_NAMES.items():
            members = {
                name: _npy_bytes(f"archive:{key}:{name}".encode("ascii"))
                for name in member_names
            }
            for name in verifier.HISTORICAL_PHASE4_VOLATILE_NPZ_MEMBERS.get(
                key, frozenset()
            ):
                members[name] = _npy_bytes(
                    f"/archive/cache/{key}/{name}".encode("ascii")
                )
            path = self.phase4_data_paths[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_npz_bytes(members, timestamp=(2026, 8, 27, 1, 2, 4)))

        target_plan = []
        for rank, name in enumerate(
            verifier.EXPECTED_HISTORICAL_PHASE4_TARGET_NAMES, start=1
        ):
            family, power_text = name.split("^", 1)
            target_plan.append(
                {
                    "rank": rank,
                    "name": name,
                    "family": family,
                    "power": int(power_text),
                    "centre": 1.0 / (rank + 1),
                    "expected_multiplicity": 2 if family == "mu" else 1,
                    "nearest_target_separation": 0.02,
                    "radius": 0.01,
                }
            )
        surface_grids = {
            "local": 25,
            "global": 45,
            "zoom": 161,
            "deep_zoom": 201,
        }
        report = {
            "map_label": verifier.EXPECTED_MAP_LABEL,
            "producer_schema": verifier.EXPECTED_HISTORICAL_PHASE4_SCHEMA,
            "diagnostic_status": verifier.EXPECTED_HISTORICAL_PHASE4_STATUS,
            "diagnostic_description": (
                "retained historical wide-radius sampled finite-section diagnostic; "
                "not interval-certified and not eligible for a theorem gate"
            ),
            "record_role": "historical_exploratory_diagnostic",
            "authoritative_for_current_thesis": False,
            "theorem_gate_eligible": False,
            "contour_interval_certified": False,
            "legacy_seed_dependency": False,
            "retained_csv_inputs": [],
            "retained_npz_inputs": [],
            "configuration": dict(verifier.EXPECTED_HISTORICAL_PHASE4_CONFIGURATION),
            "configuration_digest": configuration_digest,
            "source_digest": source_digest,
            "cache_key": cache_key,
            "producer_sources": {
                "historical_phase4_producer": {
                    "path": "Numerics/blaschke_deformation_historical_phase4.py",
                    "sha256": _sha256(
                        self.root
                        / "Numerics"
                        / "blaschke_deformation_historical_phase4.py"
                    ),
                },
                "mpmath_row_worker": {
                    "path": "Numerics/mpmath_pf_raw.py",
                    "sha256": _sha256(self.root / "Numerics" / "mpmath_pf_raw.py"),
                },
                "hardy_surface_worker": {
                    "path": "Numerics/hardy_moat_surface_worker.py",
                    "sha256": _sha256(
                        self.root / "Numerics" / "hardy_moat_surface_worker.py"
                    ),
                },
            },
            "runtime_versions": {
                "python": "test-version",
                "python_implementation": "CPython",
                "numpy": "test-version",
                "scipy": "test-version",
                "mpmath": "test-version",
                "byteorder": sys.byteorder,
            },
            "process_execution": {
                "row_block_assembly": True,
                "assembly_workers": 24,
                "assembly_row_block_size": 32,
                "surface_row_blocks": True,
                "surface_workers": 6,
                "surface_row_block_size": 2,
            },
            "cache_validation": {
                "keyed_by_configuration_digest": True,
                "keyed_by_source_digest": True,
                "keyed_by_runtime_versions": True,
                "surface_keyed_by_hardy_matrix_hash": True,
                "cache_hits": {
                    "scaled_block": False,
                    "hardy_matrix": False,
                    "eigenvalues": False,
                    "contours": False,
                    "surface_local": False,
                    "surface_global": False,
                    "surface_zoom": False,
                    "surface_deep_zoom": False,
                },
                "contour_cache_path": "/archive/cache/contour_samples.npz",
            },
            "historical_regression": {
                "contract": "source-rederived historical sampled comparison",
                "pure_scaled_binary64_boundary_after_mpmath_dps": 200,
                "hardy_similarity": "binary64 T B_scaled T_inverse historical path",
                "contour_radius_rule": (
                    "finite-eigenvalue cluster-to-nearest-outside mid-gap"
                ),
                "scaled_matrix_maximum_modulus": 1.965e9,
                "hardy_matrix_maximum_modulus": 1.775e9,
                "sampled_validation_passes": 15,
                "worst_finite_epsilon_m_gamma": 0.5,
                "theorem_gate_eligible": False,
            },
            "target_plan": target_plan,
            "target_plan_digest": "a" * 64,
            "sampled_diagnostics": {
                "target_count": 15,
                "algebraic_multiplicity": 19,
                "contour_samples": 128,
                "profile_rows": 15 * 128,
                "fragile_rows": 4,
                "sampled_validation_passes": 15,
                "worst_finite_epsilon_m_gamma": 0.5,
            },
            "surfaces": {
                name: {
                    "xlim": [0.0, 1.0],
                    "ylim": [-1.0, 1.0],
                    "grid": grid,
                    "minimum_singular_value": 0.01,
                    "cache_hit": False,
                }
                for name, grid in surface_grids.items()
            },
            "matrix_hashes": {
                "B_scaled_sha256": "b" * 64,
                "A_X_sha256": "c" * 64,
                "T_sha256": "d" * 64,
                "eigenvalues_sha256": "e" * 64,
            },
            "outputs": {
                key: {
                    "path": relative.name,
                    "sha256": _sha256(self.phase4_data_paths[key]),
                    "diagnostic_status": verifier.EXPECTED_HISTORICAL_PHASE4_STATUS,
                }
                for key, relative in verifier.HISTORICAL_PHASE4_DATA_RELATIVES.items()
            },
        }
        self._write_json(self.phase4_report_path, report)
        self._write_plot_group(
            verifier.HISTORICAL_PHASE4_PLOT_RELATIVES,
            offset=20,
        )

    def _write_diagnostic_audit_outputs(self) -> None:
        universal_items = (
            "target source and contour role",
            "Bernstein branch geometry",
            "single-space Schur envelope",
            "finite packet counts",
            "finite-section contour moats",
            "sampled small-gain test",
            "overall spectral certification",
        )
        universal_rows = []
        for ordinal, audit_item in enumerate(universal_items):
            status = (
                "theorem_certified"
                if ordinal == 0
                else (
                    "diagnostic_not_theorem_certified"
                    if ordinal == len(universal_items) - 1
                    else "sampled_not_theorem_certified"
                )
            )
            universal_rows.append(
                {
                    "map_label": verifier.EXPECTED_MAP_LABEL,
                    "audit_item": audit_item,
                    "status": status,
                    "evidence": f"source-generated evidence {ordinal}",
                }
            )
        self._write_csv(self.audit_data_paths["universal_audit"], universal_rows)

        first14_rows = []
        for ordinal, name in enumerate(
            verifier.EXPECTED_DIAGNOSTIC_TARGET_NAMES, start=1
        ):
            row = {
                field: f"{field}-{ordinal}"
                for field in verifier.EXPECTED_FIRST14_AUDIT_FIELDS
            }
            row.update(
                {
                    "name": name,
                    "target": name,
                    "contour_interval_certified": False,
                    "finite_count_certified": False,
                    "certified_small_gain_pass": False,
                    "validation_status": "sampled_not_interval_certified",
                    "finite_section_status": "sampled_not_interval_certified",
                    "riesz_rank_status": "sampled_pass_not_theorem_certified",
                }
            )
            first14_rows.append(row)
        self._write_csv(self.audit_data_paths["first14_audit"], first14_rows)

        output_specs = {
            "universal_audit": (
                verifier.EXPECTED_UNIVERSAL_AUDIT_FIELDS,
                7,
            ),
            "first14_audit": (
                verifier.EXPECTED_FIRST14_AUDIT_FIELDS,
                14,
            ),
        }
        report = {
            "producer_schema": verifier.EXPECTED_DIAGNOSTIC_AUDIT_SCHEMA,
            "map_label": verifier.EXPECTED_MAP_LABEL,
            "diagnostic_only": True,
            "diagnostic_status": verifier.EXPECTED_DIAGNOSTIC_AUDIT_STATUS,
            "legacy_seed_dependency": False,
            "source_extraction": {
                "notebook": {
                    "path": packager.SOURCE_NOTEBOOK_RELATIVE.as_posix(),
                    "sha256": _sha256(
                        self.root.joinpath(*packager.SOURCE_NOTEBOOK_RELATIVE.parts)
                    ),
                    "cells": {
                        "Cell 100": {
                            "cell_id": "cell-100",
                            "source_sha256": "2" * 64,
                        },
                        "Cell 102": {
                            "cell_id": "cell-102",
                            "source_sha256": "3" * 64,
                        },
                    },
                },
                "producer": {
                    "path": "Numerics/blaschke_deformation_diagnostic_audits.py",
                    "sha256": _sha256(
                        self.root
                        / "Numerics"
                        / "blaschke_deformation_diagnostic_audits.py"
                    ),
                },
            },
            "inputs": {
                name: {
                    "kind": "explicit_generated_path",
                    "path": f"/archive/inputs/{name}.csv",
                    "sha256": str(ordinal) * 64,
                    "rows": ordinal,
                    "columns": ["source", name],
                }
                for ordinal, name in enumerate(
                    ("targets", "geometry", "schur_rows", "moat_rows"),
                    start=5,
                )
            },
            "checks": {
                "map_lock": True,
                "schema_alignment": True,
                "target_alignment": True,
                "sampled_claims_remain_diagnostic": True,
                "target_count": 14,
                "target_names": list(verifier.EXPECTED_DIAGNOSTIC_TARGET_NAMES),
                "sampled_schur_N": list(verifier.EXPECTED_DIAGNOSTIC_SCHUR_N),
            },
            "outputs": {
                verifier.DIAGNOSTIC_AUDIT_DATA_RELATIVES[key].name: {
                    "path": (
                        "/archive/data/"
                        + verifier.DIAGNOSTIC_AUDIT_DATA_RELATIVES[key].name
                    ),
                    "rows": rows,
                    "columns": list(fields),
                    "sha256": _sha256(self.audit_data_paths[key]),
                    "diagnostic_only": True,
                }
                for key, (fields, rows) in output_specs.items()
            },
        }
        self._write_json(self.audit_report_path, report)
        self._write_plot_group(
            verifier.DIAGNOSTIC_AUDIT_PLOT_RELATIVES,
            offset=40,
        )

    def _write_semantic_outputs(self) -> None:
        contour_records = [
            {
                "rank": rank,
                "name": f"target-{rank:02d}",
                "theorem_certified": True,
                "inverse_sampling_seconds": f"{rank / 10:.1f}",
                "elapsed_seconds": f"{rank / 5:.1f}",
            }
            for rank in range(1, 25)
        ]
        self._write_csv(self.contour_path, contour_records)
        self._write_csv(
            self.phase2_row_path,
            [
                {
                    "N": 600,
                    "M": 610,
                    "epsilon_upper_text": "3.3264e-20",
                    "certified": True,
                }
            ],
        )

        baseline_record: dict[str, object] = {
            "N": 600,
            "M": 610,
            "rho": "1.5999",
            "q_gap": "0.967",
            "epsilon_X": "1.298e-8",
            "branch_data_certified": True,
            "output_tail_certified": True,
            "input_tail_certified": True,
            "transport_certified": True,
            "schur_matrix_certified": True,
            "total_certified": True,
        }
        effect_record: dict[str, object] = {
            "N": 600,
            "M": 610,
            "cells": 65536,
            "old_epsilon": "1.298e-8",
            "new_epsilon_candidate": "1.017e-8",
            "epsilon_reduction_factor": "0.7837",
        }
        self._write_csv(self.historical_baseline_path, [baseline_record])
        self._write_csv(self.historical_effect_path, [effect_record])
        historical_report = {
            "map_label": "blaschke_mu_0p3",
            "producer_schema": "phase2-historical-comparisons-v1",
            "configuration": dict(verifier.EXPECTED_HISTORICAL_CONFIGURATION),
            "process_workers": 24,
            "diagnostic_status": (
                "retained source-rebuilt historical-design comparison; "
                "not a theorem gate"
            ),
            "historical_design_inputs": {
                "rho": "1.5999",
                "q_gap": "0.967",
                "baseline_cells": 32768,
                "branch_image_cells": 65536,
            },
            "complete_boundary_cover": {
                "uses_arb_pi": True,
                "covers_zero_to_two_pi": True,
                "baseline_cells": 32768,
                "branch_image_cells": 65536,
                "branch_image_maximum_cell": 123,
            },
            "aggregation": (
                "historical triangle comparison only: two_B_out plus B_in "
                "plus collocation"
            ),
            "current_theorem_aggregation": (
                "root-sum-square of B_out and B_in plus collocation; "
                "produced elsewhere"
            ),
            "legacy_seed_dependency": False,
            "zwx13_terms_used": 801,
            "producer_sources": {
                "historical_comparisons": {
                    "path": "Numerics/historical.py",
                    "sha256": _sha256(self.root / "Numerics" / "historical.py"),
                }
            },
            "baseline_record": baseline_record,
            "effect_record": effect_record,
            "baseline_geometry_configuration_digest": "c" * 64,
            "branch_geometry_configuration_digest": "d" * 64,
            "outputs": {
                self.historical_baseline_path.name: _sha256(
                    self.historical_baseline_path
                ),
                self.historical_effect_path.name: _sha256(self.historical_effect_path),
            },
        }
        self._write_json(self.historical_report_path, historical_report)

        for ordinal, relative in enumerate(verifier.HISTORICAL_PLOT_RELATIVES, start=1):
            path = self.root.joinpath(*relative.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                _png_bytes(
                    rgba=bytes((ordinal, ordinal + 1, ordinal + 2, 255)),
                    note=f"archive-{ordinal}",
                )
            )
        self._write_historical_phase4_outputs()
        self._write_diagnostic_audit_outputs()

    def _write_source_generated_diagnostic_outputs(self) -> None:
        data_dir = self.output_dir / "data"
        report_dir = self.output_dir / "reports"
        phase1_outputs: dict[str, dict[str, object]] = {}
        for name in sorted(packager.EXPECTED_PHASE1_OUTPUT_NAMES):
            csv_path = data_dir / f"{name}.csv"
            parquet_path = data_dir / f"{name}.parquet"
            self._write_csv(
                csv_path,
                [{"map_name": packager.EXPECTED_MAP_LABEL, "value": "1"}],
            )
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            parquet_path.write_bytes(f"fixture-parquet:{name}\n".encode("ascii"))
            phase1_outputs[name] = {
                "path": str(csv_path),
                "rows": 1,
                "columns": ["map_name", "value"],
                "sha256": _sha256(csv_path),
                "parquet_path": str(parquet_path),
                "parquet_sha256": _sha256(parquet_path),
            }

        cloud_pairs = [list(pair) for pair in packager.EXPECTED_PHASE1_CLOUD_PAIRS]
        fixed_pairs = [[25, 25]]
        heat_pairs = [[1, 1]]
        unique_pairs = sorted(cloud_pairs + fixed_pairs + heat_pairs)
        worker_path = self.root / packager.PHASE1_WORKER_RELATIVE.as_posix()
        phase1_report = {
            "producer_schema": packager.EXPECTED_PHASE1_PRODUCER_SCHEMA,
            "schema_version": packager.EXPECTED_DIAGNOSTIC_SCHEMA_VERSION,
            "generation_id": "1" * 64,
            "map_label": packager.EXPECTED_MAP_LABEL,
            "diagnostic_only": True,
            "theorem_gate": False,
            "legacy_seed_dependency": False,
            "commit_protocol": "atomic fixture replacement",
            "config": {
                "cloud_pairs": cloud_pairs,
                "fixed_n": 25,
                "fixed_m_values": [25],
                "heat_n_values": [1],
                "heat_m_values": [1],
                "dps": 80,
                "max_power": 24,
                "max_clusters": 24,
                "expected_target_count": 24,
                "assembly_workers": 1,
                "row_block_size": None,
                "progress": False,
            },
            "lineage": {
                "map_spec_sha256": "2" * 64,
                "reference_clusters_sha256": None,
                "expected_target_names_sha256": "3" * 64,
                "raw_sweep": {
                    "module": "mpmath_pf_raw",
                    "qualname": "run_pair_sweep_mpmath_raw",
                    "module_path": packager.PHASE1_WORKER_RELATIVE.as_posix(),
                    "source_scope": "complete module file",
                    "source_sha256": _sha256(worker_path),
                },
                "runtime": {
                    "python": "fixture",
                    "numpy": "fixture",
                    "pandas": "fixture",
                },
            },
            "jobs": [
                {
                    "N": n_value,
                    "M": m_value,
                    "ok": True,
                    "assembly_mode": "serial",
                    "assembly_workers": 1,
                    "row_block_size": None,
                    "gram_error": "0",
                }
                for n_value, m_value in unique_pairs
            ],
            "schedules": {
                "cloud_pairs": cloud_pairs,
                "fixed_pairs": fixed_pairs,
                "heat_pairs": heat_pairs,
                "unique_pairs": unique_pairs,
            },
            "outputs": phase1_outputs,
        }
        self._write_json(
            report_dir / "blaschke_deformation_phase1_diagnostics_rebuild.json",
            phase1_report,
        )

        sampled_path = data_dir / packager.EXPECTED_SAMPLED_SCHUR_FILENAME
        sampled_records = [
            {
                "map_label": packager.EXPECTED_MAP_LABEL,
                "N": n_value,
                "M": n_value + packager.EXPECTED_SAMPLED_SCHUR_OVERSAMPLING,
                "epsilon_schur_diagnostic": "1e-6",
                "transported_Bmat_diagnostic": "1e-7",
                "status": "diagnostic",
                "geometry_status": "admissible",
            }
            for n_value in packager.EXPECTED_SAMPLED_SCHUR_N
        ]
        self._write_csv(sampled_path, sampled_records)
        envelope_path = self.root / packager.SAMPLED_SCHUR_ENVELOPE_RELATIVE.as_posix()
        kappa_module = "notebook_fixture"
        kappa_qualname = "_sampled_schur_kappa"
        kappa_digest = hashlib.sha256(
            f"{kappa_module}:{kappa_qualname}".encode("utf-8")
        ).hexdigest()
        sampled_report = {
            "producer_schema": packager.EXPECTED_SAMPLED_SCHUR_PRODUCER_SCHEMA,
            "schema_version": packager.EXPECTED_DIAGNOSTIC_SCHEMA_VERSION,
            "map_label": packager.EXPECTED_MAP_LABEL,
            "diagnostic_only": True,
            "theorem_gate": False,
            "legacy_seed_dependency": False,
            "commit_protocol": "atomic fixture replacement",
            "config": {
                "n_values": list(packager.EXPECTED_SAMPLED_SCHUR_N),
                "oversampling": packager.EXPECTED_SAMPLED_SCHUR_OVERSAMPLING,
            },
            "geometry": {"rho": 2.725, "r_tau": 2.293, "r": 2.47, "Phi": 1.0},
            "lineage": {
                "sampled_schur_envelope": {
                    "module": "transfer_spectrum_certification",
                    "qualname": "sampled_schur_envelope",
                    "module_path": packager.SAMPLED_SCHUR_ENVELOPE_RELATIVE.as_posix(),
                    "source_scope": "complete module file",
                    "source_sha256": _sha256(envelope_path),
                },
                "kappa": {
                    "module": kappa_module,
                    "qualname": kappa_qualname,
                    "module_path": packager.THESIS_NOTEBOOK_BUILDER_RELATIVE.as_posix(),
                    "source_scope": "stable callable identity",
                    "source_sha256": kappa_digest,
                },
                "runtime": {"python": "fixture", "pandas": "fixture"},
            },
            "output": {
                "path": str(sampled_path),
                "rows": len(sampled_records),
                "columns": list(sampled_records[0]),
                "sha256": _sha256(sampled_path),
            },
        }
        self._write_json(
            report_dir / "blaschke_deformation_sampled_schur_diagnostics_rebuild.json",
            sampled_report,
        )

    def initialise(self) -> None:
        self.root.mkdir(parents=True)
        self._write_text(
            ".gitignore",
            "Numerics/outputs/blaschke_deformation_certifier/reproducibility/\n"
            "__pycache__/\n*.pyc\n",
        )
        self._write_text("README.md", "# Temporary Final Deployment\n")
        self._write_text(
            "requirements.txt", "python-flint==0.8.0\nmpmath==1.3.0\n"
        )
        self._write_text(
            "pip-requirements-lock.txt",
            "pip==26.1.2\nmpmath==1.3.0\npython-flint==0.8.0\n"
            "threadpoolctl==3.6.0\n",
        )
        self._write_text("conda-explicit-lock.txt", TEST_CONDA_LOCK)
        builder = """CURATED_INLINE_HELPER_ORDINALS = (1,)

def validate_inline_helper_sync(notebook, *, helper_ordinals, allow_notebook_provenance):
    if tuple(helper_ordinals) != CURATED_INLINE_HELPER_ORDINALS:
        raise AssertionError("unexpected helper ordinals")
    if allow_notebook_provenance is not True:
        raise AssertionError("notebook provenance must be allowed")
    if notebook.get("metadata", {}).get("inline_sync") is not True:
        raise AssertionError("inline helper mismatch")

def validate_curated_counterpart(notebook):
    cells = notebook.get("cells", [])
    if len(cells) != 140:
        raise AssertionError("unexpected notebook cell count")
    first = cells[0]
    if first.get("id") != "dependency-map-0m":
        raise AssertionError("missing Cell 0M")
    if first.get("metadata", {}).get("thesis_math_dependency_map") is not True:
        raise AssertionError("missing Cell 0M provenance marker")
    validate_inline_helper_sync(
        notebook,
        helper_ordinals=CURATED_INLINE_HELPER_ORDINALS,
        allow_notebook_provenance=True,
    )
"""
        self._write_text(
            "Numerics/build_blaschke_deformation_thesis_math_notebook.py",
            builder,
        )
        self._write_json(
            self.root.joinpath(*packager.TEMPLATE_NOTEBOOK_RELATIVE.parts),
            {"cells": [], "metadata": {}},
        )
        self._write_json(
            self.root.joinpath(*packager.SOURCE_NOTEBOOK_RELATIVE.parts),
            {"cells": [], "metadata": {}},
        )
        self._write_json(self.notebook_path, self.notebook())
        self._write_json(self.plan_path, self.source_plan())
        for ordinal, path in enumerate(self.theorem_input_paths.values(), start=1):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"theorem-input-{ordinal}\n".encode("ascii"))
        report = self.report()
        report["geometry_input_hashes"] = {
            key: _sha256(path) for key, path in self.geometry_input_paths.items()
        }
        report["input_hashes"] = {
            key: _sha256(path) for key, path in self.theorem_input_paths.items()
        }
        self._write_json(self.report_path, report)
        for relative in (
            "Numerics/historical.py",
            "Numerics/blaschke_deformation_historical_phase4.py",
            "Numerics/mpmath_pf_raw.py",
            "Numerics/hardy_moat_surface_worker.py",
            "Numerics/blaschke_deformation_diagnostic_audits.py",
            "Numerics/transfer_spectrum_certification.py",
            "Numerics/blaschke_deformation_phase1_diagnostics.py",
            "Numerics/blaschke_deformation_sampled_schur_diagnostics.py",
        ):
            self._write_text(relative, f'"""Fixture source for {relative}."""\n')
        seed = self.output_dir / "data" / "seed.csv"
        seed.parent.mkdir(parents=True, exist_ok=True)
        seed.write_text("name,value\nseed,1\n", encoding="utf-8", newline="")
        self._write_source_generated_diagnostic_outputs()
        self._write_semantic_outputs()
        shutil.copyfile(
            Path(packager.__file__).resolve(),
            self.root / "Numerics" / "blaschke_deformation_reproducibility.py",
        )
        shutil.copyfile(
            Path(verifier.__file__).resolve(),
            self.root / "Numerics" / "verify_blaschke_reproducibility.py",
        )
        shutil.copyfile(
            NUMERICS_DIR / "verify_blaschke_deformation_reproducibility.py",
            self.root
            / "Numerics"
            / "verify_blaschke_deformation_reproducibility.py",
        )
        for name in (
            "blaschke_source_only_replay_policy.json",
            "prepare_blaschke_source_only_replay.py",
            "run_blaschke_clean_room_replay.py",
        ):
            shutil.copyfile(
                NUMERICS_DIR / name,
                self.root / "Numerics" / name,
            )

        self.git("init", "-q")
        self.git("config", "user.name", "Reproducibility Test")
        self.git("config", "user.email", "reproducibility@example.invalid")
        self.refresh_manifest()
        self.commit("initial deployment")

    def refresh_manifest(self) -> None:
        output = subprocess.run(
            ("git", "ls-files", "-co", "--exclude-standard", "-z"),
            cwd=self.root,
            check=True,
            capture_output=True,
        ).stdout
        selected: list[str] = []
        for label in output.decode("utf-8").split("\0"):
            if not label or label in packager.AUTHORITATIVE_EXCLUDED_PATHS:
                continue
            if any(
                label.startswith(prefix)
                for prefix in packager.AUTHORITATIVE_EXCLUDED_PREFIXES
            ):
                continue
            if (self.root / label).is_file():
                selected.append(label)
        payload = "".join(
            f"{_sha256(self.root / label)}  {label}\n"
            for label in sorted(set(selected))
        )
        (self.root / "MANIFEST.sha256").write_text(
            payload, encoding="utf-8", newline=""
        )

    def commit(self, message: str) -> None:
        self.git("add", "-A")
        env = os.environ.copy()
        env.update(
            {
                "GIT_AUTHOR_DATE": "2026-08-27T00:00:00+00:00",
                "GIT_COMMITTER_DATE": "2026-08-27T00:00:00+00:00",
            }
        )
        self.git("-c", "commit.gpgsign=false", "commit", "-q", "-m", message, env=env)

    def refresh_and_commit(self, message: str) -> None:
        self.refresh_manifest()
        self.commit(message)

    def read_plan(self) -> dict[str, object]:
        value = json.loads(self.plan_path.read_text(encoding="utf-8"))
        assert isinstance(value, dict)
        return value


class ReproducibilityBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="blaschke-repro-test-")
        self.deployment = TemporaryDeployment(
            Path(self.temporary.name) / "Final Deployment"
        )
        self.deployment.initialise()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @contextmanager
    def package_environment(self):
        versions = {name: "test-version" for name in packager.PACKAGE_NAMES}
        versions.update(
            {
                "pip": "26.1.2",
                "mpmath": "1.3.0",
                "python-flint": "0.8.0",
                "threadpoolctl": "3.6.0",
            }
        )
        with mock.patch.object(
            packager, "_conda_explicit_lock", return_value=TEST_CONDA_LOCK
        ), mock.patch.object(packager, "_package_versions", return_value=versions):
            yield

    def build(self) -> dict[str, object]:
        plan = self.deployment.read_plan()
        precision = plan["precision_settings"]
        names = plan["upstream_artifact_names"]
        assert isinstance(precision, dict)
        assert isinstance(names, list)
        with self.package_environment():
            return packager.build_reproducibility_bundle(
                repo_root=self.deployment.root,
                notebook_path=self.deployment.notebook_path,
                output_dir=self.deployment.output_dir,
                precision_settings=precision,
                upstream_artifact_names=names,
            )

    def test_legacy_verifier_reports_current_execution_contract(self) -> None:
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": count,
                    "outputs": [],
                }
                for count in range(1, verifier.EXPECTED_NOTEBOOK_CODE_CELLS + 1)
            ]
            + [
                {"cell_type": "markdown"}
                for _ in range(
                    verifier.EXPECTED_NOTEBOOK_CELLS
                    - verifier.EXPECTED_NOTEBOOK_CODE_CELLS
                )
            ]
        }
        notebook["cells"][0]["execution_count"] = 0
        with self.assertRaisesRegex(
            verifier.VerificationError,
            rf"1-{verifier.EXPECTED_NOTEBOOK_CODE_CELLS}",
        ):
            verifier._validate_notebook(notebook)

    def test_exact_reaggregated_theorem_status_is_accepted(self) -> None:
        plan = self.deployment.source_plan()
        report = self.deployment.report()
        report["status"] = (
            "theorem-certified twenty-four-target Riesz-rank package; "
            "finite moats reused and small-gain products reaggregated"
        )
        effective = packager.refresh_reproducibility_plan(plan, report)
        self.assertEqual(
            effective["precision_settings"]["contour_target_count"],
            24,
        )

        report["status"] = str(report["status"]) + "; unreviewed suffix"
        with self.assertRaisesRegex(
            packager.ReproducibilityError,
            "Unexpected spectral report status",
        ):
            packager.refresh_reproducibility_plan(plan, report)

    def test_archive_verifier_accepts_only_exact_reaggregated_status(self) -> None:
        plan = self.deployment.source_plan()
        report = self.deployment.report()
        report["status"] = (
            "theorem-certified twenty-four-target Riesz-rank package; "
            "finite moats reused and small-gain products reaggregated"
        )
        effective = packager.refresh_reproducibility_plan(plan, report)
        verifier._validate_plan_and_report(effective, report)

        report["status"] = str(report["status"]) + "; unreviewed suffix"
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "Unexpected theorem-facing report status",
        ):
            verifier._validate_plan_and_report(effective, report)

    def test_semantic_source_digest_ignores_only_docstrings(self) -> None:
        source_path = self.deployment.root / "semantic-source.py"
        source_path.write_text(
            "'''old module explanation'''\n"
            "def square(value):\n"
            "    '''old function explanation'''\n"
            "    return value * value\n",
            encoding="utf-8",
        )
        baseline = packager._semantic_ast_sha256(source_path)

        source_path.write_text(
            "'''new intuitive module explanation'''\n"
            "def square(value):\n"
            "    '''new intuitive mathematical explanation'''\n"
            "    return value * value\n",
            encoding="utf-8",
        )
        self.assertEqual(baseline, packager._semantic_ast_sha256(source_path))

        source_path.write_text(
            "def square(value):\n"
            "    return value * value + 1\n",
            encoding="utf-8",
        )
        self.assertNotEqual(baseline, packager._semantic_ast_sha256(source_path))

    def test_committed_conda_lock_must_match_live_environment(self) -> None:
        self.deployment._write_text(
            "conda-explicit-lock.txt",
            TEST_CONDA_LOCK.replace("test-package-1.0", "different-package-2.0"),
        )
        self.deployment.refresh_and_commit("change committed Conda lock")
        with self.package_environment(), self.assertRaisesRegex(
            packager.ReproducibilityError,
            "does not match the live packaging environment",
        ):
            plan = self.deployment.read_plan()
            packager.build_reproducibility_bundle(
                repo_root=self.deployment.root,
                notebook_path=self.deployment.notebook_path,
                output_dir=self.deployment.output_dir,
                precision_settings=plan["precision_settings"],
                upstream_artifact_names=plan["upstream_artifact_names"],
            )

    def test_semantic_contract_matches_current_producer_filenames(self) -> None:
        config = phase4_producer.HistoricalPhase4Config.production_n600_m610()
        producer_paths = phase4_producer._artifact_paths(
            config,
            Path("/producer/data"),
            Path("/producer/reports"),
        )
        self.assertEqual(
            {
                key: producer_paths[key].name
                for key in verifier.HISTORICAL_PHASE4_DATA_RELATIVES
            },
            {
                key: relative.name
                for key, relative in verifier.HISTORICAL_PHASE4_DATA_RELATIVES.items()
            },
        )
        self.assertEqual(
            producer_paths["rebuild_report"].name,
            verifier.HISTORICAL_PHASE4_REPORT_RELATIVE.name,
        )
        self.assertEqual(
            audit_producer.UNIVERSAL_AUDIT_FILENAME,
            verifier.DIAGNOSTIC_AUDIT_DATA_RELATIVES["universal_audit"].name,
        )
        self.assertEqual(
            audit_producer.FIRST14_AUDIT_FILENAME,
            verifier.DIAGNOSTIC_AUDIT_DATA_RELATIVES["first14_audit"].name,
        )
        self.assertEqual(
            audit_producer.REPORT_FILENAME,
            verifier.DIAGNOSTIC_AUDIT_REPORT_RELATIVE.name,
        )
        self.assertEqual(
            (
                verifier.EXPECTED_NOTEBOOK_CELLS,
                verifier.EXPECTED_NOTEBOOK_CODE_CELLS,
            ),
            (
                packager.EXPECTED_NOTEBOOK_CELLS,
                packager.EXPECTED_NOTEBOOK_CODE_CELLS,
            ),
        )

    def test_pip_only_requirements_are_exact_and_match_installed_versions(self) -> None:
        pins = packager._validate_pip_requirements(
            self.deployment.root,
            {
                "pip": "26.1.2",
                "mpmath": "1.3.0",
                "python-flint": "0.8.0",
                "threadpoolctl": "3.6.0",
            },
        )
        self.assertEqual(
            pins,
            {
                "mpmath": "1.3.0",
                "pip": "26.1.2",
                "python-flint": "0.8.0",
                "threadpoolctl": "3.6.0",
            },
        )

    def test_non_exact_pip_requirement_is_rejected(self) -> None:
        self.deployment._write_text(
            "pip-requirements-lock.txt",
            "pip==26.1.2\npython-flint==0.8.0\n"
            "threadpoolctl==3.6.0\nmpmath>=1.3,<2\n",
        )
        with self.assertRaisesRegex(
            packager.ReproducibilityError, "must be an exact pin"
        ):
            packager._validate_pip_requirements(
                self.deployment.root,
                {
                    "pip": "26.1.2",
                    "mpmath": "1.3.0",
                    "python-flint": "0.8.0",
                    "threadpoolctl": "3.6.0",
                },
            )

    def test_stale_source_plan_is_refreshed_and_bundle_is_deterministic(self) -> None:
        first = self.build()
        archive_path = Path(str(first["archive_path"]))
        summary = verifier.verify_bundle(
            archive_path,
            compare_replay_root=self.deployment.root,
        )
        self.assertEqual(summary["verification_status"], "verified")
        self.assertEqual(
            summary["certificate_schema"], "blaschke-deformation-24-contour-hybrid-v3"
        )
        self.assertEqual(summary["notebook_cell_count"], 140)
        self.assertEqual(summary["executed_code_cell_count"], 68)

        with tarfile.open(archive_path, "r:gz") as archive:
            root = packager.BUNDLE_ROOT
            source_plan_stream = archive.extractfile(
                f"{root}/{packager.PLAN_RELATIVE.as_posix()}"
            )
            effective_plan_stream = archive.extractfile(
                f"{root}/{packager.EFFECTIVE_PLAN_NAME}"
            )
            readme_stream = archive.extractfile(f"{root}/README.md")
            self.assertIsNotNone(source_plan_stream)
            self.assertIsNotNone(effective_plan_stream)
            self.assertIsNotNone(readme_stream)
            source_plan = json.loads(source_plan_stream.read().decode("utf-8"))
            effective_plan = json.loads(effective_plan_stream.read().decode("utf-8"))
            self.assertEqual(
                source_plan["precision_settings"]["contour_certificate_schema"],
                "blaschke-deformation-24-contour-hybrid-v2",
            )
            self.assertEqual(
                effective_plan["precision_settings"]["contour_certificate_schema"],
                "blaschke-deformation-24-contour-hybrid-v3",
            )
            self.assertEqual(
                readme_stream.read(),
                (self.deployment.root / "README.md").read_bytes(),
            )
            self.assertNotIn(
                f"{root}/DEPLOYMENT_README.md",
                archive.getnames(),
            )

        first_hash = str(first["archive_sha256"])
        second = self.build()
        self.assertEqual(second["archive_sha256"], first_hash)

    def test_semantic_executed_replay_accepts_only_declared_variation(self) -> None:
        result = self.build()
        archive_path = Path(str(result["archive_path"]))

        notebook = json.loads(self.deployment.notebook_path.read_text(encoding="utf-8"))
        first_code_cell = next(
            cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
        )
        first_code_cell["outputs"] = [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": ["replayed output\n"],
            }
        ]
        self.deployment._write_json(self.deployment.notebook_path, notebook)

        report = json.loads(self.deployment.report_path.read_text(encoding="utf-8"))
        report["elapsed_seconds"] = 999.0
        report["artifacts"] = {"report": "/different/replay/root/report.json"}
        report["binary64_candidate_runtime"] = {"platform": "replay-host"}
        for ordinal, path in enumerate(
            self.deployment.geometry_input_paths.values(), start=1
        ):
            path.write_bytes(f"replayed-geometry-input-{ordinal}\n".encode("ascii"))
        report["geometry_input_hashes"] = {
            key: _sha256(path)
            for key, path in self.deployment.geometry_input_paths.items()
        }
        report["input_hashes"] = {
            key: _sha256(path)
            for key, path in self.deployment.theorem_input_paths.items()
        }
        self.deployment._write_json(self.deployment.report_path, report)

        _, contour_rows = self.deployment._read_csv(self.deployment.contour_path)
        for ordinal, row in enumerate(contour_rows, start=1):
            row["inverse_sampling_seconds"] = str(1000 + ordinal)
            row["elapsed_seconds"] = str(2000 + ordinal)
        self.deployment._write_csv(self.deployment.contour_path, contour_rows)

        historical_report = json.loads(
            self.deployment.historical_report_path.read_text(encoding="utf-8")
        )
        historical_report["process_workers"] = 8
        historical_report["producer_sources"] = {
            "historical_comparisons": {
                "path": "/temporary/replay/historical.py",
                "sha256": "b" * 64,
            }
        }
        self.deployment._write_json(
            self.deployment.historical_report_path, historical_report
        )

        phase4_report = json.loads(
            self.deployment.phase4_report_path.read_text(encoding="utf-8")
        )
        phase4_report["process_execution"]["assembly_workers"] = 8
        phase4_report["process_execution"]["assembly_row_block_size"] = 75
        phase4_report["process_execution"]["surface_workers"] = 4
        phase4_report["cache_validation"][
            "contour_cache_path"
        ] = "/temporary/replay/cache/contour_samples.npz"
        for key in phase4_report["cache_validation"]["cache_hits"]:
            phase4_report["cache_validation"]["cache_hits"][key] = True
        for surface in phase4_report["surfaces"].values():
            surface["cache_hit"] = True
        for key in verifier.HISTORICAL_PHASE4_NPZ_MEMBER_NAMES:
            path = self.deployment.phase4_data_paths[key]
            members = _read_npz_members(path)
            for member_name in verifier.HISTORICAL_PHASE4_VOLATILE_NPZ_MEMBERS.get(
                key, frozenset()
            ):
                members[member_name] = _npy_bytes(
                    f"/temporary/replay/cache/{key}/{member_name}".encode("ascii")
                )
            path.write_bytes(_npz_bytes(members, timestamp=(2026, 8, 28, 2, 4, 6)))
            phase4_report["outputs"][key]["sha256"] = _sha256(path)
        self.deployment._write_json(self.deployment.phase4_report_path, phase4_report)

        audit_report = json.loads(
            self.deployment.audit_report_path.read_text(encoding="utf-8")
        )
        audit_report["source_extraction"]["notebook"][
            "path"
        ] = "/temporary/replay/Numerics/blaschke_deformation_certifier.ipynb"
        audit_report["source_extraction"]["producer"][
            "path"
        ] = "/temporary/replay/Numerics/blaschke_deformation_diagnostic_audits.py"
        for name, source in audit_report["inputs"].items():
            source["path"] = f"/temporary/replay/inputs/{name}.csv"
        for name, output in audit_report["outputs"].items():
            output["path"] = f"/temporary/replay/data/{name}"
        self.deployment._write_json(self.deployment.audit_report_path, audit_report)

        plot_groups = (
            (verifier.HISTORICAL_PLOT_RELATIVES, 1),
            (verifier.HISTORICAL_PHASE4_PLOT_RELATIVES, 20),
            (verifier.DIAGNOSTIC_AUDIT_PLOT_RELATIVES, 40),
        )
        for relatives, start in plot_groups:
            for ordinal, relative in enumerate(relatives, start=start):
                self.deployment.root.joinpath(*relative.parts).write_bytes(
                    _png_bytes(
                        rgba=bytes((ordinal, ordinal + 1, ordinal + 2, 255)),
                        note=f"replay-{ordinal}",
                    )
                )

        summary = verifier.verify_bundle(
            archive_path,
            compare_executed_replay_root=self.deployment.root,
        )
        semantic = summary["semantic_replay_summary"]
        self.assertIsInstance(semantic, dict)
        self.assertEqual(semantic["status"], "semantic-replay-match")
        self.assertEqual(semantic["contour_certificate_row_count"], 24)
        self.assertEqual(semantic["phase2_certified_row_count"], 1)
        self.assertEqual(semantic["historical_comparison"]["plot_count"], 3)
        self.assertEqual(semantic["historical_phase4"]["dataset_count"], 20)
        self.assertEqual(semantic["historical_phase4"]["csv_count"], 13)
        self.assertEqual(semantic["historical_phase4"]["npz_count"], 6)
        self.assertEqual(semantic["historical_phase4"]["plot_count"], 12)
        self.assertEqual(semantic["diagnostic_audits"]["csv_count"], 2)
        self.assertEqual(
            semantic["diagnostic_audits"]["csv_row_counts"],
            {"universal_audit": 7, "first14_audit": 14},
        )
        self.assertEqual(semantic["diagnostic_audits"]["plot_count"], 2)
        self.assertEqual(len(semantic["png_critical_chunk_visual_digests"]), 17)
        self.assertTrue(summary["executed_replay_root_compared"])
        self.assertFalse(summary["replay_root_compared"])

        with self.assertRaisesRegex(
            verifier.VerificationError, "Replay-root hash mismatch"
        ):
            verifier.verify_bundle(
                archive_path,
                compare_replay_root=self.deployment.root,
            )

    def test_semantic_executed_replay_rejects_tampered_evidence(self) -> None:
        result = self.build()
        archive_path = Path(str(result["archive_path"]))

        def expect_failure(
            path: Path,
            mutate,
            pattern: str,
        ) -> None:
            original = path.read_bytes()
            try:
                mutate(path)
                with self.assertRaisesRegex(verifier.VerificationError, pattern):
                    verifier.verify_bundle(
                        archive_path,
                        compare_executed_replay_root=self.deployment.root,
                    )
            finally:
                path.write_bytes(original)

        with self.subTest("immutable source"):
            expect_failure(
                self.deployment.root / "README.md",
                lambda path: path.write_text("tampered source\n", encoding="utf-8"),
                "immutable source differs byte-for-byte",
            )

        with self.subTest("inline helper sync"):

            def tamper_inline_sync(path: Path) -> None:
                notebook = json.loads(path.read_text(encoding="utf-8"))
                notebook["metadata"]["inline_sync"] = False
                self.deployment._write_json(path, notebook)

            expect_failure(
                self.deployment.notebook_path,
                tamper_inline_sync,
                "inline-helper validation failed",
            )

        with self.subTest("notebook execution counts"):

            def tamper_execution_count(path: Path) -> None:
                notebook = json.loads(path.read_text(encoding="utf-8"))
                first_code_cell = next(
                    cell
                    for cell in notebook["cells"]
                    if cell.get("cell_type") == "code"
                )
                first_code_cell["execution_count"] = 57
                self.deployment._write_json(path, notebook)

            expect_failure(
                self.deployment.notebook_path,
                tamper_execution_count,
                "execution counts are not contiguous",
            )

        with self.subTest("theorem gate"):

            def tamper_theorem_gate(path: Path) -> None:
                report = json.loads(path.read_text(encoding="utf-8"))
                report["all_small_gain_tests_pass"] = False
                self.deployment._write_json(path, report)

            expect_failure(
                self.deployment.report_path,
                tamper_theorem_gate,
                "Expected all_small_gain_tests_pass=True",
            )

        with self.subTest("contour stable field"):

            def tamper_contour(path: Path) -> None:
                _, records = self.deployment._read_csv(path)
                records[0]["theorem_certified"] = "False"
                self.deployment._write_csv(path, records)

            expect_failure(
                self.deployment.contour_path,
                tamper_contour,
                "contour certificate CSV stable fields differ",
            )

        with self.subTest("Phase 2 row"):
            expect_failure(
                self.deployment.phase2_row_path,
                lambda path: path.write_bytes(path.read_bytes() + b"tamper\n"),
                "Phase 2 certified row differs byte-for-byte",
            )

        with self.subTest("historical seed dependency"):

            def tamper_history(path: Path) -> None:
                report = json.loads(path.read_text(encoding="utf-8"))
                report["legacy_seed_dependency"] = True
                self.deployment._write_json(path, report)

            expect_failure(
                self.deployment.historical_report_path,
                tamper_history,
                "Expected legacy_seed_dependency=False",
            )

        with self.subTest("historical CSV"):
            expect_failure(
                self.deployment.historical_effect_path,
                lambda path: path.write_bytes(path.read_bytes() + b"tamper\n"),
                "historical effect CSV differs byte-for-byte",
            )

        with self.subTest("historical plot pixels"):
            plot_path = self.deployment.root.joinpath(
                *verifier.HISTORICAL_PLOT_RELATIVES[0].parts
            )
            expect_failure(
                plot_path,
                lambda path: path.write_bytes(
                    _png_bytes(rgba=b"\xff\x00\x00\xff", note="same-metadata")
                ),
                "plot visual digest differs",
            )

        with self.subTest("historical Phase 4 deterministic CSV"):
            expect_failure(
                self.deployment.phase4_data_paths["profiles"],
                lambda path: path.write_bytes(path.read_bytes() + b"tamper\n"),
                "historical Phase 4 deterministic artifact differs: profiles",
            )

        with self.subTest("historical Phase 4 stable NPZ member"):

            def tamper_phase4_npz(path: Path) -> None:
                members = _read_npz_members(path)
                members["B_scaled.npy"] = _npy_bytes(b"tampered-stable-array")
                path.write_bytes(_npz_bytes(members, timestamp=(2026, 8, 28, 4, 6, 8)))

            expect_failure(
                self.deployment.phase4_data_paths["scaled_block"],
                tamper_phase4_npz,
                "historical Phase 4 NPZ stable member differs",
            )

        with self.subTest("historical Phase 4 seed dependency"):

            def tamper_phase4_report(path: Path) -> None:
                report = json.loads(path.read_text(encoding="utf-8"))
                report["legacy_seed_dependency"] = True
                self.deployment._write_json(path, report)

            expect_failure(
                self.deployment.phase4_report_path,
                tamper_phase4_report,
                "Expected legacy_seed_dependency=False",
            )

        with self.subTest("historical Phase 4 plot pixels"):
            plot_path = self.deployment.root.joinpath(
                *verifier.HISTORICAL_PHASE4_PLOT_RELATIVES[0].parts
            )
            expect_failure(
                plot_path,
                lambda path: path.write_bytes(
                    _png_bytes(rgba=b"\xff\x00\x00\xff", note="same-metadata")
                ),
                "historical Phase 4 plot visual digest differs",
            )

        with self.subTest("diagnostic audit CSV"):
            expect_failure(
                self.deployment.audit_data_paths["first14_audit"],
                lambda path: path.write_bytes(path.read_bytes() + b"tamper\n"),
                "diagnostic audit deterministic CSV differs: first14_audit",
            )

        with self.subTest("diagnostic audit report status"):

            def tamper_audit_report(path: Path) -> None:
                report = json.loads(path.read_text(encoding="utf-8"))
                report["diagnostic_only"] = False
                self.deployment._write_json(path, report)

            expect_failure(
                self.deployment.audit_report_path,
                tamper_audit_report,
                "Expected diagnostic_only=True",
            )

        with self.subTest("diagnostic audit plot pixels"):
            plot_path = self.deployment.root.joinpath(
                *verifier.DIAGNOSTIC_AUDIT_PLOT_RELATIVES[0].parts
            )
            expect_failure(
                plot_path,
                lambda path: path.write_bytes(
                    _png_bytes(rgba=b"\xff\x00\x00\xff", note="same-metadata")
                ),
                "diagnostic audit plot visual digest differs",
            )

    def test_semantic_replay_rejects_stale_geometry_input_hash(self) -> None:
        result = self.build()
        archive_path = Path(str(result["archive_path"]))
        self.deployment.geometry_input_paths["matrix_midpoint"].write_bytes(
            b"tampered-matrix-container\n"
        )
        with self.assertRaisesRegex(
            verifier.VerificationError,
            "geometry input hash mismatch for matrix_midpoint",
        ):
            verifier.verify_bundle(
                archive_path,
                compare_executed_replay_root=self.deployment.root,
            )

    def test_dirty_repository_fails_before_publication(self) -> None:
        (self.deployment.root / "README.md").write_text("dirty\n", encoding="utf-8")
        with self.assertRaisesRegex(
            packager.ReproducibilityError, "requires a clean deployment commit"
        ):
            self.build()
        self.assertFalse(
            (
                self.deployment.output_dir / "reproducibility" / packager.ARCHIVE_NAME
            ).exists()
        )

    def test_subdirectory_is_not_accepted_as_repository_root(self) -> None:
        plan = self.deployment.read_plan()
        with self.package_environment(), self.assertRaisesRegex(
            packager.ReproducibilityError, "exact Git repository root"
        ):
            packager.build_reproducibility_bundle(
                repo_root=self.deployment.root / "Numerics",
                notebook_path=self.deployment.notebook_path,
                output_dir=self.deployment.output_dir,
                precision_settings=plan["precision_settings"],
                upstream_artifact_names=plan["upstream_artifact_names"],
            )

    def test_git_command_failure_is_fatal(self) -> None:
        failure = subprocess.CalledProcessError(
            128,
            ("git", "rev-parse"),
            stderr="fatal: simulated Git failure",
        )
        with mock.patch.object(
            packager.subprocess, "run", side_effect=failure
        ), self.assertRaisesRegex(
            packager.ReproducibilityError, "simulated Git failure"
        ):
            packager._capture_git_snapshot(self.deployment.root)

    def test_clean_commit_with_stale_manifest_fails(self) -> None:
        (self.deployment.root / "README.md").write_text(
            "changed without manifest refresh\n", encoding="utf-8"
        )
        self.deployment.commit("stale manifest")
        with self.assertRaisesRegex(packager.ReproducibilityError, "Checksum mismatch"):
            self.build()

    def test_unexecuted_notebook_cell_fails(self) -> None:
        notebook = json.loads(self.deployment.notebook_path.read_text(encoding="utf-8"))
        first_code_cell = next(
            cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
        )
        first_code_cell["execution_count"] = None
        self.deployment._write_json(self.deployment.notebook_path, notebook)
        self.deployment.refresh_and_commit("unexecuted notebook")
        with self.assertRaisesRegex(
            packager.ReproducibilityError, "Code cell 1 is not executed"
        ):
            self.build()

    def test_failed_theorem_gate_cannot_be_refreshed_into_success(self) -> None:
        report = json.loads(self.deployment.report_path.read_text(encoding="utf-8"))
        report["all_small_gain_tests_pass"] = False
        self.deployment._write_json(self.deployment.report_path, report)
        self.deployment.refresh_and_commit("failed theorem gate")
        with self.assertRaisesRegex(
            packager.ReproducibilityError,
            "Expected all_small_gain_tests_pass=True",
        ):
            self.build()

    def test_upstream_artifact_traversal_is_rejected(self) -> None:
        plan = self.deployment.read_plan()
        plan["upstream_artifact_names"] = ["../seed.csv"]
        self.deployment._write_json(self.deployment.plan_path, plan)
        self.deployment.refresh_and_commit("unsafe artifact path")
        with self.assertRaisesRegex(
            packager.ReproducibilityError, "Unsafe upstream artifact data path"
        ):
            self.build()

    def test_missing_source_rebuild_report_is_rejected(self) -> None:
        self.deployment.phase4_report_path.unlink()
        self.deployment.refresh_and_commit("missing source rebuild report")
        with self.assertRaisesRegex(
            packager.ReproducibilityError,
            "Authoritative path is not a regular file",
        ):
            self.build()

    def test_retained_legacy_seed_dependency_is_rejected(self) -> None:
        report = json.loads(
            self.deployment.phase4_report_path.read_text(encoding="utf-8")
        )
        report["legacy_seed_dependency"] = True
        self.deployment._write_json(self.deployment.phase4_report_path, report)
        self.deployment.refresh_and_commit("retained legacy seed dependency")
        with self.assertRaisesRegex(
            packager.ReproducibilityError,
            "does not exclude retained legacy seed inputs",
        ):
            self.build()

    def test_source_generated_output_tamper_is_rejected(self) -> None:
        output_path = self.deployment.phase4_data_paths["profiles"]
        output_path.write_bytes(output_path.read_bytes() + b"tamper\n")
        self.deployment.refresh_and_commit("tampered source-generated output")
        with self.assertRaisesRegex(
            packager.ReproducibilityError,
            "historical Phase 4 output digest mismatch",
        ):
            self.build()

    def test_verifier_detects_archive_tamper_and_replay_mismatch(self) -> None:
        result = self.build()
        archive_path = Path(str(result["archive_path"]))
        original = archive_path.read_bytes()
        archive_path.write_bytes(original + b"tamper")
        with self.assertRaisesRegex(
            verifier.VerificationError, "archive hash mismatch"
        ):
            verifier.verify_bundle(archive_path)
        archive_path.write_bytes(original)

        (self.deployment.root / "README.md").write_text(
            "different replay\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            verifier.VerificationError, "Replay-root hash mismatch"
        ):
            verifier.verify_bundle(
                archive_path,
                compare_replay_root=self.deployment.root,
            )

    def test_verifier_rejects_unsafe_tar_member(self) -> None:
        result = self.build()
        valid_external = Path(str(result["manifest_path"]))
        external = json.loads(valid_external.read_text(encoding="utf-8"))
        epoch = external["bundle"]["repository"]["commit_epoch"]
        unsafe_dir = Path(self.temporary.name) / "unsafe-publication"
        unsafe_dir.mkdir()
        unsafe_archive = unsafe_dir / verifier.ARCHIVE_NAME
        with unsafe_archive.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w") as archive:
                    root_info = tarfile.TarInfo(verifier.BUNDLE_ROOT)
                    root_info.type = tarfile.DIRTYPE
                    root_info.mode = 0o755
                    root_info.mtime = epoch
                    archive.addfile(root_info)
                    unsafe_info = tarfile.TarInfo(
                        f"{verifier.BUNDLE_ROOT}/../escape.txt"
                    )
                    unsafe_info.mode = 0o644
                    unsafe_info.mtime = epoch
                    unsafe_info.size = 1
                    import io

                    archive.addfile(unsafe_info, io.BytesIO(b"x"))
        unsafe_hash = _sha256(unsafe_archive)
        external["archive"]["sha256"] = unsafe_hash
        external["archive"]["bytes"] = unsafe_archive.stat().st_size
        unsafe_external = unsafe_dir / verifier.EXTERNAL_MANIFEST_NAME
        unsafe_external.write_text(
            json.dumps(external, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        unsafe_checksum = unsafe_dir / verifier.CHECKSUM_NAME
        unsafe_checksum.write_text(
            f"{unsafe_hash}  {verifier.ARCHIVE_NAME}\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            verifier.VerificationError, "Unsafe archive member path"
        ):
            verifier.verify_bundle(unsafe_archive)

    def test_publication_rolls_back_when_post_publish_git_check_fails(self) -> None:
        result = self.build()
        paths = (
            Path(str(result["archive_path"])),
            Path(str(result["checksum_path"])),
            Path(str(result["manifest_path"])),
        )
        before = {path: path.read_bytes() for path in paths}
        checks = 0

        def fail_second_check(*_args, **_kwargs):
            nonlocal checks
            checks += 1
            if checks == 2:
                raise packager.ReproducibilityError("simulated concurrent Git change")

        with self.package_environment(), mock.patch.object(
            packager,
            "_assert_same_clean_snapshot",
            side_effect=fail_second_check,
        ), self.assertRaisesRegex(
            packager.ReproducibilityError, "simulated concurrent Git change"
        ):
            plan = self.deployment.read_plan()
            packager.build_reproducibility_bundle(
                repo_root=self.deployment.root,
                notebook_path=self.deployment.notebook_path,
                output_dir=self.deployment.output_dir,
                precision_settings=plan["precision_settings"],
                upstream_artifact_names=plan["upstream_artifact_names"],
            )
        self.assertEqual({path: path.read_bytes() for path in paths}, before)


if __name__ == "__main__":
    unittest.main()
