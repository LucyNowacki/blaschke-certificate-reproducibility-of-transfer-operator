"""Verify Phase 1 and sampled-Schur data in a Blaschke replay bundle.

This entry point extends :mod:`verify_blaschke_reproducibility` without
duplicating its archive-safety and theorem-certificate checks.  The deployment
packager remains the source of truth for notebook dimensions and authoritative
paths; this module adds fail-closed validation of the two source-generated
diagnostic producer reports and their CSV/Parquet outputs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Callable, Iterator, Mapping

import blaschke_deformation_reproducibility as source_plan
import verify_blaschke_reproducibility as _base


VerificationError = _base.VerificationError

OUTPUT_RELATIVE = source_plan.OUTPUT_RELATIVE
PLAN_RELATIVE = source_plan.PLAN_RELATIVE
SOURCE_CONTROLLED_PLAN_RELATIVE = source_plan.SOURCE_CONTROLLED_PLAN_RELATIVE
EXPECTED_MAP_LABEL = source_plan.EXPECTED_MAP_LABEL
EXPECTED_NOTEBOOK_CELLS = source_plan.EXPECTED_NOTEBOOK_CELLS
EXPECTED_NOTEBOOK_CODE_CELLS = source_plan.EXPECTED_NOTEBOOK_CODE_CELLS

PHASE1_REPORT_RELATIVE = source_plan.PHASE1_DIAGNOSTIC_REPORT_RELATIVE
SAMPLED_SCHUR_REPORT_RELATIVE = source_plan.SAMPLED_SCHUR_DIAGNOSTIC_REPORT_RELATIVE
SAMPLED_SCHUR_DATA_RELATIVE = OUTPUT_RELATIVE / (
    f"data/{source_plan.EXPECTED_SAMPLED_SCHUR_FILENAME}"
)

EXPECTED_PHASE1_REPORT_FIELDS = frozenset(
    {
        "producer_schema",
        "schema_version",
        "generation_id",
        "map_label",
        "diagnostic_only",
        "theorem_gate",
        "legacy_seed_dependency",
        "commit_protocol",
        "config",
        "lineage",
        "jobs",
        "schedules",
        "outputs",
    }
)
EXPECTED_PHASE1_CONFIG_FIELDS = frozenset(
    {
        "cloud_pairs",
        "fixed_n",
        "fixed_m_values",
        "heat_n_values",
        "heat_m_values",
        "dps",
        "max_power",
        "max_clusters",
        "expected_target_count",
        "assembly_workers",
        "row_block_size",
        "progress",
    }
)
EXPECTED_PHASE1_JOB_FIELDS = frozenset(
    {
        "N",
        "M",
        "ok",
        "assembly_mode",
        "assembly_workers",
        "row_block_size",
        "gram_error",
    }
)
EXPECTED_PHASE1_OUTPUT_FIELDS = frozenset(
    {
        "path",
        "rows",
        "columns",
        "sha256",
        "parquet_path",
        "parquet_sha256",
    }
)
EXPECTED_SAMPLED_SCHUR_REPORT_FIELDS = frozenset(
    {
        "producer_schema",
        "schema_version",
        "map_label",
        "diagnostic_only",
        "theorem_gate",
        "legacy_seed_dependency",
        "commit_protocol",
        "config",
        "geometry",
        "lineage",
        "output",
    }
)
EXPECTED_PHASE1_CLOUD_PAIRS = source_plan.EXPECTED_PHASE1_CLOUD_PAIRS
EXPECTED_PHASE1_OUTPUT_NAMES = source_plan.EXPECTED_PHASE1_OUTPUT_NAMES
EXPECTED_SAMPLED_SCHUR_N = source_plan.EXPECTED_SAMPLED_SCHUR_N
EXPECTED_SAMPLED_SCHUR_OVERSAMPLING = source_plan.EXPECTED_SAMPLED_SCHUR_OVERSAMPLING
EXPECTED_SAMPLED_SCHUR_COLUMNS = source_plan.EXPECTED_SAMPLED_SCHUR_COLUMNS
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def __getattr__(name: str) -> object:
    """Retain the public surface of the underlying archive verifier."""

    return getattr(_base, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_base)))


def _expect_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise VerificationError(f"{label} is not a lowercase SHA-256 digest.")
    return value


def _expect_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 1:
        raise VerificationError(f"{label} must be a positive integer.")
    return value


def _expect_path_name(value: object, relative: PurePosixPath, *, label: str) -> None:
    if not isinstance(value, str) or PurePosixPath(value).name != relative.name:
        raise VerificationError(
            f"{label} must name the source-controlled output {relative.name}."
        )


def _normalised_pairs(value: object, *, label: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or not value:
        raise VerificationError(f"{label} must be a non-empty pair array.")
    pairs: list[tuple[int, int]] = []
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(type(entry) is not int or entry < 1 for entry in item)
        ):
            raise VerificationError(f"{label} contains an invalid dimension pair.")
        pairs.append((item[0], item[1]))
    if len(set(pairs)) != len(pairs):
        raise VerificationError(f"{label} contains duplicate dimension pairs.")
    return tuple(sorted(pairs))


def _positive_int_list(value: object, *, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) is not int or item < 1 for item in value)
        or len(set(value)) != len(value)
    ):
        raise VerificationError(f"{label} must contain distinct positive integers.")
    return tuple(value)


def _validate_callable_record(value: object, *, label: str) -> None:
    if not isinstance(value, dict):
        raise VerificationError(f"{label} must be a JSON object.")
    required = {"module", "qualname", "module_path", "source_scope", "source_sha256"}
    if not required.issubset(value):
        raise VerificationError(f"{label} lacks callable-source provenance.")
    module = value.get("module")
    qualname = value.get("qualname")
    if module is not None and (not isinstance(module, str) or not module):
        raise VerificationError(f"{label}.module is invalid.")
    if qualname is not None and (not isinstance(qualname, str) or not qualname):
        raise VerificationError(f"{label}.qualname is invalid.")
    if module is None and qualname is None:
        raise VerificationError(f"{label} has no callable identifier.")
    module_path = value.get("module_path")
    if module_path is not None and (
        not isinstance(module_path, str) or not PurePosixPath(module_path).name
    ):
        raise VerificationError(f"{label}.module_path is invalid.")
    if not isinstance(value.get("source_scope"), str) or not value["source_scope"]:
        raise VerificationError(f"{label}.source_scope must be non-empty.")
    _expect_sha256(value.get("source_sha256"), label=f"{label}.source_sha256")


def _stable_report(value: object) -> object:
    """Remove only fields that necessarily depend on the extraction root."""

    if isinstance(value, dict):
        return {
            key: _stable_report(item)
            for key, item in value.items()
            if key not in {"generation_id", "module_path", "path", "parquet_path"}
        }
    if isinstance(value, list):
        return [_stable_report(item) for item in value]
    return value


def _validate_notebook(notebook: Mapping[str, object]) -> dict[str, object]:
    """Validate the current 141-cell/68-code-cell notebook contract."""

    cells = notebook.get("cells")
    if not isinstance(cells, list) or len(cells) != EXPECTED_NOTEBOOK_CELLS:
        raise VerificationError(
            f"Executed notebook does not have {EXPECTED_NOTEBOOK_CELLS} cells."
        )
    code_cells = [
        cell
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "code"
    ]
    if len(code_cells) != EXPECTED_NOTEBOOK_CODE_CELLS:
        raise VerificationError(
            "Executed notebook does not have exactly "
            f"{EXPECTED_NOTEBOOK_CODE_CELLS} code cells."
        )
    counts: list[int] = []
    errors = 0
    for cell in code_cells:
        count = cell.get("execution_count")
        if type(count) is not int:
            raise VerificationError("The archive contains an unexecuted code cell.")
        counts.append(count)
        outputs = cell.get("outputs")
        if not isinstance(outputs, list):
            raise VerificationError("A code cell has a non-array outputs field.")
        errors += sum(
            isinstance(output, dict) and output.get("output_type") == "error"
            for output in outputs
        )
    expected_counts = list(range(1, EXPECTED_NOTEBOOK_CODE_CELLS + 1))
    if counts != expected_counts:
        raise VerificationError(
            "Notebook execution counts are not contiguous "
            f"1-{EXPECTED_NOTEBOOK_CODE_CELLS}."
        )
    if errors:
        raise VerificationError("The executed notebook contains error outputs.")
    return {
        "cell_count": len(cells),
        "code_cell_count": len(code_cells),
        "executed_code_cell_count": len(code_cells),
        "stored_error_output_count": errors,
    }


PayloadReader = Callable[[PurePosixPath, str], bytes]


def _validate_csv_record(
    record: Mapping[str, object],
    *,
    relative: PurePosixPath,
    payload: bytes,
    label: str,
) -> tuple[tuple[str, ...], int]:
    _expect_path_name(record.get("path"), relative, label=f"{label}.path")
    fields, rows = _base._csv_table(payload, label=label)
    if record.get("columns") != list(fields):
        raise VerificationError(f"{label} report columns differ from its CSV header.")
    if record.get("rows") != len(rows):
        raise VerificationError(f"{label} report row count differs from its CSV.")
    if record.get("sha256") != _base._sha256_bytes(payload):
        raise VerificationError(f"{label} report digest differs from its CSV.")
    if not rows:
        raise VerificationError(f"{label} CSV is empty.")
    return fields, len(rows)


def _phase1_output_names(config: Mapping[str, object]) -> tuple[str, ...]:
    cloud_pairs = _normalised_pairs(
        config.get("cloud_pairs"), label="Phase 1 cloud_pairs"
    )
    fixed_n = _expect_positive_int(config.get("fixed_n"), label="Phase 1 fixed_n")
    names: list[str] = []
    for n_value, m_value in cloud_pairs:
        names.extend(
            (
                f"phase1_eigencloud_match_N{n_value}_M{m_value}",
                f"phase1_eigencloud_eigenvalues_N{n_value}_M{m_value}",
            )
        )
    names.extend(
        (
            f"phase1_raw_error_vs_M_N{fixed_n}",
            "phase1_raw_NM_heatmap_data_near_square",
        )
    )
    if set(names) != EXPECTED_PHASE1_OUTPUT_NAMES:
        raise VerificationError(
            "Phase 1 output schedule differs from the deployment plan."
        )
    return tuple(names)


def _validate_phase1_report(
    report: Mapping[str, object], *, read_payload: PayloadReader, label: str
) -> tuple[dict[str, object], dict[str, bytes], dict[str, bytes], dict[str, int]]:
    if set(report) != EXPECTED_PHASE1_REPORT_FIELDS:
        raise VerificationError(f"Unexpected {label} schema.")
    if report.get("producer_schema") != source_plan.EXPECTED_PHASE1_PRODUCER_SCHEMA:
        raise VerificationError(f"Unexpected {label} producer_schema.")
    if report.get("schema_version") != "2.0.0":
        raise VerificationError(f"Unexpected {label} schema_version.")
    if report.get("map_label") != EXPECTED_MAP_LABEL:
        raise VerificationError(f"Unexpected {label} map_label.")
    _base._expect_bool(report, "diagnostic_only", True)
    _base._expect_bool(report, "theorem_gate", False)
    _base._expect_bool(report, "legacy_seed_dependency", False)
    _expect_sha256(report.get("generation_id"), label=f"{label}.generation_id")

    config = report.get("config")
    if not isinstance(config, dict) or set(config) != EXPECTED_PHASE1_CONFIG_FIELDS:
        raise VerificationError(f"Unexpected {label} config schema.")
    cloud_pairs = _normalised_pairs(config["cloud_pairs"], label=f"{label}.cloud_pairs")
    fixed_n = _expect_positive_int(config["fixed_n"], label=f"{label}.fixed_n")
    if cloud_pairs != EXPECTED_PHASE1_CLOUD_PAIRS or fixed_n != 25:
        raise VerificationError(f"{label} schedule differs from the deployment plan.")
    fixed_m = _positive_int_list(
        config["fixed_m_values"], label=f"{label}.fixed_m_values"
    )
    heat_n = _positive_int_list(config["heat_n_values"], label=f"{label}.heat_n_values")
    heat_m = _positive_int_list(config["heat_m_values"], label=f"{label}.heat_m_values")
    _expect_positive_int(config["dps"], label=f"{label}.dps")
    if config["dps"] < 50:
        raise VerificationError(f"{label}.dps is below the producer minimum.")
    _expect_positive_int(config["max_power"], label=f"{label}.max_power")
    _expect_positive_int(config["max_clusters"], label=f"{label}.max_clusters")
    if config["expected_target_count"] != source_plan.EXPECTED_TARGET_COUNT:
        raise VerificationError(f"Unexpected {label} expected_target_count.")
    workers = _expect_positive_int(
        config["assembly_workers"], label=f"{label}.assembly_workers"
    )
    if workers > 24:
        raise VerificationError(f"{label}.assembly_workers exceeds 24.")
    if config["row_block_size"] is not None:
        _expect_positive_int(config["row_block_size"], label=f"{label}.row_block_size")
    if type(config["progress"]) is not bool:
        raise VerificationError(f"{label}.progress must be boolean.")

    fixed_pairs = tuple(sorted((fixed_n, m_value) for m_value in fixed_m))
    heat_pairs = tuple(
        sorted(
            (n_value, m_value)
            for n_value in heat_n
            for m_value in heat_m
            if m_value >= n_value
        )
    )
    unique_pairs = tuple(sorted(set(cloud_pairs + fixed_pairs + heat_pairs)))
    schedules = report.get("schedules")
    if not isinstance(schedules, dict) or set(schedules) != {
        "cloud_pairs",
        "fixed_pairs",
        "heat_pairs",
        "unique_pairs",
    }:
        raise VerificationError(f"Unexpected {label} schedule schema.")
    expected_schedules = {
        "cloud_pairs": cloud_pairs,
        "fixed_pairs": fixed_pairs,
        "heat_pairs": heat_pairs,
        "unique_pairs": unique_pairs,
    }
    for key, expected in expected_schedules.items():
        if _normalised_pairs(schedules[key], label=f"{label}.{key}") != expected:
            raise VerificationError(f"{label} schedule differs for {key}.")

    jobs = report.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != len(unique_pairs):
        raise VerificationError(f"{label} jobs do not cover the locked pair schedule.")
    observed_job_pairs: list[tuple[int, int]] = []
    expected_mode = "process_row_blocks" if workers > 1 else "serial"
    for job in jobs:
        if not isinstance(job, dict) or set(job) != EXPECTED_PHASE1_JOB_FIELDS:
            raise VerificationError(f"Unexpected {label} job schema.")
        n_value = _expect_positive_int(job["N"], label=f"{label} job N")
        m_value = _expect_positive_int(job["M"], label=f"{label} job M")
        observed_job_pairs.append((n_value, m_value))
        if (
            job["ok"] is not True
            or job["assembly_mode"] != expected_mode
            or job["assembly_workers"] != workers
            or job["row_block_size"] != config["row_block_size"]
            or job["gram_error"] is None
        ):
            raise VerificationError(f"{label} contains inconsistent job metadata.")
    if tuple(sorted(observed_job_pairs)) != unique_pairs:
        raise VerificationError(f"{label} job pairs differ from the locked schedule.")

    lineage = report.get("lineage")
    if not isinstance(lineage, dict) or set(lineage) != {
        "map_spec_sha256",
        "reference_clusters_sha256",
        "expected_target_names_sha256",
        "raw_sweep",
        "runtime",
    }:
        raise VerificationError(f"Unexpected {label} lineage schema.")
    for key in ("map_spec_sha256", "expected_target_names_sha256"):
        _expect_sha256(lineage[key], label=f"{label}.{key}")
    reference_digest = lineage["reference_clusters_sha256"]
    if reference_digest is not None:
        _expect_sha256(reference_digest, label=f"{label}.reference_clusters_sha256")
    _validate_callable_record(lineage["raw_sweep"], label=f"{label}.raw_sweep")
    runtime = lineage["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {"python", "numpy", "pandas"}:
        raise VerificationError(f"Unexpected {label} runtime schema.")

    outputs = report.get("outputs")
    expected_names = _phase1_output_names(config)
    if not isinstance(outputs, dict) or set(outputs) != set(expected_names):
        raise VerificationError(f"Unexpected {label} output set.")
    csv_payloads: dict[str, bytes] = {}
    parquet_payloads: dict[str, bytes] = {}
    row_counts: dict[str, int] = {}
    for name in expected_names:
        record = outputs[name]
        if not isinstance(record, dict) or set(record) != EXPECTED_PHASE1_OUTPUT_FIELDS:
            raise VerificationError(f"Unexpected {label} output record for {name}.")
        csv_relative = OUTPUT_RELATIVE / f"data/{name}.csv"
        csv_payload = read_payload(csv_relative, f"{label} {name} CSV")
        _, row_count = _validate_csv_record(
            record,
            relative=csv_relative,
            payload=csv_payload,
            label=f"{label} {name}",
        )
        csv_fields, csv_rows = _base._csv_table(
            csv_payload, label=f"{label} {name} CSV"
        )
        if "map_name" not in csv_fields or {row["map_name"] for row in csv_rows} != {
            EXPECTED_MAP_LABEL
        }:
            raise VerificationError(f"{label} {name} CSV violates the map lock.")
        parquet_relative = OUTPUT_RELATIVE / f"data/{name}.parquet"
        _expect_path_name(
            record.get("parquet_path"),
            parquet_relative,
            label=f"{label} {name}.parquet_path",
        )
        parquet_payload = read_payload(parquet_relative, f"{label} {name} Parquet")
        if record.get("parquet_sha256") != _base._sha256_bytes(parquet_payload):
            raise VerificationError(
                f"{label} {name} report digest differs from its Parquet output."
            )
        csv_payloads[name] = csv_payload
        parquet_payloads[name] = parquet_payload
        row_counts[name] = row_count

    return (
        _stable_report(dict(report)),
        csv_payloads,
        parquet_payloads,
        row_counts,
    )


def _validate_sampled_schur_report(
    report: Mapping[str, object], *, read_payload: PayloadReader, label: str
) -> tuple[dict[str, object], bytes, int]:
    if set(report) != EXPECTED_SAMPLED_SCHUR_REPORT_FIELDS:
        raise VerificationError(f"Unexpected {label} schema.")
    if (
        report.get("producer_schema")
        != source_plan.EXPECTED_SAMPLED_SCHUR_PRODUCER_SCHEMA
    ):
        raise VerificationError(f"Unexpected {label} producer_schema.")
    if report.get("schema_version") != "2.0.0":
        raise VerificationError(f"Unexpected {label} schema_version.")
    if report.get("map_label") != EXPECTED_MAP_LABEL:
        raise VerificationError(f"Unexpected {label} map_label.")
    _base._expect_bool(report, "diagnostic_only", True)
    _base._expect_bool(report, "theorem_gate", False)
    _base._expect_bool(report, "legacy_seed_dependency", False)

    config = report.get("config")
    if not isinstance(config, dict) or set(config) != {"n_values", "oversampling"}:
        raise VerificationError(f"Unexpected {label} config schema.")
    if tuple(config["n_values"]) != EXPECTED_SAMPLED_SCHUR_N:
        raise VerificationError(f"Unexpected {label} N schedule.")
    if config["oversampling"] != EXPECTED_SAMPLED_SCHUR_OVERSAMPLING:
        raise VerificationError(f"Unexpected {label} oversampling.")

    geometry = report.get("geometry")
    if not isinstance(geometry, dict) or set(geometry) != {"rho", "r_tau", "r", "Phi"}:
        raise VerificationError(f"Unexpected {label} geometry schema.")
    for key, value in geometry.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise VerificationError(f"{label}.{key} must be numeric.")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise VerificationError(f"{label}.{key} must be finite and positive.")

    lineage = report.get("lineage")
    if not isinstance(lineage, dict) or set(lineage) != {
        "sampled_schur_envelope",
        "kappa",
        "runtime",
    }:
        raise VerificationError(f"Unexpected {label} lineage schema.")
    _validate_callable_record(
        lineage["sampled_schur_envelope"],
        label=f"{label}.sampled_schur_envelope",
    )
    _validate_callable_record(lineage["kappa"], label=f"{label}.kappa")
    runtime = lineage["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {"python", "pandas"}:
        raise VerificationError(f"Unexpected {label} runtime schema.")

    output = report.get("output")
    if not isinstance(output, dict) or set(output) != {
        "path",
        "rows",
        "columns",
        "sha256",
    }:
        raise VerificationError(f"Unexpected {label} output schema.")
    payload = read_payload(SAMPLED_SCHUR_DATA_RELATIVE, f"{label} CSV")
    fields, row_count = _validate_csv_record(
        output,
        relative=SAMPLED_SCHUR_DATA_RELATIVE,
        payload=payload,
        label=label,
    )
    if not EXPECTED_SAMPLED_SCHUR_COLUMNS.issubset(fields):
        raise VerificationError(f"{label} CSV lacks required diagnostic columns.")
    _, rows = _base._csv_table(payload, label=f"{label} CSV")
    if row_count != len(EXPECTED_SAMPLED_SCHUR_N):
        raise VerificationError(f"{label} CSV must contain six rows.")
    for row, n_value in zip(rows, EXPECTED_SAMPLED_SCHUR_N, strict=True):
        if row["map_label"] != EXPECTED_MAP_LABEL:
            raise VerificationError(f"{label} CSV violates the map lock.")
        try:
            observed_n = int(row["N"])
            observed_m = int(row["M"])
            epsilon = float(row["epsilon_schur_diagnostic"])
            transported = float(row["transported_Bmat_diagnostic"])
        except ValueError as exc:
            raise VerificationError(
                f"{label} CSV contains invalid numeric data."
            ) from exc
        if (
            observed_n != n_value
            or observed_m != n_value + EXPECTED_SAMPLED_SCHUR_OVERSAMPLING
        ):
            raise VerificationError(f"{label} CSV violates the locked M=N+6 schedule.")
        if (
            not math.isfinite(epsilon)
            or not math.isfinite(transported)
            or epsilon < 0.0
            or transported < 0.0
        ):
            raise VerificationError(f"{label} CSV contains invalid diagnostic bounds.")
        if not row["status"].strip() or not row["geometry_status"].strip():
            raise VerificationError(f"{label} CSV contains an empty status.")
    return _stable_report(dict(report)), payload, row_count


def _planned_tabular_relatives(plan: Mapping[str, object]) -> tuple[PurePosixPath, ...]:
    names = plan.get("upstream_artifact_names")
    if not isinstance(names, list):
        raise VerificationError(
            "The source reproducibility plan has no artifact array."
        )
    relatives: list[PurePosixPath] = []
    for value in names:
        relative = _base._safe_relative_path(value, label="source-plan artifact")
        if relative.suffix.lower() in {".csv", ".parquet"}:
            relatives.append(OUTPUT_RELATIVE / "data" / relative)
    return tuple(relatives)


def _source_plan_relative(files: Mapping[str, bytes]) -> PurePosixPath:
    source_path = SOURCE_CONTROLLED_PLAN_RELATIVE.as_posix()
    compatibility_path = PLAN_RELATIVE.as_posix()
    if source_path in files:
        if files.get(compatibility_path) != files[source_path]:
            raise VerificationError(
                "The reports-path plan is not an exact alias of the "
                "source-controlled reproducibility plan."
            )
        return SOURCE_CONTROLLED_PLAN_RELATIVE
    if compatibility_path not in files:
        raise VerificationError("Archive lacks a reproducibility plan.")
    return PLAN_RELATIVE


def _archive_reader(
    files: Mapping[str, bytes], source_entries: Mapping[str, str]
) -> PayloadReader:
    def read(relative: PurePosixPath, label: str) -> bytes:
        payload = _base._archive_payload(files, relative, label=label)
        path_text = relative.as_posix()
        expected = source_entries.get(path_text)
        if expected is None:
            raise VerificationError(
                f"Source-generated diagnostic is absent from MANIFEST.sha256: {path_text}."
            )
        if _base._sha256_bytes(payload) != expected:
            raise VerificationError(f"Source-manifest digest differs for {path_text}.")
        return payload

    return read


def _replay_reader(root: Path) -> PayloadReader:
    def read(relative: PurePosixPath, label: str) -> bytes:
        return _base._read_replay_bytes(root, relative, label=label)

    return read


def _validate_archive_producer_diagnostics(
    *,
    files: Mapping[str, bytes],
    source_entries: Mapping[str, str],
    plan: Mapping[str, object],
) -> dict[str, object]:
    read = _archive_reader(files, source_entries)
    phase1_report = _base._json_object(
        read(PHASE1_REPORT_RELATIVE, "Phase 1 producer report"),
        label="Phase 1 producer report",
    )
    sampled_report = _base._json_object(
        read(SAMPLED_SCHUR_REPORT_RELATIVE, "sampled-Schur producer report"),
        label="sampled-Schur producer report",
    )
    phase1_stable, phase1_csv, phase1_parquet, phase1_rows = _validate_phase1_report(
        phase1_report,
        read_payload=read,
        label="archive Phase 1 producer report",
    )
    sampled_stable, _, sampled_rows = _validate_sampled_schur_report(
        sampled_report,
        read_payload=read,
        label="archive sampled-Schur producer report",
    )
    planned = _planned_tabular_relatives(plan)
    for relative in planned:
        read(relative, f"source-plan tabular artifact {relative.name}")
    return {
        "phase1": {
            "report_path": PHASE1_REPORT_RELATIVE.as_posix(),
            "stable_report_field_count": len(phase1_stable),
            "csv_count": len(phase1_csv),
            "parquet_count": len(phase1_parquet),
            "csv_row_counts": phase1_rows,
        },
        "sampled_schur": {
            "report_path": SAMPLED_SCHUR_REPORT_RELATIVE.as_posix(),
            "csv_path": SAMPLED_SCHUR_DATA_RELATIVE.as_posix(),
            "stable_report_field_count": len(sampled_stable),
            "csv_row_count": sampled_rows,
        },
        "source_plan_tabular_artifact_count": len(planned),
    }


def _compare_payload_sets(
    archive: Mapping[str, bytes], replay: Mapping[str, bytes], *, label: str
) -> None:
    if set(archive) != set(replay):
        raise VerificationError(f"Replay {label} output set differs from the archive.")
    for name in archive:
        if replay[name] != archive[name]:
            raise VerificationError(
                f"Replay {label} deterministic output differs: {name}."
            )


_ORIGINAL_COMPARE_EXECUTED_REPLAY_ROOT = _base._compare_executed_replay_root


def _compare_executed_replay_root(
    root: Path,
    *,
    files: Mapping[str, bytes],
    source_entries: Mapping[str, str],
    archive_effective_plan: Mapping[str, object],
    archive_report: Mapping[str, object],
) -> dict[str, object]:
    summary = _ORIGINAL_COMPARE_EXECUTED_REPLAY_ROOT(
        root,
        files=files,
        source_entries=source_entries,
        archive_effective_plan=archive_effective_plan,
        archive_report=archive_report,
    )
    replay_root = Path(root).resolve()
    archive_read = _archive_reader(files, source_entries)
    replay_read = _replay_reader(replay_root)

    archive_phase1_report = _base._json_object(
        archive_read(PHASE1_REPORT_RELATIVE, "Phase 1 producer report"),
        label="archive Phase 1 producer report",
    )
    replay_phase1_report = _base._read_replay_json(
        replay_root, PHASE1_REPORT_RELATIVE, label="Phase 1 producer report"
    )
    archive_phase1 = _validate_phase1_report(
        archive_phase1_report,
        read_payload=archive_read,
        label="archive Phase 1 producer report",
    )
    replay_phase1 = _validate_phase1_report(
        replay_phase1_report,
        read_payload=replay_read,
        label="replay Phase 1 producer report",
    )
    if replay_phase1[0] != archive_phase1[0]:
        raise VerificationError(
            "Replay Phase 1 producer report stable fields differ from the archive."
        )
    _compare_payload_sets(archive_phase1[1], replay_phase1[1], label="Phase 1 CSV")
    _compare_payload_sets(archive_phase1[2], replay_phase1[2], label="Phase 1 Parquet")
    if replay_phase1[3] != archive_phase1[3]:
        raise VerificationError(
            "Replay Phase 1 CSV row counts differ from the archive."
        )

    archive_sampled_report = _base._json_object(
        archive_read(SAMPLED_SCHUR_REPORT_RELATIVE, "sampled-Schur producer report"),
        label="archive sampled-Schur producer report",
    )
    replay_sampled_report = _base._read_replay_json(
        replay_root,
        SAMPLED_SCHUR_REPORT_RELATIVE,
        label="sampled-Schur producer report",
    )
    archive_sampled = _validate_sampled_schur_report(
        archive_sampled_report,
        read_payload=archive_read,
        label="archive sampled-Schur producer report",
    )
    replay_sampled = _validate_sampled_schur_report(
        replay_sampled_report,
        read_payload=replay_read,
        label="replay sampled-Schur producer report",
    )
    if replay_sampled[0] != archive_sampled[0]:
        raise VerificationError(
            "Replay sampled-Schur producer report stable fields differ from the archive."
        )
    if replay_sampled[1] != archive_sampled[1]:
        raise VerificationError(
            "Replay sampled-Schur deterministic CSV differs from the archive."
        )

    plan_relative = _source_plan_relative(files)
    archive_source_plan = _base._json_object(
        archive_read(plan_relative, "source-controlled reproducibility plan"),
        label="archive source-controlled reproducibility plan",
    )
    replay_source_plan = _base._read_replay_json(
        replay_root,
        plan_relative,
        label="source-controlled reproducibility plan",
    )
    if replay_source_plan != archive_source_plan:
        raise VerificationError(
            "Replay source-controlled reproducibility plan differs from the archive."
        )
    planned_relatives = _planned_tabular_relatives(archive_source_plan)
    for relative in planned_relatives:
        archive_payload = archive_read(
            relative, f"source-plan tabular artifact {relative.name}"
        )
        replay_payload = replay_read(
            relative, f"source-plan tabular artifact {relative.name}"
        )
        if replay_payload != archive_payload:
            raise VerificationError(
                "Replay source-plan tabular artifact differs: "
                f"{relative.as_posix()}."
            )

    summary["phase1_diagnostics"] = {
        "report_stable_field_count": len(archive_phase1[0]),
        "csv_count": len(archive_phase1[1]),
        "parquet_count": len(archive_phase1[2]),
        "csv_row_counts": archive_phase1[3],
    }
    summary["sampled_schur_diagnostics"] = {
        "report_stable_field_count": len(archive_sampled[0]),
        "csv_row_count": archive_sampled[2],
    }
    summary["source_plan_tabular_artifact_count"] = len(planned_relatives)
    return summary


@contextmanager
def _patched_base_verifier() -> Iterator[None]:
    original_notebook = _base._validate_notebook
    original_replay = _base._compare_executed_replay_root
    _base._validate_notebook = _validate_notebook
    _base._compare_executed_replay_root = _compare_executed_replay_root
    try:
        yield
    finally:
        _base._validate_notebook = original_notebook
        _base._compare_executed_replay_root = original_replay


def _archive_context(
    archive_path: Path, external_manifest_path: Path | None
) -> tuple[dict[str, bytes], dict[str, str], dict[str, object]]:
    archive = Path(archive_path).resolve()
    external_path = (
        Path(external_manifest_path).resolve()
        if external_manifest_path is not None
        else archive.parent / _base.EXTERNAL_MANIFEST_NAME
    )
    external = _base._read_json_object(
        external_path, label="external reproducibility manifest"
    )
    bundle = external.get("bundle")
    if not isinstance(bundle, dict):
        raise VerificationError("External manifest lacks bundle metadata.")
    repository = bundle.get("repository")
    if (
        not isinstance(repository, dict)
        or type(repository.get("commit_epoch")) is not int
    ):
        raise VerificationError("External manifest lacks a valid repository epoch.")
    files, _ = _base._read_safe_archive(
        archive, commit_epoch=repository["commit_epoch"]
    )
    source_manifest = files.get(_base.SOURCE_MANIFEST_NAME)
    if source_manifest is None:
        raise VerificationError("Archive lacks MANIFEST.sha256.")
    source_entries = _base._parse_source_manifest(source_manifest)
    plan_relative = _source_plan_relative(files)
    plan_payload = files.get(plan_relative.as_posix())
    if plan_payload is None:
        raise VerificationError(
            f"Archive lacks the source-controlled plan {plan_relative.as_posix()}."
        )
    plan = _base._json_object(plan_payload, label="source reproducibility plan")
    return files, source_entries, plan


def verify_bundle(
    archive_path: Path,
    *,
    external_manifest_path: Path | None = None,
    checksum_path: Path | None = None,
    compare_replay_root: Path | None = None,
    compare_executed_replay_root: Path | None = None,
) -> dict[str, object]:
    """Run the base checks plus diagnostic producer and replay verification."""

    with _patched_base_verifier():
        summary = _base.verify_bundle(
            archive_path,
            external_manifest_path=external_manifest_path,
            checksum_path=checksum_path,
            compare_replay_root=compare_replay_root,
            compare_executed_replay_root=compare_executed_replay_root,
        )
    files, source_entries, plan = _archive_context(archive_path, external_manifest_path)
    summary["diagnostic_producer_validation"] = _validate_archive_producer_diagnostics(
        files=files,
        source_entries=source_entries,
        plan=plan,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--external-manifest", type=Path)
    parser.add_argument("--checksum", type=Path)
    parser.add_argument("--compare-replay-root", type=Path)
    parser.add_argument("--compare-executed-replay-root", type=Path)
    arguments = parser.parse_args()
    try:
        result = verify_bundle(
            arguments.archive,
            external_manifest_path=arguments.external_manifest,
            checksum_path=arguments.checksum,
            compare_replay_root=arguments.compare_replay_root,
            compare_executed_replay_root=arguments.compare_executed_replay_root,
        )
    except (OSError, VerificationError) as exc:
        parser.exit(1, f"verification failed: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
