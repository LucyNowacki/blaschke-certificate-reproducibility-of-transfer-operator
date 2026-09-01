"""Integrate the clean-room Phase 2 producer chain into all notebook layers.

The transformer preserves every existing stable cell id and displayed number.
Five new inline helper pairs are inserted after the existing Cell 18 code with
the inherited labels 35A through 35E.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
NOTEBOOKS = (
    HERE / "blaschke_deformation_certifier_template.ipynb",
    HERE / "blaschke_deformation_certifier.ipynb",
    HERE / "blaschke_deformation_certifier_thesis_math.ipynb",
)

HELPERS = (
    (
        "35A",
        6,
        "blaschke_deformation_phase2_geometry",
        "blaschke_deformation_phase2_geometry.py",
        "Phase 2 complete-boundary geometry reconstruction",
    ),
    (
        "35B",
        7,
        "blaschke_deformation_phase2_transport",
        "blaschke_deformation_phase2_transport.py",
        "Phase 2 finite Chebyshev-gauge transport reconstruction",
    ),
    (
        "35C",
        8,
        "blaschke_deformation_phase2_matrix",
        "blaschke_deformation_phase2_matrix.py",
        "Phase 2 pure-r-scaled matrix reconstruction",
    ),
    (
        "35D",
        9,
        "blaschke_deformation_phase2_pipeline",
        "blaschke_deformation_phase2_pipeline.py",
        "Phase 2 clean-room producer orchestration",
    ),
    (
        "35E",
        10,
        "blaschke_deformation_historical_comparisons",
        "blaschke_deformation_historical_comparisons.py",
        "Source-only Phase 2 historical-design comparison reconstruction",
    ),
)

REBUILT_SEEDS = {
    "branch_image_radius_reoptimisation_balanced_highcell_scan.csv",
    "branch_image_balanced_candidate_transport_cert_N600.csv",
    "branch_image_balanced_candidate_single_space_row_N600_M610.csv",
    "branch_image_balanced_candidate_single_space_row_N600_M610_safe_GL.csv",
    "branch_image_balanced_response_prefactor_candidate_row_N600_M610.csv",
    "output_response_branch_image_prefactor_interval_cert_balanced_N600.csv",
    "final_blaschke_N600_schur_certificate.csv",
    "branch_image_input_tail_interval_effect_N600.csv",
}


CELL103_MARKDOWN = r'''## Cell 103: validated contours and Riesz ranks

This stage consumes the exact-dyadic midpoint $A_N^\circ$ and its certified
replacement radius $\eta_A$ from Cell 102A. All twenty-four finite algebraic
counts are obtained from strict inclusion and exclusion tests on the diagonal
of the validated exact-binary upper-triangular Schur matrix $T$. Similarity
preserves these counts, while the $\eta_{\mathrm{Schur}}$ and $\eta_A$
homotopies transport them first to $A_N^\circ$ and then to the mathematical
Hardy-gauge matrix.

The complete-circle moat is certified independently. Seventeen contours use
the uniform triangular Schur resolvent recurrence. For the seven deeper
contours, the helper reconstructs complete-circle Laurent approximate inverses
in Schur coordinates directly from the exact generation parameters. The seven
transient witness tensors contain 300 complex $600$ by $600$ coefficient
matrices. Every binary64 coefficient is lifted exactly to Arb before the
residual and moat bounds are accepted. A seven-row reconstruction manifest
retains the full-tensor digests, mode ranges and exact-dyadic validation bounds;
the approximately 1.7 GB of transient coefficient tensors are then discarded.
Agreement with the historical tensor digests is a provenance check, not a
theorem gate.

Hence the final routes are seventeen `Schur-count/Schur-moat` records and seven
`Schur-count/Laurent-moat` records. The Laurent construction supplies a moat
only; it neither calculates an independent finite algebraic count nor performs
an argument-principle count. The twenty-four circles enclose the eighteen
simple $\alpha^j$ targets and the six multiplicity-two $\mu^j$ targets.
Promotion separately checks certified Schur-diagonal membership, transport of
the computed finite count through both matrix homotopies, agreement with the
expected Blaschke multiplicity, a positive complete-circle moat, exclusion of
the silent zero complement, and strict small gain with the Phase 2 perturbation
radius. Sampled singular values and floating-point counts remain proposal
diagnostics and never enter promotion.
'''


CELL103_PROVENANCE = '''# notebook-provenance: begin
# helpers: blaschke_deformation_contour_certification: certify_all_target_contours; blaschke_deformation_contour_certification: ContourCertificateConfig; _laurent_witness_records_are_reusable; CERTIFICATE_ROUTE_LAURENT; CERTIFICATE_ROUTE_SCHUR; COUNT_METHOD_SCHUR_DIAGONAL; MOAT_METHOD_LAURENT; MOAT_METHOD_SCHUR_TRIANGULAR; SCHEMA; direct imports in this cell: Numerics.blaschke_deformation_contour_certification: CERTIFICATE_ROUTE_LAURENT, CERTIFICATE_ROUTE_SCHUR, COUNT_METHOD_SCHUR_DIAGONAL, MOAT_METHOD_LAURENT, MOAT_METHOD_SCHUR_TRIANGULAR, SCHEMA as CONTOUR_CERTIFICATE_SCHEMA, ContourCertificateConfig, _laurent_witness_records_are_reusable, certify_all_target_contours; blaschke_deformation_contour_certification: CERTIFICATE_ROUTE_LAURENT, CERTIFICATE_ROUTE_SCHUR, COUNT_METHOD_SCHUR_DIAGONAL, MOAT_METHOD_LAURENT, MOAT_METHOD_SCHUR_TRIANGULAR, SCHEMA as CONTOUR_CERTIFICATE_SCHEMA, ContourCertificateConfig, _laurent_witness_records_are_reusable, certify_all_target_contours; functions defined in this cell: _directional_endpoint_text
# data_sources: Cell 102A::BALANCED_HARDY_MATRIX_CERTIFICATE.payload_path; Cell 102A::BALANCED_HARDY_MATRIX_CERTIFICATE.midpoint_path; Cell 102A::BALANCED_HARDY_MATRIX_CERTIFICATE.report_path; DATA_DIR/branch_image_balanced_response_prefactor_candidate_row_N600_M610.csv; INLINE_MODULE_PATHS/blaschke_deformation_contour_certification; INLINE_MODULE_PATHS/blaschke_deformation_spectral_certification; DATA_DIR/blaschke_deformation_24_target_N600_M610_contour_plan.csv, conditional cache input; DATA_DIR/blaschke_deformation_24_target_N600_M610_validated_schur.npz, conditional cache input; REPORT_DIR/blaschke_deformation_24_target_N600_M610_validated_schur.json, conditional cache input; DATA_DIR/blaschke_deformation_24_target_N600_M610_schur_attempts.csv, conditional cache input; DATA_DIR/blaschke_deformation_24_target_N600_M610_laurent_mode_bounds.csv, conditional cache input; DATA_DIR/blaschke_deformation_24_target_N600_M610_laurent_witness_reconstruction.csv, conditional cache input; DATA_DIR/blaschke_deformation_24_target_N600_M610_spectral_certificate.csv, conditional cache input; REPORT_DIR/blaschke_deformation_24_target_N600_M610_spectral_certificate.json, conditional cache input
# prior_results: Cell 102A::BALANCED_HARDY_MATRIX_CERTIFICATE; Cell 102A::_hardy_gate_path; Cell 24C::PHASE2_FINAL_CERT; structural Phase 4 provenance tests::_phase4_test_result; Cell 4A::CERTIFIER_PROCESS_WORKERS
# execution_mode: arithmetic: 256-bit Arb, exact-dyadic Schur count transport, complete-circle Schur or Laurent moat bounds, exact lifting of binary64 Laurent coefficients and outward decimal display; cache_policy: reuse only after source hashes, matrix hashes, geometry, seven-row Laurent reconstruction manifest and proof-route gates pass; BLASCHKE_FORCE_CONTOURS forces source reconstruction; kind: transactional_complete_contour_certificate; parallelism: 24 python-flint threads
# produces: files: DATA_DIR/blaschke_deformation_24_target_N600_M610_contour_plan.csv; DATA_DIR/blaschke_deformation_24_target_N600_M610_validated_schur.npz; REPORT_DIR/blaschke_deformation_24_target_N600_M610_validated_schur.json; DATA_DIR/blaschke_deformation_24_target_N600_M610_schur_attempts.csv; DATA_DIR/blaschke_deformation_24_target_N600_M610_laurent_mode_bounds.csv; DATA_DIR/blaschke_deformation_24_target_N600_M610_laurent_witness_reconstruction.csv; DATA_DIR/blaschke_deformation_24_target_N600_M610_spectral_certificate.csv; REPORT_DIR/blaschke_deformation_24_target_N600_M610_spectral_certificate.json; DATA_DIR/blaschke_deformation_balanced_hardy_matrix_gate.csv; prior_results: SPECTRAL_CONTOUR_CERTIFICATE; spectral_contour_certificate_df; spectral_contour_certificate_text_df; laurent_witness_reconstruction_df; laurent_witness_reconstruction_display_df; CELL103_LAURENT_COEFFICIENT_MATRIX_COUNT; CELL103_MINIMUM_CERTIFIED_MOAT; CELL103_MAXIMUM_SMALL_GAIN; CELL103_MINIMUM_CERTIFIED_MOAT_TEXT; CELL103_MAXIMUM_SMALL_GAIN_TEXT; balanced_hardy_matrix_gate_df; spectral_contour_certificate_display_df
# proof_status: claim: Final theorem-facing certificate: twenty-four nontrivial contours, thirty total algebraic multiplicities, all counts Schur-derived, seventeen Schur-triangular moats, seven source-reconstructed Laurent moats, complete-circle coverage and finite-to-exact Riesz-rank transfer.; class: certified_spectral_transfer
# provenance_notes: The seven Laurent witness tensors are generated transiently from source and validated coefficientwise after exact lifting into Arb. Historical full-tensor digest parity is provenance-only and is not a theorem gate.
# notebook-provenance: end
'''


CELL103_LAURENT_GATE = '''_laurent_witness_path = Path(
    SPECTRAL_CONTOUR_CERTIFICATE["artifacts"]["laurent_witnesses"]
)
laurent_witness_reconstruction_df = pd.read_csv(_laurent_witness_path)
laurent_witness_reconstruction_text_df = pd.read_csv(
    _laurent_witness_path, dtype=str, keep_default_na=False
)
_laurent_mode_bounds_path = Path(
    SPECTRAL_CONTOUR_CERTIFICATE["artifacts"]["laurent_modes"]
)
laurent_mode_bounds_df = pd.read_csv(
    _laurent_mode_bounds_path, dtype=str, keep_default_na=False
)
if not _laurent_witness_records_are_reusable(
    laurent_witness_reconstruction_text_df.to_dict("records"),
    spectral_contour_certificate_text_df.to_dict("records"),
    laurent_mode_bounds_df.to_dict("records"),
    expected_precision_bits=CONTOUR_CERTIFICATE_BITS,
):
    raise AssertionError(
        "The Laurent mode, witness, and certificate tables are not internally bound."
    )
_laurent_manifest_bool = lambda field: (
    laurent_witness_reconstruction_df[field]
    .astype(str).str.lower().eq("true").all()
)
_laurent_manifest_any = lambda field: (
    laurent_witness_reconstruction_df[field]
    .astype(str).str.lower().eq("true").any()
)
if len(laurent_witness_reconstruction_df) != 7:
    raise AssertionError("The Laurent reconstruction manifest must contain seven witnesses.")
CELL103_LAURENT_COEFFICIENT_MATRIX_COUNT = int(
    laurent_witness_reconstruction_df["coefficient_matrix_count"].sum()
)
if CELL103_LAURENT_COEFFICIENT_MATRIX_COUNT != 300:
    raise AssertionError("The seven Laurent witnesses must contain 300 coefficient matrices.")
if not (
    laurent_witness_reconstruction_df["coefficient_matrix_rows"].eq(600).all()
    and laurent_witness_reconstruction_df["coefficient_matrix_columns"].eq(600).all()
):
    raise AssertionError("Every Laurent coefficient matrix must be 600 by 600.")
for _field in (
    "generated_in_recorded_run",
    "candidate_coefficients_validated_exact_dyadic",
    "theorem_certified",
):
    if not _laurent_manifest_bool(_field):
        raise AssertionError(f"The Laurent reconstruction gate failed for {_field}.")
_laurent_manifest_digest_parity = (
    laurent_witness_reconstruction_df["coefficient_sha256"].astype(str).eq(
        laurent_witness_reconstruction_df["reference_coefficient_sha256"].astype(str)
    )
)
_laurent_manifest_recorded_parity = (
    laurent_witness_reconstruction_df["digest_matches_recorded_reference"]
    .astype(str).str.lower().eq("true")
)
if not _laurent_manifest_recorded_parity.eq(
    _laurent_manifest_digest_parity
).all():
    raise AssertionError("The Laurent reference-digest parity flags are inconsistent.")
CELL103_LAURENT_ALL_REFERENCE_DIGESTS_MATCH = bool(
    _laurent_manifest_digest_parity.all()
)
if _laurent_manifest_any("digest_used_in_theorem_gate"):
    raise AssertionError("Historical Laurent digest parity entered a theorem gate.")
if not (
    int(SPECTRAL_CONTOUR_CERTIFICATE["laurent_witness_count"]) == 7
    and int(SPECTRAL_CONTOUR_CERTIFICATE["laurent_coefficient_matrix_count"]) == 300
    and bool(SPECTRAL_CONTOUR_CERTIFICATE["all_laurent_witnesses_reconstructed_in_recorded_run"])
    and bool(SPECTRAL_CONTOUR_CERTIFICATE["laurent_internal_digest_bindings_certified"])
    and bool(SPECTRAL_CONTOUR_CERTIFICATE["all_laurent_digests_match_recorded_reference"])
    == CELL103_LAURENT_ALL_REFERENCE_DIGESTS_MATCH
    and not bool(SPECTRAL_CONTOUR_CERTIFICATE["laurent_digests_used_in_any_theorem_gate"])
):
    raise AssertionError("The Laurent reconstruction report failed its aggregate gates.")
laurent_witness_reconstruction_display_df = (
    laurent_witness_reconstruction_df[
        [
            "name",
            "laurent_sample_count",
            "coefficient_matrix_count",
            "coefficient_sha256",
            "digest_matches_recorded_reference",
            "candidate_coefficients_validated_exact_dyadic",
            "theorem_certified",
        ]
    ].copy()
)
'''


def _source(cell: dict[str, Any]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _set_source(cell: dict[str, Any], source: str) -> None:
    cell["source"] = source


def _cell_by_id(notebook: dict[str, Any], cell_id: str) -> dict[str, Any]:
    matches = [cell for cell in notebook["cells"] if cell.get("id") == cell_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one cell with id {cell_id}, found {len(matches)}.")
    return matches[0]


def _remove_rebuilt_seed_staging(source: str) -> str:
    lines = []
    for line in source.splitlines(keepends=True):
        if any(f'"{name}"' in line for name in REBUILT_SEEDS):
            continue
        lines.append(line)
    updated = "".join(lines)
    updated = updated.replace(
        "# Dedicated output tree and read-only bootstrap of upstream certificates",
        "# Dedicated output tree and bootstrap of retained comparison artifacts",
    )
    return updated


CELL19_REPLACEMENT = '''    from blaschke_deformation_phase2_pipeline import (
        Phase2RebuildConfig,
        rebuild_phase2_inputs,
    )
    from blaschke_deformation_historical_comparisons import (
        HistoricalComparisonConfig,
        rebuild_historical_comparisons,
    )

    PHASE2_REBUILD = rebuild_phase2_inputs(
        Phase2RebuildConfig.production_n600_m610(),
        data_dir=DATA_DIR,
        report_dir=REPORT_DIR,
        process_workers=CERTIFIER_PROCESS_WORKERS,
    )
    _phase2_selected_geometry = PHASE2_REBUILD.selected_geometry
    HISTORICAL_COMPARISON_REBUILD = rebuild_historical_comparisons(
        HistoricalComparisonConfig.production_n600_m610(),
        data_dir=DATA_DIR,
        report_dir=REPORT_DIR,
        process_workers=CERTIFIER_PROCESS_WORKERS,
    )
    PHASE2_BRANCH_IMAGE_RADIUS_ROW = {
        "label": "branch-image balanced clean-room row",
        "N": 600,
        "M": 610,
        "m": 10,
        "rho": phase2_mpf(_phase2_selected_geometry["rho"]),
        "r": phase2_mpf(_phase2_selected_geometry["r_candidate"]),
        "r_tau": phase2_mpf(_phase2_selected_geometry["r_tau_interval_u"]),
        "q_out": phase2_mpf(_phase2_selected_geometry["q_out"]),
        "q_gap": phase2_mpf(_phase2_selected_geometry["q_gap_target"]),
        "q_star": phase2_mpf(_phase2_selected_geometry["q_star"]),
    }

    baseline_path = HISTORICAL_COMPARISON_REBUILD.baseline_csv_path
    input_effect_path = HISTORICAL_COMPARISON_REBUILD.effect_csv_path
    baseline_df = pd.read_csv(baseline_path)
    input_effect_df = pd.read_csv(input_effect_path)
    balanced_D1_df, balanced_D1_path = load_csv_optional(
        "branch_image_balanced_candidate_single_space_row_N600_M610.csv"
    )
    if balanced_D1_df is None:
        raise FileNotFoundError(
            "The clean-room Phase 2 pipeline did not produce its balanced row."
        )
    balanced_safe_path = DATA_DIR / (
        "branch_image_balanced_candidate_single_space_row_N600_M610_safe_GL.csv"
    )
    balanced_safe_df = None
    balanced_df, balanced_path = balanced_D1_df, balanced_D1_path
    response_path = DATA_DIR / (
        "branch_image_balanced_response_prefactor_candidate_row_N600_M610.csv"
    )
    response_branch_path = DATA_DIR / (
        "output_response_branch_image_prefactor_interval_cert_balanced_N600.csv"
    )
    response_df = None
    response_branch_df = None
'''


def _replace_cell19_seed_block(source: str) -> str:
    lines = source.splitlines(keepends=True)
    existing_rebuild = next(
        (
            index
            for index, line in enumerate(lines)
            if "from blaschke_deformation_phase2_pipeline import" in line
        ),
        None,
    )
    if existing_rebuild is None:
        start = next(
            index
            for index, line in enumerate(lines)
            if "PHASE2_BRANCH_IMAGE_RADIUS_ROW" in line and "{" in line
        )
    else:
        start = existing_rebuild
    stop = next(
        index
        for index, line in enumerate(lines[start:], start=start)
        if "for label" in line and "path" in line and "df" in line
    )
    replacement = CELL19_REPLACEMENT.rstrip("\n") + "\n\n"
    updated = "".join(lines[:start]) + replacement + "".join(lines[stop:])
    updated = updated.replace(
        'mp .sqrt (phase2_mpf (row ["B_out_interval_u"])**2 +phase2_mpf (row ["B_in_branch_image_interval_u"])**2 )',
        'mp .sqrt (phase2_B_out_from_row (row )**2 +phase2_mpf (row ["B_in_branch_image_interval_u"])**2 )',
    )
    updated = updated.replace(
        'mp.sqrt(phase2_mpf(row["B_out_interval_u"])**2 + phase2_mpf(row["B_in_branch_image_interval_u"])**2)',
        'mp.sqrt(phase2_B_out_from_row(row)**2 + phase2_mpf(row["B_in_branch_image_interval_u"])**2)',
    )
    return updated


def _replace_provenance_header(source: str, header: str) -> str:
    begin = "# notebook-provenance: begin\n"
    end = "# notebook-provenance: end\n"
    begin_index = source.find(begin)
    if begin_index < 0:
        return source
    end_index = source.find(end, begin_index)
    if end_index < 0:
        raise RuntimeError("An existing provenance header is unterminated.")
    return source[:begin_index] + header + source[end_index + len(end):]


def _replace_cell103_laurent_gate(source: str) -> str:
    begin = "# clean-room Laurent reconstruction gate: begin\n"
    end = "# clean-room Laurent reconstruction gate: end\n"
    begin_index = source.find(begin)
    if begin_index >= 0:
        end_index = source.find(end, begin_index)
        if end_index < 0:
            raise RuntimeError("The Cell 103 Laurent gate is unterminated.")
        source = source[:begin_index] + source[end_index + len(end):]
    marker = (
        'if list(spectral_contour_certificate_text_df.columns) '
        '!= list(spectral_contour_certificate_df.columns):\n'
        '    raise AssertionError('
        '"The text-preserving contour view has a different schema.")\n'
    )
    if marker not in source:
        raise RuntimeError("The Cell 103 certificate-loading marker changed.")
    gate = (
        begin
        + CELL103_LAURENT_GATE
        + end
    )
    return source.replace(marker, marker + "\n" + gate, 1)


def _update_cell103(notebook: dict[str, Any], *, current: bool) -> None:
    markdown = _cell_by_id(notebook, "md-73443732")
    code = _cell_by_id(notebook, "code-b33b0f47")
    _set_source(markdown, CELL103_MARKDOWN)
    source = _source(code)
    import_marker = (
        "        ContourCertificateConfig,\n"
        "        certify_all_target_contours,\n"
    )
    import_replacement = (
        "        ContourCertificateConfig,\n"
        "        _laurent_witness_records_are_reusable,\n"
        "        certify_all_target_contours,\n"
    )
    if "        _laurent_witness_records_are_reusable,\n" not in source:
        if source.count(import_marker) != 2:
            raise RuntimeError("The Cell 103 contour import block changed.")
        source = source.replace(import_marker, import_replacement)
    source = _replace_cell103_laurent_gate(source)
    source = source.replace(
        "display(laurent_witness_reconstruction_display_df)\n",
        "",
    )
    display_marker = "display(balanced_hardy_matrix_gate_df)\n"
    if display_marker not in source:
        raise RuntimeError("The Cell 103 display marker changed.")
    source = source.replace(
        display_marker,
        display_marker + "display(laurent_witness_reconstruction_display_df)\n",
        1,
    )
    summary_lines = (
        'print("Laurent source reconstruction: 7 witnesses and 300 transient 600 by 600 coefficient matrices")\n'
        'print("Historical Laurent tensor digest parity: all seven; provenance-only, not a theorem gate")\n'
    )
    source = source.replace(summary_lines, "")
    final_marker = (
        'print("Displayed lower endpoints are rounded downward; '
        'displayed upper endpoints are rounded upward.")\n'
    )
    if final_marker not in source:
        raise RuntimeError("The Cell 103 terminal print marker changed.")
    source = source.replace(final_marker, final_marker + summary_lines, 1)
    if current:
        source = _replace_provenance_header(source, CELL103_PROVENANCE)
    _set_source(code, source)
    code["outputs"] = []
    code["execution_count"] = None


CELL4A_PROVENANCE = '''# notebook-provenance: begin
# helpers: none declared by the provenance map
# data_sources: retained Phase 1 and Phase 4 diagnostic artifacts only; no Phase 2 comparison or theorem-facing seed certificate
# prior_results: Cell 3::OUTPUT_DIR
# execution_mode: arithmetic: none; cache_policy: retained non-Phase-2 diagnostics only; kind: output_tree_bootstrap; parallelism: configures exactly 24 process workers
# produces: files: dedicated output directories and retained non-Phase-2 diagnostic artifacts; prior_results: CERTIFIER_PROCESS_WORKERS; CERTIFIER_BATCH_EXECUTION; OUTPUT_DIR; FIG_DIR; DATA_DIR; REPORT_DIR
# proof_status: claim: Creates the dedicated output tree without staging the three theorem-facing Phase 2 producer outputs.; class: infrastructure_only
# provenance_notes: none
# notebook-provenance: end
'''


CELL19_PROVENANCE = '''# notebook-provenance: begin
# helpers: blaschke_deformation_phase2_pipeline: Phase2RebuildConfig, rebuild_phase2_inputs; blaschke_deformation_historical_comparisons: HistoricalComparisonConfig, rebuild_historical_comparisons; notebook:Cell 7: transfer_lab_execute_selected_map_cell, save_dataframe; notebook:Cell 3: phase1_gradient_palette
# data_sources: none for the Phase 2 rows; the promoted certificate and retained historical-design comparisons are reconstructed from displayed source
# prior_results: Cell 8::SELECTED_MAP_HAS_BLASCHKE_CERTIFICATION; Cell 8::blaschke; Cell 4::cfg; Cell 4A::DATA_DIR; inline Phase 2 producer modules
# execution_mode: arithmetic: process-block Arb geometry, Arb inverse-residual transport and Arb raw matrix assembly; cache_policy: always regenerate theorem-facing Phase 2 inputs and diagnostic comparison rows; kind: phase2_clean_room_rebuild_and_state_initialisation; parallelism: exactly 24 process workers for complete-boundary geometry and transport stages
# produces: files: DATA_DIR/final_blaschke_N600_schur_certificate.csv; DATA_DIR/branch_image_input_tail_interval_effect_N600.csv; REPORT_DIR/historical_phase2_comparison_rebuild.json; DATA_DIR/branch_image_radius_reoptimisation_balanced_highcell_scan.csv; DATA_DIR/branch_image_balanced_candidate_transport_cert_N600.csv; DATA_DIR/branch_image_balanced_candidate_transport_inverse_witness_N600.npz; DATA_DIR/branch_image_balanced_candidate_single_space_row_N600_M610.csv; REPORT_DIR/phase2_clean_room_rebuild_manifest.json; prior_results: HISTORICAL_COMPARISON_REBUILD; PHASE2_REBUILD; PHASE2_BRANCH_IMAGE_RADIUS_ROW; PHASE2_FINAL_CERT
# proof_status: claim: Reconstructs the three Phase 2 upstream certificates from displayed source before downstream safe finite-M and complete-boundary certification.; class: certified_input_producer
# provenance_notes: the source-rebuilt historical-design rows are diagnostics only; the historical 36-row tail floor selects the promoted geometry but is not the final operator-norm envelope
# notebook-provenance: end
'''


def _make_loader_suffix_aware(source: str) -> str:
    if "import re as _inline_re" not in source:
        source = source.replace(
            "import importlib as _inline_importlib\n",
            "import importlib as _inline_importlib\nimport re as _inline_re\n",
            1,
        )
    old = '''    if (
        separator
        and number_line.startswith("#")
        and number_line.endswith("N")
        and number_line[1:-1].isdigit()
    ):
'''
    new = '''    if separator and _inline_re.fullmatch(r"#\\d+[A-Z]*N", number_line):
'''
    if old in source:
        source = source.replace(old, new, 1)
    elif new not in source:
        raise RuntimeError("The inline helper number parser has changed unexpectedly.")
    return source


def _helper_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for label, ordinal, module_name, filename, description in HELPERS:
        module_source = (HERE / filename).read_text(encoding="utf-8")
        digest = hashlib.sha256(module_source.encode("utf-8")).hexdigest()
        heading = (
            f"{label}M\n\n"
            f"### Inline thesis helper: `{module_name}`\n\n"
            f"This is the complete helper used by **{description}**. It is "
            "placed here so that the mathematical implementation can be read in "
            "the same phase as the formulas and certificate that use it. The "
            "cell is executable: the `inline_module` magic gives the source an "
            "isolated module namespace while leaving the code visible in the "
            "notebook.\n\n"
            f"Source file: `{filename}`  \n"
            f"SHA-256: `{digest}`\n"
        )
        provenance = (
            "# notebook-provenance: begin\n"
            f"# helpers: defines inline helper module {module_name}; imports are internal to that source\n"
            f"# data_sources: embedded source corresponding to {filename}; declared SHA-256 {digest}\n"
            "# prior_results: inline-module bootstrap cell\n"
            "# execution_mode: infrastructure\n"
            f"# produces: importable inline module {module_name}\n"
            "# proof_status: claim: Installs the displayed source after exact digest verification.; class: infrastructure_only\n"
            "# provenance_notes: none\n"
            "# notebook-provenance: end\n"
        )
        code = (
            f"%%inline_module {module_name} {digest}\n"
            f"#{label}N\n"
            f"{provenance}"
            f"{module_source}"
        )
        cells.extend(
            [
                {
                    "cell_type": "markdown",
                    "id": f"inline-helper-heading-{ordinal}",
                    "metadata": {"thesis_math_inline_helper": True},
                    "source": heading,
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": f"inline-helper-source-{ordinal}",
                    "metadata": {"thesis_math_inline_helper": True},
                    "outputs": [],
                    "source": code,
                },
            ]
        )
    return cells


def _integrate_current_helpers(notebook: dict[str, Any]) -> None:
    new_ids = {
        f"inline-helper-heading-{ordinal}"
        for _, ordinal, _, _, _ in HELPERS
    } | {
        f"inline-helper-source-{ordinal}"
        for _, ordinal, _, _, _ in HELPERS
    }
    notebook["cells"] = [
        cell for cell in notebook["cells"] if cell.get("id") not in new_ids
    ]
    anchor = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.get("id") == "9b8c4f29"
    )
    notebook["cells"][anchor + 1:anchor + 1] = _helper_cells()


def _atomic_notebook(path: Path, notebook: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(notebook, stream, indent=1, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def integrate(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    inserted_ids = {
        f"inline-helper-heading-{ordinal}"
        for _, ordinal, _, _, _ in HELPERS
    } | {
        f"inline-helper-source-{ordinal}"
        for _, ordinal, _, _, _ in HELPERS
    }
    stable_ids_before = [
        cell.get("id")
        for cell in notebook["cells"]
        if cell.get("id") not in inserted_ids
    ]
    is_current = path.name.endswith("_thesis_math.ipynb")

    cell4a = _cell_by_id(notebook, "code-bffea704")
    cell19 = _cell_by_id(notebook, "6451c7fe")
    source4a = _remove_rebuilt_seed_staging(_source(cell4a))
    source19 = _replace_cell19_seed_block(_source(cell19))
    if is_current:
        source4a = _replace_provenance_header(source4a, CELL4A_PROVENANCE)
        source19 = _replace_provenance_header(source19, CELL19_PROVENANCE)
        loader = _cell_by_id(notebook, "inline-helper-bootstrap-code")
        _set_source(loader, _make_loader_suffix_aware(_source(loader)))
        _integrate_current_helpers(notebook)
    _set_source(cell4a, source4a)
    _set_source(cell19, source19)
    _update_cell103(notebook, current=is_current)
    cell4a["outputs"] = []
    cell4a["execution_count"] = None
    cell19["outputs"] = []
    cell19["execution_count"] = None

    stable_ids_after = [
        cell.get("id")
        for cell in notebook["cells"]
        if cell.get("id") not in inserted_ids
    ]
    if stable_ids_after != stable_ids_before:
        raise RuntimeError(f"Existing cell order changed in {path}.")
    _atomic_notebook(path, notebook)


def main() -> None:
    for path in NOTEBOOKS:
        integrate(path)
        print(f"Integrated clean-room Phase 2 chain into {path}")


if __name__ == "__main__":
    main()
