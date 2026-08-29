"""Build the source Blaschke-deformation certifier from a locked template.

The deployment template contains the complete phase-ordered Markdown and code
of the certifier, but no execution outputs. This builder validates the cells
which carry theorem-level provenance and writes the independently executable
source notebook. The separate thesis-mathematics builder then adds readable
inline copies of the mathematical helper modules.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import nbformat as nbf


HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "blaschke_deformation_certifier_template.ipynb"
TARGET = HERE / "blaschke_deformation_certifier.ipynb"

# Updated after the deployment template is refreshed deliberately with
# ``--refresh-template``. Ordinary builds refuse unreviewed template drift.
LOCKED_TEMPLATE_SHA256 = (
    "85efb8203a6d3f0477c7cee2df99e8d684428b814f1bd142366de46bf0f615c9"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source(cell: dict[str, Any]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _set_source(cell: dict[str, Any], source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def _unique_cell(cells: list[dict[str, Any]], marker: str) -> dict[str, Any]:
    matches = tuple(cell for cell in cells if marker in _source(cell))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one notebook cell containing {marker!r}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _checked_replace(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected one occurrence of {label!r}; found {count}."
        )
    return source.replace(old, new, 1)


def _replace_in_order(
    source: str,
    old: str,
    replacements: tuple[str, ...],
    *,
    label: str,
) -> str:
    if source.count(old) != len(replacements):
        raise RuntimeError(
            f"Expected {len(replacements)} occurrences of {label!r}; "
            f"found {source.count(old)}."
        )
    for replacement in replacements:
        source = source.replace(old, replacement, 1)
    return source


FINAL_PHASE2_REFRESH = r'''
# Cell 24C is the authoritative Phase 2 aggregation point. Rewrite the
# provisional Cell 24B candidate and its human-readable report after the final
# direct-coherent intersection, branchwise and whole-ellipse input selection
# has been made.
import csv
import math

_phase2_export_upper_float = lambda value: math.nextafter(float(value), math.inf)
_rp_candidate.update({
    "phase2_aggregation_status": "authoritative Cell 24C refresh",
    "B_in_branch_image_interval_u": _phase2_export_upper_float(
        PHASE2_FINAL_CERT["B_in"]
    ),
    "B_in_selected_cert_u": _phase2_export_upper_float(
        PHASE2_FINAL_CERT["B_in"]
    ),
    "B_in_selected_cert_text": str(PHASE2_FINAL_CERT["B_in"]),
    "B_in_selection": str(PHASE2_FINAL_CERT["B_in_selection"]),
    "B_in_coherent_row_cert_u": _phase2_export_upper_float(
        PHASE2_FINAL_CERT["B_in_combined_row"]
    ),
    "B_in_branchwise_profile_cert_u": _phase2_export_upper_float(
        PHASE2_FINAL_CERT["B_in_branchwise"]
    ),
    "collocation_selected_cert_u": _phase2_export_upper_float(
        PHASE2_FINAL_CERT["collocation"]
    ),
    "matrix_selection": str(PHASE2_FINAL_CERT["matrix_selection"]),
    "new_epsilon_response_prefactor_candidate": _phase2_export_upper_float(
        PHASE2_FINAL_CERT["epsilon"]
    ),
    "new_epsilon_response_prefactor_candidate_text": str(
        PHASE2_FINAL_CERT["epsilon"]
    ),
    "epsilon_triangle_check_text": PHASE2_FINAL_CERT[
        "epsilon_triangle_upper_text"
    ],
    "noncollocation_rss_without_collocation": _phase2_export_upper_float(
        PHASE2_FINAL_CERT["noncollocation_rss"]
    ),
    "tail_floor_without_collocation": _phase2_export_upper_float(
        PHASE2_FINAL_CERT["noncollocation_rss"]
    ),
    "status": (
        "authoritative selected resolved-response row with the certified "
        "unresolved-input intersection and best matrix aggregation"
    ),
})
if "B_in_whole_ellipse_fallback" in PHASE2_FINAL_CERT:
    _rp_candidate["B_in_whole_ellipse_fallback_cert_u"] = (
        _phase2_export_upper_float(
            PHASE2_FINAL_CERT["B_in_whole_ellipse_fallback"]
        )
    )
response_df = pd.DataFrame([_rp_candidate])
response_df.to_csv(_rp_candidate_path, index=False)

_rp_provenance["final_phase2_refresh"] = {
    "producer_cell": "Cell 24C",
    "B_in_selection": str(PHASE2_FINAL_CERT["B_in_selection"]),
    "B_in_selected_cert_text": str(PHASE2_FINAL_CERT["B_in"]),
    "matrix_selection": str(PHASE2_FINAL_CERT["matrix_selection"]),
    "epsilon_cert_text": PHASE2_FINAL_CERT["epsilon_upper_text"],
    "epsilon_triangle_cert_text": PHASE2_FINAL_CERT[
        "epsilon_triangle_upper_text"
    ],
}
_rp_json_path.write_text(
    json.dumps(_rp_provenance, indent=2), encoding="utf-8"
)
_rp_report_lines = [
    "# Certified resolved-response prefactor and final Phase 2 aggregation",
    "",
    f"Resolved-response producer: Cell 24B; final aggregation: Cell 24C; map: {_rp_map_label}.",
    f"N={_rp_N}, M={_rp_M}, rho={_rp_balanced['rho']}, r={_rp_balanced['r']}.",
    (
        f"Arb precision: {RESPONSE_PREF_BITS} bits; boundary cells: "
        f"{RESPONSE_PREF_CELLS}; coherent prefix: "
        f"{int(_rp_coherent_summary['K'])} modes."
    ),
    "",
    f"Coherent Chebyshev-packet response upper: {_rp_upper_float(_rp_coherent_global):.17e}.",
    f"Scaled-Legendre response upper: {_rp_upper_float(_rp_scaled_response_upper):.17e}.",
    f"Whole-ellipse finite-restriction fallback: {_rp_upper_float(_rp_whole_fallback):.17e}.",
    f"Selected cellwise response upper: {_rp_upper_float(_rp_selected_response_upper):.17e}.",
    f"Certified output-tail contribution B_out: {mp.nstr(PHASE2_FINAL_CERT['B_out'], 50)}.",
    f"Final selected input contribution B_in: {mp.nstr(PHASE2_FINAL_CERT['B_in'], 50)}.",
    f"Input selection: {PHASE2_FINAL_CERT['B_in_selection']}.",
    "The selected unresolved-input intersection equals the branchwise upper "
    "on every deployed boundary cell; no cancellation gain is claimed.",
    f"Final selected matrix contribution: {mp.nstr(PHASE2_FINAL_CERT['collocation'], 50)}.",
    f"Matrix selection: {PHASE2_FINAL_CERT['matrix_selection']}.",
    f"Final deterministic epsilon upper: {PHASE2_FINAL_CERT['epsilon_upper_text']}.",
    f"Auxiliary triangle epsilon upper: {PHASE2_FINAL_CERT['epsilon_triangle_upper_text']}.",
    "",
    "The Cell 24B provisional radius has been replaced by the authoritative Cell 24C aggregation.",
    "",
    "Upstream SHA-256 hashes:",
]
for _rp_name, _rp_hash in _rp_provenance["inputs"].items():
    _rp_report_lines.append(f"- `{Path(_rp_name).name}`: `{_rp_hash}`")
_rp_report_path.write_text("\n".join(_rp_report_lines), encoding="utf-8")

_phase2_candidate_check = next(
    csv.DictReader(_rp_candidate_path.open("r", encoding="utf-8", newline=""))
)
if mp.mpf(_phase2_candidate_check["B_in_selected_cert_text"]) != PHASE2_FINAL_CERT["B_in"]:
    raise AssertionError("The refreshed candidate input factor is inconsistent.")
if mp.mpf(
    _phase2_candidate_check["new_epsilon_response_prefactor_candidate_text"]
) != PHASE2_FINAL_CERT["epsilon"]:
    raise AssertionError("The refreshed candidate radius is inconsistent.")
'''.strip()


HISTORICAL_PHASE4_REBUILD = r'''
from Numerics.blaschke_deformation_historical_phase4 import (
    HistoricalPhase4Config,
    rebuild_historical_phase4,
)

_historical_phase4_force = os.environ.get(
    "BLASCHKE_FORCE_HISTORICAL_PHASE4", "0"
) in {"1", "true", "True"}
historical_phase4_result = rebuild_historical_phase4(
    HistoricalPhase4Config.production_n600_m610(),
    data_dir=DATA_DIR,
    report_dir=REPORT_DIR,
    cache_dir=DATA_DIR / ".historical_phase4_cache",
    assembly_workers=min(24, os.cpu_count() or 1),
    surface_workers=min(6, os.cpu_count() or 1),
    force=_historical_phase4_force,
)
'''.strip()


DIAGNOSTIC_AUDIT_REBUILD = r'''
from Numerics.blaschke_deformation_diagnostic_audits import (
    REPORT_FILENAME as DIAGNOSTIC_AUDIT_REPORT_FILENAME,
    rebuild_diagnostic_audits,
)

diagnostic_audit_result = rebuild_diagnostic_audits(
    map_label=SELECTED_MAP_LABEL,
    targets=DATA_DIR / "raw_spectrum_square_N_sweep_extended_targets.csv",
    geometry=DATA_DIR / "transfer_lab_blaschke_mu_0p3_active_geometry.csv",
    schur_rows=(
        DATA_DIR
        / "transfer_lab_blaschke_mu_0p3_generic_sampled_schur_envelope.csv"
    ),
    moat_rows=(
        DATA_DIR
        / "branch_image_wide_candidate_first15_contour_moats_N600_M610_J128.csv"
    ),
    data_dir=DATA_DIR,
    report_path=REPORT_DIR / DIAGNOSTIC_AUDIT_REPORT_FILENAME,
)
selected_map_certification_audit_df = (
    diagnostic_audit_result.universal_audit.copy()
)
'''.strip()


CELL_24A_WRAPPER = r'''# Cell 24A
# ============================================================
# Phase 2G -- unconditional finite-M Gauss--Legendre prefactor
# ============================================================

from pathlib import Path
import os

import pandas as pd
from flint import arb

if Path.cwd().name == "Numerics":
    from blaschke_deformation_phase2_finite_m import (
        Phase2FiniteMConfig,
        certify_finite_m_completion,
    )
else:
    from Numerics.blaschke_deformation_phase2_finite_m import (
        Phase2FiniteMConfig,
        certify_finite_m_completion,
    )

FINITE_M_SAFE_BITS = int(os.environ.get("TRANSFER_LAB_FINITE_M_BITS", "192"))
phase2_finite_m_result = certify_finite_m_completion(
    Phase2FiniteMConfig(
        N=600,
        M=610,
        rho="2.725",
        map_label=SELECTED_MAP_LABEL,
        precision_bits=FINITE_M_SAFE_BITS,
    ),
    output_dir=OUTPUT_DIR,
)

finite_M_prefactor_df = pd.DataFrame([phase2_finite_m_result.certificate])
balanced_safe_df = pd.DataFrame([phase2_finite_m_result.safe_row])
balanced_df = balanced_safe_df.copy()
balanced_path = phase2_finite_m_result.safe_row_csv_path
FINITE_M_PREFACTOR_CERTIFIED = bool(
    phase2_finite_m_result.certificate["finite_M_prefactor_certified"]
)
FINITE_M_SAFE_D = arb(
    str(phase2_finite_m_result.certificate["D_safe_u"])
).upper()

display(finite_M_prefactor_df)
print("Stored:", phase2_finite_m_result.certificate_csv_path)
print("Stored:", phase2_finite_m_result.safe_row_csv_path)
print("Stored:", phase2_finite_m_result.report_json_path)
print("Stored:", phase2_finite_m_result.report_markdown_path)
'''


CELL_24B_WRAPPER = r'''# Cell 24B
# ============================================================
# Phase 2H -- certified resolved-response prefactor
# ============================================================

from pathlib import Path
import json
import os

import pandas as pd
from flint import arb

if Path.cwd().name == "Numerics":
    from blaschke_deformation_phase2_resolved_response import (
        Phase2ResolvedResponseConfig,
        certify_resolved_response_completion,
    )
else:
    from Numerics.blaschke_deformation_phase2_resolved_response import (
        Phase2ResolvedResponseConfig,
        certify_resolved_response_completion,
    )

RESPONSE_PREF_CELLS = int(os.environ.get("BLASCHKE_RESPONSE_CERT_CELLS", "65536"))
RESPONSE_PREF_BITS = int(os.environ.get("BLASCHKE_RESPONSE_CERT_BITS", "192"))
RESPONSE_PREF_PREFIX_TERMS = int(
    os.environ.get("BLASCHKE_RESPONSE_PREFIX_TERMS", "24")
)
_response_progress_step = max(1, RESPONSE_PREF_CELLS // 10)


def _phase2_response_progress(stage, done, total):
    if done % _response_progress_step == 0 or done == total:
        print(f"{stage}: {done}/{total}")


phase2_resolved_response_result = certify_resolved_response_completion(
    Phase2ResolvedResponseConfig(
        N=600,
        M=610,
        rho="2.725",
        r="2.473669807791324",
        mu="0.3",
        cells=RESPONSE_PREF_CELLS,
        precision_bits=RESPONSE_PREF_BITS,
        prefix_terms=RESPONSE_PREF_PREFIX_TERMS,
        map_label=SELECTED_MAP_LABEL,
    ),
    output_dir=OUTPUT_DIR,
    progress=_phase2_response_progress,
)

resolved_response_certificate = phase2_resolved_response_result.certificate
response_branch_df = pd.DataFrame(
    [phase2_resolved_response_result.response_summary]
)
response_profile_df = pd.DataFrame(
    phase2_resolved_response_result.response_profile
)
response_df = pd.DataFrame([phase2_resolved_response_result.candidate_row])

# Public compatibility names used by the retained Phase 2 audit and figures.
_rp_summary = phase2_resolved_response_result.response_summary
_rp_coherent_summary = phase2_resolved_response_result.coherent_summary
_rp_candidate = phase2_resolved_response_result.candidate_row
_rp_candidate_path = phase2_resolved_response_result.candidate_csv_path
_rp_json_path = phase2_resolved_response_result.report_json_path
_rp_report_path = phase2_resolved_response_result.report_markdown_path
_rp_provenance = json.loads(_rp_json_path.read_text(encoding="utf-8"))
_rp_map_label = SELECTED_MAP_LABEL
_rp_N = int(_rp_summary["N"])
_rp_M = 610
_rp_coherent_global = arb(str(_rp_summary["C_resp_coherent_packet_cert_u"])).upper()
_rp_scaled_response_upper = arb(
    str(_rp_summary["C_resp_scaled_legendre_cert_u"])
).upper()
_rp_whole_fallback = arb(
    str(_rp_summary["C_resp_whole_ellipse_fallback_u"])
).upper()
_rp_selected_response_upper = arb(
    str(_rp_summary["C_resp_selected_cert_u"])
).upper()
_rp_Z_tail_upper = arb(str(_rp_summary["Z_tail_N_inferred"])).upper()
_rp_B_out = (_rp_selected_response_upper * _rp_Z_tail_upper).upper()

PHASE2_FINAL_CERT = phase2_promoted_certificate()
display(response_branch_df)
display(response_df)
print("Stored:", phase2_resolved_response_result.principal_csv_path)
print("Stored:", phase2_resolved_response_result.candidate_csv_path)
print("Stored:", phase2_resolved_response_result.report_json_path)
'''


CELL_24C_WRAPPER = r'''# Cell 24C
# Complete Arb certificate for the unresolved Chebyshev input tail

from pathlib import Path
import os

import pandas as pd

if Path.cwd().name == "Numerics":
    from blaschke_deformation_phase2_final_aggregation import (
        Phase2FinalAggregationConfig,
        certify_final_phase2_aggregation,
    )
else:
    from Numerics.blaschke_deformation_phase2_final_aggregation import (
        Phase2FinalAggregationConfig,
        certify_final_phase2_aggregation,
    )

INPUT_TAIL_CERT_CELLS = int(os.environ.get("BLASCHKE_INPUT_CERT_CELLS", "65536"))
INPUT_TAIL_CERT_BITS = int(os.environ.get("BLASCHKE_INPUT_CERT_BITS", "192"))
INPUT_TAIL_PREFIX_TERMS = int(os.environ.get("BLASCHKE_INPUT_PREFIX_TERMS", "24"))
_input_progress_step = max(1, INPUT_TAIL_CERT_CELLS // 10)


def _phase2_input_progress(done, total):
    if done % _input_progress_step == 0 or done == total:
        print(f"unresolved-input rows: {done}/{total}")


phase2_final_aggregation_result = certify_final_phase2_aggregation(
    Phase2FinalAggregationConfig(
        N=600,
        M=610,
        rho="2.725",
        r="2.473669807791324",
        mu="0.3",
        cells=INPUT_TAIL_CERT_CELLS,
        precision_bits=INPUT_TAIL_CERT_BITS,
        prefix_terms=INPUT_TAIL_PREFIX_TERMS,
        map_label=SELECTED_MAP_LABEL,
    ),
    output_dir=OUTPUT_DIR,
    progress=_phase2_input_progress,
)

input_tail_certificate = phase2_final_aggregation_result.input_certificate
_input_summary = input_tail_certificate["summary"]
input_tail_certificate_df = pd.DataFrame([_input_summary])
input_tail_profile_df = pd.DataFrame(input_tail_certificate["profile"])
PHASE2_FINAL_CERT = phase2_final_aggregation_result.final_certificate
response_df = pd.DataFrame(
    [phase2_final_aggregation_result.refreshed_candidate]
)
cert_summary_df = pd.DataFrame(phase2_final_aggregation_result.summary_rows)

QSTAR_X = PHASE2_FINAL_CERT["q_star"]
N_CERT_X = int(PHASE2_FINAL_CERT["N"])
M_CERT_X = int(PHASE2_FINAL_CERT["M"])
EPS_CERT_X = PHASE2_FINAL_CERT["epsilon"]
BOUT_CERT_X = PHASE2_FINAL_CERT["B_out"]
BIN_CERT_X = PHASE2_FINAL_CERT["B_in"]
COLL_CERT_X = PHASE2_FINAL_CERT["collocation"]
NONCOLLOCATION_RSS_CERT_X = PHASE2_FINAL_CERT["noncollocation_rss"]
NONCOLLOCATION_SUM_CERT_X = PHASE2_FINAL_CERT["noncollocation_sum"]
TAIL_FLOOR_CERT_X = NONCOLLOCATION_RSS_CERT_X
C_CERT_QSTAR = EPS_CERT_X / (QSTAR_X ** N_CERT_X)

display(input_tail_certificate_df)
display(cert_summary_df)
print("Stored:", phase2_final_aggregation_result.input_certificate_csv_path)
print("Stored:", phase2_final_aggregation_result.input_profile_csv_path)
print("Stored:", phase2_final_aggregation_result.phase2_summary_csv_path)
'''


def _normalise_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    notebook = deepcopy(notebook)
    cells = notebook["cells"]

    cell24a = _unique_cell(cells, "# Cell 24A\n")
    _set_source(cell24a, CELL_24A_WRAPPER)

    cell24b = _unique_cell(cells, "# Cell 24B\n")
    _set_source(cell24b, CELL_24B_WRAPPER)

    cell24c = _unique_cell(cells, "# Cell 24C\n")
    _set_source(cell24c, CELL_24C_WRAPPER)

    cell35 = _unique_cell(cells, "# Cell 35\n")
    source = _source(cell35)
    if "historical_phase4_result = rebuild_historical_phase4(" not in source:
        anchor = "    for _d in (DATA_DIR, FIG_DIR, REPORT_DIR):\n        _d.mkdir(parents=True, exist_ok=True)\n"
        source = _checked_replace(
            source,
            anchor,
            anchor
            + "\n"
            + "\n".join(
                "    " + line if line else ""
                for line in HISTORICAL_PHASE4_REBUILD.splitlines()
            )
            + "\n",
            label="Cell 35 historical Phase 4 source rebuild anchor",
        )
    _set_source(cell35, source)

    cell100 = _unique_cell(cells, "# Cell 100\n")
    source = _source(cell100)
    if "diagnostic_audit_result = rebuild_diagnostic_audits(" not in source:
        old = '''selected_map_certification_audit_df = build_certification_audit(
    map_label=SELECTED_MAP_LABEL,
    capability=capability,
    target_count=len(targets_14),
    geometry=geometry_for_audit,
    schur_rows=schur_for_audit,
    moat_rows=moat_for_audit.head(14),
)

audit_path = DATA_DIR / f'transfer_lab_{SELECTED_MAP_LABEL}_universal_certification_audit.csv'
selected_map_certification_audit_df.to_csv(audit_path, index=False)
'''
        source = _checked_replace(
            source,
            old,
            DIAGNOSTIC_AUDIT_REBUILD + "\n\naudit_path = diagnostic_audit_result.universal_csv_path\n",
            label="Cell 100 legacy diagnostic audit writer",
        )
    _set_source(cell100, source)

    cell102 = _unique_cell(cells, "# Cell 102\n")
    source = _source(cell102)
    if "packet_table = diagnostic_audit_result.first14_audit.copy()" not in source:
        start = source.index("packet_table = targets_14.copy()")
        stop = source.index("fig, ax = plt.subplots", start)
        source = (
            source[:start]
            + "packet_table = diagnostic_audit_result.first14_audit.copy()\n"
            + "packet_path = diagnostic_audit_result.first14_csv_path\n\n"
            + source[stop:]
        )
    _set_source(cell102, source)

    for update_number in (89, 98):
        current_marker = f"## Notebook update {update_number}"
        historical_marker = f"## Historical notebook update {update_number}"
        matches = tuple(
            cell
            for cell in cells
            if current_marker in _source(cell) or historical_marker in _source(cell)
        )
        if len(matches) != 1:
            raise RuntimeError(f"Expected one notebook update {update_number}.")
        cell = matches[0]
        source = _source(cell).replace(
            current_marker, historical_marker, 1
        )
        notice = "**Historical: superseded by Notebook update 99.**\n\n"
        if notice not in source:
            source = source.replace(
                historical_marker + "\n\n",
                historical_marker + "\n\n" + notice,
                1,
            )
        _set_source(cell, source)

    hardy_cell = _unique_cell(cells, "# Cell 102A\n")
    source = _source(hardy_cell)
    old_sources = (
        "_hardy_source_files = (\n"
        "    _spectral_cert_module_path, _spectral_worker_path, _spectral_builder_path,\n"
        ")"
    )
    source = source.replace(
        old_sources,
        "_hardy_source_files = (_spectral_cert_module_path,)",
        1,
    )
    _set_source(hardy_cell, source)

    contour_cell = _unique_cell(cells, "# Cell 103\n")
    source = _source(contour_cell)
    inline_contour = 'INLINE_MODULE_PATHS["blaschke_deformation_contour_certification"]'
    inline_matrix = 'INLINE_MODULE_PATHS["blaschke_deformation_spectral_certification"]'
    inline_deterministic = 'INLINE_MODULE_PATHS["blaschke_deformation_certification"]'
    if inline_contour in source:
        source = _replace_in_order(
            source,
            inline_contour,
            (
                'Path("blaschke_deformation_contour_certification.py")',
                'Path("Numerics/blaschke_deformation_contour_certification.py")',
            ),
            label="Cell 103 contour provenance path",
        )
    if inline_matrix in source:
        source = _replace_in_order(
            source,
            inline_matrix,
            (
                'Path("blaschke_deformation_spectral_certification.py")',
                'Path("Numerics/blaschke_deformation_spectral_certification.py")',
            ),
            label="Cell 103 matrix provenance path",
        )
    if inline_deterministic in source:
        source = source.replace(
            f'    _contour_deterministic_module_path = {inline_deterministic}\n',
            "",
        )
    for deterministic_assignment in (
        '    _contour_deterministic_module_path = Path("blaschke_deformation_certification.py")\n',
        '    _contour_deterministic_module_path = Path("Numerics/blaschke_deformation_certification.py")\n',
    ):
        source = source.replace(deterministic_assignment, "")
    source = source.replace(
        "        _contour_deterministic_module_path,\n",
        "",
    )
    _set_source(contour_cell, source)

    reproducibility_cell = _unique_cell(cells, "# Cell 104\n")
    source = _source(reproducibility_cell).replace(
        '"blaschke_deformation_certifier.ipynb"',
        '"blaschke_deformation_certifier_thesis_math.ipynb"',
        1,
    )
    if "BLASCHKE_SKIP_ARCHIVE" not in source:
        source = _checked_replace(
            source,
            "from pathlib import Path\nimport pandas as pd\n",
            "from pathlib import Path\nimport json\nimport os\nimport pandas as pd\n",
            label="Cell 104 imports",
        )
        call = '''reproducibility_bundle = build_reproducibility_bundle(
    repo_root=_repro_repo_root,
    notebook_path=_repro_repo_root / "Numerics" / "blaschke_deformation_certifier_thesis_math.ipynb",
    output_dir=OUTPUT_DIR,
    precision_settings=_repro_precision_settings,
    upstream_artifact_names=_certifier_seed_names,
)'''
        guarded_call = '''_repro_plan_path = REPORT_DIR / "blaschke_deformation_reproducibility_plan.json"
_repro_plan_path.write_text(
    json.dumps(
        {
            "precision_settings": _repro_precision_settings,
            "upstream_artifact_names": list(_certifier_seed_names),
        },
        indent=2,
    ),
    encoding="utf-8",
)
def _deferred_reproducibility_record(*, execution_status, repository_dirty, reason):
    return {
        "archive_path": "deferred until clean-commit finalisation",
        "manifest_path": str(_repro_plan_path),
        "checksum_path": "deferred until clean-commit finalisation",
        "archive_sha256": "deferred",
        "archive_bytes": 0,
        "packaged_file_count": 0,
        "repository_commit": "deferred",
        "repository_dirty": repository_dirty,
        "execution_status": execution_status,
        "deferred_reason": reason,
    }


if os.environ.get("BLASCHKE_SKIP_ARCHIVE", "0") in {"1", "true", "True"}:
    reproducibility_bundle = _deferred_reproducibility_record(
        execution_status="deferred_by_environment",
        repository_dirty=None,
        reason="Archive creation was deferred by BLASCHKE_SKIP_ARCHIVE.",
    )
else:
    try:
        reproducibility_bundle = build_reproducibility_bundle(
            repo_root=_repro_repo_root,
            notebook_path=(
                _repro_repo_root
                / "Numerics"
                / "blaschke_deformation_certifier_thesis_math.ipynb"
            ),
            output_dir=OUTPUT_DIR,
            precision_settings=_repro_precision_settings,
            upstream_artifact_names=_certifier_seed_names,
        )
    except RuntimeError as exc:
        if "requires a clean deployment commit" not in str(exc):
            raise
        reproducibility_bundle = _deferred_reproducibility_record(
            execution_status="deferred_dirty_worktree",
            repository_dirty=True,
            reason=(
                "Archive creation was deferred because the deployment worktree "
                "is dirty. Commit the intended deployment state and rerun Cell 104."
            ),
        )'''
        source = _checked_replace(
            source,
            call,
            guarded_call,
            label="Cell 104 reproducibility call",
        )
    _set_source(reproducibility_cell, source)

    metadata = notebook.setdefault("metadata", {})
    metadata["blaschke_deformation_certifier"] = {
        "map_label": "blaschke_mu_0p3",
        "trivial_targets": 0,
        "nontrivial_targets": 24,
        "phase2_final_epsilon": (
            "3.3264433839017426342179265983234893950031511766904056153485116e-20"
        ),
        "count_route": "24 Schur-derived finite algebraic counts",
        "moat_routes": "17 Schur-triangular and 7 Laurent complete-circle moats",
    }
    return notebook


def _clear_runtime_state(notebook: dict[str, Any]) -> dict[str, Any]:
    notebook = deepcopy(notebook)
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook.get("metadata", {}).pop("widgets", None)
    return notebook


def _validate(notebook: dict[str, Any]) -> None:
    cells = notebook["cells"]
    cell24a = _source(_unique_cell(cells, "# Cell 24A\n"))
    cell24b = _source(_unique_cell(cells, "# Cell 24B\n"))
    cell24c = _source(_unique_cell(cells, "# Cell 24C\n"))
    hardy = _source(_unique_cell(cells, "# Cell 102A\n"))
    contour = _source(_unique_cell(cells, "# Cell 103\n"))
    cell104 = _source(_unique_cell(cells, "# Cell 104\n"))
    required_24a = (
        "Phase2FiniteMConfig",
        "certify_finite_m_completion",
        "output_dir=OUTPUT_DIR",
    )
    if not all(marker in cell24a for marker in required_24a):
        raise AssertionError("Cell 24A is not the standalone finite-M wrapper.")
    required_24b = (
        "Phase2ResolvedResponseConfig",
        "certify_resolved_response_completion",
        "output_dir=OUTPUT_DIR",
        "response_branch_df",
    )
    if not all(marker in cell24b for marker in required_24b):
        raise AssertionError("Cell 24B is not the standalone resolved-response wrapper.")
    if "INLINE_MODULE_PATHS" in cell24b:
        raise AssertionError("The source Cell 24B depends on inline-only state.")
    required_24c = (
        "Phase2FinalAggregationConfig",
        "certify_final_phase2_aggregation",
        "output_dir=OUTPUT_DIR",
        "PHASE2_FINAL_CERT",
        "cert_summary_df",
    )
    if not all(marker in cell24c for marker in required_24c):
        raise AssertionError("Cell 24C is not the standalone final-aggregation wrapper.")
    for embedded_marker in (
        "def _fm_",
        "def _rp_",
        "def _input_",
        "certify_resolved_response_rows",
        "certify_input_tail_rows",
    ):
        if embedded_marker in cell24a + cell24b + cell24c:
            raise AssertionError(
                "A Phase 2 completion implementation remains embedded in the notebook."
            )
    if "_hardy_source_files = (_spectral_cert_module_path,)" not in hardy:
        raise AssertionError("The Hardy checkpoint has extraneous source dependencies.")
    if "_contour_deterministic_module_path" in contour:
        raise AssertionError("The contour moat cache depends on deterministic epsilon code.")
    if "epsilon_report_path" in contour:
        raise AssertionError("Cell 103 still parses epsilon from a prose report.")
    if not all(
        marker in contour
        for marker in (
            "epsilon_certificate_path",
            "branch_image_balanced_response_prefactor_candidate_row_N600_M610.csv",
        )
    ):
        raise AssertionError("Cell 103 does not load the machine-readable epsilon certificate.")
    if "INLINE_MODULE_PATHS" in contour:
        raise AssertionError("The source Cell 103 depends on inline-only state.")
    if not all(
        marker in contour
        for marker in (
            "_directional_endpoint_text",
            "ROUND_FLOOR",
            "ROUND_CEILING",
            "CELL103_MINIMUM_CERTIFIED_MOAT_TEXT",
            "CELL103_MAXIMUM_SMALL_GAIN_TEXT",
        )
    ):
        raise AssertionError(
            "Cell 103 does not use directional presentation endpoints."
        )
    if "blaschke_deformation_certifier_thesis_math.ipynb" not in cell104:
        raise AssertionError("Cell 104 does not package the executed counterpart.")
    if "BLASCHKE_SKIP_ARCHIVE" not in cell104:
        raise AssertionError("Cell 104 cannot defer packaging until a clean commit.")
    if "deferred_dirty_worktree" not in cell104:
        raise AssertionError("Cell 104 does not defer a dirty-worktree archive cleanly.")
    if not all(
        marker in cell104
        for marker in (
            "CELL103_MINIMUM_CERTIFIED_MOAT_TEXT",
            "CELL103_MAXIMUM_SMALL_GAIN_TEXT",
        )
    ):
        raise AssertionError(
            "Cell 104 does not preserve directional contour summary texts."
        )
    metadata = notebook.get("metadata", {}).get(
        "blaschke_deformation_certifier", {}
    )
    if metadata.get("phase2_final_epsilon") != (
        "3.3264433839017426342179265983234893950031511766904056153485116e-20"
    ):
        raise AssertionError("The exact Phase 2 epsilon metadata is stale.")
    for update_number in (89, 98):
        update = _source(
            _unique_cell(cells, f"## Historical notebook update {update_number}")
        )
        if "superseded by Notebook update 99" not in update:
            raise AssertionError(f"Notebook update {update_number} is not historical.")


def _write_notebook(path: Path, notebook: dict[str, Any]) -> None:
    path.write_text(
        nbf.writes(nbf.from_dict(notebook), version=nbf.NO_CONVERT),
        encoding="utf-8",
    )


def refresh_template() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(f"Cannot refresh from missing source notebook {TARGET}.")
    notebook = json.loads(TARGET.read_text(encoding="utf-8"))
    notebook = _clear_runtime_state(_normalise_notebook(notebook))
    _validate(notebook)
    _write_notebook(TEMPLATE, notebook)
    print(f"Refreshed {TEMPLATE}")
    print(f"Template SHA-256: {_sha256(TEMPLATE)}")


def build() -> None:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Missing deployment template {TEMPLATE}.")
    digest = _sha256(TEMPLATE)
    if digest != LOCKED_TEMPLATE_SHA256:
        raise RuntimeError(
            "The deployment template digest changed: "
            f"expected {LOCKED_TEMPLATE_SHA256}, obtained {digest}."
        )
    notebook = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    notebook = _clear_runtime_state(_normalise_notebook(notebook))
    _validate(notebook)
    _write_notebook(TARGET, notebook)
    print(f"Wrote {TARGET}")
    print(f"Locked template SHA-256: {digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-template",
        action="store_true",
        help="Deliberately refresh the output-free template from the current source.",
    )
    arguments = parser.parse_args()
    if arguments.refresh_template:
        refresh_template()
    else:
        build()


if __name__ == "__main__":
    main()
