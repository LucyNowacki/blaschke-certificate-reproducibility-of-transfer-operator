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
    "59c3a46185212e5545d54d88984f69abf0e8612ab017e6e448d48780f9eccd4c"
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
# coherent, branchwise and whole-ellipse input selection has been made.
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
    "noncollocation_rss_without_collocation": _phase2_export_upper_float(
        PHASE2_FINAL_CERT["noncollocation_rss"]
    ),
    "tail_floor_without_collocation": _phase2_export_upper_float(
        PHASE2_FINAL_CERT["noncollocation_rss"]
    ),
    "status": (
        "authoritative selected resolved-response row with final coherent "
        "input and best certified matrix aggregation"
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
    "epsilon_cert_text": str(PHASE2_FINAL_CERT["epsilon"]),
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
    f"Final selected matrix contribution: {mp.nstr(PHASE2_FINAL_CERT['collocation'], 50)}.",
    f"Matrix selection: {PHASE2_FINAL_CERT['matrix_selection']}.",
    f"Final deterministic epsilon: {mp.nstr(PHASE2_FINAL_CERT['epsilon'], 50)}.",
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


def _normalise_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    notebook = deepcopy(notebook)
    cells = notebook["cells"]

    cell24b = _unique_cell(cells, "# Cell 24B\n")
    source = _source(cell24b)
    inline_path = (
        '_rp_cert_module_path = '
        'INLINE_MODULE_PATHS["blaschke_deformation_certification"]'
    )
    ordinary_path = (
        "_rp_cert_module_path = (\n"
        "    Path(\"blaschke_deformation_certification.py\")\n"
        "    if Path.cwd().name == \"Numerics\"\n"
        "    else Path(\"Numerics/blaschke_deformation_certification.py\")\n"
        ")"
    )
    if inline_path in source:
        source = source.replace(inline_path, ordinary_path, 1)
    source = source.replace(
        "str(_rp_cert_module_path): _rp_sha256(_rp_cert_module_path),",
        "_rp_cert_module_path.name: _rp_sha256(_rp_cert_module_path),",
        1,
    )
    if "phase2_aggregation_status" not in source:
        source = source.replace(
            '"total_certified": True,\n    "status": (',
            '"total_certified": True,\n'
            '    "phase2_aggregation_status": "provisional before Cell 24C input refresh",\n'
            '    "status": (',
            1,
        )
    source = source.replace(
        'f"Current deterministic epsilon: {_rp_upper_float(_rp_new_epsilon_current):.17e}.",',
        'f"Provisional deterministic epsilon before Cell 24C: {_rp_upper_float(_rp_new_epsilon_current):.17e}.",',
        1,
    )
    source = source.replace(
        '"  current deterministic epsilon = "',
        '"  provisional epsilon before Cell 24C = "',
        1,
    )
    _set_source(cell24b, source)

    cell24c = _unique_cell(cells, "# Cell 24C\n")
    source = _source(cell24c)
    if "# Cell 24C is the authoritative Phase 2 aggregation point." not in source:
        anchor = 'EPS_CERT_X = PHASE2_FINAL_CERT["epsilon"]\n'
        source = _checked_replace(
            source,
            anchor,
            anchor + "\n" + FINAL_PHASE2_REFRESH + "\n",
            label="Cell 24C final aggregate anchor",
        )
    _set_source(cell24c, source)

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
if os.environ.get("BLASCHKE_SKIP_ARCHIVE", "0") in {"1", "true", "True"}:
    reproducibility_bundle = {
        "archive_path": "deferred until clean-commit finalisation",
        "manifest_path": str(_repro_plan_path),
        "checksum_path": "deferred until clean-commit finalisation",
        "archive_sha256": "deferred",
        "archive_bytes": 0,
        "packaged_file_count": 0,
        "repository_commit": "deferred",
        "repository_dirty": None,
        "execution_status": "deferred_until_clean_commit",
    }
else:
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
        "phase2_final_epsilon": "3.32644338390174263421785759832e-20",
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
    cell24b = _source(_unique_cell(cells, "# Cell 24B\n"))
    cell24c = _source(_unique_cell(cells, "# Cell 24C\n"))
    hardy = _source(_unique_cell(cells, "# Cell 102A\n"))
    contour = _source(_unique_cell(cells, "# Cell 103\n"))
    cell104 = _source(_unique_cell(cells, "# Cell 104\n"))
    required_24b = (
        "ResolvedResponseCertificateConfig",
        "certify_resolved_response_rows",
        "C_resp_coherent_packet_cert_u",
        "provisional epsilon before Cell 24C",
    )
    if not all(marker in cell24b for marker in required_24b):
        raise AssertionError("Cell 24B is not the coherent resolved-response producer.")
    if "INLINE_MODULE_PATHS" in cell24b:
        raise AssertionError("The source Cell 24B depends on inline-only state.")
    if "_rp_cert_module_path.name: _rp_sha256(_rp_cert_module_path)" not in cell24b:
        raise AssertionError("Cell 24B does not use a stable helper provenance key.")
    if "authoritative Phase 2 aggregation point" not in cell24c:
        raise AssertionError("Cell 24C does not refresh the provisional artefacts.")
    if "_hardy_source_files = (_spectral_cert_module_path,)" not in hardy:
        raise AssertionError("The Hardy checkpoint has extraneous source dependencies.")
    if "_contour_deterministic_module_path" in contour:
        raise AssertionError("The contour moat cache depends on deterministic epsilon code.")
    if "INLINE_MODULE_PATHS" in contour:
        raise AssertionError("The source Cell 103 depends on inline-only state.")
    if "blaschke_deformation_certifier_thesis_math.ipynb" not in cell104:
        raise AssertionError("Cell 104 does not package the executed counterpart.")
    if "BLASCHKE_SKIP_ARCHIVE" not in cell104:
        raise AssertionError("Cell 104 cannot defer packaging until a clean commit.")
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
