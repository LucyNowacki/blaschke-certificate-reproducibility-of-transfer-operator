"""Build the phase-preserving, inline thesis-mathematics certifier.

The counterpart starts from the output-free source notebook produced by
``build_blaschke_deformation_certifier.py``.  Repository-local helpers which
implement thesis mathematics are inserted as executable cell modules beside
their first use, and two source-only diagnostic producers are inserted before
their consumers.  The appendix transformer then applies the locked provenance
and plotting rules to curate the 187-cell intermediate into the 139-cell
computational body.  A non-executable dependency map is prepended as Cell 0M
and a non-executable auditor explanation is appended as Cell 110M.  Builder-owned
Cell 108N renders a fail-closed presentation of Cell 107N's final certificate,
while Cell 109N remains a no-op.  The result is the final 141-cell
thesis-mathematics notebook without renumbering or altering source-owned code
cells.  The source notebook is required and is never
reconstructed from a generated counterpart; the provenance direction is
always locked template to source to curated counterpart.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import tomllib
from typing import Any

import nbformat as nbf


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "blaschke_deformation_certifier.ipynb"
OUTPUT = HERE / "blaschke_deformation_certifier_thesis_math.ipynb"
DEPENDENCY_MAP = HERE / "blaschke_deformation_notebook_dependency_map.md"
DEPENDENCY_MAP_SVG = HERE / "blaschke_deformation_notebook_dependency_map.svg"
DEPENDENCY_MAP_CELL_ID = "dependency-map-0m"
DEPENDENCY_MAP_STANDALONE_LINK = "./blaschke_deformation_notebook_dependency_map.svg"
CHAPTER_MATH_MAP = HERE / "numerical_certification_transfer_markdown.toml"
CHAPTER_MATH_MARKER = "<!-- NUMERICS_1_CHAPTER_MATH -->"
CHAPTER_MATH_LABEL = "chap:numerical-certification-transfer"
RESEARCH_THESIS_SOURCE_HEADING = "Research thesis source"
TERMINAL_AUDITOR_CELL_ID = "terminal-certificate-auditor-110m"
PDF_REFERENCE_KINDS = frozenset(
    {
        "Chapter",
        "Section",
        "Subsection",
        "Definition",
        "Remark",
        "Lemma",
        "Proposition",
        "Theorem",
        "Corollary",
        "Algorithm",
        "Equation",
        "Figure",
        "Table",
    }
)


CHAPTER_MATH_STATUS = {
    "infrastructure_only": (
        "Execution, import, path or provenance support; this cell alone "
        "does not certify a mathematical claim."
    ),
    "empirical_diagnostic": (
        "Numerical or sampled evidence only; it is excluded from theorem gates."
    ),
    "presentation_only": (
        "Presentation of retained results; it does not strengthen their proof status."
    ),
    "certified_input_producer": (
        "Reconstructs theorem-facing inputs from displayed source and validated "
        "arithmetic before downstream aggregation."
    ),
    "certified_component": (
        "Validated arithmetic establishes a theorem-facing component bound, "
        "subject to its stated upstream gates."
    ),
    "certified_finite_matrix": (
        "Validated arithmetic encloses the finite Hardy-gauge matrix and its "
        "exact-dyadic replacement."
    ),
    "certified_spectral_transfer": (
        "Complete-contour finite-to-exact Riesz-rank certification."
    ),
    "validation_test": (
        "Tests proof routing and fail-closed behaviour; it supports but does not "
        "replace the certificate."
    ),
}


# The insertion index is the original zero-based cell index before which the
# module must be available. This keeps every helper beside the phase that uses
# it instead of collecting detached implementation material at the end.
INLINE_HELPERS = (
    (10, "mpmath_pf_raw", "mpmath_pf_raw.py", "Phase 1 transfer assembly"),
    (
        12,
        "transfer_spectrum_certification",
        "transfer_spectrum_certification.py",
        "Map-generic certification bridge",
    ),
    (
        46,
        "blaschke_deformation_certification",
        "blaschke_deformation_certification.py",
        "Phase 2 resolved-response and unresolved-input certification",
    ),
    (
        126,
        "blaschke_deformation_spectral_certification",
        "blaschke_deformation_spectral_certification.py",
        "Phase 4 validated Hardy-gauge matrix certification",
    ),
    (
        128,
        "blaschke_deformation_contour_certification",
        "blaschke_deformation_contour_certification.py",
        "Phase 4 contour and Riesz-rank certification",
    ),
    (
        31,
        "blaschke_deformation_phase2_geometry",
        "blaschke_deformation_phase2_geometry.py",
        "Phase 2 complete-boundary geometry reconstruction",
    ),
    (
        31,
        "blaschke_deformation_phase2_transport",
        "blaschke_deformation_phase2_transport.py",
        "Phase 2 finite Chebyshev-gauge transport reconstruction",
    ),
    (
        31,
        "blaschke_deformation_phase2_matrix",
        "blaschke_deformation_phase2_matrix.py",
        "Phase 2 pure-r-scaled matrix reconstruction",
    ),
    (
        31,
        "blaschke_deformation_phase2_pipeline",
        "blaschke_deformation_phase2_pipeline.py",
        "Phase 2 clean-room producer orchestration",
    ),
    (
        31,
        "blaschke_deformation_historical_comparisons",
        "blaschke_deformation_historical_comparisons.py",
        "Source-only Phase 2 historical-design comparison reconstruction",
    ),
    (
        72,
        "hardy_moat_surface_worker",
        "hardy_moat_surface_worker.py",
        "Process-based historical Phase 4 moat-surface sampling",
    ),
    (
        72,
        "blaschke_deformation_historical_phase4",
        "blaschke_deformation_historical_phase4.py",
        "Source-only historical Phase 4 diagnostic reconstruction",
    ),
    (
        121,
        "blaschke_deformation_diagnostic_audits",
        "blaschke_deformation_diagnostic_audits.py",
        "Source-only retained diagnostic-audit reconstruction",
    ),
    (
        44,
        "blaschke_deformation_phase2_finite_m",
        "blaschke_deformation_phase2_finite_m.py",
        "Phase 2 safe finite-order Gauss--Legendre completion",
    ),
    (
        47,
        "blaschke_deformation_phase2_resolved_response",
        "blaschke_deformation_phase2_resolved_response.py",
        "Phase 2 complete-boundary resolved-response completion",
    ),
    (
        48,
        "blaschke_deformation_phase2_final_aggregation",
        "blaschke_deformation_phase2_final_aggregation.py",
        "Phase 2 unresolved-input and final deterministic aggregation",
    ),
    (
        26,
        "blaschke_deformation_phase1_diagnostics",
        "blaschke_deformation_phase1_diagnostics.py",
        "Source-only retained Phase 1 diagnostic reconstruction",
    ),
    (
        120,
        "blaschke_deformation_sampled_schur_diagnostics",
        "blaschke_deformation_sampled_schur_diagnostics.py",
        "Source-only sampled Schur diagnostic reconstruction",
    ),
)

CURATED_INLINE_HELPER_ORDINALS = tuple(range(1, len(INLINE_HELPERS) + 1))


INLINE_HELPER_CELL_LABELS = {
    "mpmath_pf_raw": ("13", "14"),
    "transfer_spectrum_certification": ("15A", "15A"),
    "blaschke_deformation_certification": ("51", "52"),
    "blaschke_deformation_spectral_certification": ("99", "100"),
    "blaschke_deformation_contour_certification": ("103", "104"),
    "blaschke_deformation_phase2_geometry": ("35A", "35A"),
    "blaschke_deformation_phase2_transport": ("35B", "35B"),
    "blaschke_deformation_phase2_matrix": ("35C", "35C"),
    "blaschke_deformation_phase2_pipeline": ("35D", "35D"),
    "blaschke_deformation_historical_comparisons": ("35E", "35E"),
    "hardy_moat_surface_worker": ("78A", "78A"),
    "blaschke_deformation_historical_phase4": ("78B", "78B"),
    "blaschke_deformation_diagnostic_audits": ("95A", "95A"),
    "blaschke_deformation_phase2_finite_m": ("48A", "48A"),
    "blaschke_deformation_phase2_resolved_response": ("52A", "52A"),
    "blaschke_deformation_phase2_final_aggregation": ("54A", "54A"),
    "blaschke_deformation_phase1_diagnostics": ("30A", "30A"),
    "blaschke_deformation_sampled_schur_diagnostics": ("94A", "94A"),
}


PHASE1_DIAGNOSTIC_REBUILD = r'''# Phase 1 retained diagnostic producer.
from Numerics.blaschke_deformation_phase1_diagnostics import (
    Phase1DiagnosticsConfig,
    rebuild_phase1_diagnostics,
)

PHASE1_DIAGNOSTICS_RESULT = rebuild_phase1_diagnostics(
    Phase1DiagnosticsConfig(
        dps=cfg.dps,
        max_power=cfg.max_power,
        max_clusters=RAW_SWEEP_TARGET_CLUSTER_COUNT,
        expected_target_count=RAW_SWEEP_TARGET_CLUSTER_COUNT,
        assembly_workers=cfg.process_workers,
        progress=cfg.progress,
    ),
    map_spec=SELECTED_MAP_SPEC,
    raw_sweep=run_pair_sweep_mpmath_raw,
    reference_clusters=None,
    data_dir=DATA_DIR,
    report_path=(
        REPORT_DIR / "blaschke_deformation_phase1_diagnostics_rebuild.json"
    ),
)
'''


SAMPLED_SCHUR_DIAGNOSTIC_REBUILD = r"""# Retained sampled Schur diagnostic producer.
from Numerics.transfer_spectrum_certification import sampled_schur_envelope
from Numerics.blaschke_deformation_sampled_schur_diagnostics import (
    REPORT_FILENAME as SAMPLED_SCHUR_DIAGNOSTIC_REPORT_FILENAME,
    SampledSchurDiagnosticsConfig,
    rebuild_sampled_schur_diagnostics,
)


def _sampled_schur_kappa(N, r):
    '''Explanation: Changing from scaled Legendre coordinates to packet coordinates can amplify finite errors. This sampled condition number estimates that amplification for diagnostic Schur rows, while the final perturbation proof uses the rigorous transport certificate.
    Functionality: Evaluate the notebook's finite connection condition number at the requested dimension and radius.'''
    value = float(kappa_T_numeric(int(N), str(r)))
    _sampled_schur_kappa.runtime_evidence = tuple(
        KAPPA_T_NUMERIC_BLAS_RUNTIME_EVIDENCE
    )
    return value


KAPPA_T_NUMERIC_BLAS_RUNTIME_EVIDENCE.clear()
_sampled_schur_kappa.runtime_evidence = ()


SAMPLED_SCHUR_DIAGNOSTICS_RESULT = rebuild_sampled_schur_diagnostics(
    SampledSchurDiagnosticsConfig(),
    map_label=SELECTED_MAP_LABEL,
    geometry=geometry_record,
    sampled_schur_envelope=sampled_schur_envelope,
    kappa=_sampled_schur_kappa,
    data_dir=DATA_DIR,
    report_path=REPORT_DIR / SAMPLED_SCHUR_DIAGNOSTIC_REPORT_FILENAME,
)
"""


BOOTSTRAP_SOURCE = r'''# Inline-module support for the thesis-mathematics counterpart.
#
# Each following helper cell contains the ordinary Python source of one local
# mathematical module. The cell magic executes that source in an isolated
# module namespace, registers the module under both import spellings used by
# the original notebook, and materialises a temporary copy so process workers
# can import the same definitions.

import hashlib as _inline_hashlib
import importlib as _inline_importlib
import re as _inline_re
import sys as _inline_sys
import tempfile as _inline_tempfile
import types as _inline_types
from pathlib import Path as _InlinePath

_INLINE_HELPER_DIRECTORY = _InlinePath(
    _inline_tempfile.mkdtemp(prefix="blaschke_thesis_math_")
)
if str(_INLINE_HELPER_DIRECTORY) not in _inline_sys.path:
    _inline_sys.path.insert(0, str(_INLINE_HELPER_DIRECTORY))

INLINE_MODULE_PATHS = {}
INLINE_MODULE_SHA256 = {}


def _install_inline_module(line, cell):
    arguments = line.split()
    if len(arguments) != 2:
        raise ValueError(
            "The inline_module magic requires a module name and SHA-256."
        )
    module_name, expected_sha256 = arguments
    if any(part == "" for part in module_name.split(".")):
        raise ValueError("The inline_module magic requires one dotted module name.")

    module_source = cell
    number_line, separator, unnumbered_source = module_source.partition("\n")
    if separator and _inline_re.fullmatch(r"#\d+[A-Z]*N", number_line):
        module_source = unnumbered_source
        cell = unnumbered_source
    provenance_begin = "# notebook-provenance: begin\n"
    provenance_end = "# notebook-provenance: end\n"
    if module_source.startswith(provenance_begin):
        marker_index = module_source.find(provenance_end)
        if marker_index < 0:
            raise RuntimeError(
                f"Unterminated notebook provenance header for {module_name}."
            )
        module_source = module_source[marker_index + len(provenance_end):]

    actual_sha256 = _inline_hashlib.sha256(module_source.encode("utf-8")).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Inline source digest mismatch for {module_name}: "
            f"expected {expected_sha256}, obtained {actual_sha256}."
        )

    short_name = module_name.rsplit(".", 1)[-1]
    module_path = _INLINE_HELPER_DIRECTORY / f"{short_name}.py"
    module_path.write_text(module_source, encoding="utf-8")

    module = _inline_types.ModuleType(module_name)
    module.__file__ = str(module_path)
    module.__package__ = module_name.rpartition(".")[0]
    module.__source__ = module_source
    module.__source_sha256__ = actual_sha256

    canonical_name = short_name
    qualified_name = f"Numerics.{short_name}"
    previous = {
        canonical_name: _inline_sys.modules.get(canonical_name),
        qualified_name: _inline_sys.modules.get(qualified_name),
    }
    _inline_sys.modules[canonical_name] = module
    _inline_sys.modules[qualified_name] = module
    try:
        exec(compile(cell, str(module_path), "exec"), module.__dict__)
    except Exception:
        for alias, old_module in previous.items():
            if old_module is None:
                _inline_sys.modules.pop(alias, None)
            else:
                _inline_sys.modules[alias] = old_module
        raise

    try:
        numerics_package = _inline_importlib.import_module("Numerics")
    except ModuleNotFoundError:
        numerics_package = _inline_types.ModuleType("Numerics")
        numerics_package.__path__ = []
        _inline_sys.modules["Numerics"] = numerics_package
    setattr(numerics_package, short_name, module)

    INLINE_MODULE_PATHS[canonical_name] = module_path
    INLINE_MODULE_SHA256[canonical_name] = module.__source_sha256__
    print(
        f"Installed inline thesis helper {canonical_name} "
        f"with SHA-256 {module.__source_sha256__}"
    )
    return module


get_ipython().register_magic_function(
    _install_inline_module,
    magic_kind="cell",
    magic_name="inline_module",
)
'''


def _markdown_cell(source: str, cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {"thesis_math_inline_helper": True},
        "source": source,
    }


def _code_cell(source: str, cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {"thesis_math_inline_helper": True},
        "outputs": [],
        "source": source,
    }


def _producer_cell(source: str, cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {"thesis_math_inline_helper": True, "source_producer": True},
        "outputs": [],
        "source": source,
    }


def dependency_map_cell() -> dict[str, Any]:
    """Return the builder-owned, non-executable Cell 0M dependency map."""

    source = DEPENDENCY_MAP.read_text(encoding="utf-8")
    svg = DEPENDENCY_MAP_SVG.read_text(encoding="utf-8")
    if not source.startswith("0M\n\n# Reproducibility dependency map\n"):
        raise AssertionError("The dependency-map source must begin with Cell 0M.")
    if DEPENDENCY_MAP_STANDALONE_LINK not in source:
        raise AssertionError("The standalone dependency map must link its sibling SVG.")
    if not svg.startswith("<svg ") or not svg.rstrip().endswith("</svg>"):
        raise AssertionError("The dependency-map SVG is not complete.")
    return {
        "cell_type": "markdown",
        "id": DEPENDENCY_MAP_CELL_ID,
        "metadata": {"thesis_math_dependency_map": True},
        "source": source,
    }


def _checked_replace(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected one occurrence of {label!r}, found {count}. "
            "The source notebook loader has changed and must be reviewed."
        )
    return source.replace(old, new)


def _source_reference_chain(
    label: str, references: dict[str, dict[str, Any]]
) -> list[str]:
    """Return one integrated-PDF locator path from chapter to cited object."""

    chain: list[str] = []
    seen: set[str] = set()
    current: str | None = label
    while current is not None:
        if current in seen:
            raise RuntimeError(f"Cyclic PDF source-reference ancestry at {current!r}.")
        seen.add(current)
        reference = references.get(current)
        if reference is None:
            raise RuntimeError(f"Missing PDF source reference {current!r}.")
        chain.append(current)
        parent = reference.get("parent")
        current = str(parent) if parent is not None else None
    chain.reverse()
    return chain


def _format_source_reference(
    label: str, reference: dict[str, Any]
) -> str:
    """Format the same numbered locator a thesis-PDF reader sees."""

    kind = str(reference["kind"])
    number = str(reference["number"])
    if kind == "Equation":
        visible = f"Equation ({number})"
    else:
        visible = f"{kind} {number}: *{reference['title']}*"
    return f"- {visible} (`{label}`)."


@lru_cache(maxsize=1)
def _chapter_math_payload() -> dict[str, Any]:
    """Load and validate the chapter-to-notebook Markdown contract."""

    if not CHAPTER_MATH_MAP.is_file():
        raise FileNotFoundError(
            f"Missing chapter mathematics map {CHAPTER_MATH_MAP}."
        )
    payload = tomllib.loads(CHAPTER_MATH_MAP.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.4.0":
        raise RuntimeError("Unsupported chapter mathematics map schema.")
    if (
        payload.get("research_thesis_source_heading")
        != RESEARCH_THESIS_SOURCE_HEADING
    ):
        raise RuntimeError("The research-thesis source heading changed.")
    if payload.get("chapter_label") != CHAPTER_MATH_LABEL:
        raise RuntimeError(
            "The Markdown map is not rooted at the numerical-certification chapter."
        )
    entries = payload.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise RuntimeError("The chapter mathematics map has no entries.")
    references = payload.get("source_references")
    if not isinstance(references, dict) or not references:
        raise RuntimeError("The chapter mathematics map has no PDF source references.")

    for label, reference in references.items():
        if not isinstance(label, str) or not label or not isinstance(reference, dict):
            raise RuntimeError("Invalid PDF source-reference record.")
        kind = reference.get("kind")
        number = reference.get("number")
        title = reference.get("title")
        parent = reference.get("parent")
        if kind not in PDF_REFERENCE_KINDS:
            raise RuntimeError(
                f"PDF source reference {label!r} has invalid kind {kind!r}."
            )
        if not isinstance(number, str) or not number:
            raise RuntimeError(
                f"PDF source reference {label!r} has no formatted number."
            )
        if kind == "Equation":
            if title is not None:
                raise RuntimeError(
                    f"Equation reference {label!r} must not invent a title."
                )
        elif not isinstance(title, str) or not title:
            raise RuntimeError(
                f"PDF source reference {label!r} has no reader-facing title."
            )
        if parent is not None and (not isinstance(parent, str) or not parent):
            raise RuntimeError(
                f"PDF source reference {label!r} has an invalid parent."
            )

    for label in references:
        chain = _source_reference_chain(label, references)
        if references[chain[0]]["kind"] != "Chapter":
            raise RuntimeError(
                f"PDF source reference {label!r} is not rooted at a chapter."
            )

    seen_code_ids: set[str] = set()
    cited_labels = {CHAPTER_MATH_LABEL}
    for markdown_id, raw_entry in entries.items():
        if not isinstance(raw_entry, dict):
            raise RuntimeError(f"Invalid Markdown-map entry {markdown_id!r}.")
        code_ids = raw_entry.get("code_cell_ids")
        labels = raw_entry.get("chapter_labels")
        statuses = raw_entry.get("evidence_status")
        mathematics = raw_entry.get("mathematics")
        if not isinstance(code_ids, list) or not code_ids:
            raise RuntimeError(f"Entry {markdown_id!r} has no code-cell IDs.")
        if not isinstance(labels, list) or not labels:
            raise RuntimeError(f"Entry {markdown_id!r} has no chapter labels.")
        if not isinstance(statuses, list) or not statuses:
            raise RuntimeError(f"Entry {markdown_id!r} has no evidence status.")
        if any(not isinstance(value, str) or not value for value in code_ids):
            raise RuntimeError(f"Entry {markdown_id!r} has invalid code-cell IDs.")
        if any(not isinstance(value, str) or not value for value in labels):
            raise RuntimeError(f"Entry {markdown_id!r} has invalid chapter labels.")
        if any(not isinstance(value, str) or not value for value in statuses):
            raise RuntimeError(f"Entry {markdown_id!r} has invalid evidence statuses.")
        if not isinstance(mathematics, str) or len(mathematics.split()) < 28:
            raise RuntimeError(
                f"Entry {markdown_id!r} needs a detailed mathematical explanation."
            )
        if "$" not in mathematics:
            raise RuntimeError(
                f"Entry {markdown_id!r} must contain rendered Markdown mathematics."
            )
        if any(delimiter in mathematics for delimiter in (r"\(", r"\)", r"\[", r"\]")):
            raise RuntimeError(
                f"Entry {markdown_id!r} must use dollar-delimited Markdown mathematics."
            )
        for placeholder in (
            "with the arguments and return contract used by this numerical stage",
            "placeholder-style Functionality",
        ):
            if placeholder in mathematics:
                raise RuntimeError(
                    f"Entry {markdown_id!r} contains placeholder explanation prose."
                )
        unknown_statuses = set(statuses) - set(CHAPTER_MATH_STATUS)
        if unknown_statuses:
            raise RuntimeError(
                f"Entry {markdown_id!r} has unknown statuses {unknown_statuses}."
            )
        duplicate_code_ids = seen_code_ids.intersection(map(str, code_ids))
        if duplicate_code_ids:
            raise RuntimeError(
                f"Code cells mapped more than once: {sorted(duplicate_code_ids)}."
            )
        seen_code_ids.update(map(str, code_ids))
        cited_labels.update(map(str, labels))

    terminal = payload.get("terminal_auditor")
    if not isinstance(terminal, dict):
        raise RuntimeError("The chapter mathematics map has no terminal auditor.")
    terminal_cell_id = terminal.get("cell_id")
    terminal_code_ids = terminal.get("code_cell_ids")
    terminal_labels = terminal.get("chapter_labels")
    terminal_statuses = terminal.get("evidence_status")
    terminal_mathematics = terminal.get("mathematics")
    if terminal_cell_id != TERMINAL_AUDITOR_CELL_ID:
        raise RuntimeError("The terminal auditor cell ID changed.")
    if not isinstance(terminal_code_ids, list) or not terminal_code_ids:
        raise RuntimeError("The terminal auditor has no interpreted code-cell IDs.")
    if any(not isinstance(value, str) or not value for value in terminal_code_ids):
        raise RuntimeError("The terminal auditor has invalid code-cell IDs.")
    unknown_terminal_code_ids = set(terminal_code_ids) - seen_code_ids
    if unknown_terminal_code_ids:
        raise RuntimeError(
            "The terminal auditor interprets unknown code cells: "
            f"{sorted(unknown_terminal_code_ids)}."
        )
    if not isinstance(terminal_labels, list) or not terminal_labels:
        raise RuntimeError("The terminal auditor has no chapter labels.")
    if any(not isinstance(value, str) or not value for value in terminal_labels):
        raise RuntimeError("The terminal auditor has invalid chapter labels.")
    if not isinstance(terminal_statuses, list) or not terminal_statuses:
        raise RuntimeError("The terminal auditor has no evidence status.")
    unknown_terminal_statuses = set(terminal_statuses) - set(CHAPTER_MATH_STATUS)
    if unknown_terminal_statuses:
        raise RuntimeError(
            "The terminal auditor has unknown statuses "
            f"{unknown_terminal_statuses}."
        )
    if (
        not isinstance(terminal_mathematics, str)
        or len(terminal_mathematics.split()) < 28
        or "$" not in terminal_mathematics
    ):
        raise RuntimeError(
            "The terminal auditor needs a detailed rendered mathematical explanation."
        )
    if any(
        delimiter in terminal_mathematics
        for delimiter in (r"\(", r"\)", r"\[", r"\]")
    ):
        raise RuntimeError(
            "The terminal auditor must use dollar-delimited Markdown mathematics."
        )
    cited_labels.update(map(str, terminal_labels))

    missing_references = cited_labels - set(references)
    if missing_references:
        raise RuntimeError(
            "Cited labels lack integrated-PDF locators: "
            f"{sorted(missing_references)}."
        )
    used_references: set[str] = set()
    for label in cited_labels:
        used_references.update(_source_reference_chain(label, references))
    orphan_references = set(references) - used_references
    if orphan_references:
        raise RuntimeError(
            "Unreferenced integrated-PDF locators: "
            f"{sorted(orphan_references)}."
        )
    fidelity_replacements = payload.get("fidelity_replacements", {})
    if not isinstance(fidelity_replacements, dict):
        raise RuntimeError("Invalid chapter-fidelity replacement table.")
    for markdown_id, replacement in fidelity_replacements.items():
        if (
            not isinstance(replacement, dict)
            or not isinstance(replacement.get("old"), str)
            or not isinstance(replacement.get("new"), str)
        ):
            raise RuntimeError(
                f"Invalid chapter-fidelity replacement for {markdown_id!r}."
            )
    return payload


def _chapter_math_entries() -> dict[str, dict[str, Any]]:
    return _chapter_math_payload()["entries"]


def _terminal_auditor_entry() -> dict[str, Any]:
    return _chapter_math_payload()["terminal_auditor"]


def _chapter_math_source_references_for_entry(entry: dict[str, Any]) -> list[str]:
    """Return deduplicated PDF locators for one mapped Markdown entry."""

    references = _chapter_math_payload()["source_references"]
    ordered: list[str] = []
    seen: set[str] = set()
    for cited_label in [CHAPTER_MATH_LABEL, *entry["chapter_labels"]]:
        for label in _source_reference_chain(cited_label, references):
            if label not in seen:
                seen.add(label)
                ordered.append(label)
    return ordered


def _chapter_math_source_references(markdown_id: str) -> list[str]:
    """Return deduplicated PDF locators, including structural ancestors."""

    return _chapter_math_source_references_for_entry(
        _chapter_math_entries()[markdown_id]
    )


def _chapter_math_block_for_entry(
    entry: dict[str, Any],
    source_references: list[str],
    *,
    code_relation: str,
) -> str:
    """Render one operation-specific chapter-derived mathematical bridge."""

    references = _chapter_math_payload()["source_references"]
    source_locators = "\n".join(
        _format_source_reference(label, references[label])
        for label in source_references
    )
    code_ids = ", ".join(f"`{cell_id}`" for cell_id in entry["code_cell_ids"])
    statuses = "  \n".join(
        f"- `{status}`: {CHAPTER_MATH_STATUS[status]}"
        for status in entry["evidence_status"]
    )
    mathematics = entry["mathematics"].strip()
    return (
        f"{CHAPTER_MATH_MARKER}\n\n"
        "#### Chapter-derived mathematical bridge\n\n"
        f"**{RESEARCH_THESIS_SOURCE_HEADING}.** Integrated `main.pdf` locator, "
        "with the visible "
        "title and number followed by the stable LaTeX label.\n\n"
        f"{source_locators}\n\n"
        f"**Mathematical reading.** {mathematics}\n\n"
        f"**Evidence status.**\n\n{statuses}\n\n"
        f"**{code_relation}.** Stable code-cell IDs: {code_ids}."
    )


def _chapter_math_block(markdown_id: str) -> str:
    """Render one operation-specific chapter-derived mathematical bridge."""

    entry = _chapter_math_entries()[markdown_id]
    return _chapter_math_block_for_entry(
        entry,
        _chapter_math_source_references(markdown_id),
        code_relation="Code covered",
    )


def terminal_auditor_cell() -> dict[str, Any]:
    """Build the visible final Markdown interpretation after Cell 109N."""

    entry = _terminal_auditor_entry()
    bridge = _chapter_math_block_for_entry(
        entry,
        _chapter_math_source_references_for_entry(entry),
        code_relation="Code interpreted",
    )
    return {
        "cell_type": "markdown",
        "id": TERMINAL_AUDITOR_CELL_ID,
        "metadata": {"thesis_math_terminal_auditor": True},
        "source": (
            "110M\n\n"
            "## Final auditor reading of the twenty-four-target spectral certificate\n\n"
            "Cell 107N forms and checks the final certificate. Cell 108N then "
            "reaggregates those checked results into a fail-closed final ladder; "
            "it adds no theorem claim. Cell 109N remains a no-op, so this terminal "
            "interpretation stays directly below the final presentation.\n\n"
            f"{bridge}\n"
        ),
    }


def _augment_markdown_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Apply the stable-ID chapter mathematics map to one Markdown cell."""

    adapted = deepcopy(cell)
    if adapted.get("cell_type") != "markdown":
        return adapted
    cell_id = str(adapted.get("id", ""))
    source = _normalise_source(adapted.get("source", ""))
    replacement = _chapter_math_payload().get("fidelity_replacements", {}).get(
        cell_id
    )
    if replacement is not None:
        source = _checked_replace(
            source,
            replacement["old"],
            replacement["new"],
            label=f"chapter-fidelity correction for {cell_id}",
        )
    entries = _chapter_math_entries()
    if cell_id not in entries:
        adapted["source"] = source
        return adapted

    entry = entries[cell_id]
    heading_from = entry.get("heading_from")
    heading_to = entry.get("heading_to")
    if (heading_from is None) != (heading_to is None):
        raise RuntimeError(
            f"Entry {cell_id!r} must provide both heading replacement fields."
        )
    if heading_from is not None:
        source = _checked_replace(
            source,
            str(heading_from),
            str(heading_to),
            label=f"stable heading for {cell_id}",
        )
    replace_old = entry.get("replace_old")
    replace_new = entry.get("replace_new")
    if (replace_old is None) != (replace_new is None):
        raise RuntimeError(
            f"Entry {cell_id!r} must provide both prose replacement fields."
        )
    if replace_old is not None:
        source = _checked_replace(
            source,
            str(replace_old),
            str(replace_new),
            label=f"chapter-fidelity prose for {cell_id}",
        )
    if CHAPTER_MATH_MARKER in source:
        raise RuntimeError(
            f"Source Markdown {cell_id!r} already contains the generated bridge."
        )
    adapted["source"] = source.rstrip() + "\n\n" + _chapter_math_block(cell_id) + "\n"
    return adapted


def _adapt_original_cell(index: int, cell: dict[str, Any]) -> dict[str, Any]:
    """Point provenance and explicit reloads at the inline module copies."""

    adapted = deepcopy(cell)
    if adapted.get("cell_type") != "code":
        return adapted
    source = "".join(adapted.get("source", []))

    if index == 10:
        source = _checked_replace(
            source,
            "MODULE_PATH_CANDIDATES = [\n",
            "MODULE_PATH_CANDIDATES = [\n"
            "    globals().get(\"INLINE_MODULE_PATHS\", {}).get(\"mpmath_pf_raw\"),\n",
            label="Phase 1 worker path list",
        )
        source = _checked_replace(
            source,
            "if path.exists()), None)",
            "if path is not None and path.exists()), None)",
            label="Phase 1 worker path filter",
        )

    if index == 126:
        replacements = {
            'Path("blaschke_deformation_spectral_certification.py")':
                'INLINE_MODULE_PATHS["blaschke_deformation_spectral_certification"]',
            'Path("Numerics/blaschke_deformation_spectral_certification.py")':
                'INLINE_MODULE_PATHS["blaschke_deformation_spectral_certification"]',
        }
        for old, new in replacements.items():
            source = _checked_replace(source, old, new, label=old)

    if index == 130:
        for short_name in (
            "blaschke_deformation_contour_certification",
            "blaschke_deformation_spectral_certification",
        ):
            for old in (
                f'Path("{short_name}.py")',
                f'Path("Numerics/{short_name}.py")',
            ):
                source = _checked_replace(
                    source,
                    old,
                    f'INLINE_MODULE_PATHS["{short_name}"]',
                    label=old,
                )

    adapted["source"] = source
    return adapted


def _helper_cells(
    module_name: str,
    filename: str,
    phase_description: str,
    ordinal: int,
) -> list[dict[str, Any]]:
    path = HERE / filename
    module_source = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(module_source.encode("utf-8")).hexdigest()
    displayed_labels = INLINE_HELPER_CELL_LABELS.get(module_name)
    if displayed_labels:
        heading_label, code_label = displayed_labels
        heading_prefix = f"{heading_label}M\n\n"
        code_prefix = f"#{code_label}N\n"
    else:
        heading_prefix = ""
        code_prefix = ""
    heading = (
        heading_prefix
        + f"### Inline thesis helper: `{module_name}`\n\n"
        f"This is the complete helper used by **{phase_description}**. It is "
        "placed here so that the mathematical implementation can be read in "
        "the same phase as the formulas and certificate that use it. The "
        "cell is executable: the `inline_module` magic gives the source an "
        "isolated module namespace while leaving the code visible in the "
        "notebook.\n\n"
        f"Source file: `{filename}`  \n"
        f"SHA-256: `{digest}`\n"
    )
    code = (
        f"%%inline_module {module_name} {digest}\n"
        f"{code_prefix}{module_source}"
    )
    return [
        _markdown_cell(heading, f"inline-helper-heading-{ordinal}"),
        _code_cell(code, f"inline-helper-source-{ordinal}"),
    ]


def build() -> dict[str, Any]:
    source_notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    original_cells = source_notebook["cells"]
    insertions: dict[int, list[dict[str, Any]]] = {}

    bootstrap_heading = _markdown_cell(
        "### Executable inline mathematical helpers\n\n"
        "The counterpart retains the complete original notebook, including "
        "its Markdown, LaTeX, phase structure, diagnostics, figures and stored "
        "results. The additional cells expose the repository-local helpers "
        "that implement the symmetric Blaschke deformation mathematics. "
        "Plotting workers and archive packaging remain external because they "
        "are outside this counterpart's mathematical scope.\n",
        "inline-helper-bootstrap-heading",
    )
    bootstrap_code = _code_cell(
        BOOTSTRAP_SOURCE,
        "inline-helper-bootstrap-code",
    )
    insertions[10] = [bootstrap_heading, bootstrap_code]

    for ordinal, (index, module_name, filename, phase_description) in enumerate(
        INLINE_HELPERS,
        start=1,
    ):
        insertions.setdefault(index, []).extend(
            _helper_cells(module_name, filename, phase_description, ordinal)
        )
    insertions.setdefault(26, []).append(
        _producer_cell(PHASE1_DIAGNOSTIC_REBUILD, "producer-phase1-diagnostics")
    )
    insertions.setdefault(120, []).append(
        _producer_cell(
            SAMPLED_SCHUR_DIAGNOSTIC_REBUILD,
            "producer-sampled-schur-diagnostics",
        )
    )
    rebuilt_cells = []
    for index, cell in enumerate(original_cells):
        rebuilt_cells.extend(insertions.get(index, []))
        rebuilt_cells.append(_adapt_original_cell(index, cell))
    rebuilt_cells.extend(insertions.get(len(original_cells), []))

    counterpart = deepcopy(source_notebook)
    counterpart["cells"] = rebuilt_cells
    # Notebook-level metadata, including widget state, is intentionally kept
    # byte-for-byte equivalent at the parsed-object level.  The human-readable
    # helper manifest is provided by the inserted phase-local headings.
    counterpart["metadata"] = deepcopy(source_notebook.get("metadata", {}))
    return counterpart


def _normalise_source(source: Any) -> str:
    return "".join(source) if isinstance(source, list) else str(source)


def _normalise_attachments(value: Any) -> dict[str, dict[str, str]]:
    """Normalise nbformat's list-versus-string MIME payload representation."""

    return {
        str(name): {
            str(mime): _normalise_source(payload)
            for mime, payload in dict(mime_bundle).items()
        }
        for name, mime_bundle in dict(value or {}).items()
    }


def _without_notebook_provenance(source: str) -> str:
    """Remove only the appendix provenance preamble following a cell magic."""

    lines = source.splitlines(keepends=True)
    if not lines or not lines[0].lstrip().startswith("%%inline_module"):
        return source
    begin = "# notebook-provenance: begin"
    end = "# notebook-provenance: end"
    begin_index = 1
    if (
        len(lines) >= 3
        and re.fullmatch(r"#\d+[A-Z]*N", lines[1].rstrip("\r\n"))
        and lines[2].rstrip("\r\n") == begin
    ):
        begin_index = 2
    if len(lines) <= begin_index or lines[begin_index].rstrip("\r\n") != begin:
        return source
    end_index = next(
        (
            index
            for index in range(begin_index + 1, len(lines))
            if lines[index].rstrip("\r\n") == end
        ),
        None,
    )
    if end_index is None:
        raise AssertionError("Inline helper has an unterminated notebook provenance header.")
    return "".join(lines[:begin_index]) + "".join(lines[end_index + 1:])


def _without_chapter_math(source: str) -> str:
    """Remove the generated chapter bridge before helper-provenance comparison."""

    separator = "\n\n" + CHAPTER_MATH_MARKER
    if separator not in source:
        return source
    prefix, _, _ = source.partition(separator)
    return prefix.rstrip() + "\n"


def validate_inline_helper_sync(
    counterpart: dict[str, Any],
    *,
    helper_ordinals: tuple[int, ...] | None = None,
    allow_notebook_provenance: bool = False,
    allow_chapter_math: bool = True,
) -> None:
    """Require every inline source and displayed digest to match its file."""

    cells = counterpart["cells"]
    positions: dict[str, list[int]] = {}
    for index, cell in enumerate(cells):
        positions.setdefault(str(cell.get("id", "")), []).append(index)

    selected_ordinals = set(helper_ordinals or range(1, len(INLINE_HELPERS) + 1))
    for ordinal, (_, module_name, filename, phase_description) in enumerate(INLINE_HELPERS, start=1):
        if ordinal not in selected_ordinals:
            continue
        heading_id = f"inline-helper-heading-{ordinal}"
        source_id = f"inline-helper-source-{ordinal}"
        heading_positions = positions.get(heading_id, [])
        source_positions = positions.get(source_id, [])
        if len(heading_positions) != 1 or len(source_positions) != 1:
            raise AssertionError(
                f"Expected one heading and one source cell for {module_name}."
            )
        heading_index = heading_positions[0]
        source_index = source_positions[0]
        if source_index != heading_index + 1:
            raise AssertionError(
                f"The heading and source cells for {module_name} are not adjacent."
            )

        expected_heading, expected_source = _helper_cells(
            module_name,
            filename,
            phase_description,
            ordinal,
        )
        actual_heading = _normalise_source(cells[heading_index].get("source", ""))
        actual_source = _normalise_source(cells[source_index].get("source", ""))
        if allow_chapter_math:
            actual_heading = _without_chapter_math(actual_heading)
        if allow_notebook_provenance:
            actual_source = _without_notebook_provenance(actual_source)
        if actual_heading != _normalise_source(expected_heading["source"]):
            raise AssertionError(
                f"Displayed source provenance does not match {filename}."
            )
        if actual_source != _normalise_source(expected_source["source"]):
            raise AssertionError(
                f"Inline source does not match standalone helper {filename}."
            )


def validate_counterpart(counterpart: dict[str, Any]) -> None:
    """Require complete preservation of the original narrative and outputs."""

    original = json.loads(SOURCE.read_text(encoding="utf-8"))
    original_cells = original["cells"]
    counterpart_cells = counterpart["cells"]
    inserted = [
        cell for cell in counterpart_cells
        if cell.get("metadata", {}).get("thesis_math_inline_helper")
    ]
    retained = [
        cell for cell in counterpart_cells
        if not cell.get("metadata", {}).get("thesis_math_inline_helper")
    ]
    if len(retained) != len(original_cells):
        raise AssertionError("An original notebook cell was added, removed or duplicated.")
    if len(inserted) != 4 + 2 * len(INLINE_HELPERS):
        raise AssertionError("The inline-helper insertion count is incorrect.")
    validate_inline_helper_sync(counterpart)
    if counterpart.get("metadata", {}) != original.get("metadata", {}):
        raise AssertionError("Notebook metadata or widget state changed.")

    for index, (before, after) in enumerate(zip(original_cells, retained)):
        if before.get("cell_type") != after.get("cell_type"):
            raise AssertionError(f"Cell type changed at original index {index}.")
        if before.get("id") != after.get("id"):
            raise AssertionError(f"Cell identifier changed at original index {index}.")
        if before.get("metadata", {}) != after.get("metadata", {}):
            raise AssertionError(f"Cell metadata changed at original index {index}.")
        if before.get("execution_count") != after.get("execution_count"):
            raise AssertionError(f"Execution count changed at original index {index}.")
        if before.get("outputs", []) != after.get("outputs", []):
            raise AssertionError(f"Stored outputs changed at original index {index}.")

        before_source = _normalise_source(before.get("source", ""))
        after_source = _normalise_source(after.get("source", ""))
        expected_source = _normalise_source(
            _adapt_original_cell(index, before).get("source", "")
        )
        if after_source != expected_source:
            raise AssertionError(f"Unexpected source change at original index {index}.")
        if index in {10, 126, 130} and before_source == after_source:
            raise AssertionError(f"Expected loader adaptation missing at index {index}.")

    original_png = []
    retained_png = []
    for cells, destination in (
        (original_cells, original_png),
        (retained, retained_png),
    ):
        for cell in cells:
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                if "image/png" in data:
                    payload = _normalise_source(data["image/png"])
                    destination.append(
                        hashlib.sha256(payload.encode("ascii")).hexdigest()
                    )
    if original_png != retained_png:
        raise AssertionError("A stored PNG plot payload changed.")


def build_curated() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the output-free 141-cell thesis notebook in memory."""

    import prepare_blaschke_deformation_thesis_appendix as preparation

    full_counterpart = build()
    validate_counterpart(full_counterpart)
    provenance_records, _ = preparation.load_provenance_records(
        preparation.DEFAULT_PROVENANCE,
        full_counterpart,
    )
    plotting_replacements, _ = preparation.load_plotting_replacements(
        preparation.DEFAULT_PLOTTING_REPLACEMENTS,
        full_counterpart,
    )
    preparation.validate_plotting_replacement_coverage(plotting_replacements)
    curated, report = preparation.transform_notebook(
        full_counterpart,
        provenance_records,
        plotting_replacements,
        require_stored_outputs=False,
    )
    validate_inline_helper_sync(
        curated,
        helper_ordinals=CURATED_INLINE_HELPER_ORDINALS,
        allow_notebook_provenance=True,
    )
    curated["cells"] = [
        _augment_markdown_cell(cell)
        if cell.get("cell_type") == "markdown"
        else cell
        for cell in curated["cells"]
    ]
    curated["cells"].insert(0, dependency_map_cell())
    curated["cells"].append(terminal_auditor_cell())
    return curated, report


def validate_chapter_math_coverage(counterpart: dict[str, Any]) -> None:
    """Require one mapped mathematical owner for every retained code-cell group."""

    entries = _chapter_math_entries()
    references = _chapter_math_payload()["source_references"]
    cells = counterpart.get("cells", [])
    actual_groups: dict[str, list[str]] = {}
    markdown_sources: dict[str, str] = {}
    owner: str | None = None
    for cell in cells:
        cell_id = str(cell.get("id", ""))
        if cell.get("cell_type") == "markdown":
            owner = cell_id
            markdown_sources[cell_id] = _normalise_source(cell.get("source", ""))
        elif cell.get("cell_type") == "code":
            if owner is None:
                raise AssertionError(f"Code cell {cell_id!r} has no Markdown owner.")
            actual_groups.setdefault(owner, []).append(cell_id)

    expected_groups = {
        markdown_id: [str(value) for value in entry["code_cell_ids"]]
        for markdown_id, entry in entries.items()
    }
    if len(expected_groups) != 52:
        raise AssertionError("The final notebook must retain 52 mapped Markdown owners.")
    if sum(map(len, expected_groups.values())) != 68:
        raise AssertionError("The Markdown map must cover all 68 code cells.")
    if actual_groups != expected_groups:
        missing = sorted(set(actual_groups) - set(expected_groups))
        stale = sorted(set(expected_groups) - set(actual_groups))
        mismatched = sorted(
            key
            for key in set(actual_groups).intersection(expected_groups)
            if actual_groups[key] != expected_groups[key]
        )
        raise AssertionError(
            "Chapter mathematics ownership drifted: "
            f"unmapped={missing}, stale={stale}, groups={mismatched}."
        )

    for markdown_id in expected_groups:
        source = markdown_sources.get(markdown_id, "")
        if source.count(CHAPTER_MATH_MARKER) != 1:
            raise AssertionError(
                f"Markdown owner {markdown_id!r} lacks exactly one generated bridge."
            )
        if f"`{CHAPTER_MATH_LABEL}`" not in source:
            raise AssertionError(
                f"Markdown owner {markdown_id!r} is not rooted at the chapter label."
            )
        if (
            f"**{RESEARCH_THESIS_SOURCE_HEADING}.** Integrated `main.pdf` locator"
            not in source
        ):
            raise AssertionError(
                f"Markdown owner {markdown_id!r} lacks its PDF-facing source heading."
            )
        for forbidden_heading in (
            "**Thesis source.**",
            "**Distilled thesis source.**",
        ):
            if forbidden_heading in source:
                raise AssertionError(
                    f"Markdown owner {markdown_id!r} mixes research and future "
                    "source headings."
                )
        for label in entries[markdown_id]["chapter_labels"]:
            if f"`{label}`" not in source:
                raise AssertionError(
                    f"Markdown owner {markdown_id!r} lost chapter label {label!r}."
                )
        for label in _chapter_math_source_references(markdown_id):
            rendered = _format_source_reference(label, references[label])
            if rendered not in source:
                raise AssertionError(
                    f"Markdown owner {markdown_id!r} lost PDF locator {label!r}."
                )


def validate_curated_counterpart(counterpart: dict[str, Any]) -> None:
    """Validate the stable size, terminal numbering and helper digests."""

    cells = counterpart.get("cells", [])
    if len(cells) != 141:
        raise AssertionError("The final thesis counterpart must contain 141 cells.")
    expected_map = dependency_map_cell()
    actual_map = cells[0] if cells else {}
    if (
        actual_map.get("cell_type") != expected_map["cell_type"]
        or actual_map.get("id") != expected_map["id"]
        or actual_map.get("metadata", {}) != expected_map["metadata"]
        or _normalise_attachments(actual_map.get("attachments", {}))
        != _normalise_attachments(expected_map.get("attachments", {}))
        or _normalise_source(actual_map.get("source", ""))
        != _normalise_source(expected_map["source"])
    ):
        raise AssertionError("The builder-owned Cell 0M dependency map is missing or stale.")
    if sum(
        cell.get("metadata", {}).get("thesis_math_dependency_map") is True
        for cell in cells
    ) != 1:
        raise AssertionError("The final notebook must contain exactly one Cell 0M map.")
    code_count = sum(
        cell.get("cell_type") == "code"
        for cell in cells
    )
    if code_count != 68:
        raise AssertionError("The curated thesis counterpart must contain 68 code cells.")
    import prepare_blaschke_deformation_thesis_appendix as preparation

    expected_terminal = dict(preparation.TERMINAL_NUMBERED_CELLS)
    actual_terminal = {
        str(cell.get("id")): _normalise_source(cell.get("source", ""))
        for cell in counterpart.get("cells", [])
        if str(cell.get("id")) in expected_terminal
    }
    if actual_terminal != expected_terminal:
        raise AssertionError("The Cell 108N or Cell 109N source changed.")
    final_cell = cells[-1]
    if (
        final_cell.get("cell_type") != "markdown"
        or str(final_cell.get("id")) != TERMINAL_AUDITOR_CELL_ID
        or final_cell.get("metadata", {}).get("thesis_math_terminal_auditor")
        is not True
        or _normalise_source(final_cell.get("source", ""))
        != _normalise_source(terminal_auditor_cell()["source"])
    ):
        raise AssertionError(
            "The visible terminal Cell 110M auditor explanation is missing or stale."
        )
    positions = {
        str(cell.get("id", "")): index
        for index, cell in enumerate(cells)
    }
    if positions["128b5369"] != len(cells) - 2:
        raise AssertionError("Cell 110M must immediately follow Cell 109N.")
    validate_inline_helper_sync(
        counterpart,
        helper_ordinals=CURATED_INLINE_HELPER_ORDINALS,
        allow_notebook_provenance=True,
        allow_chapter_math=True,
    )
    validate_chapter_math_coverage(counterpart)
    for ordinal, producer_id in (
        (17, "producer-phase1-diagnostics"),
        (18, "producer-sampled-schur-diagnostics"),
    ):
        heading_id = f"inline-helper-heading-{ordinal}"
        source_id = f"inline-helper-source-{ordinal}"
        expected = (
            positions[heading_id],
            positions[source_id],
            positions[producer_id],
        )
        if expected[1] != expected[0] + 1 or expected[2] != expected[1] + 1:
            raise AssertionError(
                f"The producer {producer_id} must immediately follow helper {ordinal}."
            )


def _contains_execution_state(notebook: dict[str, Any]) -> bool:
    """Return whether any code cell carries an execution count or output."""

    return any(
        cell.get("cell_type") == "code"
        and (
            cell.get("execution_count") is not None
            or bool(cell.get("outputs", []))
        )
        for cell in notebook.get("cells", [])
    )


def _merge_preserved_execution_state(
    current: dict[str, Any],
    preserved: dict[str, Any],
) -> dict[str, Any]:
    """Replace sources while retaining one stable-ID notebook's execution evidence."""

    current_cells = current.get("cells", [])
    preserved_cells = preserved.get("cells", [])
    current_signature = [
        (str(cell.get("id", "")), str(cell.get("cell_type", "")))
        for cell in current_cells
    ]
    preserved_signature = [
        (str(cell.get("id", "")), str(cell.get("cell_type", "")))
        for cell in preserved_cells
    ]
    appended_terminal = (
        len(current_signature) == len(preserved_signature) + 1
        and current_signature[:-1] == preserved_signature
        and current_signature[-1]
        == (TERMINAL_AUDITOR_CELL_ID, "markdown")
    )
    if current_signature != preserved_signature and not appended_terminal:
        raise RuntimeError(
            "Cannot preserve execution state: notebook cell IDs or types have drifted."
        )
    if not _contains_execution_state(preserved):
        raise RuntimeError(
            "Cannot preserve execution state from an output-free notebook."
        )

    merged = deepcopy(preserved)
    retained_current_cells = (
        current_cells[:-1] if appended_terminal else current_cells
    )
    for source_cell, merged_cell in zip(retained_current_cells, merged["cells"]):
        merged_cell["source"] = deepcopy(source_cell.get("source", ""))
        if "attachments" in source_cell:
            merged_cell["attachments"] = deepcopy(source_cell["attachments"])
        else:
            merged_cell.pop("attachments", None)

        source_metadata = deepcopy(source_cell.get("metadata", {}))
        execution_metadata = merged_cell.get("metadata", {}).get("execution")
        if execution_metadata is not None:
            source_metadata["execution"] = deepcopy(execution_metadata)
        merged_cell["metadata"] = source_metadata
    if appended_terminal:
        merged["cells"].append(deepcopy(current_cells[-1]))

    merged["nbformat"] = current.get("nbformat", merged.get("nbformat"))
    merged["nbformat_minor"] = current.get(
        "nbformat_minor", merged.get("nbformat_minor")
    )
    metadata = deepcopy(preserved.get("metadata", {}))
    metadata.update(deepcopy(current.get("metadata", {})))
    # A source refresh invalidates any statement that the preserved outputs
    # were executed from, and raw-compared against, these newly installed
    # sources.  Publication normalization may recreate this metadata only
    # after validating an external hash-bound raw-comparison receipt.
    metadata.pop("source_sync_after_execution", None)
    merged["metadata"] = metadata
    return merged


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the output-free 141-cell thesis-mathematics notebook."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="Destination notebook (default: the canonical thesis-mathematics path).",
    )
    parser.add_argument(
        "--preserve-execution-from",
        type=Path,
        help=(
            "Stable-ID executed notebook whose counts, outputs, widget state and "
            "execution metadata are retained while current sources are installed."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"Missing source notebook {SOURCE}; run "
            "build_blaschke_deformation_certifier.py first."
        )
    source_bytes_before = SOURCE.read_bytes()
    counterpart, report = build_curated()
    validate_curated_counterpart(counterpart)

    output = args.output.resolve()
    if args.preserve_execution_from is not None:
        preserved_path = args.preserve_execution_from.resolve()
        if not preserved_path.is_file():
            raise FileNotFoundError(
                f"Missing execution-state notebook {preserved_path}."
            )
        preserved = json.loads(preserved_path.read_text(encoding="utf-8"))
        counterpart = _merge_preserved_execution_state(counterpart, preserved)
        validate_curated_counterpart(counterpart)
    elif output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if _contains_execution_state(existing):
            raise RuntimeError(
                "Refusing to overwrite an executed notebook. Use --output for an "
                "isolated build or --preserve-execution-from for an explicit "
                "stable-ID source refresh."
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        nbf.writes(nbf.from_dict(counterpart), version=nbf.NO_CONVERT),
        encoding="utf-8",
    )
    serialised = json.loads(output.read_text(encoding="utf-8"))
    validate_curated_counterpart(serialised)
    source_bytes_after = SOURCE.read_bytes()
    source_digest_after = hashlib.sha256(source_bytes_after).hexdigest()
    if source_bytes_after != source_bytes_before:
        raise RuntimeError("The source notebook changed while building its counterpart.")
    print(f"Wrote {output}")
    print(f"Curated cells: {report['retained_cell_count']}")
    print(f"Source notebook SHA-256 remains {source_digest_after}")
    print(f"Counterpart cells: {len(counterpart['cells'])}")


if __name__ == "__main__":
    main()
