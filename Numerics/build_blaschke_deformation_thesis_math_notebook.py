"""Build the phase-preserving, inline thesis-mathematics certifier.

The counterpart starts from the output-free source notebook produced by
``build_blaschke_deformation_certifier.py``.  Repository-local helpers which
implement thesis mathematics are inserted as executable cell modules beside
their first use, and two source-only diagnostic producers are inserted before
their consumers.  The appendix transformer then applies the locked provenance
and plotting rules to curate the 187-cell intermediate into the 139-cell
computational body.  A non-executable dependency map is prepended as Cell 0M,
giving the final 140-cell thesis-mathematics notebook without renumbering or
altering any existing cell.  The source notebook is required and is never
reconstructed from a generated counterpart; the provenance direction is
always locked template to source to curated counterpart.
"""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import nbformat as nbf


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "blaschke_deformation_certifier.ipynb"
OUTPUT = HERE / "blaschke_deformation_certifier_thesis_math.ipynb"
DEPENDENCY_MAP = HERE / "blaschke_deformation_notebook_dependency_map.md"
DEPENDENCY_MAP_SVG = HERE / "blaschke_deformation_notebook_dependency_map.svg"
DEPENDENCY_MAP_CELL_ID = "dependency-map-0m"
DEPENDENCY_MAP_ATTACHMENT = "blaschke-deformation-dependency-map.svg"


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
    return float(kappa_T_numeric(int(N), str(r)))


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
    if f"attachment:{DEPENDENCY_MAP_ATTACHMENT}" not in source:
        raise AssertionError("Cell 0M must display its attached dependency-map SVG.")
    if not svg.startswith("<svg ") or not svg.rstrip().endswith("</svg>"):
        raise AssertionError("The dependency-map attachment is not a complete SVG.")
    return {
        "attachments": {
            DEPENDENCY_MAP_ATTACHMENT: {
                "image/svg+xml": [
                    base64.b64encode(svg.encode("utf-8")).decode("ascii")
                ],
            },
        },
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


def validate_inline_helper_sync(
    counterpart: dict[str, Any],
    *,
    helper_ordinals: tuple[int, ...] | None = None,
    allow_notebook_provenance: bool = False,
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

    adapted_indices = {10, 126, 130}
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
        if index in adapted_indices:
            if before_source == after_source:
                raise AssertionError(f"Expected loader adaptation missing at index {index}.")
        elif before_source != after_source:
            raise AssertionError(f"Unexpected source change at original index {index}.")

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
    """Build the output-free 140-cell thesis notebook in memory."""

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
    curated["cells"].insert(0, dependency_map_cell())
    return curated, report


def validate_curated_counterpart(counterpart: dict[str, Any]) -> None:
    """Validate the stable size, terminal numbering and helper digests."""

    cells = counterpart.get("cells", [])
    if len(cells) != 140:
        raise AssertionError("The final thesis counterpart must contain 140 cells.")
    expected_map = dependency_map_cell()
    actual_map = cells[0] if cells else {}
    if (
        actual_map.get("cell_type") != expected_map["cell_type"]
        or actual_map.get("id") != expected_map["id"]
        or actual_map.get("metadata", {}) != expected_map["metadata"]
        or _normalise_attachments(actual_map.get("attachments", {}))
        != _normalise_attachments(expected_map["attachments"])
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
    expected_terminal = {
        "3e8b784c": "#108N\n",
        "128b5369": "#109N\n",
    }
    actual_terminal = {
        str(cell.get("id")): _normalise_source(cell.get("source", ""))
        for cell in counterpart.get("cells", [])
        if str(cell.get("id")) in expected_terminal
    }
    if actual_terminal != expected_terminal:
        raise AssertionError("The terminal Cell 108 and Cell 109 labels changed.")
    validate_inline_helper_sync(
        counterpart,
        helper_ordinals=CURATED_INLINE_HELPER_ORDINALS,
        allow_notebook_provenance=True,
    )
    positions = {
        str(cell.get("id", "")): index
        for index, cell in enumerate(cells)
    }
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
    if current_signature != preserved_signature:
        raise RuntimeError(
            "Cannot preserve execution state: notebook cell IDs or types have drifted."
        )
    if not _contains_execution_state(preserved):
        raise RuntimeError(
            "Cannot preserve execution state from an output-free notebook."
        )

    merged = deepcopy(preserved)
    for source_cell, merged_cell in zip(current_cells, merged["cells"]):
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

    merged["nbformat"] = current.get("nbformat", merged.get("nbformat"))
    merged["nbformat_minor"] = current.get(
        "nbformat_minor", merged.get("nbformat_minor")
    )
    metadata = deepcopy(preserved.get("metadata", {}))
    metadata.update(deepcopy(current.get("metadata", {})))
    metadata["source_sync_after_execution"] = {
        "date": "2026-08-29",
        "scope": "mathematical docstrings and fail-closed Phase 2 gate propagation",
        "stored_output_status": (
            "retained historical outputs; not evidence for the post-sync sources "
            "until a full clean-room replay"
        ),
    }
    merged["metadata"] = metadata
    return merged


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the output-free 140-cell thesis-mathematics notebook."
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
