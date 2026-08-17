"""Build the phase-preserving, inline thesis-mathematics certifier.

The counterpart starts from the output-free source notebook produced by
``build_blaschke_deformation_certifier.py``. Every original Markdown cell,
code cell and metadata item is retained in its original order. The
repository-local helpers which implement thesis mathematics are then inserted
as executable cell modules immediately before their first use. The source
notebook is required and is never reconstructed from a generated counterpart;
the provenance direction is always locked template to source to counterpart.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import nbformat as nbf


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "blaschke_deformation_certifier.ipynb"
OUTPUT = HERE / "blaschke_deformation_certifier_thesis_math.ipynb"


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
        "Phase 2 deterministic unresolved-tail certification",
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
)


BOOTSTRAP_SOURCE = r'''# Inline-module support for the thesis-mathematics counterpart.
#
# Each following helper cell contains the ordinary Python source of one local
# mathematical module. The cell magic executes that source in an isolated
# module namespace, registers the module under both import spellings used by
# the original notebook, and materialises a temporary copy so process workers
# can import the same definitions.

import hashlib as _inline_hashlib
import importlib as _inline_importlib
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

    actual_sha256 = _inline_hashlib.sha256(cell.encode("utf-8")).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Inline source digest mismatch for {module_name}: "
            f"expected {expected_sha256}, obtained {actual_sha256}."
        )

    short_name = module_name.rsplit(".", 1)[-1]
    module_path = _INLINE_HELPER_DIRECTORY / f"{short_name}.py"
    module_path.write_text(cell, encoding="utf-8")

    module = _inline_types.ModuleType(module_name)
    module.__file__ = str(module_path)
    module.__package__ = module_name.rpartition(".")[0]
    module.__source__ = cell
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

    if index in (47, 48):
        source = source.replace(
            'Path("blaschke_deformation_certification.py")',
            'INLINE_MODULE_PATHS["blaschke_deformation_certification"]',
        )
        source = source.replace(
            'Path("Numerics/blaschke_deformation_certification.py")',
            'INLINE_MODULE_PATHS["blaschke_deformation_certification"]',
        )
        if source.count('INLINE_MODULE_PATHS["blaschke_deformation_certification"]') != 2:
            raise RuntimeError(
                f"Phase 2 helper provenance paths were not adapted twice at {index}."
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
    heading = (
        f"### Inline thesis helper: `{module_name}`\n\n"
        f"This is the complete helper used by **{phase_description}**. It is "
        "placed here so that the mathematical implementation can be read in "
        "the same phase as the formulas and certificate that use it. The "
        "cell is executable: the `inline_module` magic gives the source an "
        "isolated module namespace while leaving the code visible in the "
        "notebook.\n\n"
        f"Source file: `{filename}`  \n"
        f"SHA-256: `{digest}`\n"
    )
    code = f"%%inline_module {module_name} {digest}\n{module_source}"
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
        "Plotting workers, the asymmetric exact-model helper and archive "
        "packaging remain external because they are outside this counterpart's "
        "mathematical scope.\n",
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


def validate_inline_helper_sync(counterpart: dict[str, Any]) -> None:
    """Require every inline source and displayed digest to match its file."""

    cells = counterpart["cells"]
    positions: dict[str, list[int]] = {}
    for index, cell in enumerate(cells):
        positions.setdefault(str(cell.get("id", "")), []).append(index)

    for ordinal, (_, module_name, filename, phase_description) in enumerate(
        INLINE_HELPERS,
        start=1,
    ):
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
    if len(inserted) != 2 + 2 * len(INLINE_HELPERS):
        raise AssertionError("The inline-helper insertion count is incorrect.")
    validate_inline_helper_sync(counterpart)
    if counterpart.get("metadata", {}) != original.get("metadata", {}):
        raise AssertionError("Notebook metadata or widget state changed.")

    adapted_indices = {10, 47, 48, 126, 130}
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


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"Missing source notebook {SOURCE}; run "
            "build_blaschke_deformation_certifier.py first."
        )
    source_bytes_before = SOURCE.read_bytes()
    counterpart = build()
    validate_counterpart(counterpart)
    OUTPUT.write_text(
        nbf.writes(nbf.from_dict(counterpart), version=nbf.NO_CONVERT),
        encoding="utf-8",
    )
    serialised = json.loads(OUTPUT.read_text(encoding="utf-8"))
    validate_counterpart(serialised)
    source_bytes_after = SOURCE.read_bytes()
    source_digest_after = hashlib.sha256(source_bytes_after).hexdigest()
    if source_bytes_after != source_bytes_before:
        raise RuntimeError("The source notebook changed while building its counterpart.")
    print(f"Wrote {OUTPUT}")
    print(f"Source notebook SHA-256 remains {source_digest_after}")
    print(f"Counterpart cells: {len(counterpart['cells'])}")


if __name__ == "__main__":
    main()
