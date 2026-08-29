"""Refresh current notebook and standalone-helper hashes in Cell provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
DEPLOYMENT_ROOT = HERE.parent
PROVENANCE = HERE / "notebook_cell_provenance.json"
FINAL_NOTEBOOK = HERE / "blaschke_deformation_certifier_thesis_math.ipynb"
SOURCE_NOTEBOOK = HERE / "blaschke_deformation_certifier.ipynb"
BUILDER = HERE / "build_blaschke_deformation_thesis_math_notebook.py"
CHAPTER_MATH_MAP = HERE / "numerical_certification_transfer_markdown.toml"
EXPECTED_VISUAL_CELL_COUNT = 20
EXPECTED_STORED_PNG_OUTPUT_COUNT = 34
ALPHA11_PROFILE_CELL_ID = "32961420"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_recorded_source(value: str) -> Path:
    prefix = "Final Deployment/"
    if not value.startswith(prefix):
        raise RuntimeError(f"Unsupported provenance source path: {value!r}.")
    relative = Path(value.removeprefix(prefix))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe provenance source path: {value!r}.")
    path = (DEPLOYMENT_ROOT / relative).resolve()
    path.relative_to(DEPLOYMENT_ROOT.resolve())
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def refreshed_payload(payload: dict[str, object]) -> dict[str, object]:
    notebook = payload.get("notebook")
    entries = payload.get("entries")
    if not isinstance(notebook, dict) or not isinstance(entries, dict):
        raise RuntimeError("Unexpected notebook provenance schema.")
    final_notebook = json.loads(FINAL_NOTEBOOK.read_text(encoding="utf-8"))
    visual_cell_count = sum(
        cell.get("cell_type") == "code"
        and any("image/png" in output.get("data", {}) for output in cell.get("outputs", []))
        for cell in final_notebook.get("cells", [])
    )
    stored_png_count = sum(
        "image/png" in output.get("data", {})
        for cell in final_notebook.get("cells", [])
        for output in cell.get("outputs", [])
    )
    alpha11_cell = next(
        (
            cell
            for cell in final_notebook.get("cells", [])
            if str(cell.get("id", "")) == ALPHA11_PROFILE_CELL_ID
        ),
        None,
    )
    alpha11_png_count = sum(
        "image/png" in output.get("data", {})
        for output in (alpha11_cell or {}).get("outputs", [])
    )
    if visual_cell_count != EXPECTED_VISUAL_CELL_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_VISUAL_CELL_COUNT} visual cells, found "
            f"{visual_cell_count}."
        )
    if stored_png_count != EXPECTED_STORED_PNG_OUTPUT_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_STORED_PNG_OUTPUT_COUNT} stored PNG outputs, "
            f"found {stored_png_count}."
        )
    if alpha11_png_count != 2:
        raise RuntimeError(
            "Cell 80N must retain both the selected profile and alpha^11 PNG."
        )

    notebook["sha256_at_manifest_creation"] = sha256_file(FINAL_NOTEBOOK)
    notebook["transformation_source_sha256"] = sha256_file(SOURCE_NOTEBOOK)
    notebook["thesis_math_builder_sha256"] = sha256_file(BUILDER)
    notebook["chapter_math_map_sha256"] = sha256_file(CHAPTER_MATH_MAP)
    notebook["expected_visual_cell_count"] = visual_cell_count
    notebook["expected_stored_png_output_count"] = stored_png_count
    notebook["stored_png_count_note"] = (
        "The 34-payload contract includes both PNG outputs in Cell 80N: the "
        "selected sampled profile and the restored alpha^11 profile."
    )
    refreshed: dict[str, str] = {}
    for label, record in entries.items():
        if not isinstance(record, dict) or "source_file" not in record:
            continue
        source_file = record["source_file"]
        if not isinstance(source_file, str):
            raise RuntimeError(f"Malformed source_file for {label}.")
        source_path = _resolve_recorded_source(source_file)
        digest = sha256_file(source_path)
        record["source_sha256"] = digest
        refreshed[source_file] = digest
    helper_rows = runpy.run_path(str(BUILDER))["INLINE_HELPERS"]
    inline_helper_hashes = {
        filename: sha256_file(HERE / filename)
        for _, _, filename, _ in helper_rows
    }
    if len(inline_helper_hashes) != 18:
        raise RuntimeError("Expected exactly eighteen inline helper sources.")
    payload["source_hash_refresh"] = {
        "date": "2026-08-29",
        "method": "refresh_notebook_cell_provenance.py",
        "notebook_sha256": notebook["sha256_at_manifest_creation"],
        "transformation_source_sha256": notebook["transformation_source_sha256"],
        "thesis_math_builder_sha256": notebook["thesis_math_builder_sha256"],
        "chapter_math_map_sha256": notebook["chapter_math_map_sha256"],
        "visual_cell_count": visual_cell_count,
        "stored_png_output_count": stored_png_count,
        "standalone_source_count": len(refreshed),
        "standalone_source_sha256": dict(sorted(refreshed.items())),
        "inline_helper_count": len(inline_helper_hashes),
        "inline_helper_sha256": dict(sorted(inline_helper_hashes.items())),
        "stored_output_status": (
            "retained historical outputs; full clean-room execution is required "
            "before they can evidence the current source snapshot"
        ),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    observed = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    expected = refreshed_payload(json.loads(PROVENANCE.read_text(encoding="utf-8")))
    if args.check:
        if observed != expected:
            raise SystemExit("notebook_cell_provenance.json contains stale hashes.")
        print("Notebook and standalone-source provenance hashes are current.")
        return
    PROVENANCE.write_text(
        json.dumps(expected, indent=2) + "\n",
        encoding="utf-8",
        newline="",
    )
    print(
        "Refreshed notebook provenance for "
        f"{expected['source_hash_refresh']['standalone_source_count']} standalone sources."
    )


if __name__ == "__main__":
    main()
