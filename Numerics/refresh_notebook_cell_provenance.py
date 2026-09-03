"""Refresh current notebook and standalone-helper hashes in Cell provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy

import normalize_blaschke_publication as publication


HERE = Path(__file__).resolve().parent
DEPLOYMENT_ROOT = HERE.parent
PROVENANCE = HERE / "notebook_cell_provenance.json"
FINAL_NOTEBOOK = HERE / "blaschke_deformation_certifier_thesis_math.ipynb"
SOURCE_NOTEBOOK = HERE / "blaschke_deformation_certifier.ipynb"
BUILDER = HERE / "build_blaschke_deformation_thesis_math_notebook.py"
CHAPTER_MATH_MAP = HERE / "numerical_certification_transfer_markdown.toml"
EXPECTED_VISUAL_CELL_COUNT = 21
EXPECTED_STORED_PNG_OUTPUT_COUNT = 35
ALPHA11_PROFILE_CELL_ID = "32961420"
SOURCE_SYNC_SCHEMA = publication.SOURCE_SYNC_SCHEMA


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sync_metadata(
    notebook: dict[str, object],
) -> dict[str, object] | None:
    metadata = notebook.get("metadata")
    source_sync = (
        metadata.get("source_sync_after_execution")
        if isinstance(metadata, dict)
        else None
    )
    if source_sync is None:
        return None
    try:
        return publication._validated_source_sync(
            source_sync,
            relative=FINAL_NOTEBOOK.name,
        )
    except publication.PublicationPortabilityError as exc:
        raise RuntimeError(
            "Final notebook lacks valid receipt-bound "
            "source_sync_after_execution metadata."
        ) from exc


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
    source_sync = _source_sync_metadata(final_notebook)
    source_sync_sha256 = (
        hashlib.sha256(
            json.dumps(
                source_sync,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if source_sync is not None
        else None
    )

    def has_stored_plot(output: dict[str, object]) -> bool:
        data = output.get("data", {})
        return isinstance(data, dict) and any(
            mime in data for mime in ("image/png", "image/jpeg")
        )

    visual_cell_count = sum(
        cell.get("cell_type") == "code"
        and any(has_stored_plot(output) for output in cell.get("outputs", []))
        for cell in final_notebook.get("cells", [])
    )
    stored_plot_count = sum(
        has_stored_plot(output)
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
    alpha11_plot_count = sum(
        has_stored_plot(output)
        for output in (alpha11_cell or {}).get("outputs", [])
    )
    if visual_cell_count != EXPECTED_VISUAL_CELL_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_VISUAL_CELL_COUNT} visual cells, found "
            f"{visual_cell_count}."
        )
    if stored_plot_count != EXPECTED_STORED_PNG_OUTPUT_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_STORED_PNG_OUTPUT_COUNT} stored plot outputs, "
            f"found {stored_plot_count}."
        )
    if alpha11_plot_count != 2:
        raise RuntimeError(
            "Cell 80N must retain both the selected profile and alpha^11 plot."
        )

    notebook["sha256_at_manifest_creation"] = sha256_file(FINAL_NOTEBOOK)
    notebook["transformation_source_sha256"] = sha256_file(SOURCE_NOTEBOOK)
    notebook["thesis_math_builder_sha256"] = sha256_file(BUILDER)
    notebook["chapter_math_map_sha256"] = sha256_file(CHAPTER_MATH_MAP)
    notebook["expected_visual_cell_count"] = visual_cell_count
    notebook["expected_stored_png_output_count"] = stored_plot_count
    notebook["stored_png_count_note"] = (
        "The 35-position plot contract includes both plots in Cell 80N: the "
        "selected sampled profile and the restored alpha^11 profile, plus the "
        "presentation-only final certification ladder in Cell 108N. The "
        "canonical execution stores PNG payloads; the GitHub display stores "
        "JPEG previews in the same output positions."
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
    previous_refresh = payload.get("source_hash_refresh", {})
    payload["source_hash_refresh"] = {
        "date": "2026-09-03",
        "method": "refresh_notebook_cell_provenance.py",
        "notebook_sha256": notebook["sha256_at_manifest_creation"],
        "transformation_source_sha256": notebook["transformation_source_sha256"],
        "thesis_math_builder_sha256": notebook["thesis_math_builder_sha256"],
        "chapter_math_map_sha256": notebook["chapter_math_map_sha256"],
        "visual_cell_count": visual_cell_count,
        "stored_png_output_count": stored_plot_count,
        "standalone_source_count": len(refreshed),
        "standalone_source_sha256": dict(sorted(refreshed.items())),
        "inline_helper_count": len(inline_helper_hashes),
        "inline_helper_sha256": dict(sorted(inline_helper_hashes.items())),
    }
    if isinstance(previous_refresh, dict):
        for key in (
            "arithmetic_baseline_commit",
            "source_sync_scope",
            "replay_status",
            "stored_output_status",
        ):
            if key in previous_refresh:
                payload["source_hash_refresh"][key] = previous_refresh[key]
    if source_sync is not None:
        payload["source_hash_refresh"]["source_sync_after_execution"] = source_sync
        payload["source_hash_refresh"][
            "source_sync_after_execution_sha256"
        ] = source_sync_sha256
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
