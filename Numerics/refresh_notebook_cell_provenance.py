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
    notebook["sha256_at_manifest_creation"] = sha256_file(FINAL_NOTEBOOK)
    notebook["transformation_source_sha256"] = sha256_file(SOURCE_NOTEBOOK)
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
        "standalone_source_count": len(refreshed),
        "standalone_source_sha256": dict(sorted(refreshed.items())),
        "inline_helper_count": len(inline_helper_hashes),
        "inline_helper_sha256": dict(sorted(inline_helper_hashes.items())),
        "stored_output_status": (
            "retained historical outputs; full clean-room execution is required after the fail-closed source correction"
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
