"""Prepare the freshly executed thesis notebook for reliable GitHub rendering.

The numerical execution notebook embeds full-resolution PNG payloads.  That is
useful locally but makes GitHub decline to render the file.  This publication
step keeps every code cell, execution count and plot position, replaces each
embedded PNG with a small JPEG preview, and omits non-visual runtime chatter
that can contain scratch paths. Full-resolution PNG files remain in
``outputs/blaschke_deformation_certifier/figures``.

This is a presentation-only transformation.  Before writing anything it
requires exact cell/source/attachment identity with a fresh locked builder
output, complete execution of all code cells, zero error outputs, and the
fixed 20-visual-cell/34-plot contract.
"""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from PIL import Image

from build_blaschke_deformation_thesis_math_notebook import build_curated


EXPECTED_CELL_COUNT = 141
EXPECTED_CODE_CELL_COUNT = 68
EXPECTED_VISUAL_CELL_COUNT = 20
EXPECTED_PLOT_COUNT = 34
DEFAULT_MAX_WIDTH = 480
DEFAULT_JPEG_QUALITY = 40
MAX_GITHUB_NOTEBOOK_BYTES = 2_000_000


def _source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _payload_text(value: Any) -> str:
    return "".join(value) if isinstance(value, list) else str(value)


def _require_current_sources(notebook: dict[str, Any]) -> None:
    expected, _ = build_curated()
    actual_cells = notebook.get("cells", [])
    expected_cells = expected.get("cells", [])
    if len(actual_cells) != EXPECTED_CELL_COUNT or len(actual_cells) != len(
        expected_cells
    ):
        raise RuntimeError("The executed notebook does not have the locked 141 cells.")
    for index, (actual, wanted) in enumerate(
        zip(actual_cells, expected_cells, strict=True)
    ):
        for key in ("id", "cell_type"):
            if actual.get(key) != wanted.get(key):
                raise RuntimeError(f"Cell {index} has a mismatched {key}.")
        if _source_text(actual) != _source_text(wanted):
            raise RuntimeError(f"Cell {index} source differs from the locked builder.")
        if actual.get("attachments", {}) != wanted.get("attachments", {}):
            raise RuntimeError(f"Cell {index} attachments differ from the locked builder.")


def _jpeg_preview(raw_png: bytes, *, max_width: int, quality: int) -> tuple[bytes, tuple[int, int]]:
    with Image.open(io.BytesIO(raw_png)) as loaded:
        original_size = (int(loaded.width), int(loaded.height))
        image = loaded.convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        background.alpha_composite(image)
        image = background.convert("RGB")
        if image.width > max_width:
            height = max(1, round(image.height * max_width / image.width))
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        target = io.BytesIO()
        image.save(
            target,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
        )
    return target.getvalue(), original_size


def prepare(
    notebook: dict[str, Any],
    *,
    max_width: int = DEFAULT_MAX_WIDTH,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_current_sources(notebook)
    result = deepcopy(notebook)
    code_cells = [
        cell for cell in result["cells"] if cell.get("cell_type") == "code"
    ]
    if len(code_cells) != EXPECTED_CODE_CELL_COUNT:
        raise RuntimeError("The executed notebook does not have 68 code cells.")
    counts = [cell.get("execution_count") for cell in code_cells]
    if counts != list(range(1, EXPECTED_CODE_CELL_COUNT + 1)):
        raise RuntimeError("Code-cell execution counts must be exactly 1 through 68.")

    plot_count = 0
    visual_cell_count = 0
    original_png_bytes = 0
    preview_jpeg_bytes = 0
    omitted_non_plot_outputs = 0
    for cell in code_cells:
        visual = False
        retained_outputs = []
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                raise RuntimeError(
                    f"Executed cell {cell.get('id')} contains an error output."
                )
            data = output.get("data", {})
            if "image/png" not in data:
                omitted_non_plot_outputs += 1
                continue
            visual = True
            plot_count += 1
            raw_png = base64.b64decode(_payload_text(data.pop("image/png")), validate=True)
            preview, original_size = _jpeg_preview(
                raw_png,
                max_width=max_width,
                quality=quality,
            )
            original_png_bytes += len(raw_png)
            preview_jpeg_bytes += len(preview)
            data["image/jpeg"] = base64.b64encode(preview).decode("ascii")
            metadata = output.setdefault("metadata", {})
            metadata["github_plot_preview"] = {
                "full_resolution_png_sha256": hashlib.sha256(raw_png).hexdigest(),
                "original_height": original_size[1],
                "original_width": original_size[0],
                "preview_format": "jpeg",
                "preview_quality": quality,
                "preview_width_limit": max_width,
            }
            retained_outputs.append(output)
        cell["outputs"] = retained_outputs
        visual_cell_count += int(visual)

    if plot_count != EXPECTED_PLOT_COUNT:
        raise RuntimeError(f"Expected 34 plot outputs, found {plot_count}.")
    if visual_cell_count != EXPECTED_VISUAL_CELL_COUNT:
        raise RuntimeError(
            f"Expected 20 visual code cells, found {visual_cell_count}."
        )

    execution_sha256 = hashlib.sha256(
        json.dumps(notebook, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    source_projection = [
        {
            "attachments": cell.get("attachments", {}),
            "cell_type": cell.get("cell_type"),
            "id": cell.get("id"),
            "source": _source_text(cell),
        }
        for cell in result["cells"]
    ]
    source_cells_sha256 = hashlib.sha256(
        json.dumps(source_projection, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    result.setdefault("metadata", {})["github_display_artifact"] = {
        "code_cell_count": EXPECTED_CODE_CELL_COUNT,
        "canonical_full_execution_sha256": execution_sha256,
        "full_resolution_figures": (
            "outputs/blaschke_deformation_certifier/figures"
        ),
        "plot_count": EXPECTED_PLOT_COUNT,
        "presentation_only": True,
        "non_plot_outputs_omitted": omitted_non_plot_outputs,
        "source_cells_sha256": source_cells_sha256,
        "source_cells_match_locked_builder": True,
        "theorem_gate": False,
        "visual_cell_count": EXPECTED_VISUAL_CELL_COUNT,
    }
    presentation_surface = {
        "metadata": result.get("metadata", {}),
        "cells": [
            {
                "metadata": cell.get("metadata", {}),
                "outputs": cell.get("outputs", []),
            }
            for cell in result["cells"]
        ],
    }
    presentation_text = json.dumps(
        presentation_surface,
        sort_keys=True,
        separators=(",", ":"),
    )
    for forbidden in ("/tmp/", "/home/", "file://"):
        if forbidden in presentation_text:
            raise RuntimeError(
                f"The GitHub display metadata or outputs expose {forbidden!r}."
            )
    report = {
        "cell_count": len(result["cells"]),
        "code_cell_count": len(code_cells),
        "error_output_count": 0,
        "omitted_non_plot_outputs": omitted_non_plot_outputs,
        "original_png_bytes": original_png_bytes,
        "plot_count": plot_count,
        "preview_jpeg_bytes": preview_jpeg_bytes,
        "visual_cell_count": visual_cell_count,
    }
    return result, report


def _serialise(notebook: dict[str, Any]) -> bytes:
    return (json.dumps(notebook, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executed_notebook", type=Path)
    parser.add_argument("output_notebook", type=Path)
    parser.add_argument("--max-width", type=int, default=DEFAULT_MAX_WIDTH)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    args = parser.parse_args()
    if args.max_width < 200:
        raise ValueError("The preview width must be at least 200 pixels.")
    if not 20 <= args.jpeg_quality <= 95:
        raise ValueError("JPEG quality must be between 20 and 95.")

    source = args.executed_notebook.resolve(strict=True)
    output = args.output_notebook.resolve()
    notebook = json.loads(source.read_text(encoding="utf-8"))
    prepared, report = prepare(
        notebook,
        max_width=args.max_width,
        quality=args.jpeg_quality,
    )
    payload = _serialise(prepared)
    if len(payload) > MAX_GITHUB_NOTEBOOK_BYTES:
        raise RuntimeError(
            "The GitHub display notebook is still too large: "
            f"{len(payload)} > {MAX_GITHUB_NOTEBOOK_BYTES} bytes."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.chmod(0o664)
    temporary.replace(output)
    report.update(
        {
            "notebook_bytes": len(payload),
            "notebook_sha256": hashlib.sha256(payload).hexdigest(),
            "output": str(output),
            "status": "GITHUB_DISPLAY_NOTEBOOK_READY",
        }
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
