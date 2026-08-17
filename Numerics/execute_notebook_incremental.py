"""Execute a notebook with visible per-cell progress and incremental saves."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient


def cell_label(cell, index: int) -> str:
    source = "".join(cell.get("source", "")).strip().splitlines()
    first = source[0].strip() if source else cell.get("cell_type", "cell")
    return f"{index:03d} {first[:100]}"


def execute(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    started: dict[int, float] = {}

    def on_cell_start(cell, cell_index, **_):
        started[int(cell_index)] = time.monotonic()
        print(f"START {cell_label(cell, int(cell_index))}", flush=True)

    def on_cell_executed(cell, cell_index, execute_reply, **_):
        index = int(cell_index)
        elapsed = time.monotonic() - started.get(index, time.monotonic())
        nbformat.write(notebook, path)
        print(f"DONE  {cell_label(cell, index)}  {elapsed:.3f}s", flush=True)

    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name=notebook.metadata.get("kernelspec", {}).get("name", "python3"),
        resources={"metadata": {"path": str(path.parent)}},
        on_cell_start=on_cell_start,
        on_cell_executed=on_cell_executed,
        allow_errors=False,
        record_timing=True,
    )
    client.execute(cwd=str(path.parent))
    nbformat.write(notebook, path)
    print(f"COMPLETE {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    execute(args.notebook.resolve())


if __name__ == "__main__":
    main()
