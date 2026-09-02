"""Focused tests for the lossless GitHub notebook presentation transform."""

from __future__ import annotations

import base64
from collections import Counter
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import unittest

from PIL import Image


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_blaschke_deformation_thesis_math_notebook import build_curated
from prepare_blaschke_github_notebook import (
    _require_size_limit,
    _source_text,
    prepare,
)


def _png_payload() -> str:
    target = io.BytesIO()
    Image.new("RGBA", (24, 12), (20, 80, 140, 180)).save(target, format="PNG")
    return base64.b64encode(target.getvalue()).decode("ascii")


class PrepareGitHubNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        notebook, _ = build_curated()
        cls.base_notebook = notebook
        cls.png = _png_payload()

    def fixture(self) -> dict:
        notebook = deepcopy(self.base_notebook)
        code_cells = [
            cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
        ]
        self.assertEqual(len(notebook["cells"]), 141)
        self.assertEqual(len(code_cells), 68)
        for execution_count, cell in enumerate(code_cells, start=1):
            cell["execution_count"] = execution_count
            cell["outputs"] = []

        # Fourteen cells have two plots and six have one: 34 plots in 20 cells.
        for index, cell in enumerate(code_cells[:20]):
            for plot_index in range(2 if index < 14 else 1):
                cell["outputs"].append(
                    {
                        "data": {"image/png": self.png},
                        "metadata": {"fixture_plot": [index, plot_index]},
                        "output_type": "display_data",
                    }
                )

        first = code_cells[0]
        first_plot, second_plot = first["outputs"]
        first_plot["data"]["text/plain"] = ["<Figure size 24x12>"]
        first["outputs"] = [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": ["phase one\n", "phase two\n"],
            },
            first_plot,
            {
                "data": {
                    "text/html": [
                        "<table><tr><th>packet</th><th>count</th></tr>",
                        "<tr><td>alpha</td><td>1</td></tr></table>",
                    ],
                    "text/plain": ["  packet  count\n0  alpha      1"],
                },
                "metadata": {"dataframe": True},
                "output_type": "display_data",
            },
            {
                "data": {"text/plain": ["{'certified_count': 30}"]},
                "execution_count": 1,
                "metadata": {},
                "output_type": "execute_result",
            },
            second_plot,
        ]
        return notebook

    @staticmethod
    def outputs(notebook: dict) -> list[dict]:
        return [
            output
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
            for output in cell.get("outputs", [])
        ]

    def test_preserves_output_count_order_types_tables_streams_and_text(self) -> None:
        notebook = self.fixture()
        original_outputs = deepcopy(self.outputs(notebook))
        prepared, report = prepare(notebook)
        prepared_outputs = self.outputs(prepared)

        self.assertEqual(len(prepared_outputs), len(original_outputs))
        self.assertEqual(
            [output["output_type"] for output in prepared_outputs],
            [output["output_type"] for output in original_outputs],
        )
        self.assertEqual(prepared_outputs[0], original_outputs[0])
        self.assertEqual(prepared_outputs[2], original_outputs[2])
        self.assertEqual(prepared_outputs[3], original_outputs[3])
        self.assertEqual(report["output_count"], len(original_outputs))
        self.assertEqual(report["non_image_outputs_preserved"], 3)
        self.assertEqual(report["omitted_non_plot_outputs"], 0)
        display = prepared["metadata"]["github_display_artifact"]
        self.assertEqual(display["output_count"], len(original_outputs))
        self.assertEqual(display["non_image_outputs_preserved"], 3)
        self.assertEqual(display["non_plot_outputs_omitted"], 0)

    def test_png_becomes_jpeg_without_changing_sibling_mime_or_metadata(self) -> None:
        notebook = self.fixture()
        original_plot = deepcopy(self.outputs(notebook)[1])
        prepared, report = prepare(notebook)
        converted = self.outputs(prepared)[1]

        self.assertNotIn("image/png", converted["data"])
        self.assertIn("image/jpeg", converted["data"])
        self.assertEqual(converted["data"]["text/plain"], original_plot["data"]["text/plain"])
        self.assertEqual(converted["metadata"]["fixture_plot"], [0, 0])
        preview = converted["metadata"]["github_plot_preview"]
        self.assertEqual(preview["preview_format"], "jpeg")
        self.assertEqual(report["plot_count"], 34)
        self.assertEqual(report["visual_cell_count"], 20)
        with Image.open(
            io.BytesIO(base64.b64decode(converted["data"]["image/jpeg"]))
        ) as image:
            self.assertEqual(image.format, "JPEG")

    def test_private_output_paths_are_sanitized_without_structure_loss(self) -> None:
        notebook = self.fixture()
        first_code = next(
            cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
        )
        first_code["outputs"][0]["text"] = [
            "Loaded /tmp/run-123/helper.py\n",
            r"Loaded C:\Users\Lucy\Temp\worker.py" + "\n",
        ]
        original_count = len(self.outputs(notebook))
        prepared, report = prepare(notebook)
        stream = self.outputs(prepared)[0]
        serialized = json.dumps(prepared)

        self.assertEqual(len(self.outputs(prepared)), original_count)
        self.assertEqual(stream["output_type"], "stream")
        self.assertEqual(len(stream["text"]), 2)
        self.assertIn("<local-path>/helper.py", stream["text"][0])
        self.assertIn("<local-path>/worker.py", stream["text"][1])
        for marker in ("/tmp/", "/home/", "/Users/", "file://", "C:\\Users\\"):
            self.assertNotIn(marker, serialized)
        self.assertEqual(report["private_output_paths_sanitized"], 2)

    def test_private_path_in_cell_metadata_fails_closed(self) -> None:
        notebook = self.fixture()
        notebook["cells"][0].setdefault("metadata", {})["local"] = "/tmp/private/x"
        with self.assertRaisesRegex(RuntimeError, "still expose"):
            prepare(notebook)

    def test_source_and_inline_helper_identity_drift_fails_closed(self) -> None:
        notebook = self.fixture()
        helper = next(
            cell
            for cell in notebook["cells"]
            if str(cell.get("id", "")).startswith("inline-helper-source-")
        )
        source = helper.get("source", [])
        helper["source"] = [*source, "# drift\n"] if isinstance(source, list) else source + "# drift\n"
        with self.assertRaisesRegex(RuntimeError, "source differs"):
            prepare(notebook)

    def test_error_output_fails_closed(self) -> None:
        notebook = self.fixture()
        code_cells = [
            cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
        ]
        code_cells[-1]["outputs"].append(
            {
                "ename": "RuntimeError",
                "evalue": "failure",
                "output_type": "error",
                "traceback": ["failure"],
            }
        )
        with self.assertRaisesRegex(RuntimeError, "contains an error output"):
            prepare(notebook)

    def test_locked_structure_and_execution_counts_fail_closed(self) -> None:
        missing_cell = self.fixture()
        missing_cell["cells"].pop()
        with self.assertRaisesRegex(RuntimeError, "locked 141 cells"):
            prepare(missing_cell)

        bad_count = self.fixture()
        first_code = next(
            cell for cell in bad_count["cells"] if cell.get("cell_type") == "code"
        )
        first_code["execution_count"] = None
        with self.assertRaisesRegex(RuntimeError, "exactly 1 through 68"):
            prepare(bad_count)

    def test_plot_and_visual_cell_contracts_fail_closed(self) -> None:
        missing_plot = self.fixture()
        missing_plot_code = [
            cell
            for cell in missing_plot["cells"]
            if cell.get("cell_type") == "code"
        ]
        missing_plot_code[0]["outputs"].pop()
        with self.assertRaisesRegex(RuntimeError, "Expected 34 plot outputs"):
            prepare(missing_plot)

        missing_visual_cell = self.fixture()
        code_cells = [
            cell
            for cell in missing_visual_cell["cells"]
            if cell.get("cell_type") == "code"
        ]
        moved_plot = code_cells[19]["outputs"].pop()
        code_cells[0]["outputs"].append(moved_plot)
        with self.assertRaisesRegex(RuntimeError, "Expected 20 visual code cells"):
            prepare(missing_visual_cell)

    def test_existing_jpeg_alongside_png_fails_closed(self) -> None:
        notebook = self.fixture()
        plot = self.outputs(notebook)[1]
        plot["data"]["image/jpeg"] = "already-present"
        with self.assertRaisesRegex(RuntimeError, "both PNG and JPEG"):
            prepare(notebook)

    def test_size_cap_is_inclusive_and_fails_closed(self) -> None:
        _require_size_limit(b"1234", maximum_bytes=4)
        with self.assertRaisesRegex(RuntimeError, "5 > 4 bytes"):
            _require_size_limit(b"12345", maximum_bytes=4)

    def test_checked_in_notebook_retains_the_complete_executed_output_contract(
        self,
    ) -> None:
        notebook = json.loads(
            (HERE / "blaschke_deformation_certifier_thesis_math.ipynb").read_text(
                encoding="utf-8"
            )
        )
        code_cells = [
            cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
        ]
        outputs = [
            output for cell in code_cells for output in cell.get("outputs", [])
        ]
        self.assertEqual(len(outputs), 301)
        self.assertEqual(
            Counter(output.get("output_type") for output in outputs),
            Counter({"stream": 203, "display_data": 67, "execute_result": 31}),
        )
        self.assertEqual(
            sum("text/html" in output.get("data", {}) for output in outputs), 43
        )
        self.assertFalse(
            any(output.get("output_type") == "error" for output in outputs)
        )
        cells_by_label = {
            _source_text(cell).splitlines()[0]: cell
            for cell in code_cells
            if _source_text(cell).splitlines()
        }
        self.assertEqual(len(cells_by_label["#101N"].get("outputs", [])), 18)
        self.assertEqual(len(cells_by_label["#107N"].get("outputs", [])), 36)
        self.assertEqual(
            sum(
                "text/html" in output.get("data", {})
                for output in cells_by_label["#107N"].get("outputs", [])
            ),
            3,
        )
        display = notebook["metadata"]["github_display_artifact"]
        self.assertEqual(display["output_count"], 301)
        self.assertEqual(display["non_image_outputs_preserved"], 267)
        self.assertEqual(display["non_plot_outputs_omitted"], 0)


if __name__ == "__main__":
    unittest.main()
