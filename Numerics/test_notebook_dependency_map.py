"""Semantic and accessibility regression tests for the canonical Cell 0M SVG."""

from __future__ import annotations

import json
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
MARKDOWN = HERE / "blaschke_deformation_notebook_dependency_map.md"
SVG = HERE / "blaschke_deformation_notebook_dependency_map.svg"
NOTEBOOK = HERE / "blaschke_deformation_certifier_thesis_math.ipynb"
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
NS = {"svg": SVG_NAMESPACE}
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"


def _classes(element: ET.Element) -> set[str]:
    return set(element.get("class", "").split())


def _by_id(root: ET.Element, identifier: str) -> ET.Element:
    matches = [element for element in root.iter() if element.get("id") == identifier]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected exactly one SVG element with id {identifier!r}; found {len(matches)}."
        )
    return matches[0]


def _css_rule(style: str, selector: str) -> str:
    match = re.search(rf"{re.escape(selector)}\s*\{{([^}}]+)\}}", style)
    if match is None:
        raise AssertionError(f"Missing CSS rule for {selector}.")
    return match.group(1)


def _initial_move(path_data: str) -> tuple[float, float] | None:
    match = re.match(rf"\s*M\s*({NUMBER})[ ,]+({NUMBER})", path_data)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def _inside_box(point: tuple[float, float], box: ET.Element) -> bool:
    x, y = point
    left = float(box.get("x", "nan"))
    top = float(box.get("y", "nan"))
    right = left + float(box.get("width", "nan"))
    bottom = top + float(box.get("height", "nan"))
    return left <= x <= right and top <= y <= bottom


class NotebookDependencyMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = ET.parse(SVG)
        cls.root = cls.tree.getroot()
        cls.markdown = " ".join(MARKDOWN.read_text(encoding="utf-8").split())
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    def test_root_accessibility_contract_is_complete_and_unique(self) -> None:
        self.assertEqual(self.root.tag, f"{{{SVG_NAMESPACE}}}svg")
        self.assertEqual(self.root.get("viewBox"), "0 0 1600 1900")
        self.assertEqual(self.root.get("width"), "100%")
        self.assertEqual(self.root.get("role"), "img")
        labelled_by = self.root.get("aria-labelledby", "").split()
        self.assertEqual(
            labelled_by,
            ["dependency-map-title", "dependency-map-description"],
        )

        identifiers = [
            element.get("id") for element in self.root.iter() if element.get("id")
        ]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        title = _by_id(self.root, labelled_by[0])
        description = _by_id(self.root, labelled_by[1])
        self.assertEqual(title.tag, f"{{{SVG_NAMESPACE}}}title")
        self.assertEqual(description.tag, f"{{{SVG_NAMESPACE}}}desc")
        self.assertTrue("dependency graph" in "".join(title.itertext()).lower())
        self.assertTrue("four numerical phases" in "".join(description.itertext()).lower())

    def test_non_colour_edge_distinctions_are_preserved(self) -> None:
        style_elements = self.root.findall("svg:defs/svg:style", NS)
        self.assertEqual(len(style_elements), 1)
        style = "".join(style_elements[0].itertext())
        solid = _css_rule(style, ".dep-solid")
        theorem = _css_rule(style, ".dep-theorem-arrow")
        diagnostic = _css_rule(style, ".dep-diagnostic-arrow")
        provenance = _css_rule(style, ".dep-provenance-arrow")
        self.assertNotIn("stroke-dasharray", solid)
        self.assertNotIn("stroke-dasharray", theorem)
        self.assertRegex(diagnostic, r"stroke-dasharray\s*:\s*9\s+6")
        self.assertRegex(provenance, r"stroke-dasharray\s*:\s*3\s+6")
        for marker in (
            "arrow-solid",
            "arrow-theorem",
            "arrow-diagnostic",
            "arrow-provenance",
        ):
            _by_id(self.root, marker)

    def test_visible_legend_matches_markdown_dependency_contract(self) -> None:
        legend_text = [
            "".join(element.itertext())
            for element in self.root.findall("svg:text", NS)
            if element.get("y") == "52"
        ]
        self.assertEqual(
            legend_text,
            [
                "theorem-facing data dependencies",
                "diagnostic only",
                "build or provenance",
            ],
        )
        self.assertIn(
            "solid arrows are theorem-facing data dependencies",
            self.markdown.lower(),
        )
        self.assertIn(
            "dashed amber arrows are diagnostic-only dependencies",
            self.markdown.lower(),
        )
        self.assertIn(
            "dotted grey arrows are build or provenance dependencies",
            self.markdown.lower(),
        )

    def test_standalone_markdown_links_the_sibling_svg(self) -> None:
        markdown = MARKDOWN.read_text(encoding="utf-8")
        target = "./blaschke_deformation_notebook_dependency_map.svg"
        self.assertIn(f"]({target})", markdown)
        self.assertNotIn("attachment:", markdown)
        self.assertTrue((MARKDOWN.parent / target).is_file())

    def test_map_describes_the_current_semantic_release(self) -> None:
        text = " ".join(self.root.itertext())
        self.assertIn("release inventory; semantic verifier", text)
        self.assertIn("authoritative 2048-bit enclosure", text)
        self.assertIn("optional 1024-bit audit is diagnostic", text)
        self.assertIn("rho=2.725; r=2.473669807791324109321273260", text)
        self.assertIn("schema 1.4.0", text)
        self.assertNotIn("archive verifier", text)
        self.assertNotIn("24N--30N", text)

    def test_published_notebook_is_really_executed_when_map_says_so(self) -> None:
        code_cells = [
            cell
            for cell in self.notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        ]
        self.assertEqual(len(code_cells), 68)
        self.assertTrue(
            all(isinstance(cell.get("execution_count"), int) for cell in code_cells)
        )
        self.assertTrue(any(cell.get("outputs") for cell in code_cells))

    def test_phase1_retained_producer_branch_is_diagnostic_only(self) -> None:
        self.assertIn(
            "both producer branches have diagnostic-only (dashed amber) semantics",
            self.markdown,
        )
        self.assertIn(
            "neither branch enters a count, moat, small-gain or Riesz-rank theorem gate",
            self.markdown,
        )

        branch = _by_id(self.root, "phase1-retained-diagnostic-branch")
        self.assertIn("diagnostic only", branch.get("aria-label", "").lower())
        self.assertIn("do not enter theorem gates", branch.get("aria-label", "").lower())
        datasets = _by_id(self.root, "phase1-retained-diagnostic-datasets")
        self.assertIn("dep-diagnostic", _classes(datasets))
        input_edge = _by_id(self.root, "phase1-retained-diagnostic-input")
        self.assertEqual(_classes(input_edge), {"dep-diagnostic-arrow"})

        branch_edges = branch.findall(".//svg:path", NS) + branch.findall(
            ".//svg:line", NS
        )
        self.assertTrue(branch_edges)
        for edge in branch_edges:
            self.assertEqual(_classes(edge), {"dep-diagnostic-arrow"})

    def test_phase1_retained_datasets_have_no_theorem_facing_outgoing_edge(self) -> None:
        datasets = _by_id(self.root, "phase1-retained-diagnostic-datasets")
        forbidden_classes = {"dep-solid", "dep-theorem-arrow"}
        offending: list[str] = []
        for path in self.root.findall(".//svg:path", NS):
            start = _initial_move(path.get("d", ""))
            if start is None or not _inside_box(start, datasets):
                continue
            if _classes(path) & forbidden_classes:
                offending.append(path.get("d", ""))
        self.assertEqual(
            offending,
            [],
            msg="Retained Phase 1 diagnostics must not feed a theorem-facing edge.",
        )
        _by_id(self.root, "phase4-certified-branch")

    def test_executed_notebook_text_is_split_inside_its_box(self) -> None:
        group = _by_id(self.root, "executed-final-notebook")
        box = _by_id(self.root, "executed-final-notebook-box")
        text_elements = group.findall("svg:text", NS)
        self.assertEqual(len(text_elements), 4)
        detail_lines = ["".join(element.itertext()) for element in text_elements[1:]]
        self.assertEqual(
            detail_lines,
            [
                "Cell 0M; inherited Cells 1M--109N",
                "final Cell 110M; 141 total cells",
                "68 code cells; 18 inline helpers",
            ],
        )
        self.assertTrue(all(len(line) <= 38 for line in detail_lines))
        left = float(box.get("x", "nan"))
        top = float(box.get("y", "nan"))
        right = left + float(box.get("width", "nan"))
        bottom = top + float(box.get("height", "nan"))
        baselines = []
        for element in text_elements:
            x = float(element.get("x", "nan"))
            y = float(element.get("y", "nan"))
            self.assertGreaterEqual(x, left)
            self.assertLess(x, right)
            self.assertGreater(y, top)
            self.assertLess(y, bottom)
            baselines.append(y)
        self.assertEqual(baselines, sorted(set(baselines)))
        provenance_edge = _by_id(
            self.root, "executed-notebook-to-setup-provenance"
        )
        start = _initial_move(provenance_edge.get("d", ""))
        self.assertIsNotNone(start)
        assert start is not None
        self.assertGreaterEqual(start[0], left)
        self.assertLessEqual(start[0], right)
        self.assertEqual(start[1], bottom)


if __name__ == "__main__":
    unittest.main()
