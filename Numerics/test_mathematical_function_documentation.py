"""Focused AST tests for mathematical-function documentation coverage."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

import mathematical_function_inventory as inventory_module
from build_blaschke_deformation_thesis_math_notebook import (
    CURATED_INLINE_HELPER_ORDINALS,
    validate_inline_helper_sync,
)


HERE = Path(__file__).resolve().parent
INVENTORY = HERE / "mathematical_function_inventory.json"
NOTEBOOK = HERE / "blaschke_deformation_certifier_thesis_math.ipynb"


def _function_nodes(source: str) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return dict(inventory_module.discover_functions(source))


def _assert_documented(
    testcase: unittest.TestCase,
    *,
    source: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    label: str,
    expected_explanation: str,
) -> None:
    testcase.assertTrue(node.body, msg=label)
    first = node.body[0]
    testcase.assertIsInstance(first, ast.Expr, msg=label)
    testcase.assertIsInstance(first.value, ast.Constant, msg=label)
    testcase.assertIsInstance(first.value.value, str, msg=label)
    literal = ast.get_source_segment(source, first)
    testcase.assertIsNotNone(literal, msg=label)
    testcase.assertTrue(literal.startswith("'''"), msg=label)
    testcase.assertTrue(literal.endswith("'''"), msg=label)
    doc = ast.get_docstring(node, clean=True) or ""
    testcase.assertIn("Explanation:", doc, msg=label)
    testcase.assertIn("Functionality:", doc, msg=label)
    testcase.assertNotIn(
        "with the arguments and return contract used by this numerical stage",
        doc,
        msg=label,
    )
    explanation, functionality = doc.split("Functionality:", 1)
    explanation = explanation.removeprefix("Explanation:").strip()
    testcase.assertEqual(explanation, expected_explanation, msg=label)
    testcase.assertTrue(functionality.strip(), msg=label)
    testcase.assertNotEqual(explanation, functionality.strip(), msg=label)


class MathematicalFunctionDocumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stored = json.loads(INVENTORY.read_text(encoding="utf-8"))
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    def test_inventory_exhaustively_matches_current_sources(self) -> None:
        current = inventory_module.build_inventory()
        self.assertEqual(self.stored, current)
        self.assertEqual(current["inline_helper_count"], 18)
        self.assertEqual(current["notebook_cell_count"], 140)
        self.assertEqual(current["notebook_code_cell_count"], 68)
        self.assertEqual(
            sum(current["counts"].values()),
            len(current["functions"]),
        )
        self.assertGreater(current["counts"]["mathematical"], 0)
        self.assertGreater(current["counts"]["infrastructure"], 0)
        self.assertGreater(current["counts"]["plotting"], 0)

    def test_every_classified_mathematical_function_has_required_docstring(self) -> None:
        notebook_cells = {
            str(cell.get("id", "")): "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if cell.get("cell_type") == "code"
        }
        helper_cache: dict[str, tuple[str, dict[str, object]]] = {}
        notebook_cache: dict[str, tuple[str, dict[str, object]]] = {}
        for record in self.stored["functions"]:
            if record["classification"] != "mathematical":
                continue
            qualname = str(record["qualname"])
            if record["scope"] == "inline_helper":
                filename = str(record["source"])
                if filename not in helper_cache:
                    source = (HERE / filename).read_text(encoding="utf-8")
                    helper_cache[filename] = (source, _function_nodes(source))
                source, nodes = helper_cache[filename]
                label = f"{filename}:{qualname}"
                source_key = filename
            else:
                cell_id = str(record["cell_id"])
                if cell_id not in notebook_cache:
                    source = notebook_cells[cell_id]
                    notebook_cache[cell_id] = (source, _function_nodes(source))
                source, nodes = notebook_cache[cell_id]
                label = f"cell {cell_id}:{qualname}"
                source_key = f"cell:{cell_id}"
            self.assertIn(qualname, nodes, msg=label)
            _assert_documented(
                self,
                source=source,
                node=nodes[qualname],
                label=label,
                expected_explanation=inventory_module.INTUITIVE_EXPLANATIONS[
                    (source_key, qualname)
                ],
            )

    def test_intuitive_explanation_registry_is_exhaustive_and_non_generic(self) -> None:
        expected = {
            (source, qualname)
            for source, names in inventory_module.MATH_HELPER_FUNCTIONS.items()
            for qualname in names
        }
        expected.update({
            (f"cell:{cell_id}", qualname)
            for cell_id, names in inventory_module.MATH_NOTEBOOK_FUNCTIONS.items()
            for qualname in names
        })
        self.assertEqual(expected, set(inventory_module.INTUITIVE_EXPLANATIONS))
        banned = (
            "Implements the ",
            "mathematics used by",
            "with the arguments and return contract used by this numerical stage",
        )
        for key, explanation in inventory_module.INTUITIVE_EXPLANATIONS.items():
            with self.subTest(key=key):
                self.assertGreaterEqual(len(explanation.split()), 20)
                for phrase in banned:
                    self.assertNotIn(phrase, explanation)

    def test_final_phase2_functionality_names_the_complete_aggregation(self) -> None:
        source = (HERE / "blaschke_deformation_phase2_final_aggregation.py").read_text(
            encoding="utf-8"
        )
        node = _function_nodes(source)["certify_final_phase2_aggregation"]
        doc = ast.get_docstring(node, clean=True) or ""
        functionality = doc.split("Functionality:", 1)[1].lower()
        for phrase in (
            "unresolved-input",
            "resolved-response",
            "output-leakage",
            "finite-matrix-defect",
            "gate",
        ):
            self.assertIn(phrase, functionality)

    def test_inline_copies_and_displayed_digests_match_standalone_helpers(self) -> None:
        validate_inline_helper_sync(
            self.notebook,
            helper_ordinals=CURATED_INLINE_HELPER_ORDINALS,
            allow_notebook_provenance=True,
        )


if __name__ == "__main__":
    unittest.main()
