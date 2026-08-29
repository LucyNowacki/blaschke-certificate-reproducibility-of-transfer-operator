"""Structural tests for standalone-to-inline helper provenance."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import tempfile
import unittest

from build_blaschke_deformation_thesis_math_notebook import (
    CURATED_INLINE_HELPER_ORDINALS,
    DEPENDENCY_MAP_CELL_ID,
    _merge_preserved_execution_state,
    build_curated,
    dependency_map_cell,
    main as builder_main,
    validate_curated_counterpart,
    validate_inline_helper_sync,
)


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "blaschke_deformation_certifier_thesis_math.ipynb"
TEMPLATE = HERE / "blaschke_deformation_certifier_template.ipynb"
SOURCE = HERE / "blaschke_deformation_certifier.ipynb"
BUILDER = HERE / "build_blaschke_deformation_thesis_math_notebook.py"
CONTOUR_HELPER = HERE / "blaschke_deformation_contour_certification.py"
DETERMINISTIC_HELPER = HERE / "blaschke_deformation_certification.py"

OBSOLETE_TERMS = (
    "Phase 2 deterministic " + "unresolved-tail certification",
    "triangular_" + "argument_principle_count",
)
CURRENT_PHASE2_TERM = (
    "Phase 2 resolved-response and unresolved-input certification"
)
SYMMETRIC_INTERVAL_SOURCE = (
    "SBJ13 equation (21); same benchmark in ASBJ24 Section 3.2"
)
SYMMETRIC_PROVENANCE_WARNING = (
    "The external formula supplies target identities and multiplicities, "
    "not finite counts or contour moats."
)
APPENDIX_HELPER_ORDINALS = CURATED_INLINE_HELPER_ORDINALS


class InlineHelperProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    def test_current_notebook_matches_all_standalone_helpers(self) -> None:
        validate_inline_helper_sync(
            self.notebook,
            helper_ordinals=APPENDIX_HELPER_ORDINALS,
            allow_notebook_provenance=True,
        )

    def test_all_current_cell_sources_match_a_fresh_output_free_build(self) -> None:
        expected, _ = build_curated()
        self.assertEqual(len(self.notebook["cells"]), len(expected["cells"]))
        for actual_cell, expected_cell in zip(
            self.notebook["cells"], expected["cells"], strict=True
        ):
            self.assertEqual(actual_cell.get("id"), expected_cell.get("id"))
            self.assertEqual(
                "".join(actual_cell.get("source", [])),
                "".join(expected_cell.get("source", [])),
                msg=str(actual_cell.get("id")),
            )
            self.assertEqual(
                actual_cell.get("attachments", {}),
                expected_cell.get("attachments", {}),
                msg=str(actual_cell.get("id")),
            )

    def test_cell_0m_matches_its_builder_owned_source(self) -> None:
        validate_curated_counterpart(self.notebook)
        expected = dependency_map_cell()
        actual = self.notebook["cells"][0]
        actual_source = "".join(actual.get("source", []))
        self.assertEqual(actual.get("id"), DEPENDENCY_MAP_CELL_ID)
        self.assertEqual(actual_source, expected["source"])

    def test_altered_cell_0m_is_rejected(self) -> None:
        altered = deepcopy(self.notebook)
        map_cell = altered["cells"][0]
        source = map_cell["source"]
        if isinstance(source, list):
            map_cell["source"] = [
                line.replace("Reproducibility dependency map", "Unverified map", 1)
                for line in source
            ]
        else:
            map_cell["source"] = source.replace(
                "Reproducibility dependency map", "Unverified map", 1
            )
        with self.assertRaises(AssertionError):
            validate_curated_counterpart(altered)

    def test_source_refresh_preserves_execution_state_by_stable_id(self) -> None:
        current, _ = build_curated()
        merged = _merge_preserved_execution_state(current, self.notebook)
        validate_curated_counterpart(merged)
        for expected, actual in zip(
            self.notebook["cells"], merged["cells"], strict=True
        ):
            if expected.get("cell_type") != "code":
                continue
            self.assertEqual(
                actual.get("execution_count"), expected.get("execution_count")
            )
            self.assertEqual(actual.get("outputs", []), expected.get("outputs", []))
        self.assertIn("source_sync_after_execution", merged["metadata"])

    def test_builder_refuses_implicit_executed_notebook_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / NOTEBOOK.name
            output.write_bytes(NOTEBOOK.read_bytes())
            before = output.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                builder_main(["--output", str(output)])
            self.assertEqual(output.read_bytes(), before)

    def test_altered_inline_byte_is_rejected(self) -> None:
        altered = deepcopy(self.notebook)
        source_cell = next(
            cell for cell in altered["cells"]
            if cell.get("id") == "inline-helper-source-3"
        )
        source = source_cell["source"]
        if isinstance(source, list):
            source_cell["source"] = [
                line.replace("Rigorous", "rigorous", 1)
                for line in source
            ]
        else:
            source_cell["source"] = source.replace("Rigorous", "rigorous", 1)
        with self.assertRaises(AssertionError):
            validate_inline_helper_sync(
                altered,
                helper_ordinals=APPENDIX_HELPER_ORDINALS,
                allow_notebook_provenance=True,
            )

    def test_altered_displayed_digest_is_rejected(self) -> None:
        altered = deepcopy(self.notebook)
        heading = next(
            cell for cell in altered["cells"]
            if cell.get("id") == "inline-helper-heading-3"
        )
        source = heading["source"]
        if isinstance(source, list):
            joined = "".join(source)
            heading["source"] = re.sub(
                r"(?<=SHA-256: `)[0-9a-f]{64}(?=`)", "0" * 64, joined, count=1
            )
        else:
            heading["source"] = re.sub(
                r"(?<=SHA-256: `)[0-9a-f]{64}(?=`)", "0" * 64, source, count=1
            )
        with self.assertRaises(AssertionError):
            validate_inline_helper_sync(
                altered,
                helper_ordinals=APPENDIX_HELPER_ORDINALS,
                allow_notebook_provenance=True,
            )

    def test_deployment_contains_no_obsolete_certification_terms(self) -> None:
        paths = (
            TEMPLATE,
            SOURCE,
            NOTEBOOK,
            BUILDER,
            CONTOUR_HELPER,
            DETERMINISTIC_HELPER,
        )
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for obsolete in OBSOLETE_TERMS:
                self.assertNotIn(obsolete, text, msg=str(path))

    def test_current_phase2_description_is_present(self) -> None:
        for path in (TEMPLATE, SOURCE, NOTEBOOK, BUILDER):
            self.assertIn(
                CURRENT_PHASE2_TERM,
                path.read_text(encoding="utf-8"),
                msg=str(path),
            )

    def test_symmetric_interval_spectrum_provenance_is_locked(self) -> None:
        for path in (TEMPLATE, SOURCE, NOTEBOOK):
            text = path.read_text(encoding="utf-8")
            self.assertIn(SYMMETRIC_INTERVAL_SOURCE, text, msg=str(path))
            self.assertIn(SYMMETRIC_PROVENANCE_WARNING, text, msg=str(path))


if __name__ == "__main__":
    unittest.main()
