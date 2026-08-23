"""Structural tests for standalone-to-inline helper provenance."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from build_blaschke_deformation_thesis_math_notebook import (
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


class InlineHelperProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    def test_current_notebook_matches_all_standalone_helpers(self) -> None:
        validate_inline_helper_sync(self.notebook)

    def test_altered_inline_byte_is_rejected(self) -> None:
        altered = deepcopy(self.notebook)
        source_cell = next(
            cell for cell in altered["cells"]
            if cell.get("id") == "inline-helper-source-3"
        )
        source = source_cell["source"]
        if isinstance(source, list):
            source[1] = source[1].replace("Rigorous", "rigorous", 1)
        else:
            source_cell["source"] = source.replace("Rigorous", "rigorous", 1)
        with self.assertRaises(AssertionError):
            validate_inline_helper_sync(altered)

    def test_altered_displayed_digest_is_rejected(self) -> None:
        altered = deepcopy(self.notebook)
        heading = next(
            cell for cell in altered["cells"]
            if cell.get("id") == "inline-helper-heading-3"
        )
        source = heading["source"]
        if isinstance(source, list):
            heading["source"] = [
                line.replace(
                    "dbe80820b21637e2cd3f24e28f6c4ee0288af81197af40a4e6871511bfa17501",
                    "0" * 64,
                )
                for line in source
            ]
        else:
            heading["source"] = source.replace(
                "dbe80820b21637e2cd3f24e28f6c4ee0288af81197af40a4e6871511bfa17501",
                "0" * 64,
            )
        with self.assertRaises(AssertionError):
            validate_inline_helper_sync(altered)

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
