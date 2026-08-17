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


if __name__ == "__main__":
    unittest.main()
