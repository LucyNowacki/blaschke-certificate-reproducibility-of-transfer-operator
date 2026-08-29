"""Regression tests for the integrated-thesis PDF-locator verifier."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import verify_integrated_thesis_source_references as verifier


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class IntegratedThesisSourceReferenceTests(unittest.TestCase):
    def test_visible_title_normalisation_preserves_pdf_en_dash(self) -> None:
        self.assertEqual(
            verifier._normalise_visible_text("Legendre--Gauss Galerkin--EDMD"),
            "Legendre–Gauss Galerkin–EDMD",
        )

    def test_verifier_executes_hash_number_title_and_caption_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "chapters").mkdir()
            artifacts = {
                "main.tex": b"fixture driver\n",
                "main.pdf": b"fixture integrated PDF snapshot\n",
                "main.aux": (
                    b"\\newlabel{chap:test}{{1}{1}{Single-space Legendre--EDMD}"
                    b"{chapter.1}{}}\n"
                    b"\\newlabel{fig:test}{{1.1}{2}{Short caption}"
                    b"{figure.caption.1}{}}\n"
                    b"\\newlabel{eq:test}{{1.1}{2}{Context}"
                    b"{equation.1.1}{}}\n"
                ),
                "chapters/numerical.tex": (
                    b"\\chapter{Single-space Legendre--EDMD}\\label{chap:test}\n"
                    b"\\begin{figure}\\caption[Short caption]{Visible caption lead. "
                    b"Further explanation.}\\label{fig:test}\\end{figure}\n"
                    b"\\begin{equation}x=1\\label{eq:test}\\end{equation}\n"
                ),
                "chapters/theory.tex": b"fixture theory source\n",
            }
            for relative, data in artifacts.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)

            reference_map = root / "reference-map.toml"
            reference_map.write_text(
                f'''integrated_thesis_driver = "main.tex"
integrated_thesis_driver_sha256_at_mapping = "{_digest(artifacts['main.tex'])}"
integrated_thesis_pdf = "main.pdf"
integrated_thesis_pdf_sha256_at_mapping = "{_digest(artifacts['main.pdf'])}"
integrated_thesis_aux = "main.aux"
integrated_thesis_aux_sha256_at_mapping = "{_digest(artifacts['main.aux'])}"
integrated_thesis_page_count_at_mapping = 1
chapter_source = "chapters/numerical.tex"
chapter_source_sha256_at_mapping = "{_digest(artifacts['chapters/numerical.tex'])}"
direct_theory_source = "chapters/theory.tex"
direct_theory_source_sha256_at_mapping = "{_digest(artifacts['chapters/theory.tex'])}"

[source_references."chap:test"]
kind = "Chapter"
number = "1"
title = "Single-space Legendre–EDMD"

[source_references."fig:test"]
kind = "Figure"
number = "1.1"
title = "Visible caption lead"
parent = "chap:test"

[source_references."eq:test"]
kind = "Equation"
number = "1.1"
parent = "chap:test"
''',
                encoding="utf-8",
            )

            with patch.object(verifier, "REFERENCE_MAP", reference_map):
                report = verifier.verify(root)

            self.assertEqual(report["verification_status"], "verified")
            self.assertEqual(report["number_checks"], 3)
            self.assertEqual(report["named_title_checks"], 1)
            self.assertEqual(report["caption_lead_checks"], 1)
            self.assertEqual(report["equation_title_policy"], "number_only")


if __name__ == "__main__":
    unittest.main()
