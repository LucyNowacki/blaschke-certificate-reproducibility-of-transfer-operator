"""Structural tests for standalone-to-inline helper provenance."""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
import re
import tempfile
import unittest

from build_blaschke_deformation_thesis_math_notebook import (
    CHAPTER_MATH_LABEL,
    CHAPTER_MATH_MAP,
    CHAPTER_MATH_MARKER,
    CURATED_INLINE_HELPER_ORDINALS,
    DEPENDENCY_MAP_CELL_ID,
    TERMINAL_AUDITOR_CELL_ID,
    _chapter_math_entries,
    _chapter_math_payload,
    _chapter_math_source_references,
    _format_source_reference,
    _merge_preserved_execution_state,
    build_curated,
    dependency_map_cell,
    main as builder_main,
    validate_chapter_math_coverage,
    validate_curated_counterpart,
    validate_inline_helper_sync,
)


HERE = Path(__file__).resolve().parent
README = HERE.parent / "README.md"
NOTEBOOK = HERE / "blaschke_deformation_certifier_thesis_math.ipynb"
TEMPLATE = HERE / "blaschke_deformation_certifier_template.ipynb"
SOURCE = HERE / "blaschke_deformation_certifier.ipynb"
BUILDER = HERE / "build_blaschke_deformation_thesis_math_notebook.py"
INTEGRATED_REFERENCE_VERIFIER = (
    HERE / "verify_integrated_thesis_source_references.py"
)
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
            allow_chapter_math=True,
        )

    def test_chapter_math_map_covers_every_code_cell_group(self) -> None:
        validate_chapter_math_coverage(self.notebook)
        entries = _chapter_math_entries()
        self.assertEqual(CHAPTER_MATH_MAP.name, "numerical_certification_transfer_markdown.toml")
        self.assertEqual(len(entries), 52)
        mapped_code_ids = [
            str(code_id)
            for entry in entries.values()
            for code_id in entry["code_cell_ids"]
        ]
        self.assertEqual(len(mapped_code_ids), 68)
        self.assertEqual(len(set(mapped_code_ids)), 68)

        payload = _chapter_math_payload()
        references = payload["source_references"]
        cited_labels = {CHAPTER_MATH_LABEL}
        for entry in entries.values():
            cited_labels.update(entry["chapter_labels"])
        cited_labels.update(payload["terminal_auditor"]["chapter_labels"])
        self.assertEqual(payload["schema_version"], "1.2.0")
        self.assertEqual(
            payload["terminal_auditor"]["cell_id"],
            TERMINAL_AUDITOR_CELL_ID,
        )
        self.assertEqual(len(cited_labels), 57)
        self.assertEqual(len(references), 63)
        self.assertEqual(payload["integrated_thesis_driver"], "main.tex")
        self.assertEqual(payload["integrated_thesis_pdf"], "main.pdf")
        self.assertEqual(payload["integrated_thesis_page_count_at_mapping"], 278)
        self.assertRegex(
            payload["integrated_thesis_driver_sha256_at_mapping"],
            r"^[0-9a-f]{64}$",
        )
        self.assertRegex(
            payload["integrated_thesis_pdf_sha256_at_mapping"], r"^[0-9a-f]{64}$"
        )
        self.assertRegex(
            payload["integrated_thesis_aux_sha256_at_mapping"], r"^[0-9a-f]{64}$"
        )
        for label, reference in references.items():
            self.assertTrue(reference["number"], msg=label)
            if reference["kind"] == "Equation":
                self.assertNotIn("title", reference, msg=label)
            else:
                self.assertTrue(reference["title"], msg=label)
        self.assertTrue(INTEGRATED_REFERENCE_VERIFIER.is_file())

        code_ids = {
            str(cell.get("id"))
            for cell in self.notebook["cells"]
            if cell.get("cell_type") == "code"
        }
        self.assertEqual(set(mapped_code_ids), code_ids)

    def test_every_mapped_markdown_has_detailed_chapter_grounding(self) -> None:
        cells = {str(cell.get("id")): cell for cell in self.notebook["cells"]}
        for markdown_id, entry in _chapter_math_entries().items():
            source = "".join(cells[markdown_id].get("source", []))
            self.assertEqual(source.count(CHAPTER_MATH_MARKER), 1, msg=markdown_id)
            generated = source.split(CHAPTER_MATH_MARKER, 1)[1]
            self.assertIn(f"`{CHAPTER_MATH_LABEL}`", generated, msg=markdown_id)
            self.assertIn(
                "**Thesis source.** Integrated `main.pdf` locator",
                generated,
                msg=markdown_id,
            )
            self.assertIn("**Mathematical reading.**", generated, msg=markdown_id)
            self.assertIn("**Evidence status.**", generated, msg=markdown_id)
            self.assertIn("**Code covered.**", generated, msg=markdown_id)
            self.assertGreaterEqual(len(generated.split()), 55, msg=markdown_id)
            self.assertIn("$", generated, msg=markdown_id)
            for delimiter in (r"\(", r"\)", r"\[", r"\]"):
                self.assertNotIn(delimiter, generated, msg=markdown_id)
            for label in entry["chapter_labels"]:
                self.assertIn(f"`{label}`", generated, msg=markdown_id)
            references = _chapter_math_payload()["source_references"]
            for label in _chapter_math_source_references(markdown_id):
                self.assertIn(
                    _format_source_reference(label, references[label]),
                    generated,
                    msg=f"{markdown_id}: {label}",
                )

    def test_pdf_facing_titles_make_internal_labels_findable(self) -> None:
        cells = {str(cell.get("id")): cell for cell in self.notebook["cells"]}
        setup = "".join(cells["67a01ec3"].get("source", []))
        for visible_locator in (
            "Chapter 5: *Numerical certification of transfer spectra for analytic "
            "expanding interval maps*",
            "Section 5.1: *Numerical methodology for analytic expanding interval maps*",
            "Subsection 5.1.1: *Logical provenance of the certification statements*",
            "Subsection 5.1.5: *Three finite coordinate gauges*",
        ):
            self.assertIn(visible_locator, setup)

        direct_theory = "".join(cells["3f36d6e7"].get("source", []))
        for visible_locator in (
            "Chapter 4: *Single-space deterministic bounds for Legendre–EDMD "
            "transfer approximations*",
            "Section 4.3: *Single-space deterministic bridge in the Chebyshev gauge*",
            "Subsection 4.3.4: *$X$-Galerkin tail and projector mismatch*",
            "Definition 4.62: *Pointwise unresolved Chebyshev-tail kernel*",
            "Equation (4.85)",
            "Remark 4.64: *Certified finite-prefix evaluation of the pointwise kernel*",
        ):
            self.assertIn(visible_locator, direct_theory)

    def test_semantic_routes_for_matrix_tail_and_phase2_figures(self) -> None:
        entries = _chapter_math_entries()

        raw_matrix = entries["inline-helper-heading-8"]
        self.assertEqual(raw_matrix["evidence_status"], ["certified_input_producer"])
        self.assertNotIn("eq:numerics-safe-Bmat-rescaling", raw_matrix["chapter_labels"])
        self.assertIn("finite_M_certified = False", raw_matrix["mathematics"])

        sampled_kernel = entries["3f36d6e7"]
        self.assertNotIn(
            "eq:numerics-safe-chebyshev-tail",
            sampled_kernel["chapter_labels"],
        )
        for label in (
            "legEDMD:def-branch-image-Chebyshev-input-tail-kernel",
            "eq:branch-image-Chebyshev-tail-kernel",
            "legEDMD:rem-certified-finite-prefix-pointwise-kernel",
        ):
            self.assertIn(label, sampled_kernel["chapter_labels"])

        certificate_array_owners = [
            markdown_id
            for markdown_id, entry in entries.items()
            if "fig:numerics-phase2-certificate-array" in entry["chapter_labels"]
        ]
        self.assertEqual(certificate_array_owners, ["inline-helper-heading-16"])

        payload = _chapter_math_payload()
        self.assertEqual(
            payload["direct_theory_source"],
            "chapters/single_space_legendre_edmd.tex",
        )

    def test_markdown_preserves_sampled_vs_certified_and_cancellation_boundaries(self) -> None:
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        self.assertNotIn("demonstrates no cancellation gain", markdown)
        self.assertNotIn(
            "the certification table uses the sampled minimum",
            markdown,
        )
        self.assertIn(
            "equality of their endpoints does not prove that the unknown exact "
            "coherent tail has no cancellation",
            markdown,
        )
        self.assertIn(
            "the sampled minimum is an upper estimate of the true moat, not a "
            "certified lower bound",
            markdown,
        )

    def test_final_auditor_markdown_explains_every_terminal_certificate_layer(self) -> None:
        cells = {
            str(cell.get("id")): cell for cell in self.notebook["cells"]
        }
        owner_source = "".join(cells["bffa1d01"].get("source", []))
        self.assertIn("## Structural count-and-moat provenance tests", owner_source)
        self.assertNotIn(
            "## Final auditor reading of the twenty-four-target spectral certificate",
            owner_source,
        )
        self.assertEqual(
            str(self.notebook["cells"][-1].get("id")), TERMINAL_AUDITOR_CELL_ID
        )
        source = "".join(cells[TERMINAL_AUDITOR_CELL_ID].get("source", []))
        self.assertIn(
            "## Final auditor reading of the twenty-four-target spectral certificate",
            source,
        )
        for required in (
            "110M",
            "The certificate output immediately above is emitted by execution Cell 107N.",
            "Meaning of `computed_and_certified`",
            "How the deterministic perturbation envelope is obtained",
            "How the finite Hardy matrix and Schur constants are certified",
            "The two complete-circle moat routes and their transport",
            "Row-by-row meaning of the displayed twenty-four-target table",
            "Why lower displays round down and upper displays round up",
            "Laurent source reconstruction and the exact meaning of provenance-only",
            "What is proved, what is hybrid, and what is not proved",
            "1.636120221822047\\times10^{-13}",
            "2.033128947087571\\times10^{-7}",
            "2.033128947087573\\times10^{-7}<1",
            "display/provenance drift",
            "44+36+48+48+36+40+48=300",
            "`digest_used_in_theorem_gate = False`",
            "Cells 108N and 109N contain only their labels",
            "Starting from an empty output directory does not ask the digest to prove the result.",
        ):
            self.assertIn(required, source)
        self.assertEqual(
            len(re.findall(r"(?m)^\| (?:[1-9]|1[0-9]|2[0-4]) \|", source)),
            24,
        )
        for delimiter in (r"\(", r"\)", r"\[", r"\]"):
            self.assertNotIn(delimiter, source)

        certificate_cell = cells["code-b33b0f47"]
        certificate_output = "\n".join(
            "".join(output.get("text", []))
            for output in certificate_cell.get("outputs", [])
            if output.get("output_type") == "stream"
        )
        self.assertIn(
            "Twenty-four-target spectral certificate: computed_and_certified",
            certificate_output,
        )
        for terminal_id, label in (
            ("3e8b784c", "#108N\n"),
            ("128b5369", "#109N\n"),
        ):
            terminal = cells[terminal_id]
            self.assertEqual("".join(terminal.get("source", [])), label)
            self.assertEqual(terminal.get("outputs", []), [])
        self.assertEqual(
            self.notebook["cells"][-2].get("id"), "128b5369"
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

    def test_cell_3m_uses_dollar_math_delimiters(self) -> None:
        source = "".join(
            next(
                cell for cell in self.notebook["cells"]
                if cell.get("id") == "33580cd0"
            ).get("source", [])
        )
        for delimiter in (r"\(", r"\)", r"\[", r"\]"):
            self.assertNotIn(delimiter, source)
        self.assertIn("$\\tau_b$", source)
        self.assertGreaterEqual(source.count("$$"), 4)

    def test_cell_80n_embeds_the_alpha11_sampled_profile(self) -> None:
        cell = next(
            cell for cell in self.notebook["cells"]
            if cell.get("id") == "32961420"
        )
        alpha11_path = (
            HERE
            / "outputs"
            / "blaschke_deformation_certifier"
            / "figures"
            / "branch_image_wide_candidate_sampled_hardy_moat_profile_alpha11_N600_M610.png"
        )
        embedded_pngs = [
            base64.b64decode(output["data"]["image/png"])
            for output in cell.get("outputs", [])
            if "image/png" in output.get("data", {})
        ]
        self.assertEqual(len(embedded_pngs), 2)
        self.assertIn(alpha11_path.read_bytes(), embedded_pngs)
        output_text = "".join(
            str(output.get("text", ""))
            + str(output.get("data", {}).get("text/plain", ""))
            + str(output.get("data", {}).get("text/html", ""))
            for output in cell.get("outputs", [])
        )
        self.assertIn("alpha^11", output_text)

    def test_provenance_png_contract_includes_the_restored_alpha11_profile(self) -> None:
        provenance = json.loads(
            (HERE / "notebook_cell_provenance.json").read_text(encoding="utf-8")
        )
        actual_png_count = sum(
            "image/png" in output.get("data", {})
            for cell in self.notebook["cells"]
            for output in cell.get("outputs", [])
        )
        actual_visual_cells = sum(
            cell.get("cell_type") == "code"
            and any(
                "image/png" in output.get("data", {})
                for output in cell.get("outputs", [])
            )
            for cell in self.notebook["cells"]
        )
        self.assertEqual(actual_png_count, 34)
        self.assertEqual(actual_visual_cells, 20)
        self.assertEqual(
            provenance["notebook"]["expected_stored_png_output_count"],
            actual_png_count,
        )
        self.assertEqual(
            provenance["notebook"]["expected_visual_cell_count"],
            actual_visual_cells,
        )
        self.assertIn("alpha^11", provenance["notebook"]["stored_png_count_note"])

    def test_release_docs_do_not_claim_the_archive_is_still_deferred(self) -> None:
        stale_wording = "archive remains deferred"
        for path in (README, HERE / "blaschke_deformation_notebook_dependency_map.md"):
            self.assertNotIn(stale_wording, path.read_text(encoding="utf-8"), msg=str(path))
        self.assertNotIn(
            stale_wording,
            "".join(self.notebook["cells"][0].get("source", [])),
        )

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
            self.assertEqual(actual.get("id"), expected.get("id"))
            self.assertEqual(actual.get("cell_type"), expected.get("cell_type"))
            if expected.get("cell_type") != "code":
                continue
            actual_source = actual.get("source", "")
            expected_source = expected.get("source", "")
            self.assertEqual(
                "".join(actual_source)
                if isinstance(actual_source, list)
                else str(actual_source),
                "".join(expected_source)
                if isinstance(expected_source, list)
                else str(expected_source),
            )
            self.assertEqual(
                actual.get("execution_count"), expected.get("execution_count")
            )
            self.assertEqual(actual.get("outputs", []), expected.get("outputs", []))
        self.assertIn("source_sync_after_execution", merged["metadata"])

    def test_source_refresh_can_append_terminal_auditor_to_legacy_execution(self) -> None:
        current, _ = build_curated()
        legacy = deepcopy(self.notebook)
        self.assertEqual(
            legacy["cells"].pop().get("id"), TERMINAL_AUDITOR_CELL_ID
        )
        merged = _merge_preserved_execution_state(current, legacy)
        validate_curated_counterpart(merged)
        self.assertEqual(
            merged["cells"][-1].get("id"), TERMINAL_AUDITOR_CELL_ID
        )
        self.assertEqual(len(merged["cells"]), len(legacy["cells"]) + 1)

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
                allow_chapter_math=True,
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
                allow_chapter_math=True,
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
