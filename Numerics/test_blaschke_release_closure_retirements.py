"""Regression gates for the corrective generated-evidence closure."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT = HERE / "outputs" / "blaschke_deformation_certifier"

RETIRED_RELEASE_MEMBERS = (
    OUTPUT / "data/branch_image_phase1_raw_eigs_N600_M610.csv",
    OUTPUT / "data/branch_image_wide_candidate_single_space_row_N600_M610.csv",
    OUTPUT / "data/phase4_diag_A_X_N600_M610.npz",
    OUTPUT / "data/phase4_diag_hardy_gauge_eigenvalues_N600_M610.csv",
    OUTPUT / "data/phase4_hp_A_X_N600_M610.npz",
    OUTPUT / "data/branch_image_wide_candidate_single_space_row_N600_M610_D_M_1_diagnostic.csv",
    OUTPUT / "data/finite_M_safe_GL_prefactor_certificate_wide_N600_M610.csv",
    OUTPUT / "reports/finite_M_safe_GL_prefactor_certificate_wide_N600_M610.json",
    OUTPUT / "reports/finite_M_safe_GL_prefactor_certificate_wide_N600_M610.md",
    OUTPUT / "data/final_blaschke_branch-image_wide_N600_schur_certificate.csv",
    OUTPUT / "data/transfer_lab_hyperparameters.json",
    OUTPUT / "figures/branch_image_wide_candidate_sampled_hardy_moat_profile_alpha5_N600_M610.png",
    OUTPUT / "reports/branch_image_wide_candidate_first15_sampled_validation_report.md",
    HERE / "outputs/figures/branch_image_wide_candidate_first15_global_hardy_moat_surface_3d_2d_N600_M610.png",
)

RETIRED_REFERENCE_TOKENS = tuple(
    path.name
    for path in RETIRED_RELEASE_MEMBERS
    if path.name
    not in {
        "transfer_lab_hyperparameters.json",
        "branch_image_wide_candidate_first15_global_hardy_moat_surface_3d_2d_N600_M610.png",
    }
) + (
    'DATA_DIR / "transfer_lab_hyperparameters.json"',
    "DATA_DIR / 'transfer_lab_hyperparameters.json'",
    'data_dir / "transfer_lab_hyperparameters.json"',
    "data_dir / 'transfer_lab_hyperparameters.json'",
    "outputs/figures/branch_image_wide_candidate_first15_global_hardy_moat_surface_3d_2d_N600_M610.png",
    "Numerics/outputs/figures/branch_image_wide_candidate_first15_global_hardy_moat_surface_3d_2d_N600_M610.png",
)

CLEAN_ROUTE_FILES = (
    HERE / "blaschke_deformation_certifier_template.ipynb",
    HERE / "blaschke_deformation_certifier.ipynb",
    HERE / "blaschke_deformation_certifier_thesis_math.ipynb",
    HERE / "blaschke_deformation_phase2_final_aggregation.py",
    HERE / "blaschke_deformation_reproducibility.py",
    HERE / "notebook_cell_provenance.json",
    HERE / "plotting_cell_replacements.json",
    HERE / "transfer_lab_hyperparameters.json",
    HERE / "blaschke_deformation_reproducibility_plan.json",
)


class ReleaseClosureRetirementTests(unittest.TestCase):
    def test_exact_fourteen_members_are_absent(self) -> None:
        self.assertEqual(len(RETIRED_RELEASE_MEMBERS), 14)
        self.assertEqual(len(set(RETIRED_RELEASE_MEMBERS)), 14)
        self.assertEqual(
            [str(path.relative_to(ROOT)) for path in RETIRED_RELEASE_MEMBERS],
            [
                "Numerics/outputs/blaschke_deformation_certifier/data/branch_image_phase1_raw_eigs_N600_M610.csv",
                "Numerics/outputs/blaschke_deformation_certifier/data/branch_image_wide_candidate_single_space_row_N600_M610.csv",
                "Numerics/outputs/blaschke_deformation_certifier/data/phase4_diag_A_X_N600_M610.npz",
                "Numerics/outputs/blaschke_deformation_certifier/data/phase4_diag_hardy_gauge_eigenvalues_N600_M610.csv",
                "Numerics/outputs/blaschke_deformation_certifier/data/phase4_hp_A_X_N600_M610.npz",
                "Numerics/outputs/blaschke_deformation_certifier/data/branch_image_wide_candidate_single_space_row_N600_M610_D_M_1_diagnostic.csv",
                "Numerics/outputs/blaschke_deformation_certifier/data/finite_M_safe_GL_prefactor_certificate_wide_N600_M610.csv",
                "Numerics/outputs/blaschke_deformation_certifier/reports/finite_M_safe_GL_prefactor_certificate_wide_N600_M610.json",
                "Numerics/outputs/blaschke_deformation_certifier/reports/finite_M_safe_GL_prefactor_certificate_wide_N600_M610.md",
                "Numerics/outputs/blaschke_deformation_certifier/data/final_blaschke_branch-image_wide_N600_schur_certificate.csv",
                "Numerics/outputs/blaschke_deformation_certifier/data/transfer_lab_hyperparameters.json",
                "Numerics/outputs/blaschke_deformation_certifier/figures/branch_image_wide_candidate_sampled_hardy_moat_profile_alpha5_N600_M610.png",
                "Numerics/outputs/blaschke_deformation_certifier/reports/branch_image_wide_candidate_first15_sampled_validation_report.md",
                "Numerics/outputs/figures/branch_image_wide_candidate_first15_global_hardy_moat_surface_3d_2d_N600_M610.png",
            ],
        )
        self.assertEqual(
            [str(path) for path in RETIRED_RELEASE_MEMBERS if path.exists()], []
        )

    def test_clean_route_has_no_retired_member_reference(self) -> None:
        for source_path in CLEAN_ROUTE_FILES:
            text = source_path.read_text(encoding="utf-8")
            if source_path.suffix == ".ipynb":
                notebook = json.loads(text)
                text = "\n".join(
                    "".join(cell.get("source", []))
                    if isinstance(cell.get("source", []), list)
                    else str(cell.get("source", ""))
                    for cell in notebook.get("cells", [])
                )
            for token in RETIRED_REFERENCE_TOKENS:
                with self.subTest(source=source_path.name, retired=token):
                    self.assertNotIn(token, text)

    def test_hyperparameter_archive_is_source_only(self) -> None:
        source_archive = HERE / "transfer_lab_hyperparameters.json"
        generated_copy = OUTPUT / "data/transfer_lab_hyperparameters.json"
        self.assertTrue(source_archive.is_file())
        self.assertFalse(generated_copy.exists())
        record = json.loads(source_archive.read_text(encoding="utf-8"))[
            "blaschke_mu_0p3"
        ]
        self.assertFalse(record["theorem_gate_eligible"])
        self.assertNotIn("schur_row", record)
        self.assertEqual(record["deterministic_status"], "historical diagnostic only")


if __name__ == "__main__":
    unittest.main()
