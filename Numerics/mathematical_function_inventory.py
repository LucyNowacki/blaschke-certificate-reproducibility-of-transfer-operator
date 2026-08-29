"""Maintain the audited mathematical-function documentation inventory.

The inventory is deliberately selective: every function in the eighteen
inline helpers and every function in a non-inline code cell of the final
notebook is classified, while only functions listed below as ``mathematical``
receive theorem-facing docstrings. Plotting and orchestration functions remain
explicitly outside that class.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy
from typing import Iterable


HERE = Path(__file__).resolve().parent
BUILDER = HERE / "build_blaschke_deformation_thesis_math_notebook.py"
FINAL_NOTEBOOK = HERE / "blaschke_deformation_certifier_thesis_math.ipynb"
NOTEBOOK_CHAIN = (
    HERE / "blaschke_deformation_certifier_template.ipynb",
    HERE / "blaschke_deformation_certifier.ipynb",
    FINAL_NOTEBOOK,
)
INVENTORY = HERE / "mathematical_function_inventory.json"
PLOTTING_REPLACEMENTS = HERE / "plotting_cell_replacements.json"


MATH_HELPER_FUNCTIONS: dict[str, frozenset[str]] = {
    "mpmath_pf_raw.py": frozenset({
        "frobenius_norm_mp", "gauss_legendre_mp", "ortho_legendre_values",
        "_eval_expr", "FormulaTransferMap.tau", "FormulaTransferMap.phi",
        "exact_clusters_for_map", "_assemble_transfer_row_block_job",
        "assemble_transfer_block", "assemble_pure_scaled_transfer_block",
        "greedy_cluster_errors", "build_reference_clusters_mpmath_raw",
    }),
    "transfer_spectrum_certification.py": frozenset({
        "balanced_single_space_geometry", "select_sampled_geometry",
        "_theta_log", "_tail_T", "_S_infty", "_C2", "_Cinf",
        "_CGL_core", "_input_tail", "sampled_schur_envelope",
        "build_certification_audit",
    }),
    "blaschke_deformation_certification.py": frozenset({
        "tau_phi", "bernstein_radius_upper", "selected_joukowski_preimage",
        "packet_value", "packet_value_from_preimage", "local_tail_l2_upper",
        "certify_input_tail_rows", "resolved_restriction_l2_upper",
        "certify_resolved_response_rows",
    }),
    "blaschke_deformation_spectral_certification.py": frozenset({
        "_orthonormal_legendre_values", "_gauss_legendre_rule",
        "_real_branch_values", "_connection_matrix",
        "assemble_interval_hardy_matrix", "_dyadic_midpoint_payload",
        "_precision_audit", "build_or_load_hardy_matrix_certificate",
        "load_exact_dyadic_midpoint",
    }),
    "blaschke_deformation_contour_certification.py": frozenset({
        "_numpy_to_acb_exact", "_frobenius_upper", "_target_contours",
        "_validate_schur_similarity", "_certified_schur_diagonal_geometry",
        "_uniform_triangular_inverse_bound", "_schur_contour_attempt",
        "_laurent_coefficients", "_laurent_contour_certificate",
        "_validate_target_contour_plan", "_existing_geometry_is_reusable",
        "_reaggregate_small_gain_rows", "_reaggregate_existing_certificate",
        "certify_all_target_contours",
    }),
    "blaschke_deformation_phase2_geometry.py": frozenset({
        "_tau_phi", "_bernstein_radius_upper", "_geometry_block", "_maximum",
        "_minimum", "_u_tail", "_c2", "_s_infinity", "_sum_n_qn_from",
        "_zwx13_coefficient_bound", "_legendre_h2_mode_bound",
        "_zwx13_tail_after", "_zwx13_tail", "_singularity_radius_lower",
        "_rho_boundary_data", "certify_geometry_scan",
    }),
    "blaschke_deformation_phase2_transport.py": frozenset({
        "_alpha_values", "connection_matrix_float", "_alpha_values_arb",
        "connection_matrix_arb", "_connection_entry_arb",
        "_transport_row_block", "_certify_transport_by_row_blocks",
        "_float_matrix_as_arb", "_identity", "_frobenius_upper",
        "certify_transport",
    }),
    "blaschke_deformation_phase2_matrix.py": frozenset({
        "_theta_sum", "_c_gl_core", "_matrix_bound", "certify_matrix_row",
    }),
    "blaschke_deformation_phase2_pipeline.py": frozenset(),
    "blaschke_deformation_historical_comparisons.py": frozenset({
        "_fixed_boundary", "_branch_image_input_upper",
        "rebuild_historical_comparisons",
    }),
    "hardy_moat_surface_worker.py": frozenset({"sample_surface_row_block"}),
    "blaschke_deformation_historical_phase4.py": frozenset({
        "blaschke_formula_map_spec", "_alpha_array", "connection_T_matrix",
        "build_hardy_matrix", "_build_target_contours", "_packet_matches",
        "_smallest_singular_value", "_sample_contour", "_finite_geometry",
        "_moat_record", "_diagnostic_tables",
        "_validate_historical_regression_profile", "_surface_windows",
        "_sample_surface", "rebuild_historical_phase4",
    }),
    "blaschke_deformation_diagnostic_audits.py": frozenset({
        "_build_certification_audit", "_build_universal_audit",
        "_build_first14_audit", "_build_first14_audit.packet_riesz_rank_status",
        "generate_diagnostic_audit_frames",
        "generate_universal_certification_audit", "generate_first14_packet_audit",
    }),
    "blaschke_deformation_phase2_finite_m.py": frozenset({
        "certify_finite_m_completion",
    }),
    "blaschke_deformation_phase2_resolved_response.py": frozenset({
        "_tau_phi_extended", "_bernstein_radius_upper", "_legendre_kernel",
        "certify_resolved_response_completion",
    }),
    "blaschke_deformation_phase2_final_aggregation.py": frozenset({
        "_best_matrix", "_b_out", "certify_final_phase2_aggregation",
    }),
    "blaschke_deformation_phase1_diagnostics.py": frozenset({
        "_with_oversampling_columns", "rebuild_phase1_diagnostics",
    }),
    "blaschke_deformation_sampled_schur_diagnostics.py": frozenset({
        "rebuild_sampled_schur_diagnostics",
    }),
}


MATH_NOTEBOOK_FUNCTIONS: dict[str, frozenset[str]] = {
    "0e01d379": frozenset({"frobenius_mp"}),
    "d09d9963": frozenset({"blaschke_formula_map_spec"}),
    "ce7ba80a": frozenset({
        "ellipse_point", "bernstein_radius", "validate_real_branches",
        "diagnostic_branch_geometry",
    }),
    "2a6f973c": frozenset({
        "phase1_positive_target_rows", "phase1_summary_table",
        "phase1_empirical_exponential_reference_curve",
    }),
    "c1b4b6e4": frozenset({
        "add_phase1_oversampling_columns", "add_phase1_reference_error_ratios",
    }),
    "9b8c4f29": frozenset({
        "D_scaled_block", "cheb_mul_x", "legendre_to_cheb_coeffs_orthonormal",
        "C_X_matrix", "T_connection", "kappa_T_numeric",
    }),
    "6451c7fe": frozenset({
        "jouk", "jouk_np", "outer_joukowski_inverse", "bernstein_radius",
        "ellipse_points_np", "phase2_best_matrix_from_row",
        "phase2_B_out_from_row", "phase2_build_summary_rows",
        "phase2_promoted_certificate",
    }),
    "44cb0b52": frozenset({"sample_branch_boundary_profile"}),
    "0443774e": frozenset({
        "U_tail_closed", "K_tail_point", "branch_image_tail_profile",
    }),
    "producer-sampled-schur-diagnostics": frozenset({"_sampled_schur_kappa"}),
}


PLOTTING_NOTEBOOK_CELLS = frozenset({
    "6a823024", "cc338a81", "59a1c39c", "340f0292", "add52fcf", "8bf58bd5",
})
PLOTTING_NOTEBOOK_NAMES = frozenset({
    "transfer_lab_plot_status_summary", "transfer_lab_plot_phase3_diagnostic",
    "transfer_lab_plot_phase4_diagnostic", "gradient_colours",
    "phase1_gradient_palette", "display",
})


FUNCTIONALITY_OVERRIDES: dict[tuple[str, str], str] = {
    ("blaschke_deformation_contour_certification.py", "_numpy_to_acb_exact"):
        "Convert every binary64 complex matrix entry exactly to an Arb/ACB ball and return the resulting matrix.",
    ("blaschke_deformation_contour_certification.py", "_frobenius_upper"):
        "Sum outward-rounded squared entry magnitudes and return a certified upper bound for the Frobenius norm.",
    ("blaschke_deformation_contour_certification.py", "_target_contours"):
        "Construct and order the 24 exact rational alpha- and mu-family contours with separated radii, multiplicities, and Laurent fallback settings.",
    ("blaschke_deformation_contour_certification.py", "_schur_contour_attempt"):
        "Derive one Schur-diagonal count, triangular complete-circle moat, perturbation transports, and finite-to-exact small-gain verdict.",
    ("blaschke_deformation_contour_certification.py", "_laurent_coefficients"):
        "Sample the resolvent on an equispaced circle and Fourier-transform the samples into proposed Laurent inverse coefficients and residual diagnostics.",
    ("blaschke_deformation_diagnostic_audits.py", "_build_universal_audit"):
        "Build the locked diagnostic-only universal audit while preventing sampled evidence from being promoted to a theorem certificate.",
    ("blaschke_deformation_diagnostic_audits.py", "_build_first14_audit"):
        "Merge the first-fourteen targets with sampled moat and count diagnostics and label every resulting row as non-theorem evidence.",
    ("blaschke_deformation_diagnostic_audits.py", "_build_first14_audit.packet_riesz_rank_status"):
        "Classify one packet from its sampled count and small-gain outcomes without asserting certified Riesz-rank equality.",
    ("blaschke_deformation_historical_comparisons.py", "_fixed_boundary"):
        "Evaluate the complete Arb boundary cover for the fixed historical rho and q-gap design.",
    ("blaschke_deformation_historical_comparisons.py", "_branch_image_input_upper"):
        "Maximise the branch-image unresolved-input tail bound over all supplied boundary cells and return the attaining cell.",
    ("blaschke_deformation_historical_phase4.py", "_alpha_array"):
        "Generate the central-binomial coefficients alpha_j by their stable first-order recurrence.",
    ("blaschke_deformation_historical_phase4.py", "_build_target_contours"):
        "Derive the retained target contours from exact map clusters and finite-eigenvalue mid-gaps while excluding zero.",
    ("blaschke_deformation_historical_phase4.py", "_packet_matches"):
        "Assign finite eigenvalues to target packets by multiplicity and record each packet's largest sampled displacement.",
    ("blaschke_deformation_historical_phase4.py", "_smallest_singular_value"):
        "Compute the smallest singular value of the shifted Hardy matrix zeta I minus A for one sampled point.",
    ("blaschke_deformation_historical_phase4.py", "_sample_contour"):
        "Evaluate the smallest singular value on an equispaced grid around one retained circular contour.",
    ("blaschke_deformation_historical_phase4.py", "_finite_geometry"):
        "Count finite eigenvalues inside one contour and return the enclosed cluster radius and nearest exterior distance.",
    ("blaschke_deformation_historical_phase4.py", "_moat_record"):
        "Combine sampled singular minima, zero separation, packet matching, and epsilon into one diagnostic moat and small-gain row.",
    ("blaschke_deformation_historical_phase4.py", "_diagnostic_tables"):
        "Assemble the retained contour-profile, moat, packet, and robustness tables from source-generated samples.",
    ("blaschke_deformation_historical_phase4.py", "_surface_windows"):
        "Choose the locked local and global complex-plane windows used for retained Hardy-moat surface sampling.",
    ("blaschke_deformation_historical_phase4.py", "_sample_surface"):
        "Evaluate the shifted-matrix smallest singular value over a rectangular complex grid using source-keyed row blocks.",
    ("blaschke_deformation_phase1_diagnostics.py", "_with_oversampling_columns"):
        "Add finite-M excess and normalised error columns relative to the largest sampled quadrature order at each N.",
    ("blaschke_deformation_phase2_final_aggregation.py", "_best_matrix"):
        "Select the smaller available certified starred or ordinary transported matrix contribution and retain its provenance.",
    ("blaschke_deformation_phase2_final_aggregation.py", "_b_out"):
        "Recover the certified output-leakage contribution from either its direct field or the stored doubled quantity.",
    ("blaschke_deformation_phase2_geometry.py", "_tau_phi"):
        "Evaluate both inverse branches of the symmetric Blaschke map and their transfer weights in Arb arithmetic.",
    ("blaschke_deformation_phase2_geometry.py", "_bernstein_radius_upper"):
        "Return an outward-rounded Bernstein radius upper bound for a complex branch-image point.",
    ("blaschke_deformation_phase2_geometry.py", "_geometry_block"):
        "Enclose both inverse-branch images and weighted branch profiles over one block of boundary cells.",
    ("blaschke_deformation_phase2_geometry.py", "_maximum"):
        "Return an Arb upper enclosure of the maximum over a nonempty collection of interval quantities.",
    ("blaschke_deformation_phase2_geometry.py", "_minimum"):
        "Return an Arb lower enclosure of the minimum over a nonempty collection of interval quantities.",
    ("blaschke_deformation_phase2_geometry.py", "_u_tail"):
        "Evaluate the closed geometric unresolved-input tail factor beginning at mode N.",
    ("blaschke_deformation_phase2_geometry.py", "_c2"):
        "Evaluate the ellipse L2 coefficient factor C2 at the supplied certified radius.",
    ("blaschke_deformation_phase2_geometry.py", "_s_infinity"):
        "Evaluate the closed infinite weighted geometric sum used in the analytic tail bounds.",
    ("blaschke_deformation_phase2_geometry.py", "_sum_n_qn_from"):
        "Evaluate the closed tail sum of n q^n starting at the requested index.",
    ("blaschke_deformation_phase2_geometry.py", "_zwx13_coefficient_bound"):
        "Compute the ZWX13-style coefficient majorant for one Legendre mode and certified ellipse radius.",
    ("blaschke_deformation_phase2_geometry.py", "_legendre_h2_mode_bound"):
        "Bound one orthonormal Legendre mode in the Hardy/ellipse norm used by the Phase 2 tail estimate.",
    ("blaschke_deformation_phase2_geometry.py", "_zwx13_tail_after"):
        "Bound the ZWX13 output tail strictly after a selected truncation index.",
    ("blaschke_deformation_phase2_geometry.py", "_zwx13_tail"):
        "Combine the retained prefix and closed remainder into the certified ZWX13 output-tail bound.",
    ("blaschke_deformation_phase2_geometry.py", "_singularity_radius_lower"):
        "Certify a lower bound for the nearest branch singularity's Bernstein radius.",
    ("blaschke_deformation_phase2_geometry.py", "_rho_boundary_data"):
        "Evaluate and aggregate the complete certified boundary cover for one candidate rho.",
    ("blaschke_deformation_phase2_matrix.py", "_theta_sum"):
        "Evaluate the finite weighted sum Theta_N(value) with outward-rounded Arb arithmetic.",
    ("blaschke_deformation_phase2_matrix.py", "_c_gl_core"):
        "Evaluate the certified Gauss-Legendre remainder prefactor core at rho.",
    ("blaschke_deformation_phase2_matrix.py", "_matrix_bound"):
        "Assemble the certified pure-scaled matrix-defect bound from the Gauss-Legendre, profile, decay, and Theta factors.",
    ("blaschke_deformation_phase2_resolved_response.py", "_tau_phi_extended"):
        "Evaluate the inverse branches and transfer weights on the extended boundary cover used for response completion.",
    ("blaschke_deformation_phase2_resolved_response.py", "_bernstein_radius_upper"):
        "Return an outward-rounded Bernstein radius upper bound for an extended-cover branch image.",
    ("blaschke_deformation_phase2_resolved_response.py", "_legendre_kernel"):
        "Evaluate the finite orthonormal Legendre Christoffel kernel at the supplied complex point.",
    ("blaschke_deformation_phase2_transport.py", "_alpha_values"):
        "Generate binary64 central-binomial coefficients alpha_j by recurrence for the diagnostic connection matrix.",
    ("blaschke_deformation_phase2_transport.py", "_alpha_values_arb"):
        "Generate rigorously enclosed central-binomial coefficients alpha_j by exact Arb recurrence.",
    ("blaschke_deformation_phase2_transport.py", "connection_matrix_arb"):
        "Construct the rigorous finite scaled-Legendre to Chebyshev-packet connection matrix in Arb arithmetic.",
    ("blaschke_deformation_phase2_transport.py", "_float_matrix_as_arb"):
        "Embed a binary64 matrix entrywise as exact Arb values for residual certification.",
    ("blaschke_deformation_phase2_transport.py", "_identity"):
        "Construct the exact Arb identity matrix of the requested dimension.",
    ("blaschke_deformation_phase2_transport.py", "_frobenius_upper"):
        "Return an outward-rounded Frobenius-norm upper bound for an Arb matrix.",
    ("blaschke_deformation_spectral_certification.py", "_precision_audit"):
        "Compare the available Arb precision with twice the transport scaling span, requested resolution, and guard-digit budget.",
    ("transfer_spectrum_certification.py", "_theta_log"):
        "Evaluate log Theta_N(t) by log-sum-exp to avoid overflow in sampled diagnostics.",
    ("transfer_spectrum_certification.py", "_tail_T"):
        "Evaluate the closed weighted geometric tail sum beginning at N.",
    ("transfer_spectrum_certification.py", "_S_infty"):
        "Evaluate the complete weighted geometric sum (1+q)/(1-q)^2.",
    ("transfer_spectrum_certification.py", "_C2"):
        "Evaluate the L2 ellipse coefficient factor at the sampled radius.",
    ("transfer_spectrum_certification.py", "_Cinf"):
        "Evaluate the uniform ellipse coefficient factor at the sampled radius.",
    ("transfer_spectrum_certification.py", "_CGL_core"):
        "Evaluate the sampled Gauss-Legendre remainder prefactor core at rho.",
    ("transfer_spectrum_certification.py", "_input_tail"):
        "Evaluate the closed unresolved-input tail factor from N, r, and the branch-image radius.",
    ("cell:0443774e", "U_tail_closed"):
        "Evaluate the notebook's closed geometric unresolved-input tail U_N(q).",
    ("cell:0443774e", "K_tail_point"):
        "Evaluate the pointwise Chebyshev tail kernel after mode N for one complex argument.",
    ("cell:0443774e", "branch_image_tail_profile"):
        "Sample both branch images on the ellipse boundary and return the pointwise unresolved-input tail profile.",
    ("cell:0e01d379", "frobenius_mp"):
        "Compute a high-precision Frobenius norm by summing squared mpmath entry magnitudes.",
    ("cell:2a6f973c", "phase1_positive_target_rows"):
        "Select the nontrivial positive exact-target rows used in the Phase 1 empirical tables.",
    ("cell:2a6f973c", "phase1_summary_table"):
        "Aggregate Phase 1 cluster errors by truncation and quadrature order into the displayed summary table.",
    ("cell:2a6f973c", "phase1_empirical_exponential_reference_curve"):
        "Construct the explicitly diagnostic empirical exponential guide used beside the Phase 1 errors.",
    ("cell:44cb0b52", "sample_branch_boundary_profile"):
        "Sample the two inverse branches and their Bernstein radii around one ellipse boundary for diagnostic geometry plots.",
    ("cell:6451c7fe", "jouk"):
        "Evaluate the scalar Joukowski map z plus z inverse over two.",
    ("cell:6451c7fe", "jouk_np"):
        "Evaluate the Joukowski map elementwise on a NumPy complex array.",
    ("cell:6451c7fe", "outer_joukowski_inverse"):
        "Select the outer solution of the Joukowski inverse while preserving the prescribed branch convention.",
    ("cell:6451c7fe", "bernstein_radius"):
        "Compute the Bernstein radius associated with one complex point through the outer Joukowski inverse.",
    ("cell:6451c7fe", "ellipse_points_np"):
        "Generate an equispaced complex parametrisation of the Bernstein ellipse of radius rho.",
    ("cell:6451c7fe", "phase2_best_matrix_from_row"):
        "Select the smaller available certified ordinary or starred Phase 2 matrix contribution from one row.",
    ("cell:6451c7fe", "phase2_B_out_from_row"):
        "Recover the Phase 2 output-leakage contribution from its direct or doubled stored field.",
    ("cell:6451c7fe", "phase2_build_summary_rows"):
        "Build the comparison rows for historical, candidate, promoted, and wide Phase 2 certificates.",
    ("cell:6451c7fe", "phase2_promoted_certificate"):
        "Compute the promoted root-sum-square deterministic radius and expose its selected input and matrix provenance.",
    ("cell:9b8c4f29", "cheb_mul_x"):
        "Apply multiplication by x to a finite Chebyshev coefficient vector using the exact three-term rule.",
    ("cell:9b8c4f29", "legendre_to_cheb_coeffs_orthonormal"):
        "Convert one orthonormal Legendre polynomial into finite Chebyshev coefficients.",
    ("cell:9b8c4f29", "T_connection"):
        "Construct the finite scaled-Legendre to Chebyshev-packet connection matrix used by the notebook.",
    ("cell:9b8c4f29", "kappa_T_numeric"):
        "Compute the diagnostic binary64 two-norm condition number of the finite connection matrix.",
    ("cell:c1b4b6e4", "add_phase1_oversampling_columns"):
        "Add Phase 1 oversampling offsets and finite-M excess ratios relative to each truncation's largest sampled M.",
    ("cell:c1b4b6e4", "add_phase1_reference_error_ratios"):
        "Normalise Phase 1 empirical errors by the displayed reference-error curve without promoting the ratio to a certificate.",
    ("cell:ce7ba80a", "ellipse_point"):
        "Evaluate the Joukowski parametrisation of the Bernstein ellipse boundary.",
    ("cell:ce7ba80a", "bernstein_radius"):
        "Return the outer Joukowski-preimage modulus of a complex point.",
    ("cell:ce7ba80a", "validate_real_branches"):
        "Check real-interval inverse-branch ranges, positivity, and imaginary leakage.",
    ("cell:ce7ba80a", "diagnostic_branch_geometry"):
        "Sample the response ellipse to estimate branch radii and transfer-weight maxima.",
    ("cell:d09d9963", "blaschke_formula_map_spec"):
        "Construct the formula-map specification and exact alpha- and mu-power target clusters.",
    ("cell:producer-sampled-schur-diagnostics", "_sampled_schur_kappa"):
        "Compute the sampled binary64 condition number of the finite packet connection matrix.",
}


# These explanations are deliberately keyed by individual mathematical
# function.  They explain the mathematical idea and its role in the thesis
# argument; FUNCTIONALITY_OVERRIDES and the function's existing documentation
# separately record the literal operation performed by the implementation.
INTUITIVE_EXPLANATIONS: dict[tuple[str, str], str] = {}


def _register_intuitive_explanations(
    source_key: str,
    descriptions: dict[str, str],
) -> None:
    additions = {(source_key, qualname): text for qualname, text in descriptions.items()}
    overlap = set(INTUITIVE_EXPLANATIONS) & set(additions)
    if overlap:
        raise RuntimeError(f"Duplicate intuitive explanations: {sorted(overlap)}")
    INTUITIVE_EXPLANATIONS.update(additions)


_register_intuitive_explanations("blaschke_deformation_certification.py", {
    "tau_phi":
        "The transfer operator is a weighted pullback: it reads a function at an inverse image and multiplies by the Perron--Frobenius weight. Enclosing both pieces makes each branch contribution rigorous before truncation.",
    "bernstein_radius_upper":
        "A branch image's Bernstein radius controls how quickly its polynomial coordinates can grow relative to the Hardy gauge. An outward upper bound supplies the safe worst-case ratio used in the infinite tail estimate.",
    "selected_joukowski_preimage":
        "Chebyshev packets are simplest in the Joukowski coordinate, where reciprocal preimages give the same symmetric Laurent packet. The thesis can therefore choose an outer pointwise preimage without pretending that a global inverse branch exists.",
    "packet_value":
        "A normalised Chebyshev packet pairs positive and negative Laurent modes into the real-line basis used for the Hardy-space comparison. Its exact enclosure is the atomic mode value from which response norms are assembled.",
    "packet_value_from_preimage":
        "Passing through a Joukowski preimage converts evaluation on the ellipse into evaluation of a symmetric Laurent mode. This is the geometric bridge between the interval transfer operator and the packet Hardy gauge.",
    "local_tail_l2_upper":
        "Past a cutoff, packet magnitudes are dominated by a geometric sequence determined by the local branch image. Summing its squared tail replaces infinitely many unresolved modes by one certified finite number.",
    "certify_input_tail_rows":
        "The complement of the first N input modes is the unresolved part of the operator. The finite prefix keeps the two inverse branches coherent, while a geometric remainder controls all later modes, producing the thesis input-tail bound.",
    "resolved_restriction_l2_upper":
        "Because the retained packets form an orthonormal coordinate system, the squared Euclidean norm of their coefficients is the finite restriction's L2 norm. An outward enclosure makes that identity usable in a rigorous bound.",
    "certify_resolved_response_rows":
        "The resolved response must add the two branch contributions before taking a norm, since cancellation belongs to the transfer operator itself. A finite coherent prefix plus an infinite packet tail bounds this response on the whole certified boundary.",
})


_register_intuitive_explanations("blaschke_deformation_contour_certification.py", {
    "_numpy_to_acb_exact":
        "Every binary64 number is an exact dyadic rational. Embedding those dyadics exactly in ACB prevents an unrecorded conversion error from entering the finite matrix used in the Riesz-projector proof.",
    "_frobenius_upper":
        "The Frobenius norm dominates the operator norm. It therefore turns entrywise ball enclosures into a single safe perturbation size for the matrix homotopies in the contour argument.",
    "_target_contours":
        "A Riesz projector has constant rank while its contour stays in the resolvent set. The 24 separated circles isolate the alpha and mu spectral packets whose algebraic multiplicities the thesis certifies, while deliberately excluding zero.",
    "_validate_schur_similarity":
        "A Schur similarity preserves eigenvalues and algebraic multiplicity. Certifying the intertwining relation makes the triangular diagonal a legitimate source of exact finite-dimensional contour counts rather than a floating-point heuristic.",
    "_certified_schur_diagonal_geometry":
        "The eigenvalues of an upper-triangular matrix are its diagonal entries with multiplicity. Certified inside/outside distance tests therefore count the finite eigenvalues enclosed by a circle without relying on a sampled winding number.",
    "_uniform_triangular_inverse_bound":
        "On a circle separated from the Schur diagonal, the strictly upper-triangular part is nilpotent. A finite Neumann expansion then bounds the inverse uniformly around the complete circle and supplies a resolvent moat.",
    "_schur_contour_attempt":
        "This is the finite-to-infinite bridge: the Schur diagonal gives the finite count, the triangular inverse gives the moat, and a strict small-gain inequality prevents the exact operator homotopy from crossing the contour.",
    "_laurent_coefficients":
        "A matrix resolvent on a circle has a Fourier--Laurent description. Discrete samples propose coefficients for an approximate inverse, but at this stage they are only a candidate for the later whole-circle interval proof.",
    "_laurent_contour_certificate":
        "If a Laurent matrix polynomial has residual norm below one everywhere on the circle, a Neumann argument proves the true resolvent exists there. This supplies the rigorous moat when the triangular Schur bound is too pessimistic.",
    "_validate_target_contour_plan":
        "Disjoint contours prevent one spectral packet from being counted twice, and strict separation from zero prevents the silent zero cluster from contaminating the nonzero multiplicity total. These are structural hypotheses of the thesis count.",
    "_existing_geometry_is_reusable":
        "Finite Schur counts and finite-matrix moats do not depend on a later refinement of the exact-operator perturbation radius. Hash and geometry checks justify retaining those proved facts while recomputing only the changed small-gain step.",
    "_reaggregate_small_gain_rows":
        "The final rank transfer depends on the product of the updated perturbation radius and each already-certified resolvent bound. Re-evaluating that inequality is enough to decide whether every finite count still transfers to the exact operator.",
    "_reaggregate_existing_certificate":
        "A certificate is a chain of claims, not just a table of numbers. Reaggregation preserves the independently validated finite geometry and rebuilds all epsilon-dependent conclusions so stale truth flags cannot survive a changed perturbation bound.",
    "certify_all_target_contours":
        "The thesis conclusion is obtained only after every target circle has both a certified finite count and a certified moat satisfying small gain. Their 24 transferred ranks then give the complete nonzero algebraic multiplicity total.",
})


_register_intuitive_explanations("blaschke_deformation_diagnostic_audits.py", {
    "_build_certification_audit":
        "The numerical argument has logically different levels: sampled geometry, finite calculations, certified bounds, and finally rank transfer. This audit keeps those levels separate so attractive diagnostics cannot masquerade as theorem hypotheses.",
    "_build_universal_audit":
        "A universal sampled envelope is useful for seeing whether the proposed asymptotic mechanism is plausible across targets. Because sampling cannot prove a whole-boundary maximum or moat, the thesis records it only as diagnostic evidence.",
    "_build_first14_audit":
        "The first fourteen packets provide a focused stress test of counts, distances, and small-gain margins. Combining them in one table reveals agreement patterns while retaining their non-theorem status when any ingredient is merely sampled.",
    "_build_first14_audit.packet_riesz_rank_status":
        "A packet's observed count and sampled small-gain outcome describe what the finite experiment suggests. The status rule deliberately withholds Riesz-rank equality until the corresponding complete-circle and perturbation hypotheses are certified.",
    "generate_diagnostic_audit_frames":
        "Placing the diagnostic ladders side by side makes the logical provenance of every reported spectral claim visible. This supports thesis interpretation without adding any new mathematical assumption.",
    "generate_universal_certification_audit":
        "This table asks how the same sampled envelope behaves across the full target family. It is an exploratory consistency check, not a substitute for the target-by-target interval certificates used in the theorem.",
    "generate_first14_packet_audit":
        "This focused packet table compares finite spectral geometry with proposed moat margins for the first fourteen targets. Its purpose is to expose discrepancies early while leaving the certified 24-contour argument logically independent.",
})


_register_intuitive_explanations("blaschke_deformation_historical_comparisons.py", {
    "_fixed_boundary":
        "The historical design chose one ellipse radius and one branch-gap margin. Re-evaluating the complete boundary with intervals shows what that older geometric choice actually guarantees, rather than trusting its formerly sampled profile.",
    "_branch_image_input_upper":
        "The unresolved input is governed by the worst branch image over the ellipse boundary. Taking the certified maximum identifies the bottleneck cell and gives a fair theorem-aware comparison with the promoted Phase 2 design.",
    "rebuild_historical_comparisons":
        "Reconstructing the earlier designs from the same map and formulas separates mathematical improvement from cache or formatting differences. These rows explain the development of the thesis bound but are not the authoritative final certificate.",
})


_register_intuitive_explanations("blaschke_deformation_historical_phase4.py", {
    "blaschke_formula_map_spec":
        "The symmetric Blaschke map determines both inverse branches and the closed-form alpha and mu target families. Keeping those formulas together fixes the mathematical model used to name every historical spectral packet.",
    "_alpha_array":
        "Central-binomial coefficients describe the change between Legendre and symmetric Chebyshev/Laurent coordinates. Their recurrence builds this basis bridge stably without repeatedly evaluating large factorials.",
    "connection_T_matrix":
        "The connection matrix expresses the same finite polynomial in the scaled Legendre basis and in Chebyshev packets. It is the similarity transform that places the finite transfer matrix in the Hardy gauge used for moat diagnostics.",
    "build_hardy_matrix":
        "Rescaling by the basis connection changes coordinates, not eigenvalues. The resulting Hardy-gauge matrix makes coefficient decay and resolvent geometry visible in the coordinates used by the thesis perturbation argument.",
    "_build_target_contours":
        "Historical contours were proposed by placing circles between the expected packet and neighbouring finite eigenvalues. This is useful contour discovery, while the final thesis proof replaces the sampled proposal by exact rational contour data and validation.",
    "_packet_matches":
        "Matching finite eigenvalues to the formula packets measures how the truncation approximates each expected cluster. It is a convergence diagnostic and does not itself prove that an exact-operator Riesz projector has the same rank.",
    "_smallest_singular_value":
        "At a fixed point, the smallest singular value of zeta I minus A is the reciprocal of the finite resolvent norm. It measures local spectral separation, but one point cannot certify an entire contour.",
    "_sample_contour":
        "Sampling singular values around a circle visualises where the finite resolvent is weakest. The sampled minimum lies above the unknown whole-circle infimum, so the thesis treats it as diagnostic rather than a certified moat.",
    "_finite_geometry":
        "Counting finite eigenvalues and measuring nearest inside and outside distances shows whether a proposed circle separates the intended packet. These distances guide contour design but need certified arithmetic for theorem use.",
    "_moat_record":
        "A moat compares the finite resolvent separation with the exact-operator perturbation size. Here the separation is sampled, so the resulting small-gain margin explains the historical experiment without asserting the final Riesz-rank theorem.",
    "_diagnostic_tables":
        "Contour profiles, packet matches, and small-gain margins are different views of the same historical finite model. Organising them together makes their numerical relationships inspectable while preserving their diagnostic status.",
    "_validate_historical_regression_profile":
        "A historical comparison is meaningful only if the rebuilt computation stays in the originally declared parameter regime. The regression guard prevents silent methodological drift from being mistaken for reproducibility.",
    "_surface_windows":
        "Local and global complex-plane windows show the shape of finite pseudospectral valleys around the target packets. They are chosen for interpretation and visual stress testing, not as domains of a theorem proof.",
    "_sample_surface":
        "A grid of smallest singular values depicts how rapidly the finite resolvent grows near spectral clusters. Since gaps between grid points remain unchecked, the surface is an explanatory diagnostic rather than a lower-bound certificate.",
    "rebuild_historical_phase4":
        "Rebuilding the whole historical Hardy-gauge experiment from source preserves a transparent comparison with the final method. The final thesis claims still rely on the later Arb whole-circle certificates, not these sampled reconstructions.",
})


_register_intuitive_explanations("blaschke_deformation_phase1_diagnostics.py", {
    "_with_oversampling_columns":
        "For fixed truncation N, increasing the quadrature order M should remove integration error while leaving truncation error. Comparing with the largest M reveals whether the displayed Phase 1 spectrum has reached that stable regime.",
    "rebuild_phase1_diagnostics":
        "Phase 1 asks whether high-precision Legendre--Gauss matrices converge towards the formula spectral packets. Rebuilding every matrix and cluster match from the map makes that convergence study reproducible, while keeping it separate from the later theorem certificate.",
})


_register_intuitive_explanations("blaschke_deformation_phase2_final_aggregation.py", {
    "_best_matrix":
        "Two rigorously derived matrix-defect estimates may describe the same finite error in different scalings. Taking the smaller is valid only after both routes have passed their own gates, and retaining its origin keeps the thesis bound auditable.",
    "_b_out":
        "The output tail measures transfer mass that lands beyond the resolved basis. Recovering the same quantity from either stored convention ensures this analytic leakage enters the final perturbation radius exactly once.",
    "certify_final_phase2_aggregation":
        "The deterministic operator radius combines unresolved input, resolved response, finite-matrix defect, and output leakage by the proved norm inequality. Fresh fail-closed gates ensure the reported epsilon exists only when every mathematical premise is certified.",
})


_register_intuitive_explanations("blaschke_deformation_phase2_finite_m.py", {
    "certify_finite_m_completion":
        "Gauss--Legendre assembly uses finitely many nodes, so its quadrature remainder needs a safe prefactor before it can enter the matrix defect. This stage certifies that finite-M amplification rather than assuming asymptotic exactness.",
})


_register_intuitive_explanations("blaschke_deformation_phase2_geometry.py", {
    "_tau_phi":
        "Each Blaschke inverse branch supplies where the test function is evaluated and how strongly that value is weighted. Rigorous branch and weight enclosures are the geometric input to every Phase 2 operator bound.",
    "_bernstein_radius_upper":
        "The Bernstein radius of a branch image translates complex geometry into a geometric coefficient-growth factor. Bounding it from above makes every later decay ratio pessimistic in the safe direction.",
    "_geometry_block":
        "A complete ellipse boundary is divided into cells, and both inverse branches are enclosed on each cell. This turns a continuum maximum problem into finitely many interval statements without leaving gaps between sample points.",
    "_maximum":
        "Global analytic bounds use the worst boundary cell. Taking the largest upper endpoint preserves an enclosure of that continuum maximum after the boundary subdivision.",
    "_minimum":
        "Admissible decay and separation conditions often depend on the least favourable boundary value. Taking the smallest lower endpoint preserves a rigorous lower bound across the complete cover.",
    "_u_tail":
        "Once the branch-to-gauge ratio is below one, all unresolved input modes form a weighted geometric tail. Its closed sum is the analytic mechanism that turns truncation at N into an exponentially small error.",
    "_c2":
        "Holomorphy on a Bernstein ellipse forces Legendre coefficients to decay geometrically, with an L2 conversion factor depending on the ellipse radius. This factor supplies the norm-compatible constant in that coefficient estimate.",
    "_s_infinity":
        "The complete weighted geometric series represents the accumulated contribution of all polynomial degrees. Evaluating it in closed form avoids replacing an infinite analytic bound by an arbitrary numerical cutoff.",
    "_sum_n_qn_from":
        "Some Legendre mode bounds carry an additional factor of the degree n. The closed tail of n times q to the n controls all such higher modes at once when q is strictly below one.",
    "_zwx13_coefficient_bound":
        "The ZWX13 estimate converts analyticity on an ellipse into an explicit majorant for one Legendre coefficient. It is the modewise ingredient used to prove that the omitted output coordinates are small.",
    "_legendre_h2_mode_bound":
        "A single orthonormal Legendre polynomial has a computable size in the chosen Hardy/ellipse gauge. Bounding that size lets coefficient decay be summed in the same norm used for the transfer-operator perturbation.",
    "_zwx13_tail_after":
        "After a chosen mode, the ZWX13 coefficient and Hardy-mode estimates reduce to closed geometric tails. This controls the genuinely infinite remainder without evaluating infinitely many polynomials.",
    "_zwx13_tail":
        "Low omitted modes are bounded individually while the far tail is summed analytically. Joining those two regimes gives a sharp but rigorous output-leakage estimate for the thesis truncation.",
    "_singularity_radius_lower":
        "The inverse branches cease to be analytic at their nearest complex singularities. A lower bound on that singularity's Bernstein radius certifies that the chosen working ellipse lies safely inside the analytic domain.",
    "_rho_boundary_data":
        "For one candidate Hardy radius, all branch images, weights, gaps, and tail factors must hold simultaneously over the full boundary. Aggregating the interval cover produces one auditable row of geometric hypotheses and bounds.",
    "certify_geometry_scan":
        "The radius is selected from a fixed finite design only after each candidate has a complete-boundary certificate. The scan exposes the trade-off between branch-image contraction, analytic room, and tail size used in Phase 2.",
})


_register_intuitive_explanations("blaschke_deformation_phase2_matrix.py", {
    "_theta_sum":
        "Theta_N is the finite squared size of the first N scaled basis modes at a geometric factor. It converts a scalar quadrature remainder into a norm bound for the entire N-by-N matrix block.",
    "_c_gl_core":
        "Gauss--Legendre quadrature is exponentially accurate for functions analytic on an ellipse. This core constant records the non-exponential part of that rigorous remainder estimate.",
    "_matrix_bound":
        "The matrix defect is obtained by multiplying quadrature error, branch-profile size, analytic decay, and finite basis-growth factors. Their product bounds the pure scaled finite block before changing to the thesis packet gauge.",
    "certify_matrix_row":
        "Ordinary and starred estimates are two certified ways to control the same finite matrix error. Computing both exposes which scaling is sharper and supplies the matrix component later transported by the connection condition number.",
})


_register_intuitive_explanations("blaschke_deformation_phase2_resolved_response.py", {
    "_tau_phi_extended":
        "The response bound needs branch geometry on a slightly enlarged certified cover, not only at isolated boundary points. Enclosing the same weighted pullback there closes the continuum gap in the resolved-response estimate.",
    "_bernstein_radius_upper":
        "On the extended cover, the branch image radius still controls packet growth. An outward upper enclosure guarantees that the local geometric remainder is valid for every point represented by the cell.",
    "_legendre_kernel":
        "The finite Christoffel kernel is the squared norm of point evaluation on the first N Legendre modes. It turns evaluation of a resolved polynomial into a sharp finite-dimensional L2 bound.",
    "certify_resolved_response_completion":
        "This completion bounds how the resolved input block responds under both weighted inverse branches over the whole boundary. It supplies the missing finite response factor required to combine the input tail with the transfer action.",
})


_register_intuitive_explanations("blaschke_deformation_phase2_transport.py", {
    "_alpha_values":
        "Central-binomial coefficients encode the Legendre-to-Chebyshev packet change of basis. The floating recurrence gives a fast candidate matrix whose conditioning can be explored before rigorous certification.",
    "connection_matrix_float":
        "This matrix expresses finite scaled Legendre coordinates in the packet gauge used by the Hardy-space proof. Its numerical version proposes the inverse and condition scale that Arb later verifies.",
    "_alpha_values_arb":
        "The same central-binomial recurrence can be evaluated with interval arithmetic, enclosing every basis coefficient. Those enclosures remove coefficient-rounding ambiguity from the transport proof.",
    "connection_matrix_arb":
        "A rigorous change-of-basis matrix is needed because matrix errors are first bounded in scaled Legendre coordinates but the spectral moat lives in packet coordinates. This object performs that finite transport with certified entries.",
    "_connection_entry_arb":
        "Each nonzero connection coefficient combines exact combinatorial factors and gauge scalings. Certifying it individually is the atomic step from which the full transport matrix is assembled.",
    "_transport_row_block":
        "Frobenius norms are sums of squared entries, so the transport and inverse candidates can be certified independently by row blocks. This decomposition preserves the global norm while making the large finite proof reproducible.",
    "_certify_transport_by_row_blocks":
        "The basis transport cost is governed by the norms of the connection and its inverse. Summing rigorous row-block contributions proves their product without trusting a black-box floating condition number.",
    "_float_matrix_as_arb":
        "A floating inverse is only a proposal, but each binary64 entry has an exact dyadic value. Embedding it exactly lets Arb measure its residual against the rigorous connection matrix with no hidden conversion error.",
    "_identity":
        "Inverse certification compares the product of the connection and candidate inverse with the identity operator. Constructing the identity exactly fixes the reference object in that residual inequality.",
    "_frobenius_upper":
        "A Frobenius upper bound controls the operator norm of a matrix residual. If that residual is below one, the Neumann argument proves invertibility and bounds the true inverse.",
    "certify_transport":
        "The pure matrix defect must be multiplied by the conditioning of the Legendre-to-packet map. Residual-based inverse certification proves this finite condition factor and therefore makes the transported thesis defect rigorous.",
})


_register_intuitive_explanations("blaschke_deformation_sampled_schur_diagnostics.py", {
    "rebuild_sampled_schur_diagnostics":
        "Sampled Schur envelopes show how finite nonnormality and contour separation interact across selected designs. They help interpret parameter choices, but the final theorem uses complete-circle interval moats instead of these samples.",
})


_register_intuitive_explanations("blaschke_deformation_spectral_certification.py", {
    "_orthonormal_legendre_values":
        "The first N orthonormal Legendre polynomials are the trial and test coordinates of the finite transfer section. Their recurrence evaluates the whole basis while preserving the L2 normalisation used in the thesis.",
    "_gauss_legendre_rule":
        "Gauss--Legendre nodes are the roots of a Legendre polynomial and integrate low-degree polynomials exactly. Isolating every root and weight in Arb makes the finite assembly a certified quadrature object.",
    "_real_branch_values":
        "At each real quadrature node, both inverse images and transfer weights determine the two weighted pullbacks. Rigorous real enclosures ensure the assembled Galerkin entries contain their exact mathematical values.",
    "_connection_matrix":
        "The finite Legendre matrix and the Hardy-gauge matrix represent the same operator section in two bases. A rigorous connection matrix links them without changing the enclosed eigenvalue problem.",
    "assemble_interval_hardy_matrix":
        "Integrating weighted branch pullbacks against Legendre test modes gives the finite Galerkin operator. Conjugating by the certified basis connection places that entire interval matrix in the packet Hardy gauge used by the contour proof.",
    "_dyadic_midpoint_payload":
        "The deployed finite matrix is stored as exact dyadic midpoints for deterministic replay. The discarded interval radii are not forgotten: their norm becomes an explicit replacement error in the total perturbation budget.",
    "_precision_audit":
        "Large gauge scalings can consume many binary digits before the desired matrix accuracy is reached. The precision audit checks that Arb has enough resolution and guard digits for those losses, preventing a vacuous enclosure.",
    "build_or_load_hardy_matrix_certificate":
        "The contour theorem needs one authoritative finite Hardy matrix together with a proof of how accurately it represents the interval assembly. Transactional construction or hash-checked loading keeps that mathematical object reproducible.",
    "load_exact_dyadic_midpoint":
        "Reconstructing the stored dyadic entries exactly gives every auditor the identical finite matrix, independent of decimal parsing or platform rounding. That exact matrix is the anchor for the finite Schur and Laurent certificates.",
})


_register_intuitive_explanations("hardy_moat_surface_worker.py", {
    "sample_surface_row_block":
        "The smallest singular value of zeta I minus A visualises finite resolvent separation over a region of the complex plane. A row block contributes to that explanatory surface, but grid sampling does not certify the gaps between points.",
})


_register_intuitive_explanations("mpmath_pf_raw.py", {
    "frobenius_norm_mp":
        "The Frobenius norm summarises the size of all matrix entries and bounds the operator norm. At high precision it provides a stable diagnostic scale for comparing assembled finite transfer sections.",
    "gauss_legendre_mp":
        "Gauss--Legendre quadrature evaluates the Galerkin inner products at optimally chosen Legendre roots. High-precision nodes and weights suppress ordinary floating-point noise so Phase 1 exposes truncation and quadrature behaviour clearly.",
    "ortho_legendre_values":
        "The orthonormal Legendre modes form the single-space trial and test basis. Evaluating them by recurrence supplies all coordinate values needed to project the weighted inverse-branch action onto a finite section.",
    "_eval_expr":
        "The map specification stores exact-looking branch formulas as controlled symbolic expressions. Evaluating them in one restricted high-precision environment ensures the numerical experiment represents the declared Blaschke map rather than duplicated handwritten formulas.",
    "FormulaTransferMap.tau":
        "An inverse branch tells the transfer operator where to pull a function value back from. This high-precision evaluation supplies that geometric part of one branch contribution in the Phase 1 Galerkin experiment.",
    "FormulaTransferMap.phi":
        "The Perron--Frobenius weight tells how strongly an inverse image contributes after the change of variables. Evaluating it beside the branch map completes one weighted pullback term.",
    "exact_clusters_for_map":
        "For this symmetric Blaschke map, formula eigenvalues occur in alpha and mu packets with known multiplicities. These declared clusters label the convergence experiment; they do not replace the later perturbative rank proof.",
    "_assemble_transfer_row_block_job":
        "Each Galerkin entry is an inner product between a Legendre test mode and the two weighted pullbacks of a trial mode. A row block computes a disjoint portion of that finite mathematical projection.",
    "assemble_transfer_block":
        "The finite transfer section is obtained by Gauss--Legendre approximation of all projected weighted pullbacks. Assembling the whole block creates the matrix whose spectral convergence is studied in Phase 1.",
    "assemble_pure_scaled_transfer_block":
        "Conjugating by powers of a radius expresses the same finite operator in a Hardy-like scaled Legendre norm. This exposes geometric coefficient decay without changing the finite eigenvalues.",
    "greedy_cluster_errors":
        "A numerical eigenvalue cloud must be paired with the expected multiplicity packets before convergence errors can be summarised. Greedy matching gives a reproducible diagnostic pairing, not a certified spectral count.",
    "build_reference_clusters_mpmath_raw":
        "A larger high-precision finite section can serve as a numerical reference when no closed target is being used. Such reference clusters measure empirical convergence only and remain logically below the thesis's exact contour certificates.",
})


_register_intuitive_explanations("transfer_spectrum_certification.py", {
    "balanced_single_space_geometry":
        "The Hardy radius must be larger than the branch-image radius but smaller than the available analytic radius. Balancing the two resulting geometric ratios minimises the slower exponential decay mechanism in the sampled model.",
    "select_sampled_geometry":
        "Different ellipse radii trade analytic room against branch contraction. Selecting the best sampled candidate helps design the rigorous scan, but boundary samples alone cannot establish the continuum inequalities required by the thesis.",
    "_theta_log":
        "Theta_N sums the squared sizes of the retained scaled modes and may span enormous magnitudes. Log-sum-exp evaluates this finite basis-growth factor without overflow in the diagnostic envelope.",
    "_tail_T":
        "The omitted weighted modes form a geometric series with a polynomial degree factor. Its closed tail displays the exponential N-dependence expected from analytic truncation.",
    "_S_infty":
        "The full weighted geometric sum gives a closed normalisation for all degrees at once. It is the infinite-series counterpart of the finite basis-growth terms in the sampled bound.",
    "_C2":
        "Analyticity on a Bernstein ellipse yields geometrically decaying Legendre coefficients in L2, up to a radius-dependent constant. This is that norm-compatible sampled constant.",
    "_Cinf":
        "A uniform coefficient estimate pays a different ellipse constant from the L2 estimate. This factor records that stronger pointwise control in the sampled diagnostic calculation.",
    "_CGL_core":
        "The non-exponential prefactor in the analytic Gauss--Legendre remainder determines how the quadrature order M converts into matrix accuracy. It complements the dominant ellipse-decay power.",
    "_input_tail":
        "After mode N, the branch image is measured relative to the chosen Hardy radius. When that ratio is below one, a closed geometric sum bounds the unresolved input mass.",
    "sampled_schur_envelope":
        "This envelope combines sampled branch geometry, analytic tails, finite matrix error, and basis transport into a proposed perturbation size. It explains the Schur strategy but is not theorem evidence unless every input is independently enclosed.",
    "build_certification_audit":
        "Riesz-rank equality follows only after geometry, perturbation size, contour moat, finite count, and strict small gain are all certified. The audit makes that implication chain explicit and refuses to infer the conclusion from exact target formulas or samples alone.",
})


_register_intuitive_explanations("cell:0443774e", {
    "U_tail_closed":
        "The unresolved input modes decay geometrically once the branch-image radius is smaller than the chosen gauge radius. This closed sum shows intuitively why increasing N suppresses the input tail exponentially.",
    "K_tail_point":
        "At one complex point, the omitted symmetric Chebyshev packets form a pointwise geometric kernel. Plotting this kernel reveals where branch images make truncation most difficult, without claiming a certified boundary maximum.",
    "branch_image_tail_profile":
        "Following both inverse branches around an ellipse shows how local packet tails vary with boundary angle. The profile is a geometric explanation of the worst-case cells later handled by interval subdivision.",
})


_register_intuitive_explanations("cell:0e01d379", {
    "frobenius_mp":
        "The high-precision Frobenius norm measures the aggregate size of a finite matrix and dominates its operator norm. In the notebook it provides a transparent scale for numerical matrix comparisons.",
})


_register_intuitive_explanations("cell:2a6f973c", {
    "phase1_positive_target_rows":
        "The nonzero positive formula targets isolate the spectral packets of interest from their symmetric or silent-zero companions. Restricting the diagnostic table this way makes convergence rates comparable across multiplicities.",
    "phase1_summary_table":
        "Grouping cluster errors by N and M separates truncation growth from quadrature refinement. The summary lets the reader see whether spectral approximations stabilise in the pattern predicted by analyticity.",
    "phase1_empirical_exponential_reference_curve":
        "Analytic spectral approximation is expected to improve roughly exponentially with truncation. The fitted guide visualises that trend, but it is intentionally empirical and contributes no bound to the final theorem.",
})


_register_intuitive_explanations("cell:44cb0b52", {
    "sample_branch_boundary_profile":
        "The inverse branches deform a source ellipse into two complex curves whose Bernstein radii govern coefficient growth. Sampling those curves gives an intuitive picture of the contraction mechanism later certified on every boundary cell.",
})


_register_intuitive_explanations("cell:6451c7fe", {
    "jouk":
        "The Joukowski map folds reciprocal points z and one over z onto the same interval-plane point. It is the geometric coordinate change connecting Laurent modes on a circle with Chebyshev polynomials on an ellipse.",
    "jouk_np":
        "Applying the Joukowski map to an array traces how a circle becomes a Bernstein ellipse. This vector form supports the notebook's geometric diagnostics of analytic domains and branch images.",
    "outer_joukowski_inverse":
        "Every noncritical point has reciprocal Joukowski preimages. Selecting the one outside the unit circle assigns its Bernstein radius and hence the exponential scale of its polynomial coefficients.",
    "bernstein_radius":
        "A complex point lies on a unique Bernstein ellipse measured by the modulus of its outer Joukowski preimage. That radius translates complex location directly into a coefficient-decay or growth factor.",
    "ellipse_points_np":
        "A circle of radius rho in Joukowski coordinates maps to the Bernstein ellipse of the same radius. Its sampled parametrisation makes the analytic boundary and branch deformation visible in the notebook.",
    "phase2_best_matrix_from_row":
        "The ordinary and starred routes are alternative rigorous majorants for one matrix defect. Choosing the smaller certified value tightens the final radius without changing the underlying inequality.",
    "phase2_B_out_from_row":
        "B_out is the norm of transfer mass escaping the resolved output modes. Recovering it consistently lets the notebook compare historical and promoted decompositions of the same operator error.",
    "phase2_build_summary_rows":
        "Putting candidate designs into common components shows which geometric, matrix, or tail term limits each perturbation radius. This makes the numerical improvement in the thesis mathematically interpretable.",
    "phase2_promoted_certificate":
        "The promoted epsilon is the root-sum-square combination prescribed by the operator decomposition, not an empirical fit. Recording the selected certified components explains exactly why this deterministic radius controls the finite-to-exact difference.",
})


_register_intuitive_explanations("cell:9b8c4f29", {
    "D_scaled_block":
        "Diagonal powers of the Hardy radius rescale Legendre coefficients according to expected analytic decay. This finite block represents the norm change used to expose contraction in polynomial degree.",
    "cheb_mul_x":
        "Multiplication by x obeys the Chebyshev three-term relation, so it acts sparsely on Chebyshev coefficients. Repeated use constructs polynomial basis conversions without unstable pointwise interpolation.",
    "legendre_to_cheb_coeffs_orthonormal":
        "An orthonormal Legendre polynomial and a Chebyshev expansion describe the same finite polynomial in different coordinates. Computing that expansion is the elementary column of the thesis's basis connection.",
    "C_X_matrix":
        "The connection matrix collects those polynomial coordinate changes for all retained degrees. It bridges the L2 Legendre Galerkin basis and the symmetric packet coordinates adapted to ellipse analyticity.",
    "T_connection":
        "Adding the Hardy scalings to the polynomial connection transports finite vectors between the pure scaled Legendre norm and the Chebyshev-packet norm. Its conditioning determines how much a matrix defect can grow under that transport.",
    "kappa_T_numeric":
        "The two-norm condition number estimates the amplification caused by changing finite bases. Here it is a diagnostic preview; the theorem uses the separately certified Arb residual bound for the inverse.",
})


_register_intuitive_explanations("cell:c1b4b6e4", {
    "add_phase1_oversampling_columns":
        "For a fixed N, increasing M tests whether quadrature error has become negligible compared with truncation error. The excess ratios reveal this stabilisation without converting it into a rigorous remainder bound.",
    "add_phase1_reference_error_ratios":
        "Dividing observed errors by an empirical reference curve makes deviations from the expected exponential trend easy to see. The ratio is interpretive evidence only, not a certified convergence constant.",
})


_register_intuitive_explanations("cell:ce7ba80a", {
    "ellipse_point":
        "A Bernstein ellipse is the Joukowski image of a circle and parametrises a complex neighbourhood of the interval. One boundary point is the basic geometric input for inspecting inverse-branch analyticity.",
    "bernstein_radius":
        "The outer Joukowski modulus labels which Bernstein ellipse passes through a complex point. It converts the plotted branch image into the exponential coordinate used by the analytic tail formulas.",
    "validate_real_branches":
        "On the real interval, the two inverse branches should satisfy the defining Blaschke equation and the expected branch ordering. Checking these identities catches a wrong radical sign before it contaminates every numerical stage.",
    "diagnostic_branch_geometry":
        "Mapping a sampled ellipse boundary through both inverse branches estimates contraction, weight size, and separation from singularities. This guides parameter selection, while the later interval scan supplies the actual proof.",
})


_register_intuitive_explanations("cell:d09d9963", {
    "blaschke_formula_map_spec":
        "The map parameter, its two inverse branches, their transfer weights, and the alpha/mu target families define one coherent spectral model. Centralising them ensures every notebook phase studies the same thesis operator.",
})


_register_intuitive_explanations("cell:producer-sampled-schur-diagnostics", {
    "_sampled_schur_kappa":
        "Changing from scaled Legendre coordinates to packet coordinates can amplify finite errors. This sampled condition number estimates that amplification for diagnostic Schur rows, while the final perturbation proof uses the rigorous transport certificate.",
})


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FunctionVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.functions: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = ".".join((*self.stack, node.name))
        self.functions.append((qualname, node))
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


def discover_functions(source: str) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    visitor = FunctionVisitor()
    visitor.visit(ast.parse(source))
    return visitor.functions


def _clean_existing_doc(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    doc = ast.get_docstring(node, clean=True)
    if not doc:
        return None
    if "Functionality:" in doc:
        doc = doc.split("Functionality:", 1)[1].strip()
    return " ".join(doc.split())


def _fallback_functionality(qualname: str) -> str:
    name = qualname.rsplit(".", 1)[-1].strip("_").replace("_", " ")
    return f"Compute {name} with the arguments and return contract used by this numerical stage."


def _doc_literal(explanation: str, functionality: str) -> str:
    explanation = " ".join(explanation.split()).replace("'''", "three single quotes")
    functionality = " ".join(functionality.split()).replace("'''", "three single quotes")
    return (
        "'''Explanation: " + explanation + "\n"
        "Functionality: " + functionality + "'''"
    )


def add_mathematical_docstrings(
    source: str,
    mathematical: Iterable[str],
    *,
    source_key: str,
) -> str:
    selected = set(mathematical)
    discovered = dict(discover_functions(source))
    missing = selected - set(discovered)
    if missing:
        raise RuntimeError(f"Mathematical inventory names are absent: {sorted(missing)}")
    line_offsets = [0]
    for line in source.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))
    edits: list[tuple[int, int, str]] = []
    for qualname in sorted(selected):
        node = discovered[qualname]
        key = (source_key, qualname)
        if key not in INTUITIVE_EXPLANATIONS:
            raise RuntimeError(f"Missing function-specific intuitive explanation: {key}")
        explanation = INTUITIVE_EXPLANATIONS[key]
        functionality = FUNCTIONALITY_OVERRIDES.get(
            key,
            _clean_existing_doc(node) or _fallback_functionality(qualname),
        )
        literal = _doc_literal(explanation, functionality)
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            start = line_offsets[first.lineno - 1] + first.col_offset
            end = line_offsets[first.end_lineno - 1] + first.end_col_offset
            edits.append((start, end, literal))
        elif first.lineno == node.lineno:
            body_start = line_offsets[first.lineno - 1] + first.col_offset
            function_start = line_offsets[node.lineno - 1] + node.col_offset
            colon = source.rfind(":", function_start, body_start)
            if colon < 0:
                raise RuntimeError(f"Cannot expand one-line function {qualname}.")
            indent = " " * (node.col_offset + 4)
            edits.append((colon + 1, body_start, f"\n{indent}{literal}\n{indent}"))
        else:
            start = line_offsets[first.lineno - 1]
            indent = " " * first.col_offset
            edits.append((start, start, f"{indent}{literal}\n"))
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


def _inline_helpers() -> tuple[tuple[int, str, str, str], ...]:
    namespace = runpy.run_path(str(BUILDER))
    return tuple(namespace["INLINE_HELPERS"])


def apply_helper_docstrings() -> None:
    helper_names = {filename for _, _, filename, _ in _inline_helpers()}
    if helper_names != set(MATH_HELPER_FUNCTIONS):
        raise RuntimeError("The eighteen-helper inventory has drifted.")
    for filename in sorted(helper_names):
        path = HERE / filename
        updated = add_mathematical_docstrings(
            path.read_text(encoding="utf-8"),
            MATH_HELPER_FUNCTIONS[filename],
            source_key=filename,
        )
        path.write_text(updated, encoding="utf-8", newline="")


def apply_notebook_docstrings() -> None:
    for path in NOTEBOOK_CHAIN:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        seen: set[str] = set()
        for cell in notebook["cells"]:
            cell_id = str(cell.get("id", ""))
            if cell.get("cell_type") != "code" or cell_id not in MATH_NOTEBOOK_FUNCTIONS:
                continue
            source = "".join(cell.get("source", []))
            cell["source"] = add_mathematical_docstrings(
                source,
                MATH_NOTEBOOK_FUNCTIONS[cell_id],
                source_key=f"cell:{cell_id}",
            )
            seen.add(cell_id)
        if path == FINAL_NOTEBOOK and seen != set(MATH_NOTEBOOK_FUNCTIONS):
            raise RuntimeError(
                f"Final-notebook mathematical cell inventory drift: {sorted(set(MATH_NOTEBOOK_FUNCTIONS) - seen)}"
            )
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="")
    replacements = json.loads(PLOTTING_REPLACEMENTS.read_text(encoding="utf-8"))
    for cell_id, mathematical in MATH_NOTEBOOK_FUNCTIONS.items():
        if cell_id not in replacements:
            continue
        source = str(replacements[cell_id])
        selected = mathematical & {name for name, _ in discover_functions(source)}
        if selected:
            replacements[cell_id] = add_mathematical_docstrings(
                source,
                selected,
                source_key=f"cell:{cell_id}",
            )
    PLOTTING_REPLACEMENTS.write_text(
        json.dumps(replacements, indent=2) + "\n",
        encoding="utf-8",
        newline="",
    )


def sync_final_inline_helpers() -> None:
    namespace = runpy.run_path(str(BUILDER))
    notebook = json.loads(FINAL_NOTEBOOK.read_text(encoding="utf-8"))
    cells = {str(cell.get("id", "")): cell for cell in notebook["cells"]}
    helper_cells = namespace["_helper_cells"]
    for ordinal, (_, module_name, filename, phase) in enumerate(_inline_helpers(), start=1):
        expected_heading, expected_source = helper_cells(module_name, filename, phase, ordinal)
        heading = cells[f"inline-helper-heading-{ordinal}"]
        source_cell = cells[f"inline-helper-source-{ordinal}"]
        heading["source"] = expected_heading["source"]
        actual = "".join(source_cell.get("source", []))
        expected = str(expected_source["source"])
        if "# notebook-provenance: begin\n" in actual:
            actual_lines = actual.splitlines(keepends=True)
            end_index = next(
                index for index, line in enumerate(actual_lines)
                if line == "# notebook-provenance: end\n"
            )
            magic = expected.splitlines(keepends=True)[0]
            module_source = (HERE / filename).read_text(encoding="utf-8")
            source_cell["source"] = magic + "".join(actual_lines[1:end_index + 1]) + module_source
        else:
            source_cell["source"] = expected
    notebook.setdefault("metadata", {})["source_sync_after_execution"] = {
        "date": "2026-08-29",
        "scope": "mathematical docstrings and fail-closed Phase 2 gate propagation",
        "stored_output_status": (
            "retained historical outputs; not evidence for the post-sync sources until a full clean-room replay"
        ),
    }
    FINAL_NOTEBOOK.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="",
    )


def _classification(cell_id: str, qualname: str) -> str:
    if qualname in MATH_NOTEBOOK_FUNCTIONS.get(cell_id, frozenset()):
        return "mathematical"
    if cell_id in PLOTTING_NOTEBOOK_CELLS or qualname in PLOTTING_NOTEBOOK_NAMES or "plot" in qualname.lower():
        return "plotting"
    return "infrastructure"


def build_inventory() -> dict[str, object]:
    helpers = _inline_helpers()
    records: list[dict[str, object]] = []
    helper_digests: dict[str, str] = {}
    for _, _, filename, _ in helpers:
        path = HERE / filename
        source = path.read_text(encoding="utf-8")
        helper_digests[filename] = sha256_file(path)
        functions = discover_functions(source)
        names = {qualname for qualname, _ in functions}
        if not MATH_HELPER_FUNCTIONS[filename] <= names:
            raise RuntimeError(f"Stale helper classification for {filename}.")
        for qualname, node in functions:
            records.append({
                "scope": "inline_helper",
                "source": filename,
                "qualname": qualname,
                "classification": (
                    "mathematical"
                    if qualname in MATH_HELPER_FUNCTIONS[filename]
                    else "infrastructure"
                ),
                "line": node.lineno,
            })
    notebook = json.loads(FINAL_NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        cell_id = str(cell.get("id", ""))
        if cell.get("cell_type") != "code" or cell_id.startswith("inline-helper-"):
            continue
        source = "".join(cell.get("source", []))
        try:
            functions = discover_functions(source)
        except SyntaxError:
            continue
        names = {qualname for qualname, _ in functions}
        if not MATH_NOTEBOOK_FUNCTIONS.get(cell_id, frozenset()) <= names:
            raise RuntimeError(f"Stale notebook classification for cell {cell_id}.")
        for qualname, node in functions:
            records.append({
                "scope": "notebook_cell",
                "source": FINAL_NOTEBOOK.name,
                "cell_id": cell_id,
                "qualname": qualname,
                "classification": _classification(cell_id, qualname),
                "line": node.lineno,
            })
    records.sort(key=lambda row: (
        str(row["scope"]), str(row["source"]), str(row.get("cell_id", "")),
        int(row["line"]), str(row["qualname"]),
    ))
    counts = {
        category: sum(row["classification"] == category for row in records)
        for category in ("mathematical", "infrastructure", "plotting")
    }
    return {
        "schema_version": 1,
        "scope": (
            "Every function in the 18 digest-checked inline helpers and every function in non-inline code cells of the 140-cell final notebook."
        ),
        "classification_policy": {
            "mathematical": "Directly implements theorem-facing or retained diagnostic mathematics and requires the two-field triple-single-quoted docstring.",
            "infrastructure": "I/O, caching, provenance, parsing, orchestration, progress, or presentation plumbing; deliberately not relabelled as theorem mathematics.",
            "plotting": "Plotting or display-only code; deliberately outside the mathematical docstring gate.",
        },
        "inline_helper_count": len(helpers),
        "notebook_cell_count": len(notebook["cells"]),
        "notebook_code_cell_count": sum(cell.get("cell_type") == "code" for cell in notebook["cells"]),
        "notebook_sha256": sha256_file(FINAL_NOTEBOOK),
        "helper_sha256": helper_digests,
        "counts": counts,
        "functions": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-docstrings", action="store_true")
    parser.add_argument("--sync-final-inline", action="store_true")
    parser.add_argument("--write-inventory", action="store_true")
    args = parser.parse_args()
    if args.apply_docstrings:
        apply_helper_docstrings()
        apply_notebook_docstrings()
    if args.sync_final_inline:
        sync_final_inline_helpers()
    if args.write_inventory:
        INVENTORY.write_text(
            json.dumps(build_inventory(), indent=2) + "\n",
            encoding="utf-8",
            newline="",
        )
    if not any((args.apply_docstrings, args.sync_final_inline, args.write_inventory)):
        parser.error("select at least one maintenance action")


if __name__ == "__main__":
    main()
