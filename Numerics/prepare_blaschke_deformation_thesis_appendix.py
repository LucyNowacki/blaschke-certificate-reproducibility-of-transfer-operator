"""Prepare the full Blaschke certifier as the curated thesis notebook.

This is a deliberately conservative transformation of the full inline-helper
counterpart. It removes
historical and superseded material by stable Jupyter cell id, cleans deployment
metadata, adds mechanically parseable provenance headers, and preserves any
execution counts and stored outputs of every retained code cell.

The default command is a dry run. ``--apply`` is refused unless a separate,
byte-identical backup of the input notebook already exists. The script never
creates that backup itself and writes the transformed notebook atomically.

Reviewed plotting-cell replacements are loaded from a separate JSON map. This
keeps the structural transformer auditable while moving all rendering code to
``plotting.py``.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_PROVENANCE = HERE / "notebook_cell_provenance.json"
DEFAULT_PLOTTING_REPLACEMENTS = HERE / "plotting_cell_replacements.json"
EXPECTED_SOURCE_CELL_COUNT = 187
EXPECTED_SOURCE_SHA256 = ""

PROVENANCE_BEGIN = "# notebook-provenance: begin"
PROVENANCE_END = "# notebook-provenance: end"


# Stable cell ids inherited from the earlier 181-cell inline-helper counterpart. The numerical
# indices are documented only in the group names; transformation never selects
# cells by position.
REMOVAL_GROUPS: Mapping[str, frozenset[str]] = {
    "superseded sampled first-fifteen Phase 4 block, former cells 79 through 96": frozenset(
        {
            "994efac4", "24870c03", "ff9e13d1", "c313a405",
            "df121ba9", "87a2aac4", "84397864", "8627a4b2",
            "300b4aa7", "1d2cf08a", "9c5ba95d", "3d814d8a",
        }
    ),
    "duplicate archive, static, and interactive material, former cells 103 through 109": frozenset(
        {
            "6eb014d0", "b1967f19", "935920c5",
        }
    ),
    "asymmetric-map material, former cells 110 through 125": frozenset(
        {
            "9c777f9329cd", "bbad4a7c", "0ae4b43c6caa", "509d64d6",
            "119a2190de91", "4c78e4cc", "020b007a42b7", "df8d2a5a",
            "027cc226a832", "97df3d30", "d1aef7051446", "fae22ab8",
            "b7e5e94febba", "a5dd5ed4", "2775f63e8661", "514d77d49e27",
        }
    ),
    "generic unknown-spectrum and preliminary first-fourteen material, former cells 127 through 132": frozenset(
        {
            "a4227ca16d78", "cell102theory95",
        }
    ),
    "Notebook update history, former cells 143 through 155": frozenset(
        {
            "codexupd86", "codexupd87", "codexupd88", "codexupd89",
            "codexupd90", "codexupd91", "codexupd92", "codexupd93",
            "codexupd94", "codexupd95", "codexupd96", "codexupd97",
            "coherent-response-update",
        }
    ),
    "reproducibility packaging and trailing empty cell, former cells 156 through 158": frozenset(
        {"md-44483f5a", "code-ca9fe310", "3ce1e063"}
    ),
    "duplicate rendered-output marker": frozenset({"b8f52471"}),
}

REMOVE_CELL_IDS = frozenset().union(*REMOVAL_GROUPS.values())

EXPECTED_RETAINED_CELL_COUNT = 139
EXPECTED_RETAINED_CODE_CELL_COUNT = 68

FINAL_CERTIFICATION_LADDER_SOURCE = '''#108N
# notebook-provenance: begin
# helpers: plotting: display_table; plotting: phase1_gradient_palette; plotting: plot_certification_ladder; direct imports in this cell: plotting: display_table, phase1_gradient_palette, plot_certification_ladder
# data_sources: Cell 107N::spectral_contour_certificate_df; Cell 107N::SPECTRAL_CONTOUR_CERTIFICATE
# prior_results: Cell 107N::_computed_multiplicity; Cell 107N::_expected_multiplicity; Cell 107N::_schur_moat_rows; Cell 107N::_laurent_moat_rows; Cell 107N::CELL103_MINIMUM_CERTIFIED_MOAT_TEXT; Cell 107N::CELL103_MAXIMUM_SMALL_GAIN; Cell 107N::CELL103_MAXIMUM_SMALL_GAIN_TEXT
# execution_mode: arithmetic: Boolean reaggregation of Cell 107N theorem gates only; cache_policy: none; kind: final_certificate_presentation; parallelism: single process
# produces: files: FIG_DIR/transfer_lab_blaschke_mu_0p3_Cell_108_final_spectral_certification_ladder.png; prior_results: final_certification_ladder_df; FINAL_CERTIFICATION_LADDER_RESULT
# proof_status: claim: Fail-closed presentation of the final twenty-four-target certificate already formed and checked by Cell 107N.; class: presentation_only
# provenance_notes: This cell raises before plotting unless all eight displayed summaries follow from the final Cell 107N theorem gates.
# notebook-provenance: end
# Final theorem-facing certification ladder, derived only from Cell 107N.

from plotting import display_table, phase1_gradient_palette, plot_certification_ladder

_final_bool_column = lambda field: (
    spectral_contour_certificate_df[field]
    .astype(str).str.lower().eq("true").all()
)
_final_rank_values = (
    spectral_contour_certificate_df["rank"].astype(int).tolist()
)
_final_laurent_not_count = not (
    spectral_contour_certificate_df.loc[
        _laurent_moat_rows, "laurent_used_for_count"
    ].astype(str).str.lower().eq("true").any()
)
_final_laurent_used_for_moat = (
    spectral_contour_certificate_df.loc[
        _laurent_moat_rows, "laurent_used_for_moat"
    ].astype(str).str.lower().eq("true").all()
)

_final_ladder_rows = [
    {
        "audit_item": "ordered target inventory and zero exclusion",
        "passed": bool(
            len(spectral_contour_certificate_df) == 24
            and _final_rank_values == list(range(1, 25))
            and _expected_multiplicity == 30
            and _final_bool_column("zero_outside_enclosed_region")
        ),
        "evidence": (
            f"{len(spectral_contour_certificate_df)} ordered targets; "
            f"expected multiplicity {_expected_multiplicity}; "
            "zero outside every contour"
        ),
    },
    {
        "audit_item": "Schur-derived finite counts",
        "passed": bool(
            spectral_contour_certificate_df["count_method"]
            .eq(COUNT_METHOD_SCHUR_DIAGONAL).all()
            and _final_bool_column("schur_diagonal_membership_certified")
            and _computed_multiplicity == 30
            and _expected_multiplicity == 30
            and _final_bool_column("finite_count_matches_expected")
        ),
        "evidence": (
            "24 Schur-diagonal counts; "
            f"computed/expected multiplicity "
            f"{_computed_multiplicity}/{_expected_multiplicity}"
        ),
    },
    {
        "audit_item": "exact-dyadic finite-matrix count transport",
        "passed": bool(
            _final_bool_column("exact_dyadic_schur_upper_triangular_certified")
            and _final_bool_column("A_N_circ_count_transport_certified")
            and _final_bool_column(
                "mathematical_finite_count_transport_certified"
            )
            and _final_bool_column("finite_count_certified")
        ),
        "evidence": (
            "exact-dyadic triangular Schur counts transported to "
            "the mathematical Hardy matrix"
        ),
    },
    {
        "audit_item": "complete-circle moat coverage",
        "passed": bool(
            int(_schur_moat_rows.sum()) == 17
            and int(_laurent_moat_rows.sum()) == 7
            and spectral_contour_certificate_df.loc[
                _schur_moat_rows, "certificate_route"
            ].eq(CERTIFICATE_ROUTE_SCHUR).all()
            and spectral_contour_certificate_df.loc[
                _laurent_moat_rows, "certificate_route"
            ].eq(CERTIFICATE_ROUTE_LAURENT).all()
            and _final_laurent_not_count
            and _final_laurent_used_for_moat
            and _final_bool_column("complete_circle_covered")
        ),
        "evidence": (
            f"{int(_schur_moat_rows.sum())} Schur-triangular and "
            f"{int(_laurent_moat_rows.sum())} Laurent complete-circle moats"
        ),
    },
    {
        "audit_item": "positive lifted Hardy-space moats",
        "passed": bool(
            spectral_contour_certificate_df[
                "lifted_finite_section_moat_lower"
            ].gt(0).all()
        ),
        "evidence": (
            "minimum outward-safe lower endpoint "
            f"{CELL103_MINIMUM_CERTIFIED_MOAT_TEXT}"
        ),
    },
    {
        "audit_item": "sampled values excluded from theorem gates",
        "passed": bool(
            not spectral_contour_certificate_df[
                "sampled_values_used_in_theorem_gate"
            ].astype(str).str.lower().eq("true").any()
        ),
        "evidence": "0 of 24 rows use sampled values in a theorem gate",
    },
    {
        "audit_item": "certified small-gain inequality",
        "passed": bool(
            _final_bool_column("certified_small_gain_pass")
            and CELL103_MAXIMUM_SMALL_GAIN < 1
        ),
        "evidence": (
            "maximum outward-safe upper endpoint "
            f"{CELL103_MAXIMUM_SMALL_GAIN_TEXT} < 1"
        ),
    },
    {
        "audit_item": "overall finite-to-exact Riesz ranks",
        "passed": bool(
            _final_bool_column("finite_to_exact_rank_transfer_certified")
            and _final_bool_column("theorem_certified")
            and bool(
                SPECTRAL_CONTOUR_CERTIFICATE[
                    "all_24_targets_theorem_certified"
                ]
            )
        ),
        "evidence": (
            f"24 theorem-certified contours; total algebraic multiplicity "
            f"{_computed_multiplicity}"
        ),
    },
]
for _final_ladder_row in _final_ladder_rows:
    _final_ladder_row["status"] = (
        "theorem_certified"
        if _final_ladder_row["passed"]
        else "gate_failed"
    )

final_certification_ladder_df = pd.DataFrame(
    _final_ladder_rows,
    columns=("audit_item", "status", "evidence", "passed"),
)
if not final_certification_ladder_df["passed"].all():
    _failed_final_ladder_items = final_certification_ladder_df.loc[
        ~final_certification_ladder_df["passed"], "audit_item"
    ].tolist()
    raise AssertionError(
        "The final certification ladder is not theorem-certified: "
        f"{_failed_final_ladder_items}."
    )

FINAL_CERTIFICATION_LADDER_RESULT = plot_certification_ladder(
    final_certification_ladder_df,
    title=(
        r"Blaschke $\\mu=0.3$: final certification ladder"
    ),
    item_colours=phase1_gradient_palette(
        final_certification_ladder_df["audit_item"]
    ),
    display_labels={
        "ordered target inventory and zero exclusion": (
            "ordered target inventory\\nand zero exclusion"
        ),
        "exact-dyadic finite-matrix count transport": (
            "exact-dyadic finite-matrix\\ncount transport"
        ),
        "sampled values excluded from theorem gates": (
            "sampled values excluded\\nfrom theorem gates"
        ),
        "overall finite-to-exact Riesz ranks": (
            "overall finite-to-exact\\nRiesz ranks"
        ),
    },
    figsize=(11.5, 6.8),
    title_fontsize=22,
    label_fontsize=18,
    status_fontsize=14,
    bar_height=0.76,
    title_pad=14,
    output_dir=FIG_DIR,
    stem=(
        "transfer_lab_blaschke_mu_0p3_"
        "Cell_108_final_spectral_certification_ladder"
    ),
    show=True,
)
_ = display_table(
    final_certification_ladder_df,
    columns=("audit_item", "status", "evidence"),
)
'''


TERMINAL_NUMBERED_CELLS: tuple[tuple[str, str], ...] = (
    ("3e8b784c", FINAL_CERTIFICATION_LADDER_SOURCE),
    ("128b5369", "#109N\n"),
)
TERMINAL_NUMBERED_CELL_IDS = frozenset(
    cell_id for cell_id, _ in TERMINAL_NUMBERED_CELLS
)

CUSTOM_DISPLAY_LABELS = {
    "inline-helper-heading-2": "15AM",
    "inline-helper-source-2": "#15AN",
    "inline-helper-heading-6": "35AM",
    "inline-helper-source-6": "#35AN",
    "inline-helper-heading-7": "35BM",
    "inline-helper-source-7": "#35BN",
    "inline-helper-heading-8": "35CM",
    "inline-helper-source-8": "#35CN",
    "inline-helper-heading-9": "35DM",
    "inline-helper-source-9": "#35DN",
    "inline-helper-heading-10": "35EM",
    "inline-helper-source-10": "#35EN",
    "inline-helper-heading-11": "78AM",
    "inline-helper-source-11": "#78AN",
    "inline-helper-heading-12": "78BM",
    "inline-helper-source-12": "#78BN",
    "inline-helper-heading-13": "95AM",
    "inline-helper-source-13": "#95AN",
    "inline-helper-heading-14": "48AM",
    "inline-helper-source-14": "#48AN",
    "inline-helper-heading-15": "52AM",
    "inline-helper-source-15": "#52AN",
    "inline-helper-heading-16": "54AM",
    "inline-helper-source-16": "#54AN",
    "inline-helper-heading-17": "30AM",
    "inline-helper-source-17": "#30AN",
    "producer-phase1-diagnostics": "#30BN",
    "inline-helper-heading-18": "94AM",
    "inline-helper-source-18": "#94AN",
    "producer-sampled-schur-diagnostics": "#94BN",
}

NON_NUMBER_ADVANCING_DISPLAY_IDS = frozenset(
    {
        "inline-helper-heading-2",
        "inline-helper-source-2",
        "inline-helper-heading-11",
        "inline-helper-source-11",
        "inline-helper-heading-12",
        "inline-helper-source-12",
        "inline-helper-heading-13",
        "inline-helper-source-13",
        "inline-helper-heading-14",
        "inline-helper-source-14",
        "inline-helper-heading-15",
        "inline-helper-source-15",
        "inline-helper-heading-16",
        "inline-helper-source-16",
        "inline-helper-heading-17",
        "inline-helper-source-17",
        "producer-phase1-diagnostics",
        "inline-helper-heading-18",
        "inline-helper-source-18",
        "producer-sampled-schur-diagnostics",
    }
)

EXPECTED_RESTORED_MARKDOWN_CELL_IDS = frozenset(
    {
        "a81aa734",
        "3d797d1d",
        "b8581535",
        "59737fdb1e5a",
        "1c203394",
        "3b83ae378cf8",
        "42a2aee243d6",
    }
)

EXPECTED_PHASE1_DISPLACEMENT_STEMS = frozenset(
    {
        "phase1_N20_M20_to_M30_eigencloud_displacement",
        "phase1_N12_M12_to_M24_visible_eigencloud_displacement",
    }
)

# Stored visual outputs in the protected 159-cell source. The transformer keeps
# these payloads byte-for-byte; a later clean execution must regenerate the same
# visual inventory through plotting.py calls.
EXPECTED_VISUAL_OUTPUT_COUNTS: Mapping[str, int] = {
    "2a6f973c": 1,
    "cc338a81": 6,
    "e40904af": 1,
    "b4390377": 1,
    "code-7598db69": 1,
    "c92b9a83": 2,
    "df668eaa": 2,
    "5cb2c38a": 2,
    "e4cd9776": 1,
    "3410c676": 1,
    "59a1c39c": 6,
    "32961420": 1,
    "phase2-math-41": 1,
    "340f0292": 1,
    "add52fcf": 1,
    "8bf58bd5": 1,
    "ea23bc60": 1,
    "009356fa": 1,
    "a2b9f84258b4": 1,
    "fd3908b317fe": 1,
}

# This contract states exactly which plotting.py entry points the restored
# sources must call. Cell 82F additionally requires the helper-owned dashboard
# renderer factory listed here; no rendering callback may be implemented in the
# notebook itself.
EXPECTED_RESTORED_PLOTTING_HELPER_CALLS: Mapping[str, frozenset[str]] = {
    "cc338a81": frozenset(
        {
            "plot_phase1_eigencloud_comparison",
            "plot_phase1_oversampling_displacement",
            "plot_phase1_fixed_n_m_sweep",
            "plot_phase1_absolute_m_heatmaps",
        }
    ),
    "59a1c39c": frozenset(
        {
            "plot_phase4_sampled_validation",
            "plot_phase4_small_gain_values",
            "plot_phase4_moat_heatmap",
            "plot_phase4_moat_profiles",
            "plot_phase4_moat_minima_table",
            "plot_phase4_moat_minima",
        }
    ),
    "32961420": frozenset({"plot_phase4_single_packet_profile"}),
    "phase2-math-41": frozenset({"plot_phase4_fragile_robustness"}),
    "ea23bc60": frozenset({"display_saved_image"}),
    "009356fa": frozenset(
        {
            "make_phase4_interactive_dashboard_renderer",
            "create_phase4_interactive_dashboard",
        }
    ),
    "a2b9f84258b4": frozenset({"plot_certification_ladder"}),
    "fd3908b317fe": frozenset({"plot_packet_contours"}),
    "3e8b784c": frozenset(
        {
            "display_table",
            "phase1_gradient_palette",
            "plot_certification_ladder",
        }
    ),
}

PLOTTING_HELPER_SIGNATURE_CONTRACT: Mapping[str, str] = {
    "make_phase4_interactive_dashboard_renderer": (
        "make_phase4_interactive_dashboard_renderer(*, surface_paths, packet_frame, "
        "eigenvalue_path, output_dir=None, stem=...) -> callback; the callback accepts "
        "packet_count, order_mode, opacity, azimuth, elevation, xy_scale, z_scale, "
        "zoom_factor, surface_resolution and show_eigenvalues"
    ),
}


# Complete reviewed replacements for two retained cells whose historical
# map-generic machinery is inappropriate in the symmetric final deployment.
# These cells remain executable and theorem-facing; only dormant alternative-map
# configuration and registry code is removed.
STATIC_CODE_REPLACEMENTS: Mapping[str, str] = {
    "91795b99": '''# Cell 4
# ============================================================
# 1. Symmetric benchmark configuration
# ============================================================

@dataclass
class Config:
    dps: int = int(os.environ.get("TRANSFER_LAB_DPS", "180"))
    process_workers: int = min(24, os.cpu_count() or 1)
    progress: bool = os.environ.get("TRANSFER_LAB_PROGRESS", "1") not in {"0", "false", "False"}
    mu: str = "0.3"
    rho: str = "2.725"
    R: str = "2.725"
    eta: str = "2.725"
    r_target: str = "2.473669807791324109321273260"
    selected_map_label: str = "blaschke_mu_0p3"
    smoke_mode: bool = os.environ.get("TRANSFER_LAB_SMOKE", "0") in {"1", "true", "True"}
    max_power: int = 25
    max_clusters: int = 24
    include_trivial: bool = False
    full_multiplicity: bool = True
    certification_target_count: int = 24


CERTIFIER_MAP_LABEL = "blaschke_mu_0p3"
CERTIFIER_TRIVIAL_TARGETS = 0
CERTIFIER_NONTRIVIAL_TARGETS = 24
CERTIFIER_TOTAL_TARGETS = 24

cfg = Config()
if cfg.selected_map_label != CERTIFIER_MAP_LABEL:
    raise RuntimeError("This numerical appendix is locked to blaschke_mu_0p3.")
if cfg.process_workers > 24:
    raise RuntimeError("The configured process pool exceeds the certified 24-core deployment limit.")

mp.mp.dps = cfg.dps
print(cfg)
print("active benchmark:", CERTIFIER_MAP_LABEL)
print("process workers:", cfg.process_workers)
''',
    "d09d9963": """# Cell 8
# ============================================================
# Symmetric Blaschke map specification and exact target families
# ============================================================

def blaschke_formula_map_spec(mu, max_power):
    '''Explanation: The map parameter, its two inverse branches, their transfer weights, and the alpha/mu target families define one coherent spectral model. Centralising them ensures every notebook phase studies the same thesis operator.
    Functionality: Construct the formula-map specification and exact alpha- and mu-power target clusters.'''
    mu = mp.mpf(mu)
    alpha = (1 + mu) / 2
    exact_clusters = [
        {
            "name": "1",
            "value": "1",
            "multiplicity": 1,
            "family": "trivial",
            "power": 0,
        }
    ]
    for n in range(1, int(max_power) + 1):
        exact_clusters.append(
            {
                "name": f"alpha^{n}",
                "value": f"alpha**{n}",
                "multiplicity": 1,
                "family": "alpha",
                "power": n,
            }
        )
    for n in range(1, int(max_power) + 1):
        exact_clusters.append(
            {
                "name": f"mu^{n}",
                "value": f"mu**{n}",
                "multiplicity": 2,
                "family": "mu",
                "power": n,
            }
        )
    return {
        "name": "formula_inverse_branches",
        "params": {
            "label": CERTIFIER_MAP_LABEL,
            "parameters": {
                "mu": mp.nstr(mu, cfg.dps),
                "alpha": mp.nstr(alpha, cfg.dps),
            },
            "branches": [
                {
                    "label": 1,
                    "tau": "x/2 - acos(mu*cos(pi*x/2))/pi",
                    "phi": "mp.mpf('0.5') - mu/2*sin(pi*x/2)/sqrt(1 - mu**2*cos(pi*x/2)**2)",
                },
                {
                    "label": 2,
                    "tau": "x/2 + acos(mu*cos(pi*x/2))/pi",
                    "phi": "mp.mpf('0.5') + mu/2*sin(pi*x/2)/sqrt(1 - mu**2*cos(pi*x/2)**2)",
                },
            ],
            "exact_clusters": exact_clusters,
            "spectrum_status": "known",
            "spectrum_source": "SBJ13 equation (21); same benchmark in ASBJ24 Section 3.2",
            "notes": (
                "The external exact formula supplies target identities and multiplicities; "
                "finite counts and contour moats are certified independently below."
            ),
        },
    }


BLASCHKE_SPEC = serialisable_map_spec(
    map_spec=blaschke_formula_map_spec(cfg.mu, cfg.max_power)
)
SELECTED_MAP_LABEL = CERTIFIER_MAP_LABEL
SELECTED_MAP_SPEC = BLASCHKE_SPEC
SELECTED_MAP_STATUS = "known exact target families"
SELECTED_MAP_OBJECT = make_transfer_map(map_spec=SELECTED_MAP_SPEC)
SELECTED_MAP_HAS_EXACT_TARGETS = True
CERTIFIER_FULL_EXECUTION = not cfg.smoke_mode
blaschke = SELECTED_MAP_OBJECT

selected_clusters = exact_clusters_for_map(
    map_spec=SELECTED_MAP_SPEC,
    max_power=cfg.max_power,
    include_trivial=True,
    full_multiplicity=cfg.full_multiplicity,
)
SELECTED_TARGET_NAMES_WITH_TRIVIAL = [str(cluster.get("name")) for cluster in selected_clusters]
SELECTED_TARGET_NAMES_NONTRIVIAL = [
    name for name in SELECTED_TARGET_NAMES_WITH_TRIVIAL if name != "1"
]
SELECTED_SHORT_TARGET_NAMES_WITH_TRIVIAL = SELECTED_TARGET_NAMES_WITH_TRIVIAL[:11]
SELECTED_SHORT_TARGET_NAMES_NONTRIVIAL = [
    name for name in SELECTED_SHORT_TARGET_NAMES_WITH_TRIVIAL if name != "1"
]

print("selected map:", SELECTED_MAP_LABEL)
print("selected map status:", SELECTED_MAP_STATUS)
print("selected map branches:", len(SELECTED_MAP_SPEC.get("params", {}).get("branches", ())))
print("first selected targets:", SELECTED_TARGET_NAMES_WITH_TRIVIAL[:15])
""",
    "ce7ba80a": """# Cell 9
# ============================================================
# Symmetric-branch validation and Bernstein-ellipse diagnostics
# ============================================================

def ellipse_point(radius, t):
    '''Explanation: A Bernstein ellipse is the Joukowski image of a circle and parametrises a complex neighbourhood of the interval. One boundary point is the basic geometric input for inspecting inverse-branch analyticity.
    Functionality: Evaluate the Joukowski parametrisation of the Bernstein ellipse boundary.'''
    radius = mp.mpf(radius)
    z = radius * mp.e ** (1j * t)
    return (z + 1 / z) / 2


def bernstein_radius(w):
    '''Explanation: The outer Joukowski modulus labels which Bernstein ellipse passes through a complex point. It converts the plotted branch image into the exponential coordinate used by the analytic tail formulas.
    Functionality: Return the outer Joukowski-preimage modulus of a complex point.'''
    s = mp.sqrt(w * w - 1)
    return max(abs(w + s), abs(w - s))


def validate_real_branches(map_obj, grid=1001):
    '''Explanation: On the real interval, the two inverse branches should satisfy the defining Blaschke equation and the expected branch ordering. Checking these identities catches a wrong radical sign before it contaminates every numerical stage.
    Functionality: Check real-interval inverse-branch ranges, positivity, and imaginary leakage.'''
    xs = [mp.mpf(-1) + 2 * mp.mpf(j) / (grid - 1) for j in range(grid)]
    min_phi = mp.inf
    max_tau_abs = mp.mpf("0")
    max_phi_imag = mp.mpf("0")
    max_tau_imag = mp.mpf("0")
    for x in xs:
        for ell in map_obj.branches():
            tau = map_obj.tau(ell, x)
            phi = map_obj.phi(ell, x)
            min_phi = min(min_phi, mp.re(phi))
            max_tau_abs = max(max_tau_abs, abs(tau))
            max_phi_imag = max(max_phi_imag, abs(mp.im(phi)))
            max_tau_imag = max(max_tau_imag, abs(mp.im(tau)))
    return {
        "min_phi": min_phi,
        "max_tau_abs": max_tau_abs,
        "max_phi_imag": max_phi_imag,
        "max_tau_imag": max_tau_imag,
    }


def diagnostic_branch_geometry(map_obj, rho, n_boundary=4096):
    '''Explanation: Mapping a sampled ellipse boundary through both inverse branches estimates contraction, weight size, and separation from singularities. This guides parameter selection, while the later interval scan supplies the actual proof.
    Functionality: Sample the response ellipse to estimate branch radii and transfer-weight maxima.'''
    rho = mp.mpf(rho)
    r_tau = mp.mpf("0")
    phi_branch_max = {ell: mp.mpf("0") for ell in map_obj.branches()}
    phi_star = mp.mpf("0")
    for j in range(int(n_boundary)):
        t = 2 * mp.pi * j / int(n_boundary)
        omega = ellipse_point(rho, t)
        local_phi_sum = mp.mpf("0")
        for ell in map_obj.branches():
            tau_omega = map_obj.tau(ell, omega)
            phi_omega = map_obj.phi(ell, omega)
            r_tau = max(r_tau, bernstein_radius(tau_omega))
            phi_abs = abs(phi_omega)
            phi_branch_max[ell] = max(phi_branch_max[ell], phi_abs)
            local_phi_sum += phi_abs
        phi_star = max(phi_star, local_phi_sum)
    return {
        "r_tau": r_tau,
        "Phi": sum(phi_branch_max.values()),
        "Phi_star": phi_star,
        "method": "sampled symmetric branch formula; diagnostic only",
    }


if SELECTED_MAP_LABEL != CERTIFIER_MAP_LABEL:
    raise RuntimeError("This numerical appendix is locked to blaschke_mu_0p3.")

real_val = validate_real_branches(SELECTED_MAP_OBJECT)
SELECTED_MAP_GEOMETRY_SCAN = pd.DataFrame()
geo = diagnostic_branch_geometry(SELECTED_MAP_OBJECT, cfg.rho, n_boundary=2048)
SELECTED_MAP_GEOMETRY_STATUS = "diagnostic sampling; certified constants are loaded in Phase 2"
r_target = mp.mpf(cfg.r_target)
rho = mp.mpf(cfg.rho)
geometry_record = {
    "map_label": SELECTED_MAP_LABEL,
    "rho": float(rho),
    "r_tau_sampled": float(geo["r_tau"]),
    "r_target": float(r_target),
    "Phi_sampled": float(geo["Phi"]),
    "Phi_star_sampled": float(geo["Phi_star"]),
    "q_out": float(r_target / rho),
    "q_gap": float(geo["r_tau"] / r_target),
    "q_star": float(max(r_target / rho, geo["r_tau"] / r_target)),
    "admissible_sampled": bool(geo["r_tau"] < r_target < rho),
    "geometry_status": SELECTED_MAP_GEOMETRY_STATUS,
    "method": geo["method"],
}
pd.DataFrame([geometry_record]).to_csv(
    DATA_DIR / f"transfer_lab_{SELECTED_MAP_LABEL}_active_geometry.csv",
    index=False,
)

print("Real branch validation")
print("  min phi      =", mp_str(real_val["min_phi"], 25))
print("  max |tau|    =", mp_str(real_val["max_tau_abs"], 25))
print("  max Im(phi)  =", mp_str(real_val["max_phi_imag"], 8))
print("  max Im(tau)  =", mp_str(real_val["max_tau_imag"], 8))
print("Complex geometry evidence")
print("  status       =", SELECTED_MAP_GEOMETRY_STATUS)
print("  r_tau        =", mp_str(geo["r_tau"], 25))
print("  Phi          =", mp_str(geo["Phi"], 25))
print("  Phi_star     =", mp_str(geo["Phi_star"], 25))
print("  method       =", geo["method"])
print("  rho          =", mp_str(rho, 25))
print("  r_target     =", mp_str(r_target, 25))
print("  q_out        =", mp_str(r_target / rho, 25))
print("  q_gap        =", mp_str(geo["r_tau"] / r_target, 25))
print("  admissible   =", bool(geo["r_tau"] < r_target < rho))
display(pd.DataFrame([geometry_record]))
""",
}


def _retain_balanced_finite_m_certificate_only(source: str) -> str:
    """Remove wide-row and sampled first-fifteen side effects from Cell 24A."""

    wide_marker = "# Apply the same unconditional finite-M certificate to the wide Phase 4 row.\n"
    promotion_marker = "# Promote the safe row for the subsequent response-prefactor producer and Phase 2 exports.\n"
    if wide_marker not in source or promotion_marker not in source:
        raise AppendixPreparationError(
            "The reviewed Cell 24A balanced-certificate boundaries have changed."
        )
    balanced_prefix = source.split(wide_marker, 1)[0]
    balanced_promotion = source.split(promotion_marker, 1)[1]
    source = balanced_prefix + promotion_marker + balanced_promotion
    forbidden = (
        "_fm_wide_",
        "wide_df",
        "wide_path",
        "first15",
        "first-fifteen",
        "sampled_spectral_packets",
    )
    residue = tuple(token for token in forbidden if token in source)
    if residue:
        raise AppendixPreparationError(
            f"Excluded Cell 24A compatibility material remains: {residue}."
        )
    return source


def _complete_phase1_plot_exports(source: str) -> str:
    """Validate the source-regenerated Phase 1 render cell."""

    completed = r'''# Cell 15: render only source-regenerated Phase 1 diagnostics.
from plotting import (
    parse_complex_for_plot,
    parse_selected_complex_values,
    plot_phase1_absolute_m_heatmaps,
    plot_phase1_eigencloud_comparison,
    plot_phase1_fixed_n_m_sweep,
    plot_phase1_oversampling_displacement,
)


def _phase1_cloud_block(match_rows, eigenvalue_rows, N, M):
    matches = match_rows[
        (match_rows.N.astype(int) == int(N))
        & (match_rows.M.astype(int) == int(M))
    ].head(25)
    eigenvalues = eigenvalue_rows[
        (eigenvalue_rows.N.astype(int) == int(N))
        & (eigenvalue_rows.M.astype(int) == int(M))
    ]
    if matches.empty or eigenvalues.empty:
        raise RuntimeError(f"Source-regenerated Phase 1 cloud is empty for N={N}, M={M}.")
    target_source = matches.get("target", matches.get("target_float"))
    targets = np.asarray(
        [parse_complex_for_plot(value) for value in target_source],
        dtype=np.complex128,
    )
    return {
        "N": int(N),
        "M": int(M),
        "eigenvalues": np.asarray(
            [parse_complex_for_plot(value) for value in eigenvalues.eig],
            dtype=np.complex128,
        ),
        "targets": targets,
        "matched": (),
        "target_labels": tuple(zip(matches.name, targets)),
    }


eig_source = pd.DataFrame(raw_results.get("eig_rows", ()))
cloud_blocks = [
    _phase1_cloud_block(raw_df, eig_source, 20, 20),
    _phase1_cloud_block(raw_df, eig_source, max(N_VALUES_RAW), max(N_VALUES_RAW)),
]
phase1_eigencloud_result = plot_phase1_eigencloud_comparison(
    cloud_blocks,
    output_dir=FIG_DIR,
)


def _phase1_generated_cloud(N, M):
    match_rows = PHASE1_DIAGNOSTICS_RESULT.cloud_match_frames[(int(N), int(M))]
    eigenvalue_rows = PHASE1_DIAGNOSTICS_RESULT.cloud_eigenvalue_frames[(int(N), int(M))]
    block = _phase1_cloud_block(match_rows, eigenvalue_rows, N, M)
    selected_labels = tuple(
        (str(row["name"]), point)
        for _, row in match_rows.iterrows()
        for point in parse_selected_complex_values(row.get("selected"))
    )
    return block["eigenvalues"], block["targets"], selected_labels


phase1_oversampling_results = {}
for N, M_base, M_comparison, radius, stem, title in (
    (20, 20, 30, 5.0e-2, "phase1_N20_M20_to_M30_eigencloud_displacement", "Visible oversampling effect at fixed N=20"),
    (12, 12, 24, 1.6e-1, "phase1_N12_M12_to_M24_visible_eigencloud_displacement", "More visible oversampling effect at fixed N=12"),
):
    base_values, exact_targets, _ = _phase1_generated_cloud(N, M_base)
    comparison_values, _, comparison_labels = _phase1_generated_cloud(N, M_comparison)
    phase1_oversampling_results[stem] = plot_phase1_oversampling_displacement(
        base_values,
        comparison_values,
        exact_targets,
        title=title,
        base_label=f"M={M_base}",
        comparison_label=f"M={M_comparison}",
        comparison_point_labels=comparison_labels,
        radius=radius,
        magnification=12.0,
        output_dir=FIG_DIR,
        stem=stem,
    )

phase1_M_df = PHASE1_DIAGNOSTICS_RESULT.fixed_m_frame.copy()
phase1_M_sweep_result = plot_phase1_fixed_n_m_sweep(
    phase1_M_df,
    list(PHASE1_SHORT_DISPLAY_TARGET_NAMES_NONTRIVIAL)[:8],
    n_value=25,
    map_label=SELECTED_MAP_LABEL,
    output_dir=FIG_DIR,
)

phase1_heat_df = PHASE1_DIAGNOSTICS_RESULT.heatmap_frame.copy()
Ns = sorted(phase1_heat_df.N.astype(int).unique())
Ms = sorted(phase1_heat_df.M.astype(int).unique())
targets = list(PHASE1_SHORT_DISPLAY_TARGET_NAMES_NONTRIVIAL)[:6]
phase1_heatmap_result = plot_phase1_absolute_m_heatmaps(
    phase1_heat_df,
    targets,
    Ns,
    Ms,
    value_column="error",
    title_prefix="Exact-target raw error: ",
    array_title="Raw cluster diagnostics in the Legendre--Gauss transfer block",
    cbar_title=r"$\log_{10}\Delta_{cluster}(N,M)$",
    output_dir=FIG_DIR,
    stem="phase1_heatmap_near_square_raw_errors",
)
phase1_heatmap_excess_result = plot_phase1_absolute_m_heatmaps(
    phase1_heat_df,
    targets,
    Ns,
    Ms,
    value_column="log10_excess_ratio_to_max_M",
    title_prefix="Finite-M excess factor: ",
    array_title="Finite-M excess relative to the largest available M at fixed N",
    cbar_title=r"$\log_{10}(\Delta(N,M)/\Delta(N,M_{\max}))$",
    output_dir=FIG_DIR,
    stem="phase1_heatmap_near_square_excess",
)
'''
    ast.parse(completed)
    ast.parse(source)
    required_exports = (
        "phase1_eigencloud_leading_interval_comparison_points",
        "phase1_N20_M20_to_M30_cloud_displacements",
        "phase1_N12_M12_to_M24_visible_cloud_displacements",
        "save_dataframe",
    )
    missing_exports = tuple(token for token in required_exports if token not in source)
    if missing_exports:
        raise AppendixPreparationError(
            "The reviewed Cell 15 derivative exports are incomplete: "
            f"{missing_exports}."
        )
    forbidden_reads = tuple(
        token for token in ("pd.read_csv", ".exists()") if token in source
    )
    if forbidden_reads:
        raise AppendixPreparationError(
            "Cell 15 must persist only source-regenerated frames; destination-file "
            f"fallbacks remain: {forbidden_reads}."
        )
    return source


# Source is hidden while outputs remain visible. These are infrastructure or
# diagnostic presentation cells, not theorem-facing certificate computations.
INFRASTRUCTURE_CELL_IDS = frozenset(
    {
        "6a823024", "code-bffea704", "91795b99",
        "inline-helper-bootstrap-code", "0e01d379", "code-7a22b58f",
        "00e8904c", "ffc45508",
    }
)

DIAGNOSTIC_PLOTTING_CELL_IDS = frozenset(
    {
        "2a6f973c", "c1b4b6e4", "cc338a81", "e40904af", "b4390377",
        "code-7598db69", "c92b9a83", "df668eaa", "5cb2c38a",
        "e4cd9776", "3410c676", "340f0292", "add52fcf", "8bf58bd5",
        "59a1c39c", "32961420", "phase2-math-41", "ea23bc60",
        "009356fa", "a2b9f84258b4", "fd3908b317fe",
    }
)

EXPECTED_PLOTTING_REPLACEMENT_CELL_IDS = frozenset(
    {
        "6a823024", "0e01d379", "2a6f973c", "c1b4b6e4", "cc338a81",
        "e40904af", "b4390377", "code-7598db69", "c92b9a83",
        "7d75c6ff", "df668eaa", "5cb2c38a", "e4cd9776", "3410c676",
        "340f0292", "add52fcf", "8bf58bd5",
        "59a1c39c", "32961420", "phase2-math-41", "ea23bc60",
        "009356fa", "a2b9f84258b4", "fd3908b317fe",
    }
)


MARKDOWN_REPLACEMENTS: Mapping[str, str] = {
    "inline-helper-bootstrap-heading": r"""### Executable mathematical helpers

The following cells expose the repository-local mathematical helpers used by
the symmetric Blaschke certification. Rendering functions are imported from
`plotting.py`, so the numerical cells remain focused on their mathematical
inputs, computations and certified outputs.
""",
    "c28fbc71": r"""# Single-space Legendre--Gauss transfer certification

This appendix notebook implements the four-phase numerical workflow for the
ASBJ24 symmetric Blaschke deformation with \(\mu=0.3\). Its finite matrices
are Legendre--Gauss Galerkin--EDMD approximations of the Perron--Frobenius
transfer operator, assembled from inverse branches and transfer weights; they
are not Koopman matrices.

The phases comprise raw transfer spectra, the branch-image single-space
deterministic certificate, empirical-versus-deterministic rate diagnostics,
and complete-contour Hardy-gauge certification. The theorem-facing result is
the validated \(N=600\), \(M=610\) row and its twenty-four nontrivial Riesz
packets. Diagnostic plots are retained as diagnostics and are not substituted
for complete-boundary or complete-contour validated arithmetic.
""",
    "33580cd0": r"""## Main numerical algorithm: the symmetric Blaschke transfer block

The notebook is locked to the ASBJ24 symmetric Blaschke deformation
`blaschke_mu_0p3`. Its inverse branches $\tau_b$ and Perron--Frobenius
weights $\phi_b$ assemble the Legendre--Gauss transfer block
$$
  (\widehat L_N^{(M)})_{k\ell}
  =
  \sum_{j=1}^{M}w_j\overline{\widetilde P_k(x_j)}
  \sum_b\phi_b(x_j)\widetilde P_\ell(\tau_b(x_j)).
$$
The raw block supplies empirical spectra. The pure $r$-scaled block supplies
the separable-Frobenius collocation estimate, and
$$
  \mathsf A_N^X(\widehat L_N^{(M)})
  =
  \mathsf T_N(r)\mathsf D_{r,N}\widehat L_N^{(M)}
  \mathsf D_{r,N}^{-1}\mathsf T_N(r)^{-1}
$$
is the Hardy-gauge matrix used for singular values, resolvent moats and Riesz
ranks. These matrices are similar and share eigenvalues, but their Euclidean
norms have different meanings. The exact Blaschke spectral formula is used
only to name packets and exhaust the exterior tail; finite counts and moats are
certified independently.
""",
    "worker-module-maths-map": r"""## Worker-module mathematics for the symmetric benchmark

The inline copy of `mpmath_pf_raw.py` performs the high-precision
Legendre--Gauss Perron--Frobenius assembly. In this notebook its serialised
branch data are fixed to `blaschke_mu_0p3`. For each \((N,M)\), process workers
evaluate the orthonormal Legendre rows, the two inverse-branch responses and
their transfer weights, then assemble \(\widehat L_N^{(M)}\). The public
routines used below are the pair and \(N\)-sweep assemblers, the raw and pure
\(r\)-scaled block constructors, and the exact Blaschke target constructor.
""",
    "e5cdca61": r"""## Numerical helper dictionary for the symmetric benchmark

High-precision transfer quantities remain in `mpmath` until a diagnostic
explicitly crosses into NumPy, SciPy or Pandas. CSV and Parquet writers retain
the same numerical rows; plotting conversions are not interval enclosures.

The retained helpers cover four tasks: assembly and persistence of the
`blaschke_mu_0p3` Legendre--Gauss blocks; matching against its exact external
target families; construction of the pure \(r\)-scaled and Chebyshev-packet
Hardy gauges; and sampled diagnostics kept separate from the later validated
complete-boundary and complete-contour certificates. The sampled Schur rows
are diagnostic only. The theorem-facing finite counts, moats and small-gain
gates are supplied by the source-backed validated stages below.
""",
    "c756b6ef": r"""## High-precision worker module

The process workers use the repository-local helper `mpmath_pf_raw.py` for the
map-independent Legendre--Gauss Perron--Frobenius assembly. The active branch
data are fixed in this notebook to the symmetric \(\mu=0.3\) Blaschke
deformation. Each process has its own `mpmath` precision context.
""",
    "f93d42b4b1f9": r"""# Exact-spectrum provenance for the symmetric benchmark

The external exact Blaschke formula is used to name target packets and to
identify the remaining exterior spectral tail. Finite algebraic counts,
complete-circle moats and Riesz-rank transfer are certified independently by
the validated Hardy-gauge calculations below.

The external formula supplies target identities and multiplicities, not finite counts or contour moats.
""",
    "md-73443732": r"""## Validated contours and Riesz ranks

All twenty-four finite algebraic counts are Schur-derived. Seventeen contours
use triangular Schur moats; the seven deeper contours use Laurent moats whose
candidate tensors are reconstructed from source in Cell 103. Those seven
transient tensors contain 300 complex (600) by (600) coefficient matrices.
Every coefficient is lifted exactly into Arb for the residual proof, while a
seven-row manifest records the generation parameters, full-tensor digests and
validated mode bounds. Historical digest parity is a provenance check rather
than a theorem gate.
""",
    "a81aa734": r"""### Sampled first-fifteen Phase 4 diagnostics

The following retained figures reproduce the original sampled first-fifteen
Hardy-gauge diagnostics from their stored CSV artefacts. They are visual
diagnostics only and do not replace the later complete-contour certificate.
""",
    "3d797d1d": r"""### Selected sampled packet profiles

The first figure selects the first-fifteen contour profile with the largest
sampled small-gain product.  The second displays the fixed \(\alpha^{11}\)
profile for direct comparison with the earlier diagnostic.  Both record their
sampled minima and perturbation reference levels.  They are diagnostics only,
not certified moats.
""",
    "b8581535": r"""### Fragile-packet sampling comparison

This retained diagnostic compares the stored angular-sampling rows for the two
most fragile packets in the first-fifteen sampled study.
""",
    "59737fdb1e5a": r"""### Full-resolution static moat landscape

This render-only cell displays the saved global first-fifteen moat landscape at
its original resolution without recomputing it.
""",
    "1c203394": r"""### Interactive sampled moat dashboard

The dashboard is reconstructed from the retained surface caches, contour rows
and finite-eigenvalue table. Its renderer belongs entirely to `plotting.py`.
""",
    "3b83ae378cf8": r"""### Symmetric-benchmark certification ladder

This presentation-only diagnostic summarises the stored certification statuses
for the fixed symmetric Blaschke benchmark.
""",
    "42a2aee243d6": r"""### First fourteen sampled packet contours

This retained plot displays the first fourteen symmetric-benchmark packets and
their stored sampled contour geometry. The later validated twenty-four-contour
certificate remains the theorem-facing result.
""",
}


NOTEBOOK_METADATA_KEYS_TO_REMOVE = frozenset(
    {"codex_update_history", "codex_update_number", "transfer_spectrum_lab_transplant", "widgets"}
)

CELL_METADATA_KEYS_TO_REMOVE = frozenset(
    {
        "codex_update_history", "codex_update_number", "execution",
        "executionInfo", "ExecuteTime", "timestamp", "timestamps",
    }
)

UPDATE_HEADING_RE = re.compile(r"(?:Historical\s+)?Notebook\s+update\s+\d+", re.IGNORECASE)

LOCAL_HELPER_MODULES = frozenset(
    {
        "blaschke_deformation_certification",
        "blaschke_deformation_contour_certification",
        "blaschke_deformation_phase1_diagnostics",
        "blaschke_deformation_reproducibility",
        "blaschke_deformation_sampled_schur_diagnostics",
        "blaschke_deformation_spectral_certification",
        "hardy_moat_surface_worker",
        "mpmath_pf_raw",
        "plotting",
        "transfer_spectrum_certification",
    }
)


class AppendixPreparationError(RuntimeError):
    """Raised when a fail-closed preparation invariant is not satisfied."""


def _source_text(cell: Mapping[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def _store_source(cell: dict[str, Any], text: str) -> None:
    original = cell.get("source", "")
    cell["source"] = text.splitlines(keepends=True) if isinstance(original, list) else text


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_output_hashes(notebook: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(cell["id"]): _canonical_digest(cell.get("outputs", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code" and str(cell.get("id", "")) not in REMOVE_CELL_IDS
    }


def retained_execution_counts(notebook: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(cell["id"]): cell.get("execution_count")
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code" and str(cell.get("id", "")) not in REMOVE_CELL_IDS
    }


def _index_cells(notebook: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    missing = 0
    for cell in notebook.get("cells", []):
        cell_id = str(cell.get("id", ""))
        if not cell_id:
            missing += 1
            continue
        if cell_id in result:
            duplicates.append(cell_id)
        result[cell_id] = cell
    if missing or duplicates:
        raise AppendixPreparationError(
            f"Stable cell ids are invalid: missing={missing}, duplicates={sorted(set(duplicates))}."
        )
    return result


def validate_source_notebook(notebook: Mapping[str, Any]) -> None:
    cells = notebook.get("cells", [])
    if len(cells) != EXPECTED_SOURCE_CELL_COUNT:
        raise AppendixPreparationError(
            f"Expected the current {EXPECTED_SOURCE_CELL_COUNT}-cell executed notebook, found {len(cells)} cells."
        )
    index = _index_cells(notebook)
    missing = sorted(REMOVE_CELL_IDS.difference(index))
    if missing:
        raise AppendixPreparationError(
            "The source layout no longer matches the reviewed stable-id removal contract; "
            f"missing ids: {missing}."
        )
    if len(REMOVE_CELL_IDS) != sum(len(group) for group in REMOVAL_GROUPS.values()):
        raise AppendixPreparationError("A stable cell id appears in more than one removal group.")


def _normalise_provenance_records(
    payload: Any,
    notebook: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    descriptive_entries = False
    if isinstance(payload, dict) and "cells" in payload:
        payload = payload["cells"]
    elif isinstance(payload, dict) and "entries" in payload:
        payload = payload["entries"]
        descriptive_entries = True
    if isinstance(payload, dict):
        records = payload
    elif isinstance(payload, list):
        records = {
            str(record.get("cell_id", "")): record
            for record in payload
            if isinstance(record, dict) and record.get("cell_id")
        }
    else:
        raise AppendixPreparationError(
            "notebook_cell_provenance.json must contain an object or a list of records."
        )
    normalised: dict[str, dict[str, Any]] = {}
    notebook_cells = notebook.get("cells", [])

    def current_full_index(legacy_index: int) -> int:
        """Translate the reviewed 181-cell map to the 187-cell producer build."""

        adjusted = int(legacy_index)
        if legacy_index >= 32:
            adjusted += 3
        if legacy_index >= 148:
            adjusted += 2
        if legacy_index >= 151:
            adjusted += 1
        return adjusted

    for record_key, record in records.items():
        if not isinstance(record, dict):
            raise AppendixPreparationError(
                f"Provenance record {record_key!r} is not an object."
            )
        cell_id = record.get("cell_id")
        if not cell_id and descriptive_entries:
            cell_index = record.get("cell_index")
            if isinstance(cell_index, int) and len(notebook_cells) == 187:
                cell_index = current_full_index(cell_index)
            if not isinstance(cell_index, int) or not 0 <= cell_index < len(notebook_cells):
                raise AppendixPreparationError(
                    f"Provenance record {record_key!r} has no valid cell_index."
                )
            cell_id = notebook_cells[cell_index].get("id")
        if not cell_id:
            cell_id = record_key
        if not cell_id:
            raise AppendixPreparationError(
                f"Provenance record {record_key!r} cannot be resolved to a stable cell id."
            )
        if str(cell_id) in normalised:
            raise AppendixPreparationError(
                f"Multiple provenance records resolve to stable cell id {cell_id!r}."
            )
        normalised[str(cell_id)] = dict(record)
    return normalised


def load_provenance_records(
    path: Path,
    notebook: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], bool]:
    if not path.exists():
        return {}, False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AppendixPreparationError(f"Cannot read provenance map {path}: {exc}") from exc
    return _normalise_provenance_records(payload, notebook), True


def _complete_plotting_replacement_source(cell_id: str, source: str) -> str:
    """Complete data wiring while keeping every renderer in plotting.py."""

    if cell_id == "cc338a81":
        source = source.replace(
            "n_value=25,output_dir=FIG_DIR",
            "n_value=25,map_label=SELECTED_MAP_LABEL,output_dir=FIG_DIR",
            1,
        )
    elif cell_id == "a2b9f84258b4":
        source = source.replace(
            "schur_rows=DATA_DIR / 'transfer_lab_blaschke_mu_0p3_generic_sampled_schur_envelope.csv'",
            "schur_rows=SAMPLED_SCHUR_DIAGNOSTICS_RESULT.csv_path",
            1,
        )
    elif cell_id == "df668eaa":
        source = source.replace(
            "plot_phase3_raw_errors_against_qstar(phase3_raw_curve_df,PHASE3_TARGETS,output_dir=FIG_DIR)",
            "plot_phase3_raw_errors_against_qstar(phase3_raw_curve_df,PHASE3_TARGETS,q_star=QSTAR_X,output_dir=FIG_DIR)",
            1,
        )
    lines = source.splitlines()
    if cell_id == "340f0292":
        replacement = """directory=_retained_phase4_data_dir()
contour_frame,contours=_retained_contours(directory)
path,X,Y,S=_retained_surface(directory,'branch_image_wide_candidate_mu2_hardy_moat_surface_N600_M610_grid25.npz')
local=[row for row in contours if row['name']=='mu^2']
profile_frame=pd.read_csv(directory/'branch_image_wide_candidate_first15_contour_profiles_N600_M610_J128.csv')
local_profile=profile_frame.loc[profile_frame['target'].astype(str).eq('mu^2')].sort_values('theta')
eigenvalue_frame=pd.read_csv(directory/'phase4_hp_hardy_gauge_eigenvalues_N600_M610.csv')
local_eigenvalues=eigenvalue_frame['real'].to_numpy(dtype=float)+1j*eigenvalue_frame['imag'].to_numpy(dtype=float)
phase4_local_surface_result=plot_phase4_local_moat_surface(
    X,Y,S,local,
    profile_theta=local_profile['theta'].to_numpy(dtype=float),
    profile_moat=local_profile['smin'].to_numpy(dtype=float),
    eigenvalues=local_eigenvalues,
    output_dir=FIG_DIR,
)
print(path)"""
        lines = [replacement if line.startswith("directory=_retained_phase4_data_dir();") else line for line in lines]
    elif cell_id == "add52fcf":
        replacement = """directory=_retained_phase4_data_dir()
contour_frame,contours=_retained_contours(directory)
path,X,Y,S=_retained_surface(directory,'branch_image_wide_candidate_first15_global_hardy_moat_surface_N600_M610_grid45.npz')
zoom_path,ZX,ZY,ZS=_retained_surface(directory,'branch_image_wide_candidate_first15_zoom_hardy_moat_surface_N600_M610_grid161.npz')
deep_path,DX,DY,DS=_retained_surface(directory,'branch_image_wide_candidate_first15_deep_zoom_hardy_moat_surface_N600_M610_grid201.npz')
eigenvalue_frame=pd.read_csv(directory/'phase4_hp_hardy_gauge_eigenvalues_N600_M610.csv')
global_eigenvalues=eigenvalue_frame['real'].to_numpy(dtype=float)+1j*eigenvalue_frame['imag'].to_numpy(dtype=float)
phase4_global_surface_result=plot_phase4_global_moat_surface(
    X,Y,S,contours,
    zoom_surface=(ZX,ZY,ZS),
    deep_zoom_surface=(DX,DY,DS),
    eigenvalues=global_eigenvalues,
    output_dir=FIG_DIR,
)
print(path,zoom_path,deep_path)"""
        lines = [replacement if line.startswith("directory=_retained_phase4_data_dir();") else line for line in lines]
    elif cell_id == "8bf58bd5":
        replacement = """directory=_retained_phase4_data_dir()
contour_frame,contours=_retained_contours(directory)
path,X,Y,S=_retained_surface(directory,'branch_image_wide_candidate_first15_global_hardy_moat_surface_N600_M610_grid45.npz')
phase4_global_zoom_result=plot_phase4_global_zoom(X,Y,S,contours,output_dir=FIG_DIR)
print(path)"""
        lines = [replacement if line.startswith("directory=_retained_phase4_data_dir();") else line for line in lines]
    elif cell_id == "009356fa":
        snapshot_line = next((index for index, line in enumerate(lines) if line.startswith("phase4_dashboard_snapshot =")), None)
        if snapshot_line is not None:
            lines[snapshot_line:] = [
                "if globals().get('CERTIFIER_BATCH_EXECUTION', False):",
                "    phase4_dashboard_snapshot = phase4_dashboard_renderer(packet_count=7, order_mode='near-origin order', opacity=0.30, azimuth=306, elevation=24, xy_scale=1.0, z_scale=0.34, zoom_factor=2.4, surface_resolution=150, show_eigenvalues=True)",
                "    phase4_dashboard = None",
                "else:",
                "    phase4_dashboard_snapshot = None",
                "    phase4_dashboard = create_phase4_interactive_dashboard(phase4_dashboard_renderer, maximum_packets=len(phase4_dashboard_packets), default_packets=7)",
            ]
    completed = "\n".join(lines).rstrip() + "\n"
    ast.parse(completed)
    return completed


def load_plotting_replacements(
    path: Path,
    notebook: Mapping[str, Any],
) -> tuple[dict[str, str], bool]:
    """Load reviewed complete code-cell sources keyed by stable cell id."""

    if not path.exists():
        return {}, False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AppendixPreparationError(
            f"Cannot read plotting replacement map {path}: {exc}"
        ) from exc
    if isinstance(payload, dict) and "replacements" in payload:
        payload = payload["replacements"]
    if not isinstance(payload, dict):
        raise AppendixPreparationError(
            "plotting_cell_replacements.json must contain an object mapping."
        )
    notebook_cells = notebook.get("cells", [])
    notebook_index = _index_cells(notebook)
    replacements: dict[str, str] = {}
    for record_key, record in payload.items():
        if isinstance(record, str):
            cell_id = str(record_key)
            source = record
        elif isinstance(record, dict):
            cell_id = str(record.get("cell_id", record_key))
            if cell_id not in notebook_index and isinstance(record.get("cell_index"), int):
                cell_index = int(record["cell_index"])
                if not 0 <= cell_index < len(notebook_cells):
                    raise AppendixPreparationError(
                        f"Plotting replacement {record_key!r} has an invalid cell_index."
                    )
                cell_id = str(notebook_cells[cell_index].get("id", ""))
            source = record.get("source")
            source_lines = record.get("source_lines")
            if source is None and isinstance(source_lines, list):
                if not all(isinstance(line, str) for line in source_lines):
                    raise AppendixPreparationError(
                        f"Plotting replacement {record_key!r} has non-string source_lines."
                    )
                source = "\n".join(source_lines).rstrip() + "\n"
        else:
            raise AppendixPreparationError(
                f"Plotting replacement {record_key!r} is not a string or object."
            )
        if cell_id not in notebook_index:
            raise AppendixPreparationError(
                f"Plotting replacement {record_key!r} does not resolve to a stable cell id."
            )
        if cell_id in REMOVE_CELL_IDS:
            raise AppendixPreparationError(
                f"Plotting replacement {record_key!r} targets a removed cell."
            )
        if notebook_index[cell_id].get("cell_type") != "code":
            raise AppendixPreparationError(
                f"Plotting replacement {record_key!r} does not target a code cell."
            )
        if not isinstance(source, str) or not source.strip():
            raise AppendixPreparationError(
                f"Plotting replacement {record_key!r} has no complete source string."
            )
        source = _complete_plotting_replacement_source(cell_id, source)
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise AppendixPreparationError(
                f"Plotting replacement {record_key!r} is not valid Python: {exc}"
            ) from exc
        if cell_id in replacements:
            raise AppendixPreparationError(
                f"Multiple plotting replacements target stable cell id {cell_id!r}."
            )
        replacements[cell_id] = source
    return replacements, True


def validate_plotting_replacement_coverage(replacements: Mapping[str, str]) -> None:
    actual = set(replacements)
    missing = sorted(EXPECTED_PLOTTING_REPLACEMENT_CELL_IDS.difference(actual))
    unexpected = sorted(actual.difference(EXPECTED_PLOTTING_REPLACEMENT_CELL_IDS))
    if missing or unexpected:
        raise AppendixPreparationError(
            "The reviewed plotting replacement contract is incomplete or has drifted: "
            f"missing={missing}, unexpected={unexpected}."
        )


def _direct_plotting_residue(source: str) -> list[str]:
    """Return direct graphics definitions/imports which must live in plotting.py."""

    lines = source.splitlines()
    if lines and lines[0].lstrip().startswith("%%"):
        lines = lines[1:]
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return ["source is not parseable for plotting-residue validation"]
    residue: set[str] = set()
    rendering_methods = {
        "add_patch",
        "add_subplot",
        "annotate",
        "axhline",
        "axvline",
        "bar",
        "barh",
        "colorbar",
        "contour",
        "contourf",
        "figure",
        "imshow",
        "loglog",
        "pcolormesh",
        "plot",
        "plot_surface",
        "savefig",
        "scatter",
        "semilogx",
        "semilogy",
        "show",
        "subplot",
        "subplots",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(("matplotlib", "seaborn", "plotly")):
                    residue.add(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith(("matplotlib", "seaborn", "plotly")):
                residue.add(f"from {module} import ...")
        elif isinstance(node, ast.Name) and node.id == "plt":
            residue.add("direct plt reference")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in rendering_methods:
                residue.add(f"direct graphics method {node.func.attr}()")
    return sorted(residue)


def _called_function_names(source: str) -> frozenset[str]:
    """Return directly called function names from a parseable code cell."""

    lines = source.splitlines()
    if lines and lines[0].lstrip().startswith("%%"):
        lines = lines[1:]
    tree = ast.parse("\n".join(lines))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return frozenset(names)


def _stored_png_output_count(cell: Mapping[str, Any]) -> int:
    """Count stored PNG display payloads in one notebook cell."""

    count = 0
    for output in cell.get("outputs", []):
        data = output.get("data", {}) if isinstance(output, Mapping) else {}
        if isinstance(data, Mapping) and data.get("image/png") is not None:
            count += 1
    return count


def _strip_existing_provenance_header(source: str) -> str:
    lines = source.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.rstrip("\r\n") == PROVENANCE_BEGIN), None)
    if start is None:
        return source
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].rstrip("\r\n") == PROVENANCE_END),
        None,
    )
    if end is None:
        raise AppendixPreparationError("Found an unterminated notebook provenance header.")
    del lines[start : end + 1]
    while start < len(lines) and lines[start].strip() == "":
        del lines[start]
    return "".join(lines)


def _direct_helper_imports(source: str) -> list[str]:
    lines = source.splitlines()
    if lines and lines[0].lstrip().startswith("%%"):
        lines = lines[1:]
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.rsplit(".", 1)[-1]
            if root in LOCAL_HELPER_MODULES:
                names = ", ".join(
                    f"{alias.name} as {alias.asname}" if alias.asname else alias.name
                    for alias in node.names
                )
                imports.append(f"{node.module}: {names}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.rsplit(".", 1)[-1]
                if root in LOCAL_HELPER_MODULES:
                    imported = f"module as {alias.asname}" if alias.asname else "module"
                    imports.append(f"{alias.name}: {imported}")
    return sorted(set(imports))


def _direct_helper_definitions(source: str) -> list[str]:
    """Return top-level helper functions defined by one executable cell."""

    lines = source.splitlines()
    if lines and lines[0].lstrip().startswith("%%"):
        lines = lines[1:]
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return []
    return sorted(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def _inline_magic_record(source: str) -> dict[str, str] | None:
    first_line = source.splitlines()[0].strip() if source.splitlines() else ""
    match = re.fullmatch(r"%%inline_module\s+(\S+)\s+([0-9a-fA-F]{64})", first_line)
    if not match:
        return None
    module_name, digest = match.groups()
    return {
        "helpers": f"defines inline helper module {module_name}; imports are internal to that source",
        "data_sources": f"embedded source corresponding to {module_name}.py; declared SHA-256 {digest.lower()}",
        "prior_results": "inline-module bootstrap cell",
        "execution_mode": "infrastructure",
        "produces": f"importable inline module {module_name}",
        "proof_status": "infrastructure with source-digest gate",
        "provenance_notes": "none",
    }


def _format_field(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        parts = [f"{key}: {_format_field(value[key])}" for key in sorted(value)]
        return "; ".join(parts) if parts else "none"
    if isinstance(value, (list, tuple, set, frozenset)):
        return "; ".join(_format_field(item) for item in value) if value else "none"
    return " ".join(str(value).split())


def _provenance_record_for_cell(
    cell: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    *,
    source_helpers_only: bool = False,
) -> tuple[dict[str, Any], bool]:
    cell_id = str(cell["id"])
    source = _source_text(cell)
    inline_record = _inline_magic_record(source)
    if inline_record is not None:
        return inline_record, False
    if cell_id in records:
        record = dict(records[cell_id])
        helper_fields = []
        if not source_helpers_only:
            for key in ("helpers", "helper_imports", "helper_functions", "helper_symbols"):
                value = record.get(key)
                if value not in (None, {}, []):
                    helper_fields.append(value)
        direct_imports = _direct_helper_imports(source)
        if direct_imports:
            helper_fields.append({"direct imports in this cell": direct_imports})
        direct_definitions = _direct_helper_definitions(source)
        if direct_definitions:
            helper_fields.append({"functions defined in this cell": direct_definitions})
        return {
            "helpers": helper_fields or "none declared by the provenance map",
            "data_sources": record.get(
                "data_sources",
                record.get("data_inputs", record.get("file_sources")),
            ),
            "prior_results": record.get(
                "prior_results",
                record.get("prior_result_sources"),
            ),
            "execution_mode": record.get("execution_mode"),
            "produces": record.get(
                "produces",
                record.get("results", record.get("products")),
            ),
            "proof_status": record.get("proof_status"),
            "provenance_notes": record.get("unresolved_provenance", []),
        }, False
    direct_imports = _direct_helper_imports(source)
    helpers = "; ".join(direct_imports) if direct_imports else "no direct local-helper import detected in this cell"
    return {
        "helpers": helpers,
        "data_sources": f"unresolved; no provenance-map record for stable cell id {cell_id}",
        "prior_results": f"unresolved; no provenance-map record for stable cell id {cell_id}",
        "execution_mode": "unclassified fallback",
        "produces": "retained variables and stored outputs from the executed source notebook",
        "proof_status": "requires provenance review before final deployment",
        "provenance_notes": "explicit provenance-map entry is missing",
    }, True


def _insert_provenance_header(source: str, record: Mapping[str, Any]) -> str:
    source = _strip_existing_provenance_header(source)
    header = "\n".join(
        [
            PROVENANCE_BEGIN,
            f"# helpers: {_format_field(record.get('helpers'))}",
            f"# data_sources: {_format_field(record.get('data_sources'))}",
            f"# prior_results: {_format_field(record.get('prior_results'))}",
            f"# execution_mode: {_format_field(record.get('execution_mode'))}",
            f"# produces: {_format_field(record.get('produces'))}",
            f"# proof_status: {_format_field(record.get('proof_status'))}",
            f"# provenance_notes: {_format_field(record.get('provenance_notes'))}",
            PROVENANCE_END,
        ]
    ) + "\n"
    lines = source.splitlines(keepends=True)
    if lines and lines[0].lstrip().startswith("%%"):
        return lines[0] + header + "".join(lines[1:])
    return header + source


def _make_inline_loader_provenance_aware(source: str) -> str:
    """Keep notebook provenance outside the digest-gated helper source."""

    if (
        'module_source = cell' in source
        and 'provenance_begin = "# notebook-provenance: begin\\n"' in source
    ):
        return source

    old = '''    actual_sha256 = _inline_hashlib.sha256(cell.encode("utf-8")).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Inline source digest mismatch for {module_name}: "
            f"expected {expected_sha256}, obtained {actual_sha256}."
        )

    short_name = module_name.rsplit(".", 1)[-1]
    module_path = _INLINE_HELPER_DIRECTORY / f"{short_name}.py"
    module_path.write_text(cell, encoding="utf-8")

    module = _inline_types.ModuleType(module_name)
    module.__file__ = str(module_path)
    module.__package__ = module_name.rpartition(".")[0]
    module.__source__ = cell
    module.__source_sha256__ = actual_sha256
'''
    new = '''    module_source = cell
    provenance_begin = "# notebook-provenance: begin\\n"
    provenance_end = "# notebook-provenance: end\\n"
    if module_source.startswith(provenance_begin):
        marker_index = module_source.find(provenance_end)
        if marker_index < 0:
            raise RuntimeError(
                f"Unterminated notebook provenance header for {module_name}."
            )
        module_source = module_source[marker_index + len(provenance_end):]

    actual_sha256 = _inline_hashlib.sha256(module_source.encode("utf-8")).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Inline source digest mismatch for {module_name}: "
            f"expected {expected_sha256}, obtained {actual_sha256}."
        )

    short_name = module_name.rsplit(".", 1)[-1]
    module_path = _INLINE_HELPER_DIRECTORY / f"{short_name}.py"
    module_path.write_text(module_source, encoding="utf-8")

    module = _inline_types.ModuleType(module_name)
    module.__file__ = str(module_path)
    module.__package__ = module_name.rpartition(".")[0]
    module.__source__ = module_source
    module.__source_sha256__ = actual_sha256
'''
    if old not in source:
        raise AppendixPreparationError(
            "The reviewed inline-module loader implementation has changed."
        )
    return source.replace(old, new, 1)


def _clean_metadata(metadata: Mapping[str, Any], *, notebook_level: bool) -> dict[str, Any]:
    remove = NOTEBOOK_METADATA_KEYS_TO_REMOVE if notebook_level else CELL_METADATA_KEYS_TO_REMOVE
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        lower = str(key).lower()
        if key in remove or lower.startswith("codex_update"):
            continue
        if not notebook_level and ("timestamp" in lower or lower == "execution"):
            continue
        if key == "tags" and isinstance(value, list):
            retained_tags = [
                tag for tag in value
                if not re.search(r"codex|update|transplant|widget", str(tag), re.IGNORECASE)
            ]
            if retained_tags:
                cleaned[key] = retained_tags
            continue
        cleaned[key] = deepcopy(value)
    return cleaned


def _hide_code_source(cell: dict[str, Any], classification: str) -> None:
    metadata = cell.setdefault("metadata", {})
    jupyter = metadata.get("jupyter")
    if not isinstance(jupyter, dict):
        jupyter = {}
    jupyter["source_hidden"] = True
    jupyter["outputs_hidden"] = False
    metadata["jupyter"] = jupyter
    metadata["collapsed"] = True
    metadata["thesis_appendix_classification"] = classification


def _restrict_phase1_to_symmetric_targets(source: str) -> str:
    """Remove dormant reference/model target routes from the curated Phase 1 cell."""

    source = source.replace(
        "build_reference_clusters_mpmath_raw = _worker_module.build_reference_clusters_mpmath_raw\n",
        "",
        1,
    )
    start_marker = "# If the selected map has no exact clusters, build a high-resolution numerical"
    end_marker = "raw_rows_list ,raw_results =run_N_sweep_mpmath_raw ("
    start = source.find(start_marker)
    end = source.find(end_marker)
    if start < 0 or end < 0 or end <= start:
        raise AppendixPreparationError(
            "The reviewed Phase 1 exact-target routing block has changed."
        )
    canonical = '''# The benchmark has an external exact spectrum.  Phase 1 therefore uses only
# the exact alpha and mu target families fixed by BLASCHKE_SPEC.
PHASE1_TARGET_MODE = "exact"
print("Phase 1 target mode:", PHASE1_TARGET_MODE)

'''
    source = source[:start] + canonical + source[end:]
    source = source.replace(
        "reference_clusters =SELECTED_REFERENCE_CLUSTERS ,",
        "reference_clusters =None ,",
        1,
    )
    old_interpretation = '''    "interpretation": (
        "exact target error" if PHASE1_TARGET_MODE == "exact" else
        "model-target diagnostic, not theorem-certified exact interval-spectrum error" if PHASE1_TARGET_MODE == "model" else
        "reference-cluster discrepancy, not exact spectral error"
    ),'''
    if old_interpretation not in source:
        raise AppendixPreparationError(
            "The reviewed Phase 1 interpretation row has changed."
        )
    source = source.replace(
        old_interpretation,
        '    "interpretation": "exact external Blaschke target error",',
        1,
    )
    return source


def _replace_function_region(
    source: str,
    *,
    start_marker: str,
    end_marker: str,
    replacement: str,
    label: str,
) -> str:
    start = source.find(start_marker)
    end = source.find(end_marker, start + len(start_marker))
    if start < 0 or end < 0 or end <= start:
        raise AppendixPreparationError(f"The reviewed {label} region has changed.")
    return source[:start] + replacement.rstrip() + "\n\n" + source[end:]


def _normalise_symmetric_execution_source(source: str) -> str:
    """Keep smoke/full execution separate from the notebook's fixed map identity."""

    source = source.replace(
        "SELECTED_MAP_HAS_BLASCHKE_CERTIFICATION",
        "CERTIFIER_FULL_EXECUTION",
    )
    source = source.replace(
        "# The notebook supplies map specifications at run time; the worker reconstructs\n"
        "# inverse branches tau_b and Perron--Frobenius weights phi_b inside each process.",
        "# The notebook supplies the locked blaschke_mu_0p3 branch specification; each\n"
        "# process reconstructs its inverse branches and Perron--Frobenius weights.",
    )
    source = source.replace(
        'print ("Worker map contract: notebook supplies serialised inverse branches and weights.")',
        'print ("Worker benchmark contract: blaschke_mu_0p3 inverse branches and weights.")',
    )
    source = source.replace(
        '# === Map-selected refined-cell execution for non-Blaschke maps ===\n'
        '# These routines make the copied refined notebook cells do real work for the\n'
        '# selected notebook-level map.  They do not claim theorem-level certification\n'
        '# unless supplied certified geometry exists.  For asymmetric_blaschke_degree2\n'
        '# they assemble the actual Legendre--Gauss transfer block, transport it to the\n'
        '# Chebyshev-packet Hardy gauge, and sample contour moats around the exact model\n'
        '# target packets.\n',
        '# === Symmetric-benchmark diagnostic execution ===\n'
        '# These routines support smoke-mode diagnostics for blaschke_mu_0p3.\n'
        '# The later complete-boundary and complete-contour stages remain theorem-facing.\n',
    )
    if "def transfer_lab_plot_title_label():" in source:
        replacement = '''def transfer_lab_plot_title_label():
    """Return the fixed benchmark title."""
    return r'Blaschke $\\mu=0.3$'


def transfer_lab_exact_or_reference_targets(max_targets=15, include_trivial=False):
    """Return the exact alpha and mu packets of the fixed Blaschke benchmark."""
    rows = []
    clusters = exact_clusters_for_map(
        map_spec=BLASCHKE_SPEC,
        max_power=getattr(cfg, 'max_power', 25),
        include_trivial=bool(include_trivial),
        full_multiplicity=getattr(cfg, 'full_multiplicity', True),
    )
    for cluster in clusters:
        name = str(cluster.get('name'))
        if name == '1' and not include_trivial:
            continue
        centre = transfer_lab_safe_complex(cluster.get('value'))
        if not np.isfinite(centre.real) or not np.isfinite(centre.imag):
            continue
        rows.append({
            'name': name,
            'family': str(cluster.get('family', 'target')),
            'power': int(cluster.get('power', 0)),
            'multiplicity': int(cluster.get('multiplicity', 1)),
            'centre': centre,
            'target_mode': 'exact',
        })
    return pd.DataFrame(rows[:int(max_targets)])
'''
        source = _replace_function_region(
            source,
            start_marker="def transfer_lab_plot_title_label():",
            end_marker="def transfer_lab_contour_radii(target_df):",
            replacement=replacement,
            label="fixed-title and exact-target",
        )
    source = source.replace(
        "# Map-generic certification bridge",
        "# Symmetric-benchmark sampled certification bridge",
    )
    source = source.replace(
        "# Concrete map formulae remain in Cell 8.  These helpers consume only the\n"
        "# active map's sampled geometry, raw/reference packets, and finite matrices.",
        "# Cell 8 fixes blaschke_mu_0p3. These helpers consume its sampled geometry,\n"
        "# exact target packets and finite transfer matrices.",
    )
    source = source.replace("raw/reference", "exact-target")
    source = source.replace("exact/reference", "exact")
    source = source.replace("; build_reference_clusters_mpmath_raw", "")
    source = re.sub(
        r"; DATA_DIR/blaschke_mu_0p3_reference_clusters_N\{REFERENCE_N\}_M\{REFERENCE_M\}\.csv, only if a reference build is requested; DATA_DIR/blaschke_mu_0p3_reference_build_metadata\.csv, only if a reference build is requested",
        "",
        source,
    )
    source = source.replace("; prior_results: SELECTED_REFERENCE_CLUSTERS;", "; prior_results:")
    source = source.replace("unknown_map", "blaschke_mu_0p3")
    source = source.replace('"unknown"', 'CERTIFIER_MAP_LABEL')
    source = source.replace("'unknown'", "CERTIFIER_MAP_LABEL")
    return source


def _simplify_markdown(cell: dict[str, Any]) -> None:
    cell_id = str(cell["id"])
    if cell_id in MARKDOWN_REPLACEMENTS:
        _store_source(cell, MARKDOWN_REPLACEMENTS[cell_id])
        return
    if cell_id == "math-cell-8-blaschke-data":
        source = _source_text(cell)
        marker = "### Blaschke deformation: known exact spectrum"
        stop = "### Asymmetric degree-two Blaschke product"
        if marker not in source or stop not in source:
            raise AppendixPreparationError("The reviewed symmetric benchmark Markdown boundaries have changed.")
        symmetric_section = source.split(marker, 1)[1].split(stop, 1)[0].strip()
        replacement = (
            "## Symmetric Blaschke benchmark and exact target families\n\n"
            "The worker module is map-independent, but this appendix fixes its branch data "
            "to the symmetric interval Blaschke deformation with \\(\\mu=0.3\\).\n\n"
            + marker + "\n\n" + symmetric_section + "\n"
        )
        _store_source(cell, replacement)
        return
    if cell_id in {"fm24am", "resp24am"}:
        source = _source_text(cell)
        source = re.sub(r"^<div[^>]*>\s*", "", source, flags=re.IGNORECASE)
        source = re.sub(r"\s*</div>\s*$", "\n", source, flags=re.IGNORECASE)
        _store_source(cell, source)


def _apply_display_numbering(cells: Sequence[dict[str, Any]]) -> None:
    """Restore the canonical M/N labels without renumbering inherited cells."""

    code_label = re.compile(r"#(?:\d+[A-Z]*N|\s*Cell\s+\d+[A-Z]*)\s*$")
    markdown_label = re.compile(r"\d+[A-Z]*M\s*$")
    for index, cell in enumerate(cells):
        cell_id = str(cell.get("id", ""))
        label = CUSTOM_DISPLAY_LABELS.get(cell_id)
        if label is None:
            inherited_index = index - sum(
                str(prior.get("id", "")) in NON_NUMBER_ADVANCING_DISPLAY_IDS
                for prior in cells[:index]
            )
            number = (
                inherited_index + 1
                if inherited_index < 35
                else inherited_index - 9
            )
            label = f"{number}M" if cell.get("cell_type") == "markdown" else f"#{number}N"

        source = _source_text(cell)
        lines = source.splitlines(keepends=True)
        filtered: list[str] = []
        for line_index, line in enumerate(lines):
            stripped = line.rstrip("\r\n")
            if cell.get("cell_type") == "code" and code_label.fullmatch(stripped):
                continue
            if (
                cell.get("cell_type") == "markdown"
                and line_index == 0
                and markdown_label.fullmatch(stripped)
            ):
                continue
            filtered.append(line)

        if cell.get("cell_type") == "markdown":
            while filtered and not filtered[0].strip():
                filtered.pop(0)

        label_line = label + "\n"
        if (
            cell.get("cell_type") == "code"
            and filtered
            and filtered[0].lstrip().startswith("%%inline_module")
        ):
            numbered = filtered[0] + label_line + "".join(filtered[1:])
        elif cell.get("cell_type") == "markdown":
            numbered = label_line + "\n" + "".join(filtered)
        else:
            numbered = label_line + "".join(filtered)
        _store_source(cell, numbered)


def transform_notebook(
    source_notebook: Mapping[str, Any],
    provenance_records: Mapping[str, Mapping[str, Any]],
    plotting_replacements: Mapping[str, str],
    *,
    require_stored_outputs: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_source_notebook(source_notebook)
    before_hashes = code_output_hashes(source_notebook)
    before_counts = retained_execution_counts(source_notebook)
    transformed = deepcopy(source_notebook)
    transformed["metadata"] = _clean_metadata(transformed.get("metadata", {}), notebook_level=True)
    removed: dict[str, list[str]] = {reason: [] for reason in REMOVAL_GROUPS}
    reason_by_id = {cell_id: reason for reason, ids in REMOVAL_GROUPS.items() for cell_id in ids}
    retained_cells: list[dict[str, Any]] = []
    fallback_ids: list[str] = []
    hidden_ids: list[str] = []

    for cell in transformed.get("cells", []):
        cell_id = str(cell["id"])
        if cell_id in REMOVE_CELL_IDS:
            removed[reason_by_id[cell_id]].append(cell_id)
            continue
        cell["metadata"] = _clean_metadata(cell.get("metadata", {}), notebook_level=False)
        if cell.get("cell_type") == "markdown":
            _simplify_markdown(cell)
        elif cell.get("cell_type") == "code":
            if cell_id in STATIC_CODE_REPLACEMENTS and cell_id in plotting_replacements:
                raise AppendixPreparationError(
                    f"Stable cell id {cell_id!r} has conflicting static and plotting replacements."
                )
            if cell_id in STATIC_CODE_REPLACEMENTS:
                _store_source(cell, STATIC_CODE_REPLACEMENTS[cell_id])
            if cell_id in plotting_replacements:
                _store_source(cell, plotting_replacements[cell_id])
            if cell_id == "a5385979":
                _store_source(
                    cell,
                    _restrict_phase1_to_symmetric_targets(_source_text(cell)),
                )
            is_inline_helper_source = cell_id.startswith("inline-helper-source-")
            if not is_inline_helper_source and cell_id != "inline-helper-bootstrap-code":
                _store_source(
                    cell,
                    _normalise_symmetric_execution_source(_source_text(cell)),
                )
            if cell_id == "inline-helper-bootstrap-code":
                _store_source(
                    cell,
                    _make_inline_loader_provenance_aware(_source_text(cell)),
                )
            if cell_id == "cc338a81":
                _store_source(
                    cell,
                    _complete_phase1_plot_exports(_source_text(cell)),
                )
            record, used_fallback = _provenance_record_for_cell(
                cell,
                provenance_records,
                source_helpers_only=cell_id in plotting_replacements,
            )
            _store_source(cell, _insert_provenance_header(_source_text(cell), record))
            if not is_inline_helper_source and cell_id != "inline-helper-bootstrap-code":
                _store_source(
                    cell,
                    _normalise_symmetric_execution_source(_source_text(cell)),
                )
            if used_fallback:
                fallback_ids.append(cell_id)
            if cell_id in INFRASTRUCTURE_CELL_IDS:
                _hide_code_source(cell, "infrastructure")
                hidden_ids.append(cell_id)
            elif cell_id in DIAGNOSTIC_PLOTTING_CELL_IDS:
                _hide_code_source(cell, "diagnostic-plotting")
                hidden_ids.append(cell_id)
        retained_cells.append(cell)

    retained_ids = {str(cell.get("id")) for cell in retained_cells}
    for cell_id, source in TERMINAL_NUMBERED_CELLS:
        if cell_id not in retained_ids:
            retained_cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": cell_id,
                    "metadata": {},
                    "outputs": [],
                    "source": source,
                }
            )
    _apply_display_numbering(retained_cells)
    transformed["cells"] = retained_cells
    validation = validate_transformed_notebook(
        source_notebook,
        transformed,
        expected_output_hashes=before_hashes,
        expected_execution_counts=before_counts,
        require_stored_outputs=require_stored_outputs,
    )
    report = {
        "source_cell_count": len(source_notebook.get("cells", [])),
        "retained_cell_count": len(retained_cells),
        "removed_cell_count": len(REMOVE_CELL_IDS),
        "removed_by_reason": {key: sorted(value) for key, value in removed.items()},
        "retained_code_cell_count": sum(cell.get("cell_type") == "code" for cell in retained_cells),
        "provenance_fallback_cell_ids": sorted(fallback_ids),
        "hidden_code_cell_ids": sorted(hidden_ids),
        "static_code_replacement_count": len(STATIC_CODE_REPLACEMENTS),
        "plotting_replacement_count": len(plotting_replacements),
        "validation": validation,
    }
    return transformed, report


def validate_transformed_notebook(
    source_notebook: Mapping[str, Any],
    transformed: Mapping[str, Any],
    *,
    expected_output_hashes: Mapping[str, str] | None = None,
    expected_execution_counts: Mapping[str, Any] | None = None,
    require_stored_outputs: bool = True,
) -> dict[str, Any]:
    source_index = _index_cells(source_notebook)
    transformed_index = _index_cells(transformed)
    remaining_removed = sorted(REMOVE_CELL_IDS.intersection(transformed_index))
    if remaining_removed:
        raise AppendixPreparationError(f"Removed stable ids remain in transformed notebook: {remaining_removed}.")
    expected_retained_ids = (
        set(source_index).difference(REMOVE_CELL_IDS)
        | set(TERMINAL_NUMBERED_CELL_IDS)
    )
    if set(transformed_index) != expected_retained_ids:
        missing = sorted(expected_retained_ids.difference(transformed_index))
        extra = sorted(set(transformed_index).difference(expected_retained_ids))
        raise AppendixPreparationError(f"Retained stable-id mismatch: missing={missing}, extra={extra}.")
    retained_code_count = sum(
        cell.get("cell_type") == "code" for cell in transformed.get("cells", [])
    )
    if len(transformed.get("cells", [])) != EXPECTED_RETAINED_CELL_COUNT:
        raise AppendixPreparationError(
            "Retained cell count drifted: "
            f"expected {EXPECTED_RETAINED_CELL_COUNT}, "
            f"found {len(transformed.get('cells', []))}."
        )
    if retained_code_count != EXPECTED_RETAINED_CODE_CELL_COUNT:
        raise AppendixPreparationError(
            "Retained code-cell count drifted: "
            f"expected {EXPECTED_RETAINED_CODE_CELL_COUNT}, found {retained_code_count}."
        )
    missing_context_ids = sorted(
        EXPECTED_RESTORED_MARKDOWN_CELL_IDS.difference(transformed_index)
    )
    if missing_context_ids:
        raise AppendixPreparationError(
            "Concise Markdown context is missing for restored visual cells: "
            f"{missing_context_ids}."
        )
    required_symmetric_replacements = {
        "33580cd0", "worker-module-maths-map", "e5cdca61", "a5385979"
    }
    missing_symmetric_replacements = sorted(
        required_symmetric_replacements.difference(transformed_index)
    )
    if missing_symmetric_replacements:
        raise AppendixPreparationError(
            "The symmetric-only notebook replacements are missing: "
            f"{missing_symmetric_replacements}."
        )
    update_cells = [
        cell_id for cell_id, cell in transformed_index.items()
        if UPDATE_HEADING_RE.search(_source_text(cell))
    ]
    if update_cells:
        raise AppendixPreparationError(f"Notebook update prose remains in stable cells: {sorted(update_cells)}.")
    forbidden_metadata = NOTEBOOK_METADATA_KEYS_TO_REMOVE.intersection(transformed.get("metadata", {}))
    if forbidden_metadata:
        raise AppendixPreparationError(f"Forbidden notebook metadata remains: {sorted(forbidden_metadata)}.")

    forbidden_source_phrases = (
        "toy example",
        "toy illustration",
        "asymmetric blaschke",
        "asymmetric_blaschke",
        "asymmetric degree-two",
        "diagnostic_asymmetric_blaschke_geometry",
        "polynomial_warped",
        "arctan_warped",
        "tanh_warped",
        "mobius_warped",
        "sinusoidal_inverse",
        "explicit_polynomial",
        "tent_orientation",
        "warped_tripling",
        "warped_quadrupling",
        "selected_map_has_blaschke_certification",
    )
    forbidden_source_hits: dict[str, list[str]] = {}
    for cell_id, cell in transformed_index.items():
        source_lower = _source_text(cell).lower()
        hits = [phrase for phrase in forbidden_source_phrases if phrase in source_lower]
        if hits:
            forbidden_source_hits[cell_id] = hits
    if forbidden_source_hits:
        raise AppendixPreparationError(
            "Excluded toy or asymmetric-benchmark source remains in the transformed notebook: "
            f"{forbidden_source_hits}."
        )

    missing_headers: list[str] = []
    misplaced_magic_headers: list[str] = []
    timestamp_metadata: list[str] = []
    plotting_residue: dict[str, list[str]] = {}
    for cell_id, cell in transformed_index.items():
        metadata = cell.get("metadata", {})
        if any(
            key in CELL_METADATA_KEYS_TO_REMOVE
            or "timestamp" in str(key).lower()
            or str(key).lower() == "execution"
            for key in metadata
        ):
            timestamp_metadata.append(cell_id)
        if cell.get("cell_type") != "code":
            continue
        source = _source_text(cell)
        if cell_id in TERMINAL_NUMBERED_CELL_IDS:
            expected_terminal_source = dict(TERMINAL_NUMBERED_CELLS)[cell_id]
            if source != expected_terminal_source:
                raise AppendixPreparationError(
                    f"Terminal numbered cell {cell_id!r} changed unexpectedly."
                )
            if cell_id == "128b5369":
                continue
        residue = _direct_plotting_residue(source)
        if residue:
            plotting_residue[cell_id] = residue
        if PROVENANCE_BEGIN not in source or PROVENANCE_END not in source:
            missing_headers.append(cell_id)
        lines = source.splitlines()
        if lines and lines[0].lstrip().startswith("%%"):
            begin_index = (
                2
                if len(lines) >= 2
                and re.fullmatch(r"#\d+[A-Z]*N", lines[1].strip())
                else 1
            )
            if len(lines) <= begin_index or lines[begin_index].strip() != PROVENANCE_BEGIN:
                misplaced_magic_headers.append(cell_id)
        else:
            begin_index = (
                1
                if lines and re.fullmatch(r"#\d+[A-Z]*N", lines[0].strip())
                else 0
            )
            if len(lines) <= begin_index or lines[begin_index].strip() != PROVENANCE_BEGIN:
                missing_headers.append(cell_id)
    if timestamp_metadata:
        raise AppendixPreparationError(f"Execution/update timestamp metadata remains: {sorted(timestamp_metadata)}.")
    if missing_headers:
        raise AppendixPreparationError(f"Code cells lack a leading provenance header: {sorted(set(missing_headers))}.")
    if misplaced_magic_headers:
        raise AppendixPreparationError(
            "Inline magic headers are not immediately below the magic line: "
            f"{sorted(misplaced_magic_headers)}."
        )
    if plotting_residue:
        raise AppendixPreparationError(
            "Direct plotting code remains in the transformed notebook instead of plotting.py: "
            f"{plotting_residue}."
        )

    actual_visual_output_counts = {
        cell_id: count
        for cell_id, cell in transformed_index.items()
        if (count := _stored_png_output_count(cell)) > 0
    }
    if require_stored_outputs:
        if actual_visual_output_counts != dict(EXPECTED_VISUAL_OUTPUT_COUNTS):
            raise AppendixPreparationError(
                "The protected 33-PNG visual inventory was not retained: "
                f"expected {dict(EXPECTED_VISUAL_OUTPUT_COUNTS)}, "
                f"found {actual_visual_output_counts}."
            )
        if sum(EXPECTED_VISUAL_OUTPUT_COUNTS.values()) != 33:
            raise AppendixPreparationError(
                "The reviewed visual-output contract no longer totals 33 PNGs."
            )
    elif actual_visual_output_counts:
        raise AppendixPreparationError(
            "An output-free builder counterpart unexpectedly contains stored PNG output."
        )

    helper_call_mismatches: dict[str, list[str]] = {}
    for cell_id, expected_calls in EXPECTED_RESTORED_PLOTTING_HELPER_CALLS.items():
        actual_calls = _called_function_names(_source_text(transformed_index[cell_id]))
        missing_calls = sorted(expected_calls.difference(actual_calls))
        if missing_calls:
            helper_call_mismatches[cell_id] = missing_calls
    if helper_call_mismatches:
        raise AppendixPreparationError(
            "Restored visual cells do not call every reviewed plotting.py entry point: "
            f"{helper_call_mismatches}."
        )
    phase1_source = _source_text(transformed_index["cc338a81"])
    missing_displacement_stems = sorted(
        stem for stem in EXPECTED_PHASE1_DISPLACEMENT_STEMS if stem not in phase1_source
    )
    if missing_displacement_stems:
        raise AppendixPreparationError(
            "Cell 15 no longer regenerates both reviewed displacement figures: "
            f"{missing_displacement_stems}."
        )

    plotting_tree = ast.parse((HERE / "plotting.py").read_text(encoding="utf-8"))
    available_plotting_helpers = {
        node.name
        for node in plotting_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required_plotting_helpers = set().union(
        *EXPECTED_RESTORED_PLOTTING_HELPER_CALLS.values()
    )
    missing_plotting_helper_definitions = sorted(
        required_plotting_helpers.difference(available_plotting_helpers)
    )

    actual_hashes = code_output_hashes(transformed)
    actual_counts = retained_execution_counts(transformed)
    if expected_output_hashes is None:
        expected_output_hashes = code_output_hashes(source_notebook)
    if expected_execution_counts is None:
        expected_execution_counts = retained_execution_counts(source_notebook)
    actual_hashes = {
        key: value for key, value in actual_hashes.items()
        if key not in TERMINAL_NUMBERED_CELL_IDS
    }
    actual_counts = {
        key: value for key, value in actual_counts.items()
        if key not in TERMINAL_NUMBERED_CELL_IDS
    }
    if actual_hashes != dict(expected_output_hashes):
        changed = sorted(
            cell_id for cell_id in set(actual_hashes).union(expected_output_hashes)
            if actual_hashes.get(cell_id) != expected_output_hashes.get(cell_id)
        )
        raise AppendixPreparationError(f"Stored output hashes changed for retained cells: {changed}.")
    if actual_counts != dict(expected_execution_counts):
        changed = sorted(
            cell_id for cell_id in set(actual_counts).union(expected_execution_counts)
            if actual_counts.get(cell_id) != expected_execution_counts.get(cell_id)
        )
        raise AppendixPreparationError(f"Execution counts changed for retained cells: {changed}.")
    return {
        "stable_cell_ids_valid": True,
        "notebook_update_cells_absent": True,
        "toy_and_asymmetric_benchmark_source_absent": True,
        "forbidden_metadata_absent": True,
        "all_retained_code_cells_have_provenance_headers": True,
        "inline_magic_header_placement_valid": True,
        "direct_plotting_code_absent": True,
        "retained_cell_count": EXPECTED_RETAINED_CELL_COUNT,
        "retained_code_cell_count": EXPECTED_RETAINED_CODE_CELL_COUNT,
        "retained_visual_cell_count": len(actual_visual_output_counts),
        "retained_png_output_count": sum(actual_visual_output_counts.values()),
        "restored_markdown_context_count": len(EXPECTED_RESTORED_MARKDOWN_CELL_IDS),
        "phase1_displacement_pair_valid": True,
        "restored_plotting_helper_contract_valid": True,
        "missing_plotting_helper_definitions": missing_plotting_helper_definitions,
        "pending_plotting_helper_signatures": {
            name: PLOTTING_HELPER_SIGNATURE_CONTRACT[name]
            for name in missing_plotting_helper_definitions
            if name in PLOTTING_HELPER_SIGNATURE_CONTRACT
        },
        "retained_output_hashes_unchanged": True,
        "retained_execution_counts_unchanged": True,
        "retained_output_hash_set_digest": _canonical_digest(actual_hashes),
    }


def matching_backup(notebook_path: Path, explicit_backup: Path | None = None) -> tuple[Path | None, str]:
    source_digest = file_sha256(notebook_path)
    if explicit_backup is not None:
        candidates = [explicit_backup]
    else:
        candidates = sorted(
            path for path in notebook_path.parent.glob(f"{notebook_path.stem}_backup*.ipynb")
            if path.resolve() != notebook_path.resolve()
        )
    for candidate in candidates:
        if candidate.is_file() and candidate.resolve() != notebook_path.resolve():
            if file_sha256(candidate) == source_digest:
                return candidate, source_digest
    return None, source_digest


def _atomic_write_json(path: Path, payload: Mapping[str, Any], original_mode: int) -> None:
    serialised = json.dumps(payload, ensure_ascii=False, indent=1) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialised)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, stat.S_IMODE(original_mode))
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _read_notebook(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AppendixPreparationError(f"Cannot read notebook {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("cells"), list):
        raise AppendixPreparationError(f"File is not a valid notebook object: {path}")
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        type=Path,
        required=True,
        help="Explicit 187-cell full inline-helper and source-producer notebook to curate.",
    )
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument(
        "--plotting-replacements",
        type=Path,
        default=DEFAULT_PLOTTING_REPLACEMENTS,
        help="Reviewed complete code-cell sources which call plotting.py.",
    )
    parser.add_argument(
        "--backup", type=Path, default=None,
        help="Explicit separate backup; it must be byte-identical to the input notebook.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true",
        help="Transform and validate in memory without writing; this is the default.",
    )
    mode.add_argument(
        "--apply", action="store_true",
        help="Atomically replace the notebook after verifying a byte-identical backup.",
    )
    parser.add_argument(
        "--require-explicit-provenance", action="store_true",
        help="Fail if any retained code cell lacks an explicit provenance-map record.",
    )
    parser.add_argument(
        "--require-complete-provenance", action="store_true",
        help=(
            "Fail if any retained code cell lacks a provenance-map record or retains "
            "an unresolved upstream provenance limitation."
        ),
    )
    parser.add_argument(
        "--report-json", type=Path, default=None,
        help="Optional report path. In dry-run mode no report file is written.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    notebook_path = args.notebook.resolve()
    provenance_path = args.provenance.resolve()
    plotting_replacements_path = args.plotting_replacements.resolve()
    if not notebook_path.is_file():
        raise AppendixPreparationError(f"Notebook does not exist: {notebook_path}")
    source_digest = file_sha256(notebook_path)
    if EXPECTED_SOURCE_SHA256 and source_digest != EXPECTED_SOURCE_SHA256:
        raise AppendixPreparationError(
            "The input notebook does not match the reviewed source snapshot: "
            f"expected {EXPECTED_SOURCE_SHA256}, found {source_digest}."
        )
    source = _read_notebook(notebook_path)
    provenance_records, provenance_file_present = load_provenance_records(
        provenance_path,
        source,
    )
    plotting_replacements, plotting_replacements_present = load_plotting_replacements(
        plotting_replacements_path,
        source,
    )
    validate_plotting_replacement_coverage(plotting_replacements)
    require_stored_outputs = any(
        _stored_png_output_count(cell) > 0
        for cell in source.get("cells", [])
    )
    transformed, report = transform_notebook(
        source,
        provenance_records,
        plotting_replacements,
        require_stored_outputs=require_stored_outputs,
    )
    backup_path, source_digest = matching_backup(
        notebook_path, args.backup.resolve() if args.backup is not None else None
    )
    report.update(
        {
            "mode": "apply" if args.apply else "dry-run",
            "notebook": str(notebook_path),
            "source_sha256": source_digest,
            "matching_backup": str(backup_path) if backup_path else None,
            "provenance_file": str(provenance_path),
            "provenance_file_present": provenance_file_present,
            "plotting_replacements_file": str(plotting_replacements_path),
            "plotting_replacements_file_present": plotting_replacements_present,
        }
    )
    fallback_ids = report["provenance_fallback_cell_ids"]
    retained_code_ids = {
        str(cell["id"])
        for cell in transformed.get("cells", [])
        if cell.get("cell_type") == "code"
    }
    unresolved_provenance_ids = sorted(
        cell_id
        for cell_id in retained_code_ids
        if provenance_records.get(cell_id, {}).get("unresolved_provenance")
    )
    report["unresolved_provenance_cell_ids"] = unresolved_provenance_ids
    if (args.require_explicit_provenance or args.require_complete_provenance) and fallback_ids:
        raise AppendixPreparationError(
            "Explicit provenance was required, but safe fallback headers are needed for "
            f"{len(fallback_ids)} retained code cells."
        )
    if args.require_complete_provenance and unresolved_provenance_ids:
        raise AppendixPreparationError(
            "Clean-room-complete provenance was required, but unresolved upstream "
            f"limitations remain for stable cell ids {unresolved_provenance_ids}."
        )
    if args.apply:
        if backup_path is None:
            raise AppendixPreparationError(
                "Apply mode requires a separate byte-identical backup. Create one first or pass its path with --backup."
            )
        original_mode = notebook_path.stat().st_mode
        _atomic_write_json(notebook_path, transformed, original_mode)
        written = _read_notebook(notebook_path)
        validate_transformed_notebook(source, written)
        report["written_sha256"] = file_sha256(notebook_path)
        if args.report_json is not None:
            args.report_json.resolve().write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    elif args.report_json is not None:
        report["report_note"] = "Dry-run mode does not write the requested report path."
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except AppendixPreparationError as exc:
        print(f"appendix preparation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
