0M

# Reproducibility dependency map

This diagram records the effective dependency graph of the executed notebook. Cell labels are
the stable display labels used below. Solid arrows are theorem-facing data dependencies, dashed
amber arrows are diagnostic-only dependencies, and dotted grey arrows are build or provenance
dependencies. Plotting products are shown separately because they present results but do not
enter a theorem gate. The final builder contract is 141 cells, including 68 code cells, 18
digest-checked inline helper modules and the visible terminal auditor Markdown Cell 110M.

**Chapter-mathematics mapping boundary.** The source-controlled contract
`Numerics/numerical_certification_transfer_markdown.toml` maps the mathematics established and
applied under `chap:numerical-certification-transfer` to all 52 stable Markdown owners and, through
their contiguous groups, all 68 code cells. The builder appends an operation-specific mathematical
bridge, exact chapter labels and an evidence-status statement without changing any pre-existing
cell ID or type. It appends Cell 110M after Cell 109N so the final certificate explanation is
visible below the stored output. The contract also keeps sampled diagnostics, presentation cells and theorem-facing
producers distinct. Its regression rejects an unmapped code cell, a duplicated owner, a missing
chapter label or an evidence-class drift.

![Effective dependency graph for the Blaschke deformation thesis certifier](attachment:blaschke-deformation-dependency-map.svg)

**Principal persisted paths.** Phase outputs are stored under
`Numerics/outputs/blaschke_deformation_certifier`. The theorem-facing chain ends in
`data/branch_image_balanced_response_prefactor_candidate_row_N600_M610.csv`,
`data/blaschke_deformation_balanced_hardy_matrix_gate.csv`,
`data/blaschke_deformation_24_target_N600_M610_spectral_certificate.csv` and the matching JSON
report. The validated Schur enclosure, Schur attempts, Laurent mode bounds and Laurent witness
reconstruction are stored beside that final certificate. Files below `figures` and the retained
first-fifteen surface datasets are explanatory diagnostics and are not theorem inputs.

**Diagnostic source-producer boundary.** Inline helper Cell 30AN exposes
`blaschke_deformation_phase1_diagnostics.py`, and producer Cell 30BN recreates the retained Phase 1
sweeps, eigencloud matches and heatmap inputs immediately before displayed Cell 31N. Inline helper
Cell 94AN exposes `blaschke_deformation_sampled_schur_diagnostics.py`, and producer Cell 94BN
recreates the six retained `M=N+6` sampled Schur rows upstream of displayed Cell 95N. Both
producer reports record `legacy_seed_dependency: false`, `diagnostic_only: true` and
`theorem_gate: false`. Under the map legend, both producer branches have diagnostic-only (dashed
amber) semantics: the Phase 1 products support empirical comparison, and the sampled Schur rows
support the diagnostic ladder, but neither branch enters a count, moat, small-gain or Riesz-rank
theorem gate. No new solid-arrow theorem dependency is introduced; the theorem-facing Phase 2 and
Phase 4 chain is unchanged.

**Clean-room diagnostic rule.** A generated diagnostic is recreated from the selected map,
explicit schedules and displayed mathematical routines rather than accepted from its destination
CSV, Parquet, JSON, report or figure as a seed. Explicit upstream frames may feed downstream
diagnostic audits, and source-keyed transient row-block checkpoints may resume interrupted work;
those are controlled dependencies, not theorem evidence. Digest equality and regression checks
establish provenance or retained numerical behaviour only.

**Phase 2 completion boundary.** Cells 24A, 24B and 24C are thin orchestration cells. Their
mathematical implementations are the visible standalone helpers
`blaschke_deformation_phase2_finite_m.py`,
`blaschke_deformation_phase2_resolved_response.py` and
`blaschke_deformation_phase2_final_aggregation.py`. Each helper receives the same explicit output
directory and communicates with the next stage only through the named CSV and JSON artefacts shown
in the diagram. No private Cell 24B state is required by Cell 24C.

**Archive boundary.** Packaging is outside the notebook dependency graph and is always a separate
clean-worktree operation after the deployment evidence has been committed. For NUMERICS_1 that
operation has completed; its external release manifest and checksum, rather than this notebook,
authenticate the published archive. A replay from an extracted archive must first run
`prepare_blaschke_source_only_replay.py`, whose archive-specific inventory removes the generated
notebook counterparts and `Numerics/outputs` evidence while hash-preserving all immutable source
and environment records. The final Phase 2 row carries every fresh input-tail and response
component gate; contour loading fails closed if any component gate is missing or false.
