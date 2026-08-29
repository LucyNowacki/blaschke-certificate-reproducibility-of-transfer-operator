# Final Deployment

This folder is the self-contained deployment of the symmetric Blaschke
deformation experiment `blaschke_mu_0p3`.

The principal executed record is
`Numerics/blaschke_deformation_certifier_thesis_math.ipynb`.  The builder locks
the final counterpart to 140 cells, including 68 code cells.  Its eighteen
retained helper modules are included inline beside the phases that use them.
They cover the theorem-facing Phase 2 and Phase 4 calculations together with
source-only Phase 1 diagnostics, historical comparisons, process-based moat-
surface sampling, historical Phase 4 diagnostics, sampled Schur diagnostics
and diagnostic-audit reconstruction.  The Phase 1 and sampled Schur producer
calls appear upstream of displayed Cells 31N and 95N, respectively.
Byte-identical standalone copies remain available for direct inspection,
process workers, digest validation and packaging.

This deployment directory is the authoritative certification record.  Copies
of similarly named builders, helpers or notebooks elsewhere in the working
repository are development or historical artefacts and are not inputs to this
builder chain.  In particular, deployment terminology and helper provenance
must be assessed from the files under this directory.

The exact symmetric interval target families are external benchmark data from
Slipantschuk--Bandtlow--Just 2013, equation 21, and are recorded again for the
same map in Akindji--Slipantschuk--Bandtlow--Just 2026, Section 3.2.  They
propose and identify the target windows.  The finite counts and complete-circle
moats in this deployment are certified independently from the validated Schur
and Laurent calculations.

The reproducible builder chain is:

1. locked output-free template
   `Numerics/blaschke_deformation_certifier_template.ipynb`;
2. output-free source notebook
   `Numerics/blaschke_deformation_certifier.ipynb`;
3. inline thesis-mathematics counterpart
   `Numerics/blaschke_deformation_certifier_thesis_math.ipynb`.

The chapter-to-notebook mathematics is a source-controlled part of that third
stage. `Numerics/numerical_certification_transfer_markdown.toml` binds all 52
stable Markdown owners, covering all 68 code cells, to exact labels under
`chap:numerical-certification-transfer`, an operation-specific mathematical
explanation and an explicit evidence class. The builder applies this map after
curation, preserving the locked 140-cell ID/type signature and all code-cell
sources. Focused regressions reject missing chapter grounding, ownership drift
and sampled-to-certified claim upgrades.

The recorded execution uses the Conda environment `lucy`; the commands below
assume that environment has been activated.

The clean-room archive carries both `conda-explicit-lock.txt` and the exact
pip-only pins in `pip-requirements-lock.txt`.  After recreating the Conda
prefix, bootstrap the bundled installer with `python -m ensurepip --upgrade`,
then install the exact pip, mpmath, python-flint and threadpoolctl wheels with
`python -m pip install --no-deps --only-binary=:all: --requirement
pip-requirements-lock.txt` before running the builders.

An extracted release archive is not source-only merely because its notebook
builders emit output-free notebooks: the archive also carries the published
numerical evidence for comparison.  Before any producer is run, prepare the
exact extracted bundle with:

`python -B Numerics/prepare_blaschke_source_only_replay.py --bundle-root .`

The command accepts only the exact
`blaschke_deformation_certifier_reproducibility` bundle layout, verifies the
archive-specific `source-only-replay-inventory.json` and all immutable hashes,
rejects traversal, symlinks, broad roots and inventory drift, then removes only
the listed generated notebook counterparts and files below `Numerics/outputs`.
For this benchmark there are no immutable external numerical input files: the
map, constants and exact target families are encoded in source.  The plan's
named upstream artifacts are outputs of earlier producers in the same replay.
The preparation receipt is `source-only-replay-preparation.json`.

Rebuild the first two executable stages from the deployment root with:

`python -B Numerics/build_blaschke_deformation_certifier.py`

`python -B Numerics/build_blaschke_deformation_thesis_math_notebook.py`

The builders reject a mismatch between a standalone helper, its displayed
SHA-256 digest and its inline source.  The notebook bootstrap repeats the digest
check before executing each inline module.  Rebuilding the counterpart produces
an output-free notebook; execute it afterwards to reconstruct the complete
Phase 2 geometry, transport and raw matrix inputs before refreshing the
downstream complete-boundary and complete-contour certificates.

The clean-room rule also applies to generated diagnostics: each destination
table, report or figure is recreated from the selected map, explicit schedules
and displayed mathematical producers rather than accepted as a seed.  Explicit
upstream frames may feed downstream diagnostic audits, and source-keyed
transient row-block checkpoints may resume interrupted work, but neither route
promotes a sampled diagnostic into theorem evidence.

Immediately before displayed Cell 31N,
`Numerics/blaschke_deformation_phase1_diagnostics.py` and its producer cell
rebuild the retained empirical Legendre--Gauss sweeps, eigencloud matches and
near-square heatmap inputs from the selected map specification, the raw mpmath
transfer-block worker and explicit schedules.  The producer never reads its
destination CSV or Parquet files as inputs and records
`legacy_seed_dependency: false`.  These outputs are diagnostic-only and do not
enter the theorem-facing single-space perturbation or contour-certification
gates.

Phase 2 is reconstructed from displayed source: a 36-configuration,
complete-boundary Arb geometry scan selects the deployed row by the historical
tail-floor criterion; a 24-process parity-block calculation certifies the
finite Chebyshev-gauge transport; and Arb assembly rebuilds the raw pure-
`r`-scaled matrix certificate.  The resulting outputs are hash-bound in
`reports/phase2_clean_room_rebuild_manifest.json` before the notebook applies
the safe finite-`M`, resolved-response and unresolved-input stages.

The two retained historical-design comparison rows are also regenerated from
displayed source by
`Numerics/blaschke_deformation_historical_comparisons.py`.  They use the fixed
historical design `rho=1.5999`, `q_gap=0.967`, complete-boundary Arb scans and
the same finite transport and matrix routines.  They remain diagnostic-only
and do not enter the promoted theorem-facing certificate.

The retained wide-radius Phase 4 comparison data and plots are regenerated
from the map and displayed source by
`Numerics/blaschke_deformation_historical_phase4.py`.  Its process-based
row-block assembly rederives the historical sampled comparison at the locked
200-digit precision and applies the original finite-eigenvalue mid-gap contour
rule, while
`Numerics/hardy_moat_surface_worker.py` samples the retained local and global
moat surfaces.  The resulting contour profiles, fragile-packet comparisons,
eigenvalue data, surface arrays and reports are explicitly diagnostic-only and
cannot enter a theorem gate.  A fail-closed regression record requires the
matrix scales, all fifteen sampled passes and the worst sampled small-gain
product to remain in the retained historical regime; it does not claim
bit-for-bit equality across different numerical-library builds.  The two
retained audit tables are likewise
regenerated from their explicit upstream frames by
`Numerics/blaschke_deformation_diagnostic_audits.py`; retained audit CSV files
are never accepted as seeds.

Upstream of displayed Cell 95N,
`Numerics/blaschke_deformation_sampled_schur_diagnostics.py` and its producer
cell rebuild the six retained `M=N+6` sampled Schur rows from the selected
sampled geometry, the displayed Schur--Frobenius expression and the `kappa_T`
transport factor.  The producer never reads its destination CSV as an input or
cache and records `legacy_seed_dependency: false`.  Because the geometry is
sampled and the auxiliary finite-order factor `D_M` is fixed to one, these rows
remain diagnostic-only: no sampled Schur value enters a count, moat,
small-gain or Riesz-rank theorem gate.

Phase 4 is reconstructed from displayed source as well.  On a cache miss, or
when `BLASCHKE_FORCE_CONTOURS=1` is set, the contour helper rebuilds the
exact-binary Schur data and the seven Laurent witness tensors.  These witnesses
contain 300 transient `600` by `600` complex coefficient matrices.  Every
matrix is lifted exactly from binary64 into Arb for the complete-circle
residual proof.  A seven-row reconstruction manifest records the proposal
toolchain, full-tensor digest, exact-dyadic residual bound and parity with the
historical reference digest.  The coefficient tensors are then discarded to
avoid adding roughly 1.7 GB of redundant candidate data; the source, exact
generation parameters, full-tensor digests and validated mode bounds are
retained.  Digest parity is a provenance check and is not a theorem gate.

For a complete source replay, rebuild the retained historical Phase 4
diagnostics first with
`python -u -B Numerics/blaschke_deformation_historical_phase4.py --production
--assembly-workers 24 --surface-workers 6 --force`.  Then execute the notebook
with `BLASCHKE_FORCE_HARDY_MATRIX=1 BLASCHKE_FORCE_CONTOURS=1`.  The rebuilt Hardy-matrix digest
invalidates the dependent contour cache, so the same notebook pass also
rebuilds the exact-binary Schur data and seven Laurent moat witnesses.  This
staged route avoids placing the long historical surface calculation and the
theorem-facing matrix calculation in one execution session.  Exact
byte-for-byte digest parity for the binary64 Laurent proposals is
toolchain-sensitive, so the certificate report records Python, NumPy, SciPy,
python-flint, byte order and BLAS metadata.  If a different pinned toolchain
produces different proposal bytes but passes every exact-dyadic residual and
homotopy gate, it has produced a new valid Laurent witness rather than a
byte-identical reconstruction of the recorded witness.

The historical surface producer may resume validated internal row-block
checkpoints after interruption.  Those blocks are keyed by the exact source,
configuration and Hardy-matrix digests, are excluded from the archive, and are
not accepted as completed diagnostic artefacts.  A fresh archive extraction
therefore begins without them, while an interrupted local source rebuild can
continue without discarding already verified work.

Some compatibility artefact names retain the historical word `schur` in their
file name or schema.  In Phase 2 those names refer to the separable-Frobenius
collocation-defect route, not to a Schur factorisation.  Schur factorisation is
used theoremically only by the validated Phase 4 count and moat machinery.

Run Jupyter from this deployment root so that the expected `Numerics` path
layout is retained.  After the standalone historical rebuild, the complete
incremental notebook execution is:

`BLASCHKE_FORCE_HARDY_MATRIX=1 BLASCHKE_FORCE_CONTOURS=1 python -u -B Numerics/execute_notebook_incremental.py Numerics/blaschke_deformation_certifier_thesis_math.ipynb`

For current-release acceptance, use a separate extracted scratch bundle and
the source-only preparation/orchestration path rather than the deployment
working tree:

`python -B Numerics/run_blaschke_clean_room_replay.py --bundle-root . --kernel-name blaschke-replay --assembly-workers 24 --surface-workers 6 --published-archive /absolute/path/to/blaschke_deformation_certifier_reproducibility.tar.gz`

This command runs preparation before either builder, forces the historical
Phase 4, 1024/2048-bit Hardy-matrix and contour producers, and optionally runs
the replay-versus-published verifier.  The uncached N=600 replay is expected to
take substantially longer than ten minutes.

The exact documentation scope for directly mathematical functions is recorded
in `Numerics/mathematical_function_inventory.json`.  Its focused AST test
classifies every function in the eighteen inline helpers and every non-inline
final-notebook code cell, requires explicit `Explanation:` and `Functionality:`
fields inside triple-single-quoted docstrings only for the mathematical class,
and verifies standalone/inline digest parity:

`python -B -m unittest Numerics/test_mathematical_function_documentation.py`

Archive creation is a separate clean-worktree release operation after the
executed notebook and evidence have been committed; the curated
thesis-mathematics notebook intentionally contains no packaging cell.  The
NUMERICS_1 release has completed that operation.  For a new release snapshot,
once the same clean-commit gate holds, create the referee archive with:

`python -B Numerics/blaschke_deformation_reproducibility.py`

That command rejects a missing or dirty Git worktree, validates the inline
helper provenance, executed-notebook state, source manifest and theorem-facing
machine records again.  It derives the current reproducibility settings from
those records rather than trusting a stale notebook plan, captures a complete
Conda explicit lock, and writes the generated archive, checksum and external
manifest under
`Numerics/outputs/blaschke_deformation_certifier/reproducibility`.  This
generated directory is ignored so packaging does not dirty the clean source
commit recorded by the archive.

The resulting archive can be checked without extraction with
`python -B Numerics/verify_blaschke_deformation_reproducibility.py <archive> \
--external-manifest <external-manifest> --checksum <checksum>`.  After a replay
in a separate extracted copy, add
`--compare-executed-replay-root <replay-root>`.  The replay check requires exact
identity for immutable source files, reruns every theorem gate, compares the
stable Phase 2 and Phase 4 certificate data, and checks the retained historical
comparison rows and plots, including the source-generated historical Phase 4
and diagnostic-audit records.  It deliberately ignores non-mathematical
variation such as wall-clock timings, temporary paths and container metadata.

The validated data, reports and figures are stored under
`Numerics/outputs/blaschke_deformation_certifier`.  Refresh and verify the
working deployment checksum manifest with:

`python -B Numerics/refresh_deployment_manifest.py`

`python -B Numerics/refresh_deployment_manifest.py --check`

`MANIFEST.sha256` covers the tracked files and the current non-ignored
deployment sources and stored evidence, excluding itself and the two explicitly
historical notebook copies.  The generated reproducibility directory is
covered separately by its own external manifest and archive checksum.
