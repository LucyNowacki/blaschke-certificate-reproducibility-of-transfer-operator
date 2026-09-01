# Blaschke certificate-equivalent reproduction

This Git tree is the public release unit for the symmetric Blaschke deformation
experiment `blaschke_mu_0p3` (`N=600`, `M=610`).  Reproduction is judged by the
mathematical certificate, not by byte-for-byte equality of paths, archives,
notebooks, plots, or toolchain-sensitive witness arrays.

The authoritative command requires a release identity obtained independently
from the mutable checkout.  For a GitHub tag or Release, use the commit ID
published by GitHub:

```bash
EXPECTED_RELEASE_COMMIT='<40-hex commit from the GitHub tag or Release>'
./reproduce-certificate.sh full --expected-git-commit "$EXPECTED_RELEASE_COMMIT"
```

For an extracted source archive without `.git`, use the inventory SHA-256
published alongside the Release with
`--expected-release-inventory-sha256`.  Never derive either expected value
from the checkout being authenticated.

It creates the exact locked environment when needed, authenticates the finite
Git-tree inventory, stages a fresh source-only scratch copy, forces the existing
clean-room numerical replay, and then runs the semantic verifier.  Success is a
canonical JSON receipt whose status is exactly:

```text
CERTIFICATION_CONFIRMED
```

Only this full mode is a source-level recomputation of the theorem certificate,
independent of retained generated artifacts.  It is not a new human derivation
or an alternate mathematical proof.  It does not run the old raw-byte
comparator, notebook normalizer, archive comparator, or
publication-provenance refresh.

## What the verifier proves

The verifier does not trust stored summary booleans in place of their primitive
one-sided data.  Among other integrity and schema checks, it requires:

- one exact Decimal Hardy radius,
  `2.473669807791324109321273260`, across the selected geometry, Phase 2,
  authoritative 2048-bit Hardy, Schur, and contour artifacts;
- the exact rational gate `r_tau_upper / r <= 0.927`;
- exact rational Phase 2 RSS-square, triangle, and epsilon inequalities;
- the ordered 24 target families, multiplicity total 30, and theorem routes
  `17` Schur-moat plus `7` Laurent-moat;
- all theorem gates, finite-count transports, zero exclusion, and complete
  circle coverage;
- all 24 Schur diagonal counts recomputed from decoded binary64 `T` and exact
  rational circle centres and radii, together with an exactly zero lower
  triangle;
- Schur lower bounds rederived as `1 / (inverse_upper * Q_condition_upper)`;
- Laurent lower bounds rederived as
  `(1 - residual_upper) / (coefficient_upper * Q_condition_upper)` from the
  replay's exact-dyadic witness summaries;
- the `eta_schur` and `eta_A` subtractions, a positive lifted moat, and the
  rederived small-gain ratio `epsilon / lifted_moat < 1`.

The stored aggregate small-gain display is diagnostic: independently
outward-rounded component displays need not reaggregate to the same final ULP.
The theorem decision is made from primitive one-sided fields with exact
rational arithmetic.

## Modes

### Full source replay

```bash
EXPECTED_RELEASE_COMMIT='<40-hex commit from the GitHub tag or Release>'
./reproduce-certificate.sh full --expected-git-commit "$EXPECTED_RELEASE_COMMIT"
```

The wrapper creates `.certificate-replay-env/` from
`conda-explicit-lock.txt`, installs the seven exact hash-locked pip overrides,
and binds a local Jupyter kernel to that interpreter.  It then uses a new
scratch directory.  The theorem-only branch freshly creates and hash-binds the
fixed 27-file theorem closure.  It deliberately skips historical Phase 4 and
the non-authoritative 1024-bit Hardy starting audit; the authoritative
2048-bit Hardy interval and all seven Laurent witnesses are regenerated.
The wrapper may create only the gitignored local
environment beside the checkout; the receipt records
`tracked_source_bytes_mutated=false`.

Expected platform and resources:

- Linux x86_64;
- at least 8 GiB RAM and 8 GiB free scratch disk;
- locked environment size about 3.1 GB.

The theorem-only duration and peak-memory measurement are pending the first
authorized replay; the earlier 45-minute broader publication replay is not
presented as a timing for this new route.  The full route regenerates the
authoritative Hardy interval matrices and all seven Laurent witness families
in memory.  Large transient coefficient tensors are verified and discarded
rather than committed.

An already authenticated pinned runtime can be reused:

```bash
EXPECTED_RELEASE_COMMIT='<40-hex commit from the GitHub tag or Release>'
./reproduce-certificate.sh full \
  --expected-git-commit "$EXPECTED_RELEASE_COMMIT" \
  --existing-python /absolute/path/to/bin/python
```

This trusted-runtime shortcut is still fail-closed: Python, packages,
`pip check`, single-thread OpenBLAS identity and location, and the exact
ephemeral kernel specification are checked again.  The default GitHub-auditor
path remains self-contained environment creation.

### Stored-artifact check

```bash
EXPECTED_RELEASE_COMMIT='<40-hex commit from the GitHub tag or Release>'
./reproduce-certificate.sh quick --expected-git-commit "$EXPECTED_RELEASE_COMMIT"
```

This checks source and artifact hashes, schemas, exact discrete structure,
primitive inequalities, theorem gates, and internal bindings in the artifacts
already present in the checkout.  Its success status is:

```text
ARTIFACT_SEMANTICS_CONFIRMED_NON_INDEPENDENT
```

Artifact-only mode is useful for inspection, but it is **not an independent
theorem proof** because it does not regenerate the Hardy intervals or Laurent
witnesses.  A pre-contract checkout is expected to fail closed rather than
silently upgrade older stored gates.

### No-compute preflight

```bash
EXPECTED_RELEASE_COMMIT='<40-hex commit from the GitHub tag or Release>'
./reproduce-certificate.sh dry-run --expected-git-commit "$EXPECTED_RELEASE_COMMIT"
```

This validates the finite source inventory, locks, interpreter, `pip check`,
BLAS identity, kernel binding, safe staging, and replay/verifier command-line
interfaces.  It starts no numerical producer and reports:

```text
DRY_RUN_READY_NO_CERTIFICATION
```

It never claims certification.

## Strict and intentionally variable evidence

Acceptance remains strict for source authentication, environment identity,
canonical radius and q-gap, exact Phase 2 inequalities, ordered targets and
multiplicities, routes and counts, complete-circle constructions, perturbation
subtractions, zero exclusion, positive lifted moats, and every theorem gate.

The comparison deliberately ignores only absolute or temporary paths,
wall-clock timings, widget/notebook runtime metadata, plots and rendered image
bytes, cache placement, archive/container formatting, and explicitly
diagnostic-only sampled tables or provenance digests.  Sampled extrema and a
digest alone are never promoted into theorem evidence.

## Release integrity and CI

`release/source-only-replay-inventory.json` lists the complete finite Git-tree
release and binds every member by SHA-256.  The stager validates it before
copying regular files into a new narrow scratch root; symlinks, traversal,
missing files, extra files, and hash drift fail closed.  The original checkout
is preserved.

Pull-request CI performs source compilation, lock and inventory checks, and the
focused semantic negative tests only.  GitHub's event-provided `github.sha`
acts as the external identity for the checked-out CI tree.  The full replay is
intentionally a manual job on a Linux x86_64 self-hosted runner.  See
`release/REPLAY.md` for the auditor-oriented command sequence and receipt
interpretation.

## Mathematical provenance

The exact symmetric interval target families are benchmark data from
Slipantschuk--Bandtlow--Just (2013, equation 21), also recorded for this map in
Akindji--Slipantschuk--Bandtlow--Just (2026, Section 3.2).  Those references
identify the target windows.  This release recomputes the finite counts,
complete-circle moats, perturbation transports, and small-gain gates from the
authenticated sources, independently of checked-in generated evidence.
