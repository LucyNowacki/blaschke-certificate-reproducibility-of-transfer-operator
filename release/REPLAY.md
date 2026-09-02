# Auditor replay guide

The release unit is the checked-out Git tree.  Do not substitute the historical
raw-comparison archive for it, and do not load retained pickle evidence before
the release inventory has been authenticated by the supplied stager/verifier.

## Source-level certificate recomputation

From the checkout root on Linux x86_64:

```bash
EXPECTED_RELEASE_COMMIT='<40-hex commit from the GitHub tag or Release>'
./reproduce-certificate.sh full --expected-git-commit "$EXPECTED_RELEASE_COMMIT"
```

The expected commit must come from an external authority such as the GitHub tag
or Release page, never from `git rev-parse HEAD` in the tree under test.  For an
archive without `.git`, use the separately published inventory digest through
`--expected-release-inventory-sha256` instead.

Allow at least 8 GiB RAM and 8 GiB free scratch storage.  The exact environment
occupies about 3.1 GB.  Duration and peak-memory figures for this theorem-only
route are pending the first authorized replay; the historical broader replay's
timing is not reused as a claim for this route.

The wrapper performs, in order:

1. exact Conda-lock and hash-locked pip environment creation;
2. Python/package, `pip check`, single-thread OpenBLAS, and Jupyter-kernel
   validation;
3. finite Git-tree inventory authentication and safe staging into a fresh
   scratch root;
4. theorem-only clean-room replay with the fixed 27 theorem outputs absent
   before compute, freshly recreated and hash-bound, including the authoritative
   2048-bit Hardy interval and all Laurent witness regeneration;
5. exact semantic verification and canonical receipt creation.

Historical Phase 4 and the non-authoritative 1024-bit Hardy starting audit are
explicitly skipped and recorded as diagnostic/non-authoritative.  They are not
members of the theorem closure.

The raw-byte comparator, notebook normalizer, archive comparator, and
publication-provenance refresh are not invoked.  The source checkout is
preserved; the wrapper may create only its gitignored local environment, and
the receipt records `tracked_source_bytes_mutated=false`.

The final JSON is successful only when both fields are present:

```json
{
  "independent_theorem_recomputation": true,
  "status": "CERTIFICATION_CONFIRMED"
}
```

Any exception, stale schema, missing field, extra or changed release member,
failed primitive inequality, failed theorem gate, or malformed evidence exits
nonzero and cannot produce that status.

Here `independent_theorem_recomputation` means independent of retained
generated artifacts: the authenticated source computation is rerun.  It does
not claim a new human derivation or an alternate proof.

## Reusing an authenticated runtime

For an already installed exact runtime and its bound local kernel:

```bash
EXPECTED_RELEASE_COMMIT='<40-hex commit from the GitHub tag or Release>'
./reproduce-certificate.sh full \
  --expected-git-commit "$EXPECTED_RELEASE_COMMIT" \
  --existing-python /absolute/path/to/bin/python
```

This skips environment creation only.  The runner still checks Python 3.13.2,
all theorem-critical pinned package versions, `python -m pip check`, OpenBLAS 0.3.30 pthreads in
single-thread mode and inside the validated prefix, plus the complete
release-local ephemeral kernelspec argv/environment.  The theorem-only driver
does not depend on a user or global kernel.

## Preflight without computation

```bash
EXPECTED_RELEASE_COMMIT='<40-hex commit from the GitHub tag or Release>'
./reproduce-certificate.sh dry-run --expected-git-commit "$EXPECTED_RELEASE_COMMIT"
```

Expected status:

```text
DRY_RUN_READY_NO_CERTIFICATION
```

This validates release files, locks, staging, runtime identity, dependency
consistency, kernel binding, and command-line entry points.  It runs no theorem
producer and is not proof evidence.

## Inspecting retained artifacts

```bash
EXPECTED_RELEASE_COMMIT='<40-hex commit from the GitHub tag or Release>'
./reproduce-certificate.sh quick --expected-git-commit "$EXPECTED_RELEASE_COMMIT"
```

Expected success status:

```text
ARTIFACT_SEMANTICS_CONFIRMED_NON_INDEPENDENT
```

This is a fail-closed semantic and binding audit of stored artifacts.  It does
not regenerate the theorem witnesses and therefore cannot claim an independent
proof.  In particular, stored gate booleans, sampled extrema, and provenance
digests are not accepted as substitutes for primitive theorem evidence.

## Display notebook and dependency map

The checked-in
[`blaschke_deformation_certifier_thesis_math.ipynb`](../Numerics/blaschke_deformation_certifier_thesis_math.ipynb)
is a generated presentation copy with all plot positions retained as compact
GitHub previews. Full-resolution images are stored in the adjacent
[`figures`](../Numerics/outputs/blaschke_deformation_certifier/figures/) directory.
Non-visual runtime chatter is omitted from that GitHub-facing copy so scratch
paths cannot become publication metadata.
The [dependency-map explanation](../Numerics/blaschke_deformation_notebook_dependency_map.md)
and [standalone SVG](../Numerics/blaschke_deformation_notebook_dependency_map.svg)
record which branches are theorem-facing and which are diagnostic.

Those stored notebook outputs are never accepted as proof input. Source replay
deletes the generated display notebook in its scratch tree, rebuilds the locked
output-free source counterpart, and recreates the theorem closure independently.

## Receipt interpretation

Semantic equivalence requires the same exact radius and q-gap inequality,
Phase 2 one-sided inequalities, ordered targets and multiplicities, route and
count structure, complete-circle Schur/Laurent lower bounds, perturbation
subtractions, zero exclusions, positive lifted moats, and small-gain gates.
Paths, timings, notebook widgets, plots, cache locations, container formatting,
and explicitly diagnostic-only data may vary.  Such variation does not weaken
the mathematical gates and is never used to manufacture a confirmation.
