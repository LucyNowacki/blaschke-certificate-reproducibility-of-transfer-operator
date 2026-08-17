# Final Deployment

This folder is the self-contained deployment of the symmetric Blaschke
deformation experiment `blaschke_mu_0p3`.  It deliberately excludes the
asymmetric degree-two Blaschke helper.

The principal executed record is
`Numerics/blaschke_deformation_certifier_thesis_math.ipynb`.  Its five
theorem-facing mathematical helpers are included inline beside the phases that
use them, while byte-identical standalone copies remain available for direct
inspection, process workers, digest validation and packaging.

The reproducible builder chain is:

1. locked output-free template
   `Numerics/blaschke_deformation_certifier_template.ipynb`;
2. output-free source notebook
   `Numerics/blaschke_deformation_certifier.ipynb`;
3. inline thesis-mathematics counterpart
   `Numerics/blaschke_deformation_certifier_thesis_math.ipynb`.

The recorded execution uses the Conda environment `lucy`; the commands below
assume that environment has been activated.

Rebuild the first two executable stages from the deployment root with:

`python -B Numerics/build_blaschke_deformation_certifier.py`

`python -B Numerics/build_blaschke_deformation_thesis_math_notebook.py`

The builders reject a mismatch between a standalone helper, its displayed
SHA-256 digest and its inline source.  The notebook bootstrap repeats the digest
check before executing each inline module.  Rebuilding the counterpart produces
an output-free notebook; execute it afterwards to recreate the stored outputs.

Run Jupyter from this deployment root so that the expected `Numerics` path
layout is retained.  The complete incremental command-line execution is:

`BLASCHKE_SKIP_ARCHIVE=1 python -u -B Numerics/execute_notebook_incremental.py Numerics/blaschke_deformation_certifier_thesis_math.ipynb`

The environment variable deliberately defers archive creation.  After the
executed notebook and evidence have been committed and the deployment worktree
is clean, create the referee archive with:

`python -B Numerics/blaschke_deformation_reproducibility.py`

That command rejects a missing or dirty Git worktree, validates the inline
helper provenance again, and writes the generated archive, checksum and
external manifest under
`Numerics/outputs/blaschke_deformation_certifier/reproducibility`.  This
generated directory is ignored so packaging does not dirty the clean source
commit recorded by the archive.

The validated data, reports and figures are stored under
`Numerics/outputs/blaschke_deformation_certifier`.  `MANIFEST.sha256` records
the SHA-256 digest of every versioned deployment file other than itself; the
generated reproducibility directory is covered by its own external manifest
and archive checksum.
