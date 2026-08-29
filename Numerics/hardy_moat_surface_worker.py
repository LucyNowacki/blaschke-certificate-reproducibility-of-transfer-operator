"""Process workers for Hardy-gauge singular-value surface sampling.

The notebook calls these helpers to assemble rows of
sigma_min((x + i y) I - A) in independent processes.  Each completed
row block is written to disk, so interrupted surface runs can resume
without recomputing finished blocks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import scipy.linalg as spla

_A = None
_I = None


def initialise_surface_worker(matrix_path: str) -> None:
    """Load the Hardy-gauge matrix once per worker process."""

    global _A, _I
    _A = np.load(matrix_path, mmap_mode="r")
    _I = np.eye(_A.shape[0], dtype=np.complex128)


def sample_surface_row_block(task: tuple) -> str:
    '''Explanation: The smallest singular value of zeta I minus A visualises finite resolvent separation over a region of the complex plane. A row block contributes to that explanatory surface, but grid sampling does not certify the gaps between points.
Functionality: Compute and save one row block of the singular-value surface.'''

    if _A is None or _I is None:
        raise RuntimeError("Surface worker was not initialised with a matrix.")

    block_id, y_start, y_stop, x_values, y_values, block_path = task
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    block_path = Path(block_path)

    s_block = np.empty((len(y_values), len(x_values)), dtype=np.float64)
    for local_iy, y_val in enumerate(y_values):
        for ix, x_val in enumerate(x_values):
            zeta = x_val + 1j * y_val
            s_block[local_iy, ix] = float(
                spla.svdvals(zeta * _I - _A, check_finite=False)[-1]
            )

    block_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = block_path.with_suffix(block_path.suffix + ".tmp")
    with open(tmp_path, "wb") as fh:
        np.savez_compressed(
            fh,
            block_id=np.asarray(block_id, dtype=np.int64),
            y_start=np.asarray(y_start, dtype=np.int64),
            y_stop=np.asarray(y_stop, dtype=np.int64),
            s_block=s_block,
        )
    os.replace(tmp_path, block_path)
    return str(block_path)
