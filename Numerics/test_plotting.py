"""Focused tests for the Phase 4 local moat surface renderer."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plotting


class Phase4LocalMoatSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.axis = np.linspace(0.06, 0.12, 25)
        self.contours = [
            {
                "centre": 0.09 + 0.0j,
                "radius": 0.015,
                "name": "mu^2",
                "family": "mu",
            }
        ]

    def test_generates_surface_for_positive_grid_below_fixed_floor(self) -> None:
        moat = np.geomspace(1.0e-33, 1.0e-21, 25 * 25).reshape(25, 25)

        with tempfile.TemporaryDirectory(prefix="phase4-local-moat-test-") as temporary:
            result = plotting.plot_phase4_local_moat_surface(
                self.axis,
                self.axis,
                moat,
                self.contours,
                output_dir=temporary,
                stem="local-moat-small-positive-values",
                show=False,
            )
            try:
                self.assertEqual(len(result.paths), 1)
                self.assertTrue(result.paths[0].is_file())
                self.assertGreater(result.paths[0].stat().st_size, 0)
                result.figure.canvas.draw()
            finally:
                plt.close(result.figure)

    def test_rejects_non_positive_and_non_finite_moat_grids(self) -> None:
        valid = np.geomspace(1.0e-33, 1.0e-21, 25 * 25).reshape(25, 25)
        invalid_cases = (
            (0.0, "strictly positive"),
            (-1.0e-30, "strictly positive"),
            (np.nan, "finite"),
            (np.inf, "finite"),
        )

        for value, message in invalid_cases:
            with self.subTest(value=value):
                moat = valid.copy()
                moat[0, 0] = value
                with self.assertRaisesRegex(ValueError, message):
                    plotting.plot_phase4_local_moat_surface(
                        self.axis,
                        self.axis,
                        moat,
                        self.contours,
                        show=False,
                    )


if __name__ == "__main__":
    unittest.main()
