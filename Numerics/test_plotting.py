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
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plotting


def _assert_pairwise_disjoint(
    testcase: unittest.TestCase,
    artists: list[object],
    renderer: object,
) -> None:
    boxes = [artist.get_window_extent(renderer=renderer) for artist in artists]
    for index, left in enumerate(boxes):
        for right in boxes[index + 1:]:
            testcase.assertFalse(left.overlaps(right))


class Phase3PlotLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = pd.DataFrame(
            {
                "name": ["old", "branch-image", "response", "wide"],
                "short_source": ["old", "branch-image", "response", "wide"],
                "fitted_base": [0.97, 0.925, 0.924, 0.932],
                "epsilon_over_qstar_power": [4.4, 2.4, 1.9, 2.0],
                "B_out": [2.8e-9, 2.4e-20, 2.9e-23, 1.8e-20],
                "B_in": [7.3e-9, 3.5e-20, 3.3e-20, 1.2e-17],
                "collocation": [8.6e-14, 1.1e-28, 1.1e-28, 7.5e-28],
                "epsilon": [7.0e-9, 4.0e-20, 3.3e-20, 1.0e-17],
                "qstar_power": [1.6e-9, 1.7e-20, 1.7e-20, 5.0e-18],
            }
        )

    def test_certificate_component_fonts_are_four_points_larger(self) -> None:
        result = plotting.plot_phase3_certificate_components(
            self.rows,
            show=False,
        )
        try:
            result.figure.canvas.draw()
            axis = result.axes
            self.assertEqual(axis.title.get_fontsize(), 24)
            self.assertEqual(axis.xaxis.label.get_fontsize(), 21)
            self.assertEqual(axis.yaxis.label.get_fontsize(), 21)
            self.assertTrue(all(label.get_fontsize() == 21 for label in axis.get_xticklabels()))
            self.assertTrue(all(label.get_fontsize() == 21 for label in axis.get_yticklabels()))
        finally:
            plt.close(result.figure)

    def test_rate_legend_occupies_its_own_figure_margin(self) -> None:
        result = plotting.plot_phase3_rate_diagnostics(
            self.rows,
            q_star=0.925,
            show=False,
        )
        try:
            result.figure.canvas.draw()
            self.assertIsNone(result.axes[1].get_legend())
            self.assertEqual(len(result.figure.legends), 1)
            renderer = result.figure.canvas.get_renderer()
            legend_box = result.figure.legends[0].get_window_extent(renderer=renderer)
            for axis in result.axes:
                self.assertFalse(
                    legend_box.overlaps(axis.get_window_extent(renderer=renderer))
                )
                self.assertEqual(axis.title.get_fontsize(), 24)
                self.assertEqual(axis.yaxis.label.get_fontsize(), 21)
                self.assertTrue(all(label.get_fontsize() == 18 for label in axis.get_xticklabels()))
                self.assertTrue(all(label.get_fontsize() == 19 for label in axis.get_yticklabels()))
        finally:
            plt.close(result.figure)

    def test_bridge_component_tick_labels_are_disjoint(self) -> None:
        raw = pd.DataFrame(
            {
                "N": [10, 20, 10, 20],
                "name": ["alpha^1", "alpha^1", "mu^1", "mu^1"],
                "error": [1.0e-8, 1.0e-12, 2.0e-8, 2.0e-13],
            }
        )
        components = {
            "output leakage": 2.0e-23,
            "input leakage": 3.0e-20,
            "collocation defect": 8.0e-29,
            "complete radius": 3.1e-20,
        }
        result = plotting.plot_phase3_empirical_deterministic_bridge(
            raw,
            self.rows,
            components,
            target_names=("alpha^1", "mu^1"),
            epsilon=3.1e-20,
            q_star=0.925,
            show=False,
        )
        try:
            result.figure.canvas.draw()
            renderer = result.figure.canvas.get_renderer()
            _assert_pairwise_disjoint(
                self,
                list(result.axes[2].get_xticklabels()),
                renderer,
            )
            self.assertEqual(result.figure._suptitle.get_fontsize(), 24)
            for axis in result.axes:
                self.assertEqual(axis.title.get_fontsize(), 22)
                self.assertEqual(axis.yaxis.label.get_fontsize(), 20)
            self.assertEqual(result.axes[0].xaxis.label.get_fontsize(), 20)
            expected_x_tick_sizes = (20, 17, 16.5)
            for axis, expected in zip(result.axes, expected_x_tick_sizes, strict=True):
                self.assertTrue(
                    all(label.get_fontsize() == expected for label in axis.get_xticklabels())
                )
                self.assertTrue(
                    all(label.get_fontsize() == 20 for label in axis.get_yticklabels())
                )
        finally:
            plt.close(result.figure)


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


class CertificationLadderTests(unittest.TestCase):
    def test_gate_failed_has_a_distinct_default_colour(self) -> None:
        frame = pd.DataFrame(
            {
                "audit_item": ["certified gate", "failed gate"],
                "status": ["theorem_certified", "gate_failed"],
            }
        )
        result = plotting.plot_certification_ladder(
            frame,
            title="Final certificate",
            show=False,
        )
        try:
            face_colours = [patch.get_facecolor() for patch in result.axes.patches]
            self.assertEqual(
                face_colours[0],
                matplotlib.colors.to_rgba(plotting.THESIS_PALETTE["real"], alpha=0.86),
            )
            self.assertEqual(
                face_colours[1],
                matplotlib.colors.to_rgba(
                    plotting.THESIS_EXTRA_PALETTE["vermillion"], alpha=0.86
                ),
            )
        finally:
            plt.close(result.figure)


if __name__ == "__main__":
    unittest.main()
