"""Reduced-size tests for the source-only historical Phase 4 producer."""

from __future__ import annotations

import csv
from dataclasses import replace
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import blaschke_deformation_historical_phase4 as phase4


class HistoricalPhase4SourceTests(unittest.TestCase):
    def test_public_notebook_entrypoint_signature(self) -> None:
        signature = inspect.signature(phase4.rebuild_historical_phase4)
        self.assertEqual(
            tuple(signature.parameters),
            (
                "config",
                "data_dir",
                "report_dir",
                "cache_dir",
                "assembly_workers",
                "surface_workers",
                "force",
            ),
        )
        self.assertEqual(signature.parameters["data_dir"].kind.name, "KEYWORD_ONLY")
        self.assertEqual(signature.parameters["force"].default, False)

    def test_production_configuration_is_locked(self) -> None:
        config = phase4.HistoricalPhase4Config.production_n600_m610()
        config.validate()
        with self.assertRaisesRegex(ValueError, "Production configuration is locked"):
            replace(config, rho="2.46").validate()
        with self.assertRaisesRegex(ValueError, "dps must be 200"):
            replace(config, dps=100).validate()

    def test_target_plan_comes_from_map_mid_gaps(self) -> None:
        config = phase4.HistoricalPhase4Config.small_test()
        identity, map_spec = phase4._build_identity(config)
        worker, _ = phase4._load_helpers()
        clusters = worker.exact_clusters_for_map(
            map_spec,
            max_power=config.maximum_target_power,
            include_trivial=False,
            full_multiplicity=True,
        )
        eigenvalues = []
        for cluster in clusters[: config.target_count]:
            eigenvalues.extend(
                [complex(cluster["value"])] * int(cluster["multiplicity"])
            )
        eigenvalues.append(0.0)
        targets = phase4._build_target_contours(
            config,
            map_spec,
            worker,
            np.asarray(eigenvalues, dtype=np.complex128),
        )

        self.assertEqual(tuple(target.name for target in targets), phase4._PRODUCTION_TARGETS)
        self.assertEqual(len(targets), 15)
        self.assertEqual(targets[0].expected_multiplicity, 1)
        self.assertEqual(
            next(target for target in targets if target.name == "mu^2").expected_multiplicity,
            2,
        )
        for target in targets:
            self.assertAlmostEqual(
                2.0 * target.radius,
                target.nearest_target_separation,
                places=15,
            )
            self.assertGreater(target.centre - target.radius, 0.0)
        self.assertEqual(len(identity.configuration_digest), 64)
        self.assertEqual(len(identity.source_digest), 64)
        self.assertTrue(
            all(len(record["sha256"]) == 64 for record in identity.source_records.values())
        )

    def test_cache_key_changes_with_numerical_configuration(self) -> None:
        first = phase4.HistoricalPhase4Config.small_test()
        second = replace(first, r="2.2")
        first_identity, _ = phase4._build_identity(first)
        second_identity, _ = phase4._build_identity(second)
        self.assertNotEqual(first_identity.configuration_digest, second_identity.configuration_digest)
        self.assertNotEqual(first_identity.cache_key, second_identity.cache_key)
        self.assertEqual(first_identity.source_digest, second_identity.source_digest)

    def test_production_regression_gate_rejects_underresolved_matrix_scale(self) -> None:
        config = phase4.HistoricalPhase4Config.production_n600_m610()
        good_tables = {
            "moats": [
                {
                    "sampled_validation_pass": True,
                    "epsilon_m_gamma": 0.5,
                }
                for _ in range(config.target_count)
            ]
        }
        profile = phase4._validate_historical_regression_profile(
            config,
            np.asarray([[1.965e9]], dtype=np.complex128),
            np.asarray([[1.775e9]], dtype=np.complex128),
            good_tables,
        )
        self.assertEqual(profile["sampled_validation_passes"], 15)
        with self.assertRaisesRegex(RuntimeError, "matrix scale"):
            phase4._validate_historical_regression_profile(
                config,
                np.asarray([[1.0e49]], dtype=np.complex128),
                np.asarray([[1.0e49]], dtype=np.complex128),
                good_tables,
            )


class HistoricalPhase4SmallRebuildTests(unittest.TestCase):
    def test_forced_surface_rebuild_resumes_source_keyed_row_blocks(self) -> None:
        config = phase4.HistoricalPhase4Config.small_test(
            N=4,
            M=5,
            dps=35,
            local_surface_grid=3,
            global_surface_grid=3,
            zoom_surface_grid=3,
            deep_zoom_surface_grid=3,
        )
        identity, _ = phase4._build_identity(config)
        _, surface_worker = phase4._load_helpers()
        window = {
            "xlim": (-0.25, 0.25),
            "ylim": (-0.25, 0.25),
            "grid": 3,
        }
        matrix = np.asarray(
            [[0.1 + 0.0j, 0.2 + 0.1j], [0.0 + 0.0j, -0.1 + 0.0j]],
            dtype=np.complex128,
        )
        with tempfile.TemporaryDirectory(prefix="historical-surface-resume-") as temporary:
            root = Path(temporary)
            output_path = root / "surface.npz"
            cache_dir = root / "cache"
            first = phase4._sample_surface(
                tag="resume-test",
                output_path=output_path,
                window=window,
                config=config,
                identity=identity,
                hardy_matrix=matrix,
                surface_worker=surface_worker,
                cache_dir=cache_dir,
                surface_workers=2,
                force=False,
            )
            expected = first[2].copy()
            output_path.unlink()

            resumed = phase4._sample_surface(
                tag="resume-test",
                output_path=output_path,
                window=window,
                config=config,
                identity=identity,
                hardy_matrix=matrix,
                surface_worker=None,
                cache_dir=cache_dir,
                surface_workers=2,
                force=True,
            )

            np.testing.assert_array_equal(resumed[2], expected)
            self.assertFalse(resumed[3])

    def test_small_source_only_rebuild_rejects_legacy_caches_and_reuses_keyed_caches(
        self,
    ) -> None:
        config = phase4.HistoricalPhase4Config.small_test(
            N=6,
            M=8,
            dps=45,
            contour_samples=8,
            fragile_samples=(8, 12),
            local_surface_grid=5,
            global_surface_grid=5,
            zoom_surface_grid=7,
            deep_zoom_surface_grid=7,
        )
        with tempfile.TemporaryDirectory(prefix="historical-phase4-test-") as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            report_dir = root / "reports"
            cache_dir = root / "cache"
            data_dir.mkdir(parents=True)
            report_dir.mkdir(parents=True)
            paths = phase4._artifact_paths(config, data_dir, report_dir)

            # These mimic retained files but carry no producer/config/source hashes.
            np.savez_compressed(
                paths["scaled_block"],
                B_scaled=np.zeros((config.N, config.N), dtype=np.complex128),
            )
            np.savez_compressed(
                paths["hardy_matrix"],
                A_X=np.zeros((config.N, config.N), dtype=np.complex128),
            )
            paths["eigenvalues"].write_text("real,imag,abs\n0,0,0\n", encoding="utf-8")
            for key, grid in (
                ("local_surface", config.local_surface_grid),
                ("global_surface", config.global_surface_grid),
                ("zoom_surface", config.zoom_surface_grid),
                ("deep_zoom_surface", config.deep_zoom_surface_grid),
            ):
                np.savez_compressed(
                    paths[key],
                    x=np.linspace(-1.0, 1.0, grid),
                    y=np.linspace(-1.0, 1.0, grid),
                    s_min=np.zeros((grid, grid)),
                )

            first = phase4.rebuild_historical_phase4(
                config,
                data_dir=data_dir,
                report_dir=report_dir,
                cache_dir=cache_dir,
                assembly_workers=2,
                surface_workers=2,
            )
            self.assertTrue(all(not hit for hit in first.cache_hits.values()))

            report = json.loads(first.report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["legacy_seed_dependency"])
            self.assertEqual(report["retained_csv_inputs"], [])
            self.assertEqual(report["retained_npz_inputs"], [])
            self.assertFalse(report["theorem_gate_eligible"])
            self.assertEqual(report["diagnostic_status"], phase4.DIAGNOSTIC_STATUS)
            self.assertTrue(report["process_execution"]["row_block_assembly"])
            self.assertTrue(report["process_execution"]["surface_row_blocks"])
            self.assertEqual(report["process_execution"]["assembly_workers"], 2)
            self.assertEqual(report["process_execution"]["surface_workers"], 2)
            self.assertEqual(report["sampled_diagnostics"]["target_count"], 15)
            self.assertEqual(report["sampled_diagnostics"]["profile_rows"], 15 * 8)
            self.assertEqual(report["sampled_diagnostics"]["fragile_rows"], 4)

            with np.load(paths["scaled_block"], allow_pickle=False) as payload:
                self.assertEqual(payload["B_scaled"].shape, (6, 6))
                self.assertFalse(np.array_equal(payload["B_scaled"], np.zeros((6, 6))))
                self.assertEqual(str(payload["cache_key"].item()), first.cache_key)
            with np.load(paths["hardy_matrix"], allow_pickle=False) as payload:
                self.assertEqual(payload["A_X"].shape, (6, 6))
                self.assertEqual(payload["B_scaled"].shape, (6, 6))
                self.assertEqual(payload["T"].shape, (6, 6))
                self.assertEqual(payload["eigenvalues"].shape, (6,))
                self.assertEqual(
                    str(payload["diagnostic_status"].item()),
                    phase4.DIAGNOSTIC_STATUS,
                )

            with paths["profiles"].open("r", encoding="utf-8", newline="") as stream:
                profile_rows = list(csv.DictReader(stream))
            self.assertEqual(len(profile_rows), 15 * config.contour_samples)
            self.assertTrue(
                all(row["diagnostic_status"] == phase4.DIAGNOSTIC_STATUS for row in profile_rows)
            )
            with paths["fragile"].open("r", encoding="utf-8", newline="") as stream:
                fragile_rows = list(csv.DictReader(stream))
            self.assertEqual(
                {(int(row["J"]), row["target"]) for row in fragile_rows},
                {(8, "alpha^11"), (8, "mu^4"), (12, "alpha^11"), (12, "mu^4")},
            )

            expected_surface_shapes = {
                "local_surface": (5, 5),
                "global_surface": (5, 5),
                "zoom_surface": (7, 7),
                "deep_zoom_surface": (7, 7),
            }
            for key, shape in expected_surface_shapes.items():
                with np.load(paths[key], allow_pickle=False) as payload:
                    self.assertEqual(payload["s_min"].shape, shape)
                    self.assertTrue(np.isfinite(payload["s_min"]).all())
                    self.assertEqual(str(payload["cache_key"].item()), first.cache_key)

            second = phase4.rebuild_historical_phase4(
                config,
                data_dir=data_dir,
                report_dir=report_dir,
                cache_dir=cache_dir,
                assembly_workers=2,
                surface_workers=2,
            )
            self.assertEqual(second.cache_key, first.cache_key)
            self.assertTrue(all(second.cache_hits.values()))
            second_report = json.loads(second.report_path.read_text(encoding="utf-8"))
            self.assertTrue(all(second_report["cache_validation"]["cache_hits"].values()))


if __name__ == "__main__":
    unittest.main()
