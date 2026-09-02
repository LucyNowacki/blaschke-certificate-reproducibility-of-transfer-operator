"""Focused regressions for source-only replay and surface-worker BLAS limits."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_blaschke_deformation_thesis_math_notebook as notebook_builder
import blaschke_deformation_phase2_transport as phase2_transport
import hardy_moat_surface_worker as surface_worker
import run_blaschke_clean_room_replay as replay


class ReplayThreadEnvironmentTests(unittest.TestCase):
    def test_every_official_child_receives_forced_single_thread_environment(self) -> None:
        with tempfile.TemporaryDirectory(prefix="threaded-replay-env-") as temporary:
            root = Path(temporary) / replay.BUNDLE_ROOT_NAME
            numerics = root / "Numerics"
            numerics.mkdir(parents=True)
            (root / "source-only-replay-preparation.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (numerics / "blaschke_deformation_reproducibility_plan.json").write_text(
                "{}\n", encoding="utf-8"
            )
            raw_comparison_receipt = Path(temporary) / "raw-comparison.json"
            raw_comparison_receipt.write_text("{}\n", encoding="utf-8")
            calls: list[tuple[list[str], dict[str, str]]] = []

            def record(
                command: list[str], *, root: Path, environment: dict[str, str]
            ) -> dict[str, object]:
                calls.append((command, dict(environment)))
                return {
                    "command": command,
                    "elapsed_seconds": 0.0,
                    "returncode": 0,
                }

            inherited = {key: "8" for key in replay.BLAS_THREAD_ENVIRONMENT}
            with mock.patch.dict(os.environ, inherited, clear=False), mock.patch.object(
                replay,
                "_authenticate_inventory_before_preparation",
                return_value="0" * 64,
            ), mock.patch.object(
                replay,
                "check_prepared_bundle",
                return_value={"status": "source-only replay root prepared"},
            ), mock.patch.object(
                replay,
                "_verify_generated_closure",
                return_value={"status": "declared generated output closure complete"},
            ), mock.patch.object(replay, "_run", side_effect=record):
                result = replay.run_replay(
                    bundle_root=root,
                    kernel_name="test-kernel",
                    assembly_workers=2,
                    surface_workers=2,
                    prepare_only=False,
                    published_archive=None,
                    expected_inventory_sha256="0" * 64,
                    raw_comparison_receipt=raw_comparison_receipt,
                )

            self.assertTrue(calls)
            self.assertEqual(calls[0][0][-2], "-c")
            self.assertEqual(calls[0][0][-1], replay.BLAS_RUNTIME_PREFLIGHT)
            for _, environment in calls:
                for key in replay.BLAS_THREAD_ENVIRONMENT:
                    self.assertEqual(environment[key], "1")
            for key in replay.BLAS_THREAD_ENVIRONMENT:
                self.assertEqual(result["forced_rebuild_environment"][key], "1")

    def test_live_preflight_accepts_one_thread_and_rejects_four(self) -> None:
        base = dict(os.environ)
        base["BLASCHKE_AUTHENTICATED_REPLAY_ROOT"] = str(HERE.parent)
        base["PYTHONPATH"] = os.pathsep.join((str(HERE.parent), str(HERE)))
        one = dict(base)
        one.update({key: "1" for key in replay.BLAS_THREAD_ENVIRONMENT})
        accepted = subprocess.run(
            [sys.executable, "-B", "-c", replay.BLAS_RUNTIME_PREFLIGHT],
            cwd=HERE.parent,
            env=one,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(accepted.returncode, 0, msg=accepted.stderr)
        report = json.loads(accepted.stdout.strip().splitlines()[-1])
        self.assertEqual(report["status"], "single-threaded BLAS runtime verified")
        self.assertTrue(report["blas"])
        self.assertTrue(all(record["num_threads"] == 1 for record in report["blas"]))

        four = dict(base)
        four.update({key: "4" for key in replay.BLAS_THREAD_ENVIRONMENT})
        rejected = subprocess.run(
            [sys.executable, "-B", "-c", replay.BLAS_RUNTIME_PREFLIGHT],
            cwd=HERE.parent,
            env=four,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("exactly one thread", rejected.stderr)


class GeneratedKernelSetupTests(unittest.TestCase):
    def test_generated_cell_5_caps_threads_before_numerical_imports(self) -> None:
        replacements = json.loads(
            (HERE / "plotting_cell_replacements.json").read_text(encoding="utf-8")
        )
        authoritative = replacements["6a823024"]
        cap_position = authoritative.index("os.environ[key] = '1'")
        for import_text in (
            "import mpmath as mp",
            "import numpy as np",
            "import pandas as pd",
            "from plotting import",
        ):
            self.assertLess(cap_position, authoritative.index(import_text))
        self.assertNotIn("os.environ.setdefault", authoritative)
        limiter_position = authoritative.index("threadpool_limits(limits=1, user_api='blas')")
        info_position = authoritative.index("threadpool_info()")
        self.assertLess(limiter_position, info_position)
        self.assertIn("_BLAS_THREAD_LIMIT", authoritative)
        self.assertIn("threadpool_info()", authoritative)
        self.assertIn("Cell 5N could not constrain every loaded BLAS runtime", authoritative)

        generated, _ = notebook_builder.build_curated()
        generated_cell = next(
            cell for cell in generated["cells"] if cell.get("id") == "6a823024"
        )
        generated_source = "".join(generated_cell.get("source", []))
        self.assertIn(authoritative, generated_source)

    def test_generated_cell_5_constrains_an_already_loaded_blas_runtime(self) -> None:
        program = textwrap.dedent(
            f"""
            import json
            import os
            from pathlib import Path
            import sys

            sys.path.insert(0, {str(HERE)!r})
            import numpy as np
            from threadpoolctl import threadpool_info, threadpool_limits

            np.linalg.svd(np.eye(2))
            before = [r for r in threadpool_info() if r.get('user_api') == 'blas']
            replacement = json.loads(
                (Path({str(HERE)!r}) / 'plotting_cell_replacements.json').read_text(
                    encoding='utf-8'
                )
            )['6a823024']
            namespace = {{'__name__': '__main__'}}
            exec(replacement, namespace)
            after_first = [r for r in threadpool_info() if r.get('user_api') == 'blas']
            exec(replacement, namespace)
            after_second = [r for r in threadpool_info() if r.get('user_api') == 'blas']
            with threadpool_limits(limits=24, user_api='blas'):
                inside = [r for r in threadpool_info() if r.get('user_api') == 'blas']
            restored = [r for r in threadpool_info() if r.get('user_api') == 'blas']
            print('WARM_KERNEL_REPORT=' + json.dumps({{
                'before': before,
                'after_first': after_first,
                'after_second': after_second,
                'inside': inside,
                'restored': restored,
                'limiter_retained': '_BLAS_THREAD_LIMIT' in namespace,
            }}, sort_keys=True))
            """
        )
        environment = dict(os.environ)
        environment.update({key: "24" for key in replay.BLAS_THREAD_ENVIRONMENT})
        with tempfile.TemporaryDirectory(prefix="warm-notebook-kernel-") as temporary:
            completed = subprocess.run(
                [sys.executable, "-B", "-c", program],
                cwd=temporary,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        line = next(
            item
            for item in completed.stdout.splitlines()
            if item.startswith("WARM_KERNEL_REPORT=")
        )
        report = json.loads(line.removeprefix("WARM_KERNEL_REPORT="))
        self.assertTrue(report["before"])
        self.assertTrue(any(row["num_threads"] == 24 for row in report["before"]))
        self.assertTrue(all(row["num_threads"] == 1 for row in report["after_first"]))
        self.assertTrue(all(row["num_threads"] == 1 for row in report["after_second"]))
        self.assertTrue(all(row["num_threads"] == 24 for row in report["inside"]))
        self.assertTrue(all(row["num_threads"] == 1 for row in report["restored"]))
        self.assertTrue(report["limiter_retained"])


class ScopedCompatibilityKernelTests(unittest.TestCase):
    def test_transport_inverse_uses_inner_24_and_restores_outer_one(self) -> None:
        inverse, evidence = phase2_transport._invert_midpoint_with_locked_blas(
            np.eye(3, dtype=np.float64)
        )

        np.testing.assert_array_equal(inverse, np.eye(3, dtype=np.float64))
        self.assertEqual(evidence["scope_threads"], 24)
        self.assertEqual(evidence["outer_threads"], 1)
        self.assertEqual(
            evidence["outer_before"]["runtime"]["num_threads"], 1
        )
        self.assertEqual(evidence["inside"]["runtime"]["num_threads"], 24)
        self.assertEqual(evidence["outer_after"]["runtime"]["num_threads"], 1)

    def test_transport_inverse_restores_outer_one_after_linalg_failure(self) -> None:
        with self.assertRaises(np.linalg.LinAlgError):
            phase2_transport._invert_midpoint_with_locked_blas(
                np.zeros((2, 2), dtype=np.float64)
            )

        restored = phase2_transport._verified_openblas_runtime(
            1,
            stage="test-after-failed-inverse",
        )
        self.assertEqual(restored["runtime"]["num_threads"], 1)

    def test_transport_inverse_fails_closed_before_linalg_without_runtime(self) -> None:
        with mock.patch.object(
            phase2_transport,
            "threadpool_info",
            return_value=[],
        ), mock.patch.object(
            phase2_transport.np.linalg,
            "inv",
        ) as inverse:
            with self.assertRaisesRegex(RuntimeError, "exactly one live OpenBLAS"):
                phase2_transport._invert_midpoint_with_locked_blas(
                    np.eye(2, dtype=np.float64)
                )
        inverse.assert_not_called()


class SurfaceWorkerRuntimeTests(unittest.TestCase):
    def test_worker_limits_loaded_blas_and_writes_equivalent_atomic_block(self) -> None:
        with tempfile.TemporaryDirectory(prefix="surface-worker-runtime-") as temporary:
            root = Path(temporary)
            matrix_path = root / "matrix.npy"
            block_path = root / "blocks" / "block-000.npz"
            matrix = np.asarray(
                [[0.1 + 0.0j, 0.2 + 0.1j], [0.0 + 0.0j, -0.1 + 0.0j]],
                dtype=np.complex128,
            )
            np.save(matrix_path, matrix)
            program = textwrap.dedent(
                f"""
                import json
                from pathlib import Path
                import numpy as np
                import scipy.linalg as spla
                from threadpoolctl import threadpool_info

                import sys
                sys.path.insert(0, {str(HERE)!r})
                before = [r for r in threadpool_info() if r.get('user_api') == 'blas']
                import hardy_moat_surface_worker as worker
                worker.initialise_surface_worker({str(matrix_path)!r})
                after = [r for r in threadpool_info() if r.get('user_api') == 'blas']
                x_values = np.asarray([-0.2, 0.3], dtype=np.float64)
                y_values = np.asarray([0.15], dtype=np.float64)
                returned = worker.sample_surface_row_block((
                    7, 0, 1, x_values, y_values, {str(block_path)!r}
                ))
                matrix = np.load({str(matrix_path)!r})
                identity = np.eye(matrix.shape[0], dtype=np.complex128)
                expected = np.asarray([[
                    float(spla.svdvals((x + 1j * y_values[0]) * identity - matrix,
                                      check_finite=False)[-1])
                    for x in x_values
                ]])
                with np.load({str(block_path)!r}, allow_pickle=False) as payload:
                    actual = payload['s_block'].copy()
                    block_id = int(payload['block_id'])
                    y_start = int(payload['y_start'])
                    y_stop = int(payload['y_stop'])
                print('THREADING_REPORT=' + json.dumps({{
                    'before': before,
                    'after': after,
                    'returned': returned,
                    'block_id': block_id,
                    'y_start': y_start,
                    'y_stop': y_stop,
                    'max_abs_difference': float(np.max(np.abs(actual - expected))),
                    'temporary_exists': Path({str(block_path.with_suffix('.npz.tmp'))!r}).exists(),
                }}, sort_keys=True))
                """
            )
            environment = dict(os.environ)
            environment.update({key: "4" for key in replay.BLAS_THREAD_ENVIRONMENT})
            completed = subprocess.run(
                [sys.executable, "-B", "-c", program],
                cwd=HERE.parent,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            line = next(
                item
                for item in completed.stdout.splitlines()
                if item.startswith("THREADING_REPORT=")
            )
            report = json.loads(line.removeprefix("THREADING_REPORT="))
            self.assertTrue(report["before"])
            self.assertTrue(
                any(record["num_threads"] > 1 for record in report["before"])
            )
            self.assertTrue(report["after"])
            self.assertTrue(
                all(record["num_threads"] == 1 for record in report["after"])
            )
            self.assertEqual(report["returned"], str(block_path))
            self.assertEqual(
                (report["block_id"], report["y_start"], report["y_stop"]),
                (7, 0, 1),
            )
            self.assertEqual(report["max_abs_difference"], 0.0)
            self.assertFalse(report["temporary_exists"])

    def test_worker_fails_closed_when_no_blas_runtime_is_reported(self) -> None:
        with tempfile.TemporaryDirectory(prefix="surface-worker-no-blas-") as temporary:
            matrix_path = Path(temporary) / "matrix.npy"
            np.save(matrix_path, np.eye(2, dtype=np.complex128))
            with mock.patch.object(
                surface_worker, "threadpool_limits", return_value=object()
            ), mock.patch.object(surface_worker, "threadpool_info", return_value=[]):
                with self.assertRaisesRegex(RuntimeError, "exactly one thread"):
                    surface_worker.initialise_surface_worker(str(matrix_path))


if __name__ == "__main__":
    unittest.main()
