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
import build_blaschke_deformation_certifier as certifier_builder
import blaschke_deformation_phase2_transport as phase2_transport
import hardy_moat_surface_worker as surface_worker
import run_blaschke_clean_room_replay as replay


def _plotting_replacement_source(record: object) -> str:
    if isinstance(record, str):
        return record
    if isinstance(record, dict) and isinstance(record.get("source_lines"), list):
        return "\n".join(record["source_lines"]).rstrip() + "\n"
    raise AssertionError("Unexpected plotting replacement record.")


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
        authoritative = _plotting_replacement_source(replacements["6a823024"])
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

    def test_generated_cell_5_has_guest_portable_colab_bootstrap(self) -> None:
        replacements = json.loads(
            (HERE / "plotting_cell_replacements.json").read_text(encoding="utf-8")
        )
        authoritative = _plotting_replacement_source(replacements["6a823024"])

        for marker in (
            "import google.colab as _google_colab",
            "os.environ['BLASCHKE_COLAB'] = '1'",
            "BLASCHKE_DRIVE_ROOT = _colab_mount / 'MyDrive' / 'Thesis Numerical'",
            "BLASCHKE_DRIVE_NUMERICS = BLASCHKE_DRIVE_ROOT / 'Numerics'",
            "BLASCHKE_PUBLIC_REPOSITORY = 'LucyNowacki/blaschke-certificate-reproducibility-of-transfer-operator'",
            "BLASCHKE_PUBLIC_REF = 'main'",
            "https://raw.githubusercontent.com/",
            "urllib.request.urlretrieve",
            "BLASCHKE_RUNTIME_NUMERICS / name",
            "'requirements-colab.txt'",
            "sys.path.insert(0, _colab_import_text)",
            "os.chdir(BLASCHKE_RUNTIME_NUMERICS)",
            "Colab runtime source:",
            "'flint': 'python-flint==0.8.0'",
            "'pyarrow': 'pyarrow>=14,<25'",
            "'threadpoolctl': 'threadpoolctl>=3.5,<4'",
            "'tqdm': 'tqdm>=4.66,<5'",
            "if importlib.util.find_spec(module_name) is None",
        ):
            self.assertIn(marker, authoritative)
        self.assertNotIn("_colab_drive.mount", authoritative)
        self.assertNotIn("from google.colab import drive", authoritative)
        self.assertNotIn("shutil.copytree", authoritative)

        generated, _ = notebook_builder.build_curated()
        generated_cell = next(
            cell for cell in generated["cells"] if cell.get("id") == "6a823024"
        )
        generated_source = "".join(generated_cell.get("source", []))
        self.assertIn("#5N", generated_source)
        self.assertIn(authoritative, generated_source)

    def test_generated_cell_5_constrains_an_already_loaded_blas_runtime(self) -> None:
        scope_threads = max(1, min(24, os.cpu_count() or 1))
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
            replacement_record = json.loads(
                (Path({str(HERE)!r}) / 'plotting_cell_replacements.json').read_text(
                    encoding='utf-8'
                )
            )['6a823024']
            replacement = (
                replacement_record
                if isinstance(replacement_record, str)
                else '\\n'.join(replacement_record['source_lines']).rstrip() + '\\n'
            )
            namespace = {{'__name__': '__main__'}}
            exec(replacement, namespace)
            after_first = [r for r in threadpool_info() if r.get('user_api') == 'blas']
            exec(replacement, namespace)
            after_second = [r for r in threadpool_info() if r.get('user_api') == 'blas']
            scope_threads = max(1, min(24, os.cpu_count() or 1))
            with threadpool_limits(limits=scope_threads, user_api='blas'):
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
        environment.update(
            {key: str(scope_threads) for key in replay.BLAS_THREAD_ENVIRONMENT}
        )
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
        self.assertTrue(
            any(row["num_threads"] == scope_threads for row in report["before"])
        )
        self.assertTrue(all(row["num_threads"] == 1 for row in report["after_first"]))
        self.assertTrue(all(row["num_threads"] == 1 for row in report["after_second"]))
        self.assertTrue(
            all(row["num_threads"] == scope_threads for row in report["inside"])
        )
        self.assertTrue(all(row["num_threads"] == 1 for row in report["restored"]))
        self.assertTrue(report["limiter_retained"])


class AdaptiveNotebookWorkerTests(unittest.TestCase):
    @staticmethod
    def _cell_6_source(path: Path) -> str:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cell = next(
            item for item in notebook["cells"] if item.get("id") == "code-bffea704"
        )
        return "".join(cell.get("source", []))

    def test_source_and_template_use_the_same_adaptive_worker_policy(self) -> None:
        source = self._cell_6_source(HERE / "blaschke_deformation_certifier.ipynb")
        template = self._cell_6_source(
            HERE / "blaschke_deformation_certifier_template.ipynb"
        )
        self.assertEqual(source, template)
        self.assertIn(
            "CERTIFIER_PROCESS_WORKERS = max(1, min(24, os.cpu_count() or 1))",
            source,
        )
        self.assertNotIn("requires 24 process workers", source)

    def test_cell_6_runs_with_two_or_one_available_cpus(self) -> None:
        source = self._cell_6_source(HERE / "blaschke_deformation_certifier.ipynb")
        for cpu_count, expected in ((2, 2), (None, 1)):
            with self.subTest(cpu_count=cpu_count), tempfile.TemporaryDirectory(
                prefix="adaptive-notebook-workers-"
            ) as temporary, mock.patch("os.cpu_count", return_value=cpu_count), mock.patch.dict(
                os.environ, {}, clear=False
            ):
                namespace = {"OUTPUT_DIR": Path(temporary) / "outputs"}
                exec(source, namespace)
                self.assertEqual(namespace["CERTIFIER_PROCESS_WORKERS"], expected)
                self.assertEqual(os.environ["MPMATH_PF_ASSEMBLY_WORKERS"], str(expected))
                self.assertTrue(
                    (Path(temporary) / "outputs" / "blaschke_deformation_certifier").is_dir()
                )

    def test_generated_cell_6_provenance_describes_adaptive_workers(self) -> None:
        generated, _ = notebook_builder.build_curated()
        cell = next(
            item for item in generated["cells"] if item.get("id") == "code-bffea704"
        )
        source = "".join(cell.get("source", []))
        self.assertIn(
            "parallelism: configures between 1 and 24 process workers according "
            "to the available CPU count",
            source,
        )
        self.assertIn(
            "CERTIFIER_PROCESS_WORKERS = max(1, min(24, os.cpu_count() or 1))",
            source,
        )


class ScopedPortableKernelTests(unittest.TestCase):
    def test_transport_inverse_uses_available_threads_and_restores_outer_one(
        self,
    ) -> None:
        self.assertEqual(
            phase2_transport.INVERSE_BLAS_THREADS,
            max(1, min(24, os.cpu_count() or 1)),
        )
        scope_threads = min(2, phase2_transport.INVERSE_BLAS_THREADS)
        with phase2_transport.threadpool_limits(limits=1, user_api="blas"), mock.patch.object(
            phase2_transport,
            "INVERSE_BLAS_THREADS",
            scope_threads,
        ):
            inverse, evidence = phase2_transport._invert_midpoint_with_locked_blas(
                np.eye(3, dtype=np.float64)
            )

        np.testing.assert_array_equal(inverse, np.eye(3, dtype=np.float64))
        self.assertEqual(evidence["scope_threads"], scope_threads)
        self.assertEqual(
            evidence["scope_thread_policy"],
            "max(1,min(24,os.cpu_count() or 1))",
        )
        self.assertEqual(evidence["outer_threads"], 1)
        self.assertGreaterEqual(evidence["outer_before"]["runtime_count"], 1)
        self.assertTrue(
            all(
                runtime["num_threads"] == 1
                for runtime in evidence["outer_before"]["runtimes"]
            )
        )
        self.assertTrue(
            all(
                runtime["num_threads"] == scope_threads
                for runtime in evidence["inside"]["runtimes"]
            )
        )
        self.assertTrue(
            all(
                runtime["num_threads"] == 1
                for runtime in evidence["outer_after"]["runtimes"]
            )
        )

    def test_transport_inverse_restores_outer_one_after_linalg_failure(self) -> None:
        with phase2_transport.threadpool_limits(limits=1, user_api="blas"):
            with self.assertRaises(np.linalg.LinAlgError):
                phase2_transport._invert_midpoint_with_locked_blas(
                    np.zeros((2, 2), dtype=np.float64)
                )

            restored = phase2_transport._verified_openblas_runtime(
                1,
                stage="test-after-failed-inverse",
            )
            self.assertTrue(
                all(
                    runtime["num_threads"] == 1
                    for runtime in restored["runtimes"]
                )
            )

    def test_transport_inverse_fails_closed_before_linalg_without_runtime(self) -> None:
        with mock.patch.object(
            phase2_transport,
            "threadpool_info",
            return_value=[],
        ), mock.patch.object(
            phase2_transport.np.linalg,
            "inv",
        ) as inverse:
            with self.assertRaisesRegex(RuntimeError, "at least one usable loaded BLAS"):
                phase2_transport._invert_midpoint_with_locked_blas(
                    np.eye(2, dtype=np.float64)
                )
        inverse.assert_not_called()

    def test_runtime_evidence_accepts_multiple_blas_implementations(self) -> None:
        runtimes = [
            {
                "user_api": "blas",
                "internal_api": "openblas",
                "version": "0.3.29",
                "threading_layer": "pthreads",
                "architecture": "SkylakeX",
                "prefix": "libopenblas",
                "num_threads": 2,
            },
            {
                "user_api": "blas",
                "internal_api": "mkl",
                "version": "2025.0",
                "threading_layer": "intel",
                "architecture": None,
                "prefix": "libmkl_rt",
                "num_threads": 2,
            },
        ]
        with mock.patch.object(
            phase2_transport,
            "threadpool_info",
            return_value=runtimes,
        ):
            evidence = phase2_transport._verified_openblas_runtime(
                2,
                stage="synthetic-multiple-runtime-test",
            )

        self.assertEqual(evidence["runtime_count"], 2)
        self.assertEqual(
            {runtime["internal_api"] for runtime in evidence["runtimes"]},
            {"openblas", "mkl"},
        )

    def test_runtime_evidence_rejects_one_misconstrained_loaded_runtime(self) -> None:
        runtimes = [
            {"user_api": "blas", "internal_api": "openblas", "num_threads": 2},
            {"user_api": "blas", "internal_api": "mkl", "num_threads": 1},
        ]
        with mock.patch.object(
            phase2_transport,
            "threadpool_info",
            return_value=runtimes,
        ), self.assertRaisesRegex(RuntimeError, "every loaded BLAS runtime"):
            phase2_transport._verified_openblas_runtime(
                2,
                stage="synthetic-mismatched-runtime-test",
            )

    def test_builder_emits_portable_kappa_runtime_contract(self) -> None:
        source = certifier_builder.KAPPA_T_NUMERIC_SCOPED_BLAS
        for marker in (
            "scope_threads = max(1, min(24, os.cpu_count() or 1))",
            "requires every loaded BLAS runtime",
            "numerics1-portable-scoped-blas-runtime-v2",
            "'outer_before': [portable(record) for record in outer_before]",
            "'inside': [portable(record) for record in inside]",
            "'outer_after': [portable(record) for record in outer_after]",
        ):
            self.assertIn(marker, source)
        for forbidden in (
            "OpenBLAS 0.3.30",
            "pthreads Haswell",
            "threadpool_limits(limits=24, user_api='blas')",
        ):
            self.assertNotIn(forbidden, source)

    def test_transport_still_fails_closed_when_arb_residual_is_not_below_one(
        self,
    ) -> None:
        config = phase2_transport.Phase2TransportConfig(
            N=1,
            r="1.1",
            rho="1.2",
            r_tau="1.1",
            geometry_configuration_digest="a" * 64,
            precision_bits=64,
            flint_threads=1,
        )
        runtime_evidence = {
            "schema": "test-portable-blas-runtime",
            "scope_threads": 1,
        }
        with tempfile.TemporaryDirectory(prefix="transport-residual-gate-") as temporary, mock.patch.object(
            phase2_transport,
            "_invert_midpoint_with_locked_blas",
            return_value=(np.eye(1, dtype=np.float64), runtime_evidence),
        ), mock.patch.object(
            phase2_transport,
            "_certify_transport_by_row_blocks",
            return_value=(
                phase2_transport.arb(1),
                phase2_transport.arb(1),
                phase2_transport.arb(1),
                1,
            ),
        ), self.assertRaisesRegex(ArithmeticError, "delta less than one"):
            root = Path(temporary)
            phase2_transport.certify_transport(
                config,
                data_dir=root / "data",
                report_dir=root / "reports",
                require_canonical_radius=False,
            )

    def test_transport_still_fails_closed_when_arb_lower_bound_is_not_positive(
        self,
    ) -> None:
        config = phase2_transport.Phase2TransportConfig(
            N=1,
            r="1.1",
            rho="1.2",
            r_tau="1.1",
            geometry_configuration_digest="b" * 64,
            precision_bits=64,
            flint_threads=1,
        )
        runtime_evidence = {
            "schema": "test-portable-blas-runtime",
            "scope_threads": 1,
        }
        with tempfile.TemporaryDirectory(prefix="transport-positivity-gate-") as temporary, mock.patch.object(
            phase2_transport,
            "_invert_midpoint_with_locked_blas",
            return_value=(np.eye(1, dtype=np.float64), runtime_evidence),
        ), mock.patch.object(
            phase2_transport,
            "_certify_transport_by_row_blocks",
            return_value=(
                phase2_transport.arb(1),
                phase2_transport.arb("inf"),
                phase2_transport.arb("0.5"),
                1,
            ),
        ), self.assertRaisesRegex(ArithmeticError, "minimum Gram eigenvalue"):
            root = Path(temporary)
            phase2_transport.certify_transport(
                config,
                data_dir=root / "data",
                report_dir=root / "reports",
                require_canonical_radius=False,
            )


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
