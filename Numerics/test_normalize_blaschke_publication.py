from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import re
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import blaschke_deformation_contour_certification as contour
import blaschke_deformation_reproducibility as packager
import normalize_blaschke_publication as portability


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PublicationPortabilityTests(unittest.TestCase):
    def test_canonical_plain_and_compressed_closure_is_portable(self) -> None:
        portability.assert_publication_portable(ROOT)

    def test_normalizer_is_idempotent_on_canonical_closure(self) -> None:
        before = {
            relative: _sha256(ROOT / relative)
            for relative in portability.NOTEBOOK_RELATIVES
        }
        self.assertEqual(portability.normalize_publication(ROOT), [])
        self.assertEqual(
            before,
            {
                relative: _sha256(ROOT / relative)
                for relative in portability.NOTEBOOK_RELATIVES
            },
        )

    def test_notebook_normalization_preserves_source_count_and_numbers(self) -> None:
        local_root = "/" + "tmp" + "/example/Numerics/"
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": 17,
                    "metadata": {},
                    "outputs": [
                        {
                            "output_type": "execute_result",
                            "execution_count": 17,
                            "data": {
                                "text/plain": [
                                    "value=3.141592653589793 at "
                                    + local_root
                                    + "outputs/value.csv"
                                ]
                            },
                        }
                    ],
                    "source": ["answer = 3.141592653589793\n", "answer\n"],
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "private environment",
                    "language": "python",
                    "name": "private",
                },
                "language_info": {"name": "python", "version": "3.13.5"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        with tempfile.TemporaryDirectory(prefix="publication-notebook-") as directory:
            path = Path(directory) / "notebook.ipynb"
            path.write_text(json.dumps(notebook), encoding="utf-8")
            self.assertTrue(portability._normalise_notebook(path))
            normalised = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(normalised["cells"][0]["source"], notebook["cells"][0]["source"])
        self.assertEqual(normalised["cells"][0]["execution_count"], 17)
        self.assertEqual(normalised["cells"][0]["outputs"][0]["execution_count"], 17)
        output = normalised["cells"][0]["outputs"][0]["data"]["text/plain"][0]
        self.assertIn("3.141592653589793", output)
        self.assertIn("Numerics/outputs/value.csv", output)
        self.assertEqual(
            normalised["metadata"]["kernelspec"], portability.PORTABLE_KERNELSPEC
        )

    def test_npz_normalization_preserves_every_numeric_array(self) -> None:
        local_root = "/" + "tmp" + "/example/Numerics/"
        with tempfile.TemporaryDirectory(prefix="publication-npz-") as directory:
            path = Path(directory) / "surface.npz"
            np.savez_compressed(
                path,
                x=np.asarray([0.125, 0.25], dtype=np.float64),
                s_min=np.asarray([[1.0e-9, 2.0e-9]], dtype=np.float64),
                A_X_sha256=np.asarray("a" * 64),
                matrix_path=np.asarray(local_root + "outputs/cache/A.npy"),
            )
            with np.load(path, allow_pickle=False) as before:
                numeric_before = {
                    key: np.array(before[key], copy=True) for key in ("x", "s_min")
                }
            self.assertTrue(
                portability._rewrite_npz_strings(
                    path,
                    {
                        "matrix_path": lambda arrays: (
                            "transient-cache/A_X_"
                            f"{str(np.asarray(arrays['A_X_sha256']).item())}.npy"
                        )
                    },
                )
            )
            with np.load(path, allow_pickle=False) as after:
                for key, expected in numeric_before.items():
                    np.testing.assert_array_equal(after[key], expected)
                self.assertEqual(
                    str(after["matrix_path"].item()),
                    "transient-cache/A_X_" + "a" * 64 + ".npy",
                )

    def test_pickle_opcode_scan_rejects_decoded_local_path(self) -> None:
        marker = "/" + "tmp" + "/private-cache/value.pkl"
        with tempfile.TemporaryDirectory(prefix="publication-pickle-") as directory:
            root = Path(directory)
            artifact = root / "checkpoint.pkl.gz"
            with gzip.open(artifact, "wb") as stream:
                pickle.dump({"cache_path": marker}, stream, protocol=5)
            digest = _sha256(artifact)
            (root / "MANIFEST.sha256").write_text(
                f"{digest}  {artifact.name}\n", encoding="utf-8"
            )
            with self.assertRaises(portability.PublicationPortabilityError):
                portability._assert_zero_host_markers(root)

    def test_pickle_opcode_scan_rejects_byte_valued_local_path(self) -> None:
        marker = ("/" + "home" + "/" + "another-user/cache/value.pkl").encode()
        with tempfile.TemporaryDirectory(prefix="publication-pickle-bytes-") as directory:
            root = Path(directory)
            artifact = root / "checkpoint.pkl.gz"
            with gzip.open(artifact, "wb") as stream:
                pickle.dump({"cache_path": marker}, stream, protocol=5)
            (root / "MANIFEST.sha256").write_text(
                f"{_sha256(artifact)}  {artifact.name}\n", encoding="utf-8"
            )
            with self.assertRaises(portability.PublicationPortabilityError):
                portability._assert_zero_host_markers(root)

    def test_compressed_parquet_scan_rejects_local_path(self) -> None:
        text_marker = "/" + "Users" + "/another-user/private/value.csv"
        binary_marker = (
            "C:" + "\\" + "Users" + "\\another-user\\private\\value.csv"
        ).encode()
        with tempfile.TemporaryDirectory(prefix="publication-parquet-") as directory:
            root = Path(directory)
            artifact = root / "records.parquet"
            pq.write_table(
                pa.table(
                    {
                        "artifact_text": [text_marker],
                        "artifact_binary": [binary_marker],
                    }
                ),
                artifact,
                compression="zstd",
            )
            (root / "MANIFEST.sha256").write_text(
                f"{_sha256(artifact)}  {artifact.name}\n", encoding="utf-8"
            )
            with self.assertRaises(
                portability.PublicationPortabilityError
            ) as raised:
                portability._assert_zero_host_markers(root)
            self.assertIn("artifact_text[0]", str(raised.exception))
            self.assertIn("artifact_binary[0]", str(raised.exception))

    def test_notebook_scan_ignores_binary_payload_and_rejects_text(self) -> None:
        marker = "/" + "tmp" + "/private/value.csv"
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "metadata": {},
                    "outputs": [
                        {
                            "data": {
                                "image/png": "opaque-prefix" + marker,
                                "text/plain": "portable display",
                            },
                            "execution_count": 1,
                            "metadata": {},
                            "output_type": "execute_result",
                        }
                    ],
                    "source": ["value = 1\n"],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        with tempfile.TemporaryDirectory(prefix="publication-notebook-scan-") as directory:
            root = Path(directory)
            artifact = root / "notebook.ipynb"
            artifact.write_text(json.dumps(notebook), encoding="utf-8")
            manifest = root / "MANIFEST.sha256"
            manifest.write_text(
                f"{_sha256(artifact)}  {artifact.name}\n", encoding="utf-8"
            )
            portability._assert_zero_host_markers(root)

            notebook["cells"][0]["outputs"][0]["data"]["text/plain"] = marker
            artifact.write_text(json.dumps(notebook), encoding="utf-8")
            with self.assertRaises(portability.PublicationPortabilityError):
                portability._assert_zero_host_markers(root)

    def test_forbidden_host_path_detection_covers_user_profiles(self) -> None:
        values = (
            "/" + "tmp" + "/private/value",
            "/" + "home" + "/another-user/private/value",
            "/" + "Users" + "/another-user/private/value",
            "C:" + "/" + "Users" + "/another-user/private/value",
            "C:" + "\\" + "Users" + "\\another-user\\private\\value",
        )
        for value in values:
            with self.subTest(value=value):
                self.assertTrue(portability._contains_forbidden_host_path(value))
        self.assertFalse(
            portability._contains_forbidden_host_path(
                "outputs/blaschke_deformation_certifier/data/value.csv"
            )
        )

    def test_relative_artifact_resolution_rejects_traversal_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory(prefix="publication-artifacts-") as directory:
            root = Path(directory)
            numerics = root / "Numerics"
            output = numerics / "outputs" / "certificate.json"
            output.parent.mkdir(parents=True)
            output.write_text("{}\n", encoding="utf-8")
            outside = root / "outside.json"
            outside.write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                portability._resolve_bundle_artifact(
                    numerics,
                    label="valid",
                    value="outputs/certificate.json",
                ),
                output,
            )
            with self.assertRaises(portability.PublicationPortabilityError):
                portability._resolve_bundle_artifact(
                    numerics,
                    label="traversal",
                    value="../outside.json",
                )
            link = numerics / "outputs" / "linked.json"
            link.symlink_to(outside)
            with self.assertRaises(portability.PublicationPortabilityError):
                portability._resolve_bundle_artifact(
                    numerics,
                    label="symlink",
                    value="outputs/linked.json",
                )

    def test_kernel_metadata_matches_lock_and_replay_contract(self) -> None:
        lock = (ROOT / "conda-explicit-lock.txt").read_text(encoding="utf-8")
        match = re.search(r"/python-(\d+\.\d+\.\d+)-", lock)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), portability.LOCKED_PYTHON_VERSION)
        self.assertEqual(
            portability.PORTABLE_KERNELSPEC,
            {
                "display_name": "Python 3.13.2 (reproducibility)",
                "language": "python",
                "name": "blaschke-replay",
            },
        )
        replay_text = packager._replay_text()
        self.assertIn(
            f"--name {portability.PORTABLE_KERNELSPEC['name']}", replay_text
        )
        self.assertIn(
            f'--display-name "{portability.PORTABLE_KERNELSPEC["display_name"]}"',
            replay_text,
        )
        self.assertIn(
            f"--kernel-name {portability.PORTABLE_KERNELSPEC['name']}", replay_text
        )
        for relative in portability.NOTEBOOK_RELATIVES:
            notebook = json.loads((ROOT / relative).read_text(encoding="utf-8"))
            self.assertEqual(
                notebook["metadata"]["kernelspec"],
                portability.PORTABLE_KERNELSPEC,
            )
            self.assertEqual(
                notebook["metadata"]["language_info"]["version"],
                portability.LOCKED_PYTHON_VERSION,
            )

    def test_public_replay_instructions_verify_before_cache_loading(self) -> None:
        for text in (
            (ROOT / "README.md").read_text(encoding="utf-8"),
            packager._replay_text(),
        ):
            verification = text.index("Before extracting")
            pickle_warning = text.index("pickle cache")
            preparation = text.index(
                "Numerics/prepare_blaschke_source_only_replay.py"
            )
            self.assertLess(verification, pickle_warning)
            self.assertLess(pickle_warning, preparation)

    def test_direct_cache_reuse_resolves_from_official_notebook_cwd(self) -> None:
        portability.assert_publication_portable(ROOT)
        with tempfile.TemporaryDirectory(prefix="publication-cache-reuse-") as directory:
            replay_root = Path(directory) / "blaschke_deformation_certifier_reproducibility"
            replay_numerics = replay_root / "Numerics"
            replay_output = replay_numerics / portability.OUTPUT_DIRECTORY_RELATIVE
            replay_output.mkdir(parents=True)

            source_names = (
                "blaschke_deformation_contour_certification.py",
                "blaschke_deformation_spectral_certification.py",
            )
            for name in source_names:
                shutil.copy2(HERE / name, replay_numerics / name)

            canonical_report = json.loads(
                (ROOT / portability.SPECTRAL_REPORT_RELATIVE).read_text(
                    encoding="utf-8"
                )
            )
            artifact_relatives = tuple(canonical_report["artifacts"].values())
            for relative in artifact_relatives:
                source = ROOT / "Numerics" / relative
                destination = replay_numerics / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

            canonical_matrix = portability._hardy_paths(
                ROOT, portability.HARDY_STEMS[-1]
            )
            replay_matrix = portability._hardy_paths(
                replay_root, portability.HARDY_STEMS[-1]
            )
            for key in ("payload", "midpoint", "report"):
                replay_matrix[key].parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(canonical_matrix[key], replay_matrix[key])
            epsilon_relative = (
                "outputs/blaschke_deformation_certifier/data/"
                "branch_image_balanced_response_prefactor_candidate_row_N600_M610.csv"
            )
            epsilon = replay_numerics / epsilon_relative
            shutil.copy2(ROOT / "Numerics" / epsilon_relative, epsilon)

            replay_report_path = replay_root / portability.SPECTRAL_REPORT_RELATIVE
            replay_report = json.loads(replay_report_path.read_text(encoding="utf-8"))
            replay_report["source_hashes"] = {
                name: _sha256(replay_numerics / name) for name in source_names
            }
            replay_report_path.write_text(
                json.dumps(replay_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            watched = (
                replay_report_path,
                replay_numerics / replay_report["artifacts"]["certificate"],
                replay_numerics / replay_report["artifacts"]["laurent_witnesses"],
            )
            before = {path: _sha256(path) for path in watched}
            with contextlib.chdir(replay_numerics):
                result = contour.certify_all_target_contours(
                    config=contour.ContourCertificateConfig(
                        N=600,
                        M=610,
                        rho="2.725",
                        r="2.473669807791324",
                        precision_bits=256,
                        flint_threads=24,
                    ),
                    output_dir=replay_output,
                    matrix_payload_path=replay_matrix["payload"],
                    matrix_midpoint_path=replay_matrix["midpoint"],
                    matrix_report_path=replay_matrix["report"],
                    epsilon_certificate_path=epsilon,
                    source_files=tuple(replay_numerics / name for name in source_names),
                    force=False,
                    progress=False,
                )
                self.assertEqual(
                    result["execution_status"], "reused_validated_checkpoint"
                )
                for key in ("certificate", "laurent_witnesses"):
                    value = Path(str(result["artifacts"][key]))
                    self.assertFalse(value.is_absolute())
                    self.assertTrue(value.is_file(), f"unresolved cache artifact {key}")
            self.assertEqual(before, {path: _sha256(path) for path in watched})

    def test_kernel_visual_assets_remain_exact(self) -> None:
        self.assertEqual(
            _sha256(HERE / "blaschke_deformation_notebook_dependency_map.svg"),
            "9248d5609302629b15215b44cb0550c4b54be6bbe04e32f1534340eb395b89d1",
        )
        self.assertEqual(
            _sha256(HERE / "test_notebook_dependency_map.py"),
            "155748516da9ba39d5cb827996ae0fee066a8e98caac1f1da23c2843a5b23b97",
        )


if __name__ == "__main__":
    unittest.main()
