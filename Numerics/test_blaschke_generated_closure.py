"""Focused tests for the post-compute declared generated-output closure gate."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import prepare_blaschke_source_only_replay as preparation
import normalize_blaschke_publication as portability
import run_blaschke_clean_room_replay as replay


HERE = Path(__file__).resolve().parent
POLICY = HERE / "blaschke_source_only_replay_policy.json"


class GeneratedClosureTests(unittest.TestCase):
    def test_cell106_contour_harness_is_source_only_and_provenance_is_exact(
        self,
    ) -> None:
        provenance = json.loads(
            (HERE / "notebook_cell_provenance.json").read_text(encoding="utf-8")
        )
        record = provenance["entries"]["structural Phase 4 provenance tests"]
        self.assertEqual(
            record["helper_functions"][
                "test_blaschke_deformation_contour_certification."
                "CountAndMoatProvenanceTests"
            ],
            [
                "test_checkpoint_requires_all_seven_reconstruction_records",
                "test_epsilon_loading_requires_every_fresh_phase2_gate",
                "test_failed_laurent_residual_preserves_count_provenance",
                "test_failed_schur_homotopy_blocks_count_transport",
                "test_fixed_flint_precision_restores_after_failure",
                "test_laurent_candidate_generation_is_repeatable",
                "test_laurent_proposal_table_has_recorded_reference_digests",
                "test_logical_source_hashes_ignore_temporary_parent_paths",
                "test_repeated_diagonal_count_survives_nonnormality",
                "test_small_gain_precision_roles_match_legacy_and_explicit_256",
                "test_small_gain_reaggregation_preserves_finite_geometry",
                "test_wrong_expected_multiplicity_does_not_invalidate_count",
                "test_zero_complement_requires_strict_exclusion",
            ],
        )
        self.assertEqual(
            record["file_sources"],
            [
                "Final Deployment/Numerics/"
                "test_blaschke_deformation_contour_certification.py",
                "Final Deployment/Numerics/"
                "blaschke_deformation_contour_certification.py",
                "Final Deployment/Numerics/"
                "blaschke_deformation_spectral_certification.py",
            ],
        )
        self.assertEqual(
            record["prior_result_sources"],
            [
                "inline contour and spectral helper modules",
                "test-owned immutable authenticated 5c0 epsilon/moat/product "
                "constants embedded in "
                "test_blaschke_deformation_contour_certification.py",
            ],
        )
        self.assertEqual(
            record["execution_mode"]["cache_policy"],
            "source-resident fixtures only; no Numerics/outputs reads",
        )

        source_names = (
            "test_blaschke_deformation_contour_certification.py",
            "blaschke_deformation_contour_certification.py",
            "blaschke_deformation_spectral_certification.py",
            "blaschke_deformation_phase2_geometry.py",
        )
        with tempfile.TemporaryDirectory(
            prefix="detached-contour-test-harness-"
        ) as temporary:
            numerics = Path(temporary) / "Numerics"
            numerics.mkdir()
            for name in source_names:
                shutil.copyfile(HERE / name, numerics / name)

            outputs = numerics / "outputs"
            self.assertFalse(outputs.exists())
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["PYTHONNOUSERSITE"] = "1"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "unittest",
                    "-q",
                    "test_blaschke_deformation_contour_certification",
                ],
                cwd=numerics,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
            )

            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            self.assertIn("Ran 17 tests", completed.stderr)
            self.assertIn("OK", completed.stderr)
            self.assertFalse(outputs.exists())

    def _archive_bundle(self, parent: Path) -> Path:
        root = parent / preparation.BUNDLE_ROOT_NAME
        numerics = root / "Numerics"
        data = numerics / "outputs/blaschke_deformation_certifier/data"
        reports = numerics / "outputs/blaschke_deformation_certifier/reports"
        data.mkdir(parents=True)
        reports.mkdir(parents=True)
        files = {
            root / ".final_deployment_generated": "marker\n",
            root / "REPLAY.md": "replay\n",
            numerics / "blaschke_deformation_certifier_template.ipynb": "{}\n",
            numerics / "blaschke_deformation_certifier.ipynb": "generated\n",
            numerics / "blaschke_deformation_certifier_thesis_math.ipynb": (
                "executed\n"
            ),
            numerics / "blaschke_deformation_reproducibility_plan.json": "{}\n",
            numerics / "helper.py": "VALUE = 1\n",
            data / "generated.csv": "x\n1\n",
            reports / "blaschke_deformation_reproducibility_plan.json": "{}\n",
        }
        for path, text in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        shutil.copyfile(POLICY, numerics / POLICY.name)

        staged_paths = sorted(path for path in root.rglob("*") if path.is_file())
        manifest_records = [
            {
                "archive_path": path.relative_to(root).as_posix(),
                "sha256": preparation.sha256_file(path),
            }
            for path in staged_paths
        ]
        inventory = preparation.make_archive_inventory(
            staging_root=root,
            file_records=manifest_records,
        )
        inventory_path = root / preparation.INVENTORY_NAME
        inventory_path.write_text(
            json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
        )
        manifest_records.append(
            {
                "archive_path": preparation.INVENTORY_NAME,
                "sha256": preparation.sha256_file(inventory_path),
            }
        )
        (root / preparation.INTERNAL_MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "bundle_format": (
                        "blaschke-deformation-certifier-reproducibility-v3"
                    ),
                    "files": manifest_records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return root

    def _prepared_bundle(
        self, parent: Path
    ) -> tuple[Path, dict[str, object]]:
        root = self._archive_bundle(parent)
        receipt = preparation.prepare_bundle(root)
        return root, receipt

    def _restore_declared(
        self, root: Path, receipt: dict[str, object]
    ) -> None:
        for value in receipt["removed_paths"]:
            path = root.joinpath(*Path(value).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"reconstructed {value}\n".encode("utf-8"))

    def _generated_data_path(
        self, root: Path, receipt: dict[str, object]
    ) -> Path:
        value = next(
            value
            for value in receipt["removed_paths"]
            if value.endswith("/data/generated.csv")
        )
        return root.joinpath(*Path(value).parts)

    def _mutated_inventory_payload(self, root: Path, mutation) -> tuple[bytes, str]:
        path = root / preparation.INVENTORY_NAME
        inventory = json.loads(path.read_text(encoding="utf-8"))
        mutation(inventory)
        payload = (json.dumps(inventory, indent=2) + "\n").encode("utf-8")
        return payload, hashlib.sha256(payload).hexdigest()

    def _assert_unprepared_authority_failure(
        self,
        root: Path,
        *,
        expected_inventory_sha256: str | None,
        error_type,
        pattern: str,
    ) -> None:
        inventory = json.loads(
            (root / preparation.INVENTORY_NAME).read_text(encoding="utf-8")
        )
        generated = {
            row["path"]: preparation.sha256_file(
                root.joinpath(*Path(row["path"]).parts)
            )
            for row in inventory["files"]
            if row["classification"] in preparation.GENERATED_CLASSES
        }
        self.assertTrue(generated)
        self.assertFalse((root / preparation.RECEIPT_NAME).exists())
        with mock.patch.object(
            replay, "prepare_bundle", wraps=replay.prepare_bundle
        ) as prepare_call, mock.patch.object(
            replay, "check_prepared_bundle", wraps=replay.check_prepared_bundle
        ) as check_call, mock.patch.object(replay, "_run") as producer:
            with self.assertRaisesRegex(error_type, pattern):
                replay.run_replay(
                    bundle_root=root,
                    kernel_name="fixture-kernel",
                    assembly_workers=2,
                    surface_workers=2,
                    prepare_only=False,
                    published_archive=None,
                    compute_only=True,
                    expected_inventory_sha256=expected_inventory_sha256,
                )
        prepare_call.assert_not_called()
        check_call.assert_not_called()
        producer.assert_not_called()
        self.assertFalse((root / preparation.RECEIPT_NAME).exists())
        self.assertFalse((root / "clean-room-compute-only-evidence.json").exists())
        for value, digest in generated.items():
            path = root.joinpath(*Path(value).parts)
            self.assertTrue(path.is_file())
            self.assertEqual(preparation.sha256_file(path), digest)

    def _run_mocked_compute(
        self,
        root: Path,
        receipt: dict[str, object],
        *,
        mutate_after_compute=None,
        compute_only: bool = True,
        expected_inventory_sha256: str | None = None,
        bypass_preparation_authority_preflight: bool = False,
        raw_comparison_receipt: Path | None = None,
    ) -> tuple[dict[str, object] | None, list[list[str]], Exception | None]:
        commands: list[list[str]] = []

        def record(
            command: list[str], *, root: Path, environment: dict[str, str]
        ) -> dict[str, object]:
            commands.append(command)
            if len(commands) == 1:
                self._restore_declared(root, receipt)
            if len(commands) == 5 and mutate_after_compute is not None:
                mutate_after_compute(root, receipt)
            return {
                "command": command,
                "elapsed_seconds": 0.0,
                "returncode": 0,
            }

        # Structural post-compute tests can isolate the terminal gate after the
        # real external-authority preflight has been covered independently.
        preflight = (
            mock.patch.object(
                replay,
                "_authenticate_inventory_before_preparation",
                return_value=(
                    expected_inventory_sha256
                    if expected_inventory_sha256 is not None
                    else str(receipt["inventory_sha256"])
                ),
            )
            if bypass_preparation_authority_preflight
            else contextlib.nullcontext()
        )
        try:
            with preflight, mock.patch.object(replay, "_run", side_effect=record):
                result = replay.run_replay(
                    bundle_root=root,
                    kernel_name="fixture-kernel",
                    assembly_workers=2,
                    surface_workers=2,
                    prepare_only=False,
                    published_archive=None,
                    compute_only=compute_only,
                    expected_inventory_sha256=(
                        expected_inventory_sha256
                        if expected_inventory_sha256 is not None
                        else str(receipt["inventory_sha256"])
                    ),
                    raw_comparison_receipt=raw_comparison_receipt,
                )
        except Exception as exc:  # returned for concise failure-mode assertions
            return None, commands, exc
        return result, commands, None

    def _assert_compute_receipt_blocked(
        self,
        root: Path,
        receipt: dict[str, object],
        mutation,
        pattern: str,
        *,
        expected_inventory_sha256: str | None = None,
        bypass_preparation_authority_preflight: bool = False,
    ) -> None:
        result, commands, error = self._run_mocked_compute(
            root,
            receipt,
            mutate_after_compute=mutation,
            expected_inventory_sha256=expected_inventory_sha256,
            bypass_preparation_authority_preflight=(
                bypass_preparation_authority_preflight
            ),
        )
        self.assertIsNone(result)
        self.assertIsInstance(error, replay.SourceOnlyReplayError)
        self.assertRegex(str(error), pattern)
        self.assertEqual(len(commands), 5)
        self.assertFalse((root / "clean-room-compute-only-evidence.json").exists())
        self.assertFalse(
            any(
                "Numerics/normalize_blaschke_publication.py" in command
                for command in commands
            )
        )

    def test_complete_fixture_binds_closure_and_allows_undeclared_transients(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-closure-pass-") as temp:
            root, receipt = self._prepared_bundle(Path(temp))

            def add_transients(
                bundle: Path, unused_receipt: dict[str, object]
            ) -> None:
                row_block = bundle / (
                    "Numerics/outputs/blaschke_deformation_certifier/data/"
                    "phase4_row_blocks/worker-00.npz"
                )
                cache = bundle / (
                    "Numerics/outputs/blaschke_deformation_certifier/.cache/"
                    "surface.tmp"
                )
                row_block.parent.mkdir(parents=True, exist_ok=True)
                cache.parent.mkdir(parents=True, exist_ok=True)
                row_block.write_bytes(b"transient row block")
                cache.write_bytes(b"transient cache")

            result, commands, error = self._run_mocked_compute(
                root,
                receipt,
                mutate_after_compute=add_transients,
            )
            self.assertIsNone(error)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(len(commands), 5)
            closure = result["generated_closure"]
            expected_paths = receipt["removed_paths"]
            self.assertEqual(closure["generated_paths"], expected_paths)
            self.assertEqual(closure["generated_path_count"], len(expected_paths))
            self.assertEqual(closure["missing"], [])
            self.assertEqual(
                closure["expected_inventory_sha256"],
                receipt["inventory_sha256"],
            )
            payload = json.dumps(
                {"count": len(expected_paths), "paths": expected_paths},
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            self.assertEqual(
                closure["generated_paths_sha256"],
                hashlib.sha256(payload).hexdigest(),
            )
            evidence = json.loads(
                (root / "clean-room-compute-only-evidence.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(evidence["generated_closure"], closure)

    def test_missing_declared_member_blocks_compute_receipt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-closure-missing-") as temp:
            root, receipt = self._prepared_bundle(Path(temp))

            def remove_member(bundle: Path, current: dict[str, object]) -> None:
                self._generated_data_path(bundle, current).unlink()

            self._assert_compute_receipt_blocked(
                root, receipt, remove_member, r"closure is incomplete; missing="
            )

    def test_symlink_declared_member_blocks_compute_receipt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-closure-symlink-") as temp:
            root, receipt = self._prepared_bundle(Path(temp))

            def replace_with_symlink(
                bundle: Path, current: dict[str, object]
            ) -> None:
                member = self._generated_data_path(bundle, current)
                member.unlink()
                member.symlink_to(
                    bundle / "Numerics/blaschke_deformation_certifier_template.ipynb"
                )

            self._assert_compute_receipt_blocked(
                root, receipt, replace_with_symlink, r"Symlinks are forbidden"
            )

    def test_nonregular_declared_member_blocks_compute_receipt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-closure-nonregular-") as temp:
            root, receipt = self._prepared_bundle(Path(temp))

            def replace_with_directory(
                bundle: Path, current: dict[str, object]
            ) -> None:
                member = self._generated_data_path(bundle, current)
                member.unlink()
                member.mkdir()

            self._assert_compute_receipt_blocked(
                root,
                receipt,
                replace_with_directory,
                r"not a regular file",
            )

    def test_inventory_path_escape_blocks_compute_receipt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-closure-escape-") as temp:
            root, receipt = self._prepared_bundle(Path(temp))

            def mutate(inventory: dict[str, object]) -> None:
                inventory["files"][0]["path"] = "../outside.txt"

            payload, forged_sha256 = self._mutated_inventory_payload(root, mutate)

            def inject_escape(bundle: Path, unused: dict[str, object]) -> None:
                (bundle / preparation.INVENTORY_NAME).write_bytes(payload)

            self._assert_compute_receipt_blocked(
                root,
                receipt,
                inject_escape,
                r"Unsafe source-only inventory path",
                expected_inventory_sha256=forged_sha256,
                bypass_preparation_authority_preflight=True,
            )

    def test_malformed_inventory_blocks_compute_receipt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-closure-malformed-") as temp:
            root, receipt = self._prepared_bundle(Path(temp))

            def mutate(inventory: dict[str, object]) -> None:
                inventory["files"][0] = "not-a-record"

            payload, forged_sha256 = self._mutated_inventory_payload(root, mutate)

            def inject_malformed(bundle: Path, unused: dict[str, object]) -> None:
                (bundle / preparation.INVENTORY_NAME).write_bytes(payload)

            self._assert_compute_receipt_blocked(
                root,
                receipt,
                inject_malformed,
                r"Malformed source-only replay record",
                expected_inventory_sha256=forged_sha256,
                bypass_preparation_authority_preflight=True,
            )

    def test_duplicate_inventory_path_blocks_compute_receipt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-closure-duplicate-") as temp:
            root, receipt = self._prepared_bundle(Path(temp))

            def mutate(inventory: dict[str, object]) -> None:
                inventory["files"].append(dict(inventory["files"][0]))

            payload, forged_sha256 = self._mutated_inventory_payload(root, mutate)

            def inject_duplicate(bundle: Path, unused: dict[str, object]) -> None:
                (bundle / preparation.INVENTORY_NAME).write_bytes(payload)

            self._assert_compute_receipt_blocked(
                root,
                receipt,
                inject_duplicate,
                r"Duplicate source-only inventory path",
                expected_inventory_sha256=forged_sha256,
                bypass_preparation_authority_preflight=True,
            )

    def test_missing_expected_inventory_authority_blocks_before_compute(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-closure-no-authority-") as temp:
            root = self._archive_bundle(Path(temp))
            self._assert_unprepared_authority_failure(
                root,
                expected_inventory_sha256=None,
                error_type=ValueError,
                pattern=r"expected-inventory-sha256",
            )

    def test_invalid_expected_inventory_authority_blocks_before_compute(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-closure-bad-authority-") as temp:
            root = self._archive_bundle(Path(temp))
            self._assert_unprepared_authority_failure(
                root,
                expected_inventory_sha256="not-a-sha256",
                error_type=ValueError,
                pattern=r"expected-inventory-sha256",
            )

    def test_expected_inventory_digest_mismatch_blocks_before_preparation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-closure-wrong-digest-") as temp:
            root = self._archive_bundle(Path(temp))
            self._assert_unprepared_authority_failure(
                root,
                expected_inventory_sha256="0" * 64,
                error_type=replay.SourceOnlyReplayError,
                pattern=r"refusing source-only preparation",
            )

    def test_forged_receipt_count_blocks_compute_receipt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-closure-receipt-count-") as temp:
            root, receipt = self._prepared_bundle(Path(temp))
            receipt["removed_file_count"] = int(receipt["removed_file_count"]) + 1
            (root / preparation.RECEIPT_NAME).write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
            self._assert_compute_receipt_blocked(
                root,
                receipt,
                lambda bundle, current: None,
                r"receipt does not match the inventory",
            )

    def test_forged_receipt_inventory_hash_blocks_compute_receipt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-closure-receipt-hash-") as temp:
            root, receipt = self._prepared_bundle(Path(temp))
            authenticated_sha256 = str(receipt["inventory_sha256"])
            receipt["inventory_sha256"] = "0" * 64
            (root / preparation.RECEIPT_NAME).write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
            self._assert_compute_receipt_blocked(
                root,
                receipt,
                lambda bundle, current: None,
                r"receipt does not authenticate the inventory",
                expected_inventory_sha256=authenticated_sha256,
            )

    def test_inventory_count_drift_blocks_compute_receipt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-closure-count-drift-") as temp:
            root, receipt = self._prepared_bundle(Path(temp))

            def mutate(inventory: dict[str, object]) -> None:
                inventory["counts"]["generated_evidence"] += 1

            payload, forged_sha256 = self._mutated_inventory_payload(root, mutate)

            def inject_count_drift(bundle: Path, unused: dict[str, object]) -> None:
                (bundle / preparation.INVENTORY_NAME).write_bytes(payload)

            self._assert_compute_receipt_blocked(
                root,
                receipt,
                inject_count_drift,
                r"inventory counts are malformed or inconsistent",
                expected_inventory_sha256=forged_sha256,
                bypass_preparation_authority_preflight=True,
            )

    def test_internal_manifest_binding_drift_blocks_compute_receipt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-closure-manifest-drift-") as temp:
            root, receipt = self._prepared_bundle(Path(temp))

            def inject_manifest_drift(
                bundle: Path, unused: dict[str, object]
            ) -> None:
                path = bundle / preparation.INTERNAL_MANIFEST_NAME
                manifest = json.loads(path.read_text(encoding="utf-8"))
                binding = next(
                    row
                    for row in manifest["files"]
                    if row.get("archive_path") == preparation.INVENTORY_NAME
                )
                binding["sha256"] = "0" * 64
                path.write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )

            self._assert_compute_receipt_blocked(
                root,
                receipt,
                inject_manifest_drift,
                r"internal manifest does not authenticate",
            )

    def test_expected_path_fingerprint_drift_blocks_compute_receipt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-closure-path-drift-") as temp:
            root, receipt = self._prepared_bundle(Path(temp))

            def inject_path_drift(bundle: Path, unused: dict[str, object]) -> None:
                path = bundle / preparation.INVENTORY_NAME
                inventory = json.loads(path.read_text(encoding="utf-8"))
                row = next(
                    row
                    for row in inventory["files"]
                    if row.get("classification") in preparation.GENERATED_CLASSES
                )
                row["path"] = str(row["path"]) + ".drift"
                path.write_text(
                    json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
                )

            self._assert_compute_receipt_blocked(
                root,
                receipt,
                inject_path_drift,
                r"externally authenticated expected SHA-256",
            )

    def test_full_composite_metadata_rewrite_cannot_forge_closure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-closure-composite-") as temp:
            root, receipt = self._prepared_bundle(Path(temp))

            def rewrite_all_local_authorities(
                bundle: Path, unused: dict[str, object]
            ) -> None:
                inventory_path = bundle / preparation.INVENTORY_NAME
                inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
                row_index = next(
                    index
                    for index, row in enumerate(inventory["files"])
                    if row.get("classification") in preparation.GENERATED_CLASSES
                )
                removed_row = inventory["files"].pop(row_index)
                classification = removed_row["classification"]
                inventory["counts"][classification] -= 1
                inventory_path.write_text(
                    json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
                )
                forged_inventory_sha256 = preparation.sha256_file(inventory_path)

                receipt_path = bundle / preparation.RECEIPT_NAME
                forged_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                forged_receipt["inventory_sha256"] = forged_inventory_sha256
                forged_receipt["removed_paths"].remove(removed_row["path"])
                forged_receipt["removed_file_count"] -= 1
                receipt_path.write_text(
                    json.dumps(forged_receipt, indent=2) + "\n", encoding="utf-8"
                )

                manifest_path = bundle / preparation.INTERNAL_MANIFEST_NAME
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                binding = next(
                    row
                    for row in manifest["files"]
                    if row.get("archive_path") == preparation.INVENTORY_NAME
                )
                binding["sha256"] = forged_inventory_sha256
                manifest_path.write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )

            self._assert_compute_receipt_blocked(
                root,
                receipt,
                rewrite_all_local_authorities,
                r"externally authenticated expected SHA-256",
            )

    def test_missing_member_blocks_full_replay_before_normalizer(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-closure-normalizer-") as temp:
            root, receipt = self._prepared_bundle(Path(temp))
            raw_comparison_receipt = Path(temp) / "raw-comparison-pass.json"
            raw_comparison_receipt.write_text(
                json.dumps(
                    {
                        "schema": portability.RAW_COMPARISON_RECEIPT_SCHEMA,
                        "phase": "raw-pre-normalization",
                        "status": "PASS",
                        "comparison_run": True,
                        "normalization_run": False,
                        "provenance_refresh_run": False,
                        "production_identity_checked": True,
                        "failures": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            def remove_member(bundle: Path, current: dict[str, object]) -> None:
                self._generated_data_path(bundle, current).unlink()

            result, commands, error = self._run_mocked_compute(
                root,
                receipt,
                mutate_after_compute=remove_member,
                compute_only=False,
                raw_comparison_receipt=raw_comparison_receipt,
            )
            self.assertIsNone(result)
            self.assertIsInstance(error, replay.SourceOnlyReplayError)
            self.assertFalse(
                any(
                    "Numerics/normalize_blaschke_publication.py" in command
                    for command in commands
                )
            )
            self.assertFalse((root / "clean-room-replay-evidence.json").exists())


if __name__ == "__main__":
    unittest.main()
