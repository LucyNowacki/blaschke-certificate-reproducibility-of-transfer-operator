"""Focused tests for the post-compute declared generated-output closure gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import prepare_blaschke_source_only_replay as preparation
import run_blaschke_clean_room_replay as replay


HERE = Path(__file__).resolve().parent
POLICY = HERE / "blaschke_source_only_replay_policy.json"


class GeneratedClosureTests(unittest.TestCase):
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

    def _run_mocked_compute(
        self,
        root: Path,
        receipt: dict[str, object],
        *,
        mutate_after_compute=None,
        compute_only: bool = True,
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

        try:
            with mock.patch.object(replay, "_run", side_effect=record):
                result = replay.run_replay(
                    bundle_root=root,
                    kernel_name="fixture-kernel",
                    assembly_workers=2,
                    surface_workers=2,
                    prepare_only=False,
                    published_archive=None,
                    compute_only=compute_only,
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
    ) -> None:
        result, commands, error = self._run_mocked_compute(
            root,
            receipt,
            mutate_after_compute=mutation,
        )
        self.assertIsNone(result)
        self.assertIsInstance(error, preparation.SourceOnlyReplayError)
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

            def inject_escape(bundle: Path, unused: dict[str, object]) -> None:
                path = bundle / preparation.INVENTORY_NAME
                inventory = json.loads(path.read_text(encoding="utf-8"))
                inventory["files"][0]["path"] = "../outside.txt"
                path.write_text(
                    json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
                )

            self._assert_compute_receipt_blocked(
                root, receipt, inject_escape, r"Unsafe source-only inventory path"
            )

    def test_malformed_inventory_blocks_compute_receipt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-closure-malformed-") as temp:
            root, receipt = self._prepared_bundle(Path(temp))

            def inject_malformed(bundle: Path, unused: dict[str, object]) -> None:
                path = bundle / preparation.INVENTORY_NAME
                inventory = json.loads(path.read_text(encoding="utf-8"))
                inventory["files"][0] = "not-a-record"
                path.write_text(
                    json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
                )

            self._assert_compute_receipt_blocked(
                root, receipt, inject_malformed, r"Malformed source-only replay record"
            )

    def test_duplicate_inventory_path_blocks_compute_receipt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-closure-duplicate-") as temp:
            root, receipt = self._prepared_bundle(Path(temp))

            def inject_duplicate(bundle: Path, unused: dict[str, object]) -> None:
                path = bundle / preparation.INVENTORY_NAME
                inventory = json.loads(path.read_text(encoding="utf-8"))
                inventory["files"].append(dict(inventory["files"][0]))
                path.write_text(
                    json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
                )

            self._assert_compute_receipt_blocked(
                root,
                receipt,
                inject_duplicate,
                r"Duplicate source-only inventory path",
            )

    def test_missing_member_blocks_full_replay_before_normalizer(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-closure-normalizer-") as temp:
            root, receipt = self._prepared_bundle(Path(temp))

            def remove_member(bundle: Path, current: dict[str, object]) -> None:
                self._generated_data_path(bundle, current).unlink()

            result, commands, error = self._run_mocked_compute(
                root,
                receipt,
                mutate_after_compute=remove_member,
                compute_only=False,
            )
            self.assertIsNone(result)
            self.assertIsInstance(error, preparation.SourceOnlyReplayError)
            self.assertFalse(
                any(
                    "Numerics/normalize_blaschke_publication.py" in command
                    for command in commands
                )
            )
            self.assertFalse((root / "clean-room-replay-evidence.json").exists())


if __name__ == "__main__":
    unittest.main()
