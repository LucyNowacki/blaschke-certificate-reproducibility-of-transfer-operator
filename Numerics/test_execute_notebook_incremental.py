"""Focused tests for environment-explicit incremental notebook execution."""

from __future__ import annotations

import os
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import nbformat

_EXECUTOR_PATH = Path(__file__).with_name("execute_notebook_incremental.py")
_SPEC = importlib.util.spec_from_file_location(
    "_blaschke_execute_notebook_incremental_test", _EXECUTOR_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load executor from {_EXECUTOR_PATH}.")
executor = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(executor)


class IncrementalExecutorTests(unittest.TestCase):
    def test_explicit_replay_kernel_and_timing_policy_reach_nbclient(self) -> None:
        notebook = nbformat.v4.new_notebook(
            metadata={
                "kernelspec": {
                    "name": "private-kernel",
                    "display_name": "Python (private-kernel)",
                    "language": "python",
                }
            }
        )
        client = mock.Mock()
        with mock.patch.object(executor.nbformat, "read", return_value=notebook), \
             mock.patch.object(executor.nbformat, "write"), \
             mock.patch.object(executor, "NotebookClient", return_value=client) as ctor:
            executor.execute(
                Path("certificate.ipynb"),
                kernel_name="blaschke-replay",
                record_timing=False,
            )

        self.assertEqual(ctor.call_args.kwargs["kernel_name"], "blaschke-replay")
        self.assertIs(ctor.call_args.kwargs["record_timing"], False)
        client.execute.assert_called_once()

    def test_environment_kernel_overrides_stored_notebook_kernel(self) -> None:
        notebook = nbformat.v4.new_notebook(
            metadata={
                "kernelspec": {
                    "name": "private-kernel",
                    "display_name": "Python (private-kernel)",
                    "language": "python",
                }
            }
        )
        client = mock.Mock()
        with mock.patch.dict(os.environ, {"BLASCHKE_KERNEL_NAME": "fresh-env"}), \
             mock.patch.object(executor.nbformat, "read", return_value=notebook), \
             mock.patch.object(executor.nbformat, "write"), \
             mock.patch.object(executor, "NotebookClient", return_value=client) as ctor:
            executor.execute(Path("certificate.ipynb"))

        self.assertEqual(ctor.call_args.kwargs["kernel_name"], "fresh-env")

    def test_portable_stored_kernel_is_fail_closed_fallback(self) -> None:
        notebook = nbformat.v4.new_notebook(
            metadata={
                "kernelspec": {
                    "name": "blaschke-replay",
                    "display_name": "Python 3.13.2 (reproducibility)",
                    "language": "python",
                }
            }
        )
        client = mock.Mock()
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(executor.nbformat, "read", return_value=notebook), \
             mock.patch.object(executor.nbformat, "write"), \
             mock.patch.object(executor, "NotebookClient", return_value=client) as ctor:
            executor.execute(Path("certificate.ipynb"))

        self.assertEqual(ctor.call_args.kwargs["kernel_name"], "blaschke-replay")
        client.execute.assert_called_once()


if __name__ == "__main__":
    unittest.main()
