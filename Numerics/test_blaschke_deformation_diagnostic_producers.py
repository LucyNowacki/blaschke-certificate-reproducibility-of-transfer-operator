"""Focused tests for the source-only Blaschke diagnostic producers."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import blaschke_deformation_phase1_diagnostics as phase1
import blaschke_deformation_sampled_schur_diagnostics as sampled_schur
import build_blaschke_deformation_thesis_math_notebook as notebook_builder


MAP_LABEL = "blaschke_mu_0p3"


def _normalise_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _locked_map_label(module: Any) -> str | None:
    for name in ("LOCKED_MAP_LABEL", "MAP_LABEL", "EXPECTED_MAP_LABEL"):
        value = getattr(module, name, None)
        if value is not None:
            return str(value)
    return None


def _symmetric_map_spec(max_power: int = 25) -> dict[str, Any]:
    exact_clusters = [
        {
            "name": "1",
            "value": "1",
            "multiplicity": 1,
            "family": "trivial",
            "power": 0,
        }
    ]
    for family, base, multiplicity in (
        ("alpha", "alpha", 1),
        ("mu", "mu", 2),
    ):
        for power in range(1, max_power + 1):
            exact_clusters.append(
                {
                    "name": f"{family}^{power}",
                    "value": f"{base}**{power}",
                    "multiplicity": multiplicity,
                    "family": family,
                    "power": power,
                }
            )
    return {
        "name": "formula_inverse_branches",
        "params": {
            "label": MAP_LABEL,
            "parameters": {"mu": "0.3", "alpha": "0.65"},
            "branches": [
                {
                    "label": 1,
                    "tau": "x/2 - acos(mu*cos(pi*x/2))/pi",
                    "phi": (
                        "mp.mpf('0.5') - mu/2*sin(pi*x/2)/"
                        "sqrt(1 - mu**2*cos(pi*x/2)**2)"
                    ),
                },
                {
                    "label": 2,
                    "tau": "x/2 + acos(mu*cos(pi*x/2))/pi",
                    "phi": (
                        "mp.mpf('0.5') + mu/2*sin(pi*x/2)/"
                        "sqrt(1 - mu**2*cos(pi*x/2)**2)"
                    ),
                },
            ],
            "exact_clusters": exact_clusters,
            "spectrum_status": "known",
        },
    }


def _reference_clusters(
    map_spec: dict[str, Any], max_clusters: int = 24
) -> tuple[dict[str, Any], ...]:
    bases = {"alpha": 0.65, "mu": 0.3}
    clusters = [
        deepcopy(cluster)
        for cluster in map_spec["params"]["exact_clusters"]
        if cluster["family"] != "trivial"
    ]
    clusters.sort(
        key=lambda cluster: bases[str(cluster["family"])]
        ** int(cluster["power"]),
        reverse=True,
    )
    return tuple(clusters[:max_clusters])


def _cluster_value(cluster: dict[str, Any]) -> float:
    base = {"alpha": 0.65, "mu": 0.3}[str(cluster["family"])]
    return base ** int(cluster["power"])


class FakePhase1Sweep:
    """Return source-shaped rows for every pair requested by the helper."""

    def __init__(
        self,
        *,
        omit_last_pair: bool = False,
        omit_last_target: bool = False,
    ) -> None:
        self.omit_last_pair = omit_last_pair
        self.omit_last_target = omit_last_target
        self.pairs: tuple[tuple[int, int], ...] = ()
        self.kwargs: dict[str, Any] = {}

    def __call__(
        self, pairs: Any, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self.pairs = tuple((int(n_value), int(m_value)) for n_value, m_value in pairs)
        self.kwargs = kwargs
        emitted_pairs = self.pairs[:-1] if self.omit_last_pair else self.pairs
        clusters = tuple(kwargs["reference_clusters"])
        emitted_clusters = clusters[:-1] if self.omit_last_target else clusters
        rows: list[dict[str, Any]] = []
        eigenvalues: list[dict[str, Any]] = []
        jobs: list[dict[str, Any]] = []
        for n_value, m_value in emitted_pairs:
            for cluster_index, cluster in enumerate(emitted_clusters):
                target = _cluster_value(cluster)
                error = (cluster_index + 1) / (1000.0 + 10 * n_value + m_value)
                rows.append(
                    {
                        "N": n_value,
                        "M": m_value,
                        "map_name": MAP_LABEL,
                        "name": cluster["name"],
                        "family": cluster["family"],
                        "power": cluster["power"],
                        "multiplicity": cluster["multiplicity"],
                        "target": target,
                        "error": error,
                        "selected": [
                            target + error
                            for _ in range(int(cluster["multiplicity"]))
                        ],
                        "gram_error": 1.0e-30,
                    }
                )
            for eigenvalue_index in range(n_value):
                eigenvalue = 1.0 / (eigenvalue_index + 1)
                eigenvalues.append(
                    {
                        "N": n_value,
                        "M": m_value,
                        "map_name": MAP_LABEL,
                        "eig_index": eigenvalue_index,
                        "eig": eigenvalue,
                        "eig_re": eigenvalue,
                        "eig_im": 0.0,
                        "eig_abs": eigenvalue,
                    }
                )
            assembly_workers = int(kwargs["assembly_workers"])
            jobs.append(
                {
                    "N": n_value,
                    "M": m_value,
                    "ok": True,
                    "assembly_mode": (
                        "process_row_blocks" if assembly_workers > 1 else "serial"
                    ),
                    "assembly_workers": assembly_workers,
                    "row_block_size": kwargs["row_block_size"],
                    "gram_error": "1e-30",
                }
            )
        return rows, {"jobs": jobs, "eig_rows": eigenvalues}


def _phase1_config() -> phase1.Phase1DiagnosticsConfig:
    return phase1.Phase1DiagnosticsConfig(
        cloud_pairs=((2, 2), (2, 3)),
        fixed_n=3,
        fixed_m_values=(3, 4),
        heat_n_values=(2, 3),
        heat_m_values=(2, 3, 4),
        dps=80,
        max_power=25,
        max_clusters=24,
        assembly_workers=2,
        row_block_size=2,
        progress=False,
    )


def _sampled_geometry() -> dict[str, Any]:
    return {
        "rho": 2.725,
        "r_tau_sampled": 2.29,
        "r_target": 2.47,
        "Phi_sampled": 4.76,
        "geometry_status": "source-generated fake geometry",
    }


class FakeSampledSchurEnvelope:
    def __init__(self, *, mode: str = "valid") -> None:
        self.mode = mode
        self.n_values: tuple[int, ...] = ()
        self.oversampling: int | None = None
        self.geometry: dict[str, float] = {}
        self.kappa: Any = None

    def __call__(
        self,
        *,
        N_values: Any,
        oversampling: int,
        geometry: dict[str, float],
        kappa: Any,
    ) -> pd.DataFrame:
        self.n_values = tuple(int(value) for value in N_values)
        self.oversampling = int(oversampling)
        self.geometry = dict(geometry)
        self.kappa = kappa
        frame = pd.DataFrame(
            {
                "N": self.n_values,
                "M": tuple(value + self.oversampling for value in self.n_values),
                "rho": geometry["rho"],
                "r_tau_sampled": geometry["r_tau"],
                "r": geometry["r"],
                "Phi_sampled": geometry["Phi"],
                "transported_Bmat_diagnostic": [
                    1.0 / value for value in self.n_values
                ],
                "epsilon_schur_diagnostic": [
                    2.0 / value for value in self.n_values
                ],
                "status": [
                    "sampled_schur_diagnostic_not_theorem_certified"
                    for _ in self.n_values
                ],
            }
        )
        if self.mode == "missing_row":
            return frame.iloc[:-1].reset_index(drop=True)
        if self.mode == "wrong_m":
            frame.loc[0, "M"] = int(frame.loc[0, "M"]) - 1
        return frame


class Phase1DiagnosticProducerTests(unittest.TestCase):
    def test_schedule_is_source_generated_and_stale_outputs_are_replaced(self) -> None:
        config = _phase1_config()
        map_spec = _symmetric_map_spec(config.max_power)
        references = _reference_clusters(map_spec, config.max_clusters)
        producer = FakePhase1Sweep()
        expected_pairs = ((2, 2), (2, 3), (2, 4), (3, 3), (3, 4))

        with tempfile.TemporaryDirectory(prefix="phase1-diagnostic-test-") as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            report_path = root / "reports" / phase1.REPORT_FILENAME
            data_dir.mkdir(parents=True)
            report_path.parent.mkdir(parents=True)
            stale_csv = data_dir / "phase1_raw_error_vs_M_N3.csv"
            stale_parquet = data_dir / "phase1_raw_error_vs_M_N3.parquet"
            stale_csv.write_text("stale phase1 seed\n", encoding="utf-8")
            stale_parquet.write_bytes(b"stale phase1 parquet seed\n")
            report_path.write_text('{"stale": true}\n', encoding="utf-8")

            result = phase1.rebuild_phase1_diagnostics(
                config,
                map_spec=map_spec,
                raw_sweep=producer,
                reference_clusters=references,
                data_dir=data_dir,
                report_path=report_path,
            )

            self.assertEqual(producer.pairs, expected_pairs)
            self.assertEqual(producer.kwargs["map_spec"], map_spec)
            self.assertEqual(tuple(producer.kwargs["reference_clusters"]), references)
            self.assertEqual(producer.kwargs["workers"], 1)
            self.assertEqual(producer.kwargs["assembly_workers"], 2)
            self.assertEqual(producer.kwargs["row_block_size"], 2)
            self.assertFalse(producer.kwargs["include_trivial"])
            self.assertTrue(producer.kwargs["full_multiplicity"])
            self.assertEqual(result.csv_paths["phase1_raw_error_vs_M_N3"], stale_csv)
            self.assertNotIn("stale phase1 seed", stale_csv.read_text(encoding="utf-8"))
            self.assertTrue(stale_parquet.is_file())
            self.assertNotEqual(
                stale_parquet.read_bytes(), b"stale phase1 parquet seed\n"
            )
            self.assertIn("phase1_raw_error_vs_M_N3", result.parquet_paths)
            self.assertEqual(len(pd.read_csv(stale_csv)), 2 * config.max_clusters)
            self.assertEqual(len(result.fixed_m_frame), 2 * config.max_clusters)
            self.assertEqual(len(result.heatmap_frame), 5 * config.max_clusters)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["producer_schema"], phase1.PRODUCER_SCHEMA)
            self.assertIs(report["legacy_seed_dependency"], False)
            self.assertIs(report["diagnostic_only"], True)
            self.assertIs(report["theorem_gate"], False)
            self.assertEqual(
                tuple(tuple(pair) for pair in report["schedules"]["unique_pairs"]),
                expected_pairs,
            )

    def test_missing_scheduled_pair_is_rejected(self) -> None:
        config = _phase1_config()
        map_spec = _symmetric_map_spec(config.max_power)
        with tempfile.TemporaryDirectory(
            prefix="phase1-cardinality-test-"
        ) as temporary:
            with self.assertRaisesRegex(
                RuntimeError, r"omitted pairs|cardinality|schedule"
            ):
                phase1.rebuild_phase1_diagnostics(
                    config,
                    map_spec=map_spec,
                    raw_sweep=FakePhase1Sweep(omit_last_pair=True),
                    reference_clusters=_reference_clusters(
                        map_spec, config.max_clusters
                    ),
                    data_dir=Path(temporary) / "data",
                )

    def test_phase1_target_cardinality_is_locked(self) -> None:
        config = _phase1_config()
        map_spec = _symmetric_map_spec(config.max_power)
        with tempfile.TemporaryDirectory(
            prefix="phase1-target-count-test-"
        ) as temporary:
            with self.assertRaisesRegex(
                RuntimeError, r"target packet is incomplete|cardinality|targets"
            ):
                phase1.rebuild_phase1_diagnostics(
                    config,
                    map_spec=map_spec,
                    raw_sweep=FakePhase1Sweep(omit_last_target=True),
                    reference_clusters=_reference_clusters(
                        map_spec, config.max_clusters
                    ),
                    data_dir=Path(temporary) / "data",
                )

    def test_strict_symmetric_map_lock_when_exposed(self) -> None:
        locked_label = _locked_map_label(phase1)
        if locked_label is None:
            return
        self.assertEqual(locked_label, MAP_LABEL)
        config = _phase1_config()
        map_spec = _symmetric_map_spec(config.max_power)
        map_spec["params"]["label"] = "asymmetric_blaschke_degree2"
        with tempfile.TemporaryDirectory(prefix="phase1-map-lock-test-") as temporary:
            with self.assertRaises((ValueError, RuntimeError)):
                phase1.rebuild_phase1_diagnostics(
                    config,
                    map_spec=map_spec,
                    raw_sweep=FakePhase1Sweep(),
                    reference_clusters=_reference_clusters(
                        _symmetric_map_spec(config.max_power), config.max_clusters
                    ),
                    data_dir=Path(temporary) / "data",
                )


class SampledSchurDiagnosticProducerTests(unittest.TestCase):
    def test_locked_schedule_replaces_stale_output_and_records_seed_free_report(
        self,
    ) -> None:
        config = sampled_schur.SampledSchurDiagnosticsConfig()
        producer = FakeSampledSchurEnvelope()
        kappa = lambda n_value, r_value: float(n_value) + float(r_value)

        with tempfile.TemporaryDirectory(prefix="sampled-schur-test-") as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            report_path = root / "reports" / sampled_schur.REPORT_FILENAME
            data_dir.mkdir(parents=True)
            report_path.parent.mkdir(parents=True)
            stale_csv = data_dir / sampled_schur.CSV_FILENAME_TEMPLATE.format(
                map_label=MAP_LABEL
            )
            stale_csv.write_text("stale sampled Schur seed\n", encoding="utf-8")
            report_path.write_text('{"stale": true}\n', encoding="utf-8")

            result = sampled_schur.rebuild_sampled_schur_diagnostics(
                config,
                map_label=MAP_LABEL,
                geometry=_sampled_geometry(),
                sampled_schur_envelope=producer,
                kappa=kappa,
                data_dir=data_dir,
                report_path=report_path,
            )

            self.assertEqual(producer.n_values, config.n_values)
            self.assertEqual(producer.oversampling, 6)
            self.assertEqual(
                producer.geometry,
                {"rho": 2.725, "r_tau": 2.29, "r": 2.47, "Phi": 4.76},
            )
            self.assertIs(producer.kappa, kappa)
            self.assertEqual(tuple(result.frame["N"]), config.n_values)
            self.assertEqual(
                tuple(result.frame["M"]),
                tuple(n_value + 6 for n_value in config.n_values),
            )
            self.assertEqual(result.csv_path, stale_csv)
            self.assertNotIn(
                "stale sampled Schur seed", stale_csv.read_text(encoding="utf-8")
            )
            self.assertEqual(tuple(pd.read_csv(stale_csv)["N"]), config.n_values)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                report["producer_schema"], sampled_schur.PRODUCER_SCHEMA
            )
            self.assertIs(report["legacy_seed_dependency"], False)
            self.assertIs(report["diagnostic_only"], True)
            self.assertIs(report["theorem_gate"], False)
            self.assertEqual(tuple(report["config"]["n_values"]), config.n_values)
            self.assertEqual(report["config"]["oversampling"], 6)

    def test_sampled_row_cardinality_is_locked(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="sampled-cardinality-test-"
        ) as temporary:
            with self.assertRaisesRegex(
                RuntimeError, r"locked dimensions|cardinality|rows"
            ):
                sampled_schur.rebuild_sampled_schur_diagnostics(
                    sampled_schur.SampledSchurDiagnosticsConfig(),
                    map_label=MAP_LABEL,
                    geometry=_sampled_geometry(),
                    sampled_schur_envelope=FakeSampledSchurEnvelope(
                        mode="missing_row"
                    ),
                    kappa=lambda n_value, r_value: 1.0,
                    data_dir=Path(temporary) / "data",
                )

    def test_sampled_m_equals_n_plus_six_is_locked(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sampled-m-lock-test-") as temporary:
            with self.assertRaisesRegex(RuntimeError, r"locked M=N\+6 law"):
                sampled_schur.rebuild_sampled_schur_diagnostics(
                    sampled_schur.SampledSchurDiagnosticsConfig(),
                    map_label=MAP_LABEL,
                    geometry=_sampled_geometry(),
                    sampled_schur_envelope=FakeSampledSchurEnvelope(mode="wrong_m"),
                    kappa=lambda n_value, r_value: 1.0,
                    data_dir=Path(temporary) / "data",
                )

    def test_strict_symmetric_map_lock_when_exposed(self) -> None:
        locked_label = _locked_map_label(sampled_schur)
        if locked_label is None:
            return
        self.assertEqual(locked_label, MAP_LABEL)
        with tempfile.TemporaryDirectory(prefix="sampled-map-lock-test-") as temporary:
            with self.assertRaises((ValueError, RuntimeError)):
                sampled_schur.rebuild_sampled_schur_diagnostics(
                    sampled_schur.SampledSchurDiagnosticsConfig(),
                    map_label="asymmetric_blaschke_degree2",
                    geometry=_sampled_geometry(),
                    sampled_schur_envelope=FakeSampledSchurEnvelope(),
                    kappa=lambda n_value, r_value: 1.0,
                    data_dir=Path(temporary) / "data",
                )


class DiagnosticProducerNotebookBuilderTests(unittest.TestCase):
    def test_helpers_are_inlined_before_their_producers_and_consumers(self) -> None:
        built = notebook_builder.build()
        entries = {
            module_name: (ordinal, source_index, filename)
            for ordinal, (source_index, module_name, filename, _) in enumerate(
                notebook_builder.INLINE_HELPERS, start=1
            )
        }
        selected_modules = (
            "blaschke_deformation_phase1_diagnostics",
            "blaschke_deformation_sampled_schur_diagnostics",
        )
        ordinals = tuple(entries[module_name][0] for module_name in selected_modules)
        notebook_builder.validate_inline_helper_sync(
            built, helper_ordinals=ordinals
        )

        positions = {
            str(cell.get("id")): index for index, cell in enumerate(built["cells"])
        }
        original = json.loads(notebook_builder.SOURCE.read_text(encoding="utf-8"))

        def original_code_id(marker: str) -> str:
            matches = [
                str(cell.get("id"))
                for cell in original["cells"]
                if cell.get("cell_type") == "code"
                and _normalise_source(cell).startswith(marker)
            ]
            self.assertEqual(len(matches), 1, msg=marker)
            return matches[0]

        phase1_ordinal = entries[selected_modules[0]][0]
        sampled_ordinal = entries[selected_modules[1]][0]
        phase1_source_position = positions[f"inline-helper-source-{phase1_ordinal}"]
        sampled_source_position = positions[f"inline-helper-source-{sampled_ordinal}"]
        phase1_producer_position = positions["producer-phase1-diagnostics"]
        sampled_producer_position = positions["producer-sampled-schur-diagnostics"]
        phase1_consumer_position = positions[original_code_id("# Cell 15\n")]
        sampled_consumer_position = positions[original_code_id("# Cell 100\n")]

        self.assertEqual(phase1_source_position + 1, phase1_producer_position)
        self.assertEqual(phase1_producer_position + 1, phase1_consumer_position)
        self.assertEqual(sampled_source_position + 1, sampled_producer_position)
        self.assertLess(sampled_producer_position, sampled_consumer_position)

        for module_name in selected_modules:
            ordinal, _, filename = entries[module_name]
            inline_source = _normalise_source(
                built["cells"][positions[f"inline-helper-source-{ordinal}"]]
            )
            standalone_source = (HERE / filename).read_text(encoding="utf-8")
            self.assertTrue(
                inline_source.startswith(f"%%inline_module {module_name} ")
            )
            self.assertTrue(inline_source.endswith(standalone_source))

        phase1_producer = _normalise_source(
            built["cells"][phase1_producer_position]
        )
        sampled_producer = _normalise_source(
            built["cells"][sampled_producer_position]
        )
        self.assertIn("raw_sweep=run_pair_sweep_mpmath_raw", phase1_producer)
        self.assertIn("map_spec=SELECTED_MAP_SPEC", phase1_producer)
        self.assertIn(
            "sampled_schur_envelope=sampled_schur_envelope", sampled_producer
        )
        self.assertIn("map_label=SELECTED_MAP_LABEL", sampled_producer)


if __name__ == "__main__":
    unittest.main()
