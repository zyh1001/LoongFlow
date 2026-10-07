from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np


# The bundled verification runtime may not include SciPy. This deterministic
# stub leaves parameters at their declared initialization while exercising the
# evaluator, AST checks, scoring, and archive behavior. Production still uses
# scipy.optimize.least_squares exactly as imported by eval_program.py.
try:
    import scipy.optimize  # noqa: F401
except ModuleNotFoundError:
    scipy_module = types.ModuleType("scipy")
    optimize_module = types.ModuleType("scipy.optimize")

    def _least_squares_stub(function, initial, **_kwargs):
        vector = np.asarray(initial, dtype=np.float64)
        return types.SimpleNamespace(
            x=vector,
            fun=function(vector),
            success=True,
            nfev=1,
            message="test stub",
        )

    optimize_module.least_squares = _least_squares_stub
    scipy_module.optimize = optimize_module
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.optimize"] = optimize_module


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("fqhe_eval_program", HERE / "eval_program.py")
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


class EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        rng = np.random.default_rng(1234)
        features = rng.normal(size=(64, 4))
        z1 = features[:, 0] - 1j * features[:, 1]
        z2 = features[:, 2] - 1j * features[:, 3]
        np.savez(
            root / "dataset.npz", features=features, psi_scaled=z1 - z2,
            kapa=np.resize([1.0, 10.0], features.shape[0]),
        )
        os.environ["LOONGFLOW_SR_DATASET"] = str(root / "dataset.npz")
        os.environ["LOONGFLOW_SR_PARETO_ARCHIVE"] = str(root / "pareto.json")
        os.environ["LOONGFLOW_SR_PARETO_CSV"] = str(root / "pareto.csv")
        self.root = root

    def tearDown(self) -> None:
        os.environ.pop("LOONGFLOW_SR_DATASET", None)
        os.environ.pop("LOONGFLOW_SR_PARETO_ARCHIVE", None)
        os.environ.pop("LOONGFLOW_SR_PARETO_CSV", None)
        EVALUATOR._DATA_CACHE.clear()
        self.temporary.cleanup()

    def test_all_island_seeds_are_valid_and_distinct(self) -> None:
        expected_complexities = [3, 4, 7, 13, 8, 21]
        expressions = set()
        for island, expected_complexity in enumerate(expected_complexities):
            result = EVALUATOR.evaluate(
                HERE / "initial_programs" / f"island_{island}.py"
            )
            self.assertEqual(result["status"], "success", result)
            self.assertEqual(
                result["metrics"]["expression_complexity"], expected_complexity
            )
            self.assertNotIn("exchange_error", result["metrics"])
            self.assertNotIn("rotation_error", result["metrics"])
            expressions.add(result["artifacts"]["expression"])
        self.assertEqual(len(expressions), 6)

        archive = json.loads(
            Path(os.environ["LOONGFLOW_SR_PARETO_ARCHIVE"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(archive["schema_version"], 2)
        self.assertTrue(archive["pareto_front"])
        self.assertIn("max_accuracy_id", archive["selections"])

        with Path(os.environ["LOONGFLOW_SR_PARETO_CSV"]).open(
            encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(rows)
        self.assertIn("original_expression", rows[0])
        self.assertIn("score", rows[0])
        self.assertIn("fidelity", rows[0])
        self.assertIn("complexity", rows[0])

    def test_decimal_literal_is_rejected(self) -> None:
        candidate = self.root / "decimal_candidate.py"
        candidate.write_text(
            "import numpy as np\n\n"
            "def wavefunction(z1, z2, kapa):\n"
            "    return 0.5 * (z1 - z2)\n",
            encoding="utf-8",
        )
        result = EVALUATOR.evaluate(candidate)
        self.assertEqual(result["status"], "validation_failed")
        self.assertIn("Decimal literals are not allowed", result["summary"])

    def test_fitted_parameter_decimal_default_is_allowed(self) -> None:
        candidate = self.root / "parameter_candidate.py"
        candidate.write_text(
            "import numpy as np\n\n"
            "def wavefunction(z1, z2, kapa, alpha=0.5):\n"
            "    return (z1 - z2) * np.exp(-alpha * (np.abs(z1) ** 2 + np.abs(z2) ** 2))\n",
            encoding="utf-8",
        )
        result = EVALUATOR.evaluate(candidate)
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(result["metrics"]["parameter_count"], 1)

    def test_parameter_bound_expands_when_optimum_is_outside_initial_span(
        self,
    ) -> None:
        with np.load(self.root / "dataset.npz") as dataset:
            features = np.asarray(dataset["features"])
        z1 = features[:, 0] - 1j * features[:, 1]
        z2 = features[:, 2] - 1j * features[:, 3]
        radius = np.abs(z1 + z2) ** 2
        target = (z1 - z2) * (1.0 + 50.0 * radius)
        np.savez(
            self.root / "adaptive_dataset.npz",
            features=features,
            psi_scaled=target,
            kapa=np.resize([1.0, 10.0], features.shape[0]),
        )
        os.environ["LOONGFLOW_SR_DATASET"] = str(
            self.root / "adaptive_dataset.npz"
        )

        candidate = self.root / "adaptive_candidate.py"
        candidate.write_text(
            "import numpy as np\n\n"
            "def wavefunction(z1, z2, kapa, alpha=0.0):\n"
            "    return (z1 - z2) * (1 + alpha * np.abs(z1 + z2) ** 2)\n",
            encoding="utf-8",
        )
        result = EVALUATOR.evaluate(candidate)
        self.assertEqual(result["status"], "success", result)
        optimizer = result["artifacts"]["optimizer"]
        self.assertGreaterEqual(optimizer["boundary_expansions"], 1)
        self.assertGreater(result["artifacts"]["fitted_parameters"]["alpha"], 20.0)
        self.assertGreater(result["metrics"]["full_data_fidelity"], 0.999)

    def test_shared_kapa_dependence_with_independent_system_phases(self) -> None:
        with np.load(self.root / "dataset.npz") as dataset:
            base = np.asarray(dataset["features"])
        # Unequal sizes and normalization scales must not change system weights.
        sizes = [16, 32, 64]
        features = np.concatenate([base[:size] for size in sizes])
        kapa = np.repeat([1.0, 10.0, 100.0], sizes)
        scales = np.repeat([2j, -1000.0, 0.001 - 0.002j], sizes)
        z1 = features[:, 0] - 1j * features[:, 1]
        z2 = features[:, 2] - 1j * features[:, 3]
        target = scales * (z1 - z2) * (
            1 + 1.7 * kapa / (1 + kapa) * np.abs(z1 + z2) ** 2
        )
        path = self.root / "multiple_systems.npz"
        np.savez(path, features=features, psi_scaled=target, kapa=kapa)
        os.environ["LOONGFLOW_SR_DATASET"] = str(path)
        candidate = self.root / "shared_candidate.py"
        candidate.write_text(
            "import numpy as np\n\n"
            "def wavefunction(z1, z2, kapa, alpha=0.1):\n"
            "    strength = alpha * kapa / (1 + kapa)\n"
            "    return (z1 - z2) * (1 + strength * np.abs(z1 + z2) ** 2)\n",
            encoding="utf-8",
        )
        result = EVALUATOR.evaluate(candidate)
        self.assertEqual(result["status"], "success", result)
        self.assertGreater(result["score"], 0.999999)
        self.assertEqual(result["metrics"]["parameter_count"], 1)
        self.assertEqual(result["metrics"]["kapa_systems"], 3)
        self.assertEqual(result["artifacts"]["kapa_values"], [1.0, 10.0, 100.0])
        self.assertAlmostEqual(result["artifacts"]["fitted_parameters"]["alpha"], 1.7, places=4)
        self.assertNotIn("kapa", result["artifacts"]["fitted_parameters"])
        for item in result["artifacts"]["per_kapa"].values():
            self.assertGreater(item["fidelity"], 0.999999)

    def test_initialization_and_seeds_for_fixed_n_six(self) -> None:
        rng = np.random.default_rng(7)
        features = rng.normal(size=(64, 12))
        z = features[:, ::2] - 1j * features[:, 1::2]
        target = np.prod(
            [z[:, i] - z[:, j] for i in range(6) for j in range(i + 1, 6)], axis=0
        )
        path = self.root / "n6.npz"
        np.savez(path, features=features, psi_scaled=target, kapa=100.0)
        os.environ["LOONGFLOW_SR_DATASET"] = str(path)
        initial, seeds = EVALUATOR.initialize_programs(
            (HERE / "initial_program.py").read_text(),
            [(HERE / "initial_programs" / f"island_{i}.py").read_text() for i in range(6)],
        )
        self.assertIn("def wavefunction(z1, z2, z3, z4, z5, z6, kapa)", initial)
        for i, source in enumerate(seeds):
            self.assertNotIn("COORDINATES", source)
            candidate = self.root / f"expanded_{i}.py"
            candidate.write_text(source, encoding="utf-8")
            result = EVALUATOR.evaluate(candidate)
            self.assertEqual(result["status"], "success", result)
            self.assertEqual(result["metrics"]["n_electrons"], 6)

    def test_invalid_system_has_equal_weight_despite_unequal_sample_counts(self) -> None:
        with np.load(self.root / "dataset.npz") as dataset:
            base = dataset["features"]
        features = np.concatenate([base[:8], base])
        target = (features[:, 0] - 1j * features[:, 1]) - (
            features[:, 2] - 1j * features[:, 3]
        )
        path = self.root / "one_invalid_system.npz"
        np.savez(path, features=features, psi_scaled=target,
                 kapa=np.repeat([1.0, 10.0], [8, 64]))
        os.environ["LOONGFLOW_SR_DATASET"] = str(path)
        candidate = self.root / "singular_candidate.py"
        candidate.write_text(
            "import numpy as np\n\n"
            "def wavefunction(z1, z2, kapa):\n"
            "    return (z1 - z2) / (kapa - 1)\n",
            encoding="utf-8",
        )
        result = EVALUATOR.evaluate(candidate)
        self.assertEqual(result["status"], "success", result)
        self.assertAlmostEqual(result["score"], 0.5)
        self.assertAlmostEqual(result["metrics"]["finite_fraction"], 0.5)
        self.assertEqual(result["artifacts"]["per_kapa"]["1.0"]["fidelity"], 0.0)

    def test_kapa_is_required_and_cannot_be_defaulted_or_overwritten(self) -> None:
        candidates = [
            "def wavefunction(z1, z2):\n    return z1 - z2\n",
            "def wavefunction(z1, z2, kapa=1.0):\n    return kapa * (z1 - z2)\n",
            "def wavefunction(z1, z2, kapa):\n    kapa = 1\n    return z1 - z2\n",
        ]
        for source in candidates:
            with self.subTest(source=source):
                path = self.root / "bad_input.py"
                path.write_text("import numpy as np\n\n" + source, encoding="utf-8")
                result = EVALUATOR.evaluate(path)
                self.assertEqual(result["status"], "validation_failed", result)

    def test_dataset_kapa_validation_and_kappa_alias(self) -> None:
        with np.load(self.root / "dataset.npz") as dataset:
            features, target = dataset["features"], dataset["psi_scaled"]
        candidate = HERE / "initial_program.py"
        cases = [
            ({}, "execution_failed"),
            ({"kapa": np.ones((64, 1))}, "execution_failed"),
            ({"kapa": -1.0}, "execution_failed"),
            ({"kapa": 1 + 1j}, "execution_failed"),
            ({"kappa": 3.0}, "success"),
        ]
        for i, (fields, expected) in enumerate(cases):
            with self.subTest(fields=list(fields), expected=expected):
                path = self.root / f"data_case_{i}.npz"
                np.savez(path, features=features, psi_scaled=target, **fields)
                os.environ["LOONGFLOW_SR_DATASET"] = str(path)
                result = EVALUATOR.evaluate(candidate)
                self.assertEqual(result["status"], expected, result)
        self.assertEqual(result["artifacts"]["kapa_values"], [3.0])

    def test_nonfinite_kapa_filter_preserves_coordinate_label_alignment(self) -> None:
        with np.load(self.root / "dataset.npz") as dataset:
            features, target = dataset["features"], dataset["psi_scaled"]
        kapa = np.resize([1.0, 10.0], len(target))
        kapa[[1, 2]] = [np.nan, np.inf]
        mask = np.ones(len(target), dtype=bool)
        mask[3] = False
        path = self.root / "filtered.npz"
        np.savez(path, features=features, psi_real=target.real, psi_imag=target.imag,
                 kapa=kapa, finite_mask=mask)
        data = EVALUATOR._load_evolution_data(path)
        usable = np.isfinite(kapa) & mask
        np.testing.assert_array_equal(data["coordinates"], features[usable])
        np.testing.assert_array_equal(data["target"], target[usable])
        np.testing.assert_array_equal(data["kapa"], kapa[usable])


if __name__ == "__main__":
    unittest.main()
