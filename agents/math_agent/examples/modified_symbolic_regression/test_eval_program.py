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
        np.savez(root / "dataset.npz", features=features, psi_scaled=z1 - z2)
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
        expected_complexities = [3, 4, 7, 13, 8, 17]
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
        self.assertEqual(archive["schema_version"], 1)
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
            "def wavefunction(z1, z2):\n"
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
            "def wavefunction(z1, z2, alpha=0.5):\n"
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
        )
        os.environ["LOONGFLOW_SR_DATASET"] = str(
            self.root / "adaptive_dataset.npz"
        )

        candidate = self.root / "adaptive_candidate.py"
        candidate.write_text(
            "import numpy as np\n\n"
            "def wavefunction(z1, z2, alpha=0.0):\n"
            "    return (z1 - z2) * (1 + alpha * np.abs(z1 + z2) ** 2)\n",
            encoding="utf-8",
        )
        result = EVALUATOR.evaluate(candidate)
        self.assertEqual(result["status"], "success", result)
        optimizer = result["artifacts"]["optimizer"]
        self.assertGreaterEqual(optimizer["boundary_expansions"], 1)
        self.assertGreater(result["artifacts"]["fitted_parameters"]["alpha"], 20.0)
        self.assertGreater(result["metrics"]["full_data_fidelity"], 0.999)


if __name__ == "__main__":
    unittest.main()
