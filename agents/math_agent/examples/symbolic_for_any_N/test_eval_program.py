from __future__ import annotations

import argparse
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
        _, seeds = EVALUATOR.initialize_programs(
            (HERE / "initial_program.py").read_text(encoding="utf-8"),
            [(HERE / "initial_programs" / f"island_{i}.py").read_text(
                encoding="utf-8"
            ) for i in range(6)],
        )
        for island, expected_complexity in enumerate(expected_complexities):
            candidate = self.root / f"island_{island}.py"
            candidate.write_text(seeds[island], encoding="utf-8")
            result = EVALUATOR.evaluate(candidate)
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

    def test_dataset_initialization_and_evaluation_for_fixed_n(self) -> None:
        initial_template = (HERE / "initial_program.py").read_text(encoding="utf-8")
        templates = [
            (HERE / "initial_programs" / f"island_{i}.py").read_text(encoding="utf-8")
            for i in range(6)
        ]
        for n_electrons in (1, 3, 6, 12):
            with self.subTest(n_electrons=n_electrons):
                features = np.random.default_rng(1234).normal(
                    size=(32, 2 * n_electrons)
                )
                z = features[:, ::2] - 1j * features[:, 1::2]
                target = np.ones(32, dtype=np.complex128)
                for i in range(n_electrons):
                    for j in range(i + 1, n_electrons):
                        target *= z[:, i] - z[:, j]
                dataset = self.root / f"n{n_electrons}.npz"
                np.savez(dataset, features=features, psi_scaled=target)
                os.environ["LOONGFLOW_SR_DATASET"] = str(dataset)
                initial_code, seeds = EVALUATOR.initialize_programs(initial_template, templates)
                self.assertEqual(initial_code, seeds[0])
                for island, source in enumerate(seeds):
                    candidate = self.root / f"island_{island}.py"
                    candidate.write_text(source, encoding="utf-8")
                    result = EVALUATOR.evaluate(candidate)
                    self.assertEqual(result["status"], "success", result)
                    self.assertEqual(result["artifacts"]["n_electrons"], n_electrons)
                    if island == 0:
                        self.assertAlmostEqual(result["score"], 1.0)
                if n_electrons > 2:
                    candidate.write_text("import numpy as np\ndef wavefunction(z1, z2):\n"
                                         "    return z1 - z2\n", encoding="utf-8")
                    self.assertEqual(EVALUATOR.evaluate(candidate)["status"], "validation_failed")

    def test_existing_runner_initializes_templates_without_rewriting_files(self) -> None:
        import yaml

        from agents.math_agent.math_evolve_agent import MathPESAgent

        features = np.random.default_rng(1234).normal(size=(32, 12))
        dataset = self.root / "runner_n6.npz"
        np.savez(dataset, features=features, psi_scaled=np.ones(32))
        os.environ["LOONGFLOW_SR_DATASET"] = str(dataset)
        config_path = HERE / "task_config.yaml"
        original_config = config_path.read_text(encoding="utf-8")
        config = yaml.safe_load(original_config)
        # Exercise the same custom merge used by run_math.sh, without starting LLMs.
        args = argparse.Namespace(
            config=str(config_path), eval_file=str(HERE / "eval_program.py"),
            initial_file=str(HERE / "initial_program.py"),
        )
        merged = MathPESAgent.__new__(MathPESAgent)._merge_custom_configs(args, config)
        initial_code = merged["evolve"]["initial_code"]
        self.assertIn("def wavefunction(z1, z2, z3, z4, z5, z6):", initial_code)
        self.assertNotIn("PAIR_PRODUCT", initial_code)
        self.assertEqual(config["evolve"]["database"]["num_islands"], 4)
        self.assertEqual(len(merged["evolve"]["metadata"]["island_initial_codes"]), 4)
        self.assertEqual(config_path.read_text(encoding="utf-8"), original_config)
        self.assertIn("COORDINATES", (HERE / "initial_program.py").read_text(encoding="utf-8"))
        candidate = self.root / "runner_initial.py"
        candidate.write_text(initial_code, encoding="utf-8")
        result = EVALUATOR.evaluate(candidate)
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(result["artifacts"]["n_electrons"], 6)

    def test_explicit_initial_program_is_preserved(self) -> None:
        source = "import numpy as np\ndef wavefunction(z1, z2):\n    return z1 - z2\n"
        initial, _ = EVALUATOR.initialize_programs(source, [])
        self.assertEqual(initial, source)

    def test_fitted_program_is_persisted_and_reproduces_six_electron_score(self) -> None:
        from agents.math_agent.executor.execute_fuse.execute_agent_fuse import (
            EvolveExecuteAgentFuse,
        )
        from loongflow.framework.pes.context import Context, Workspace

        features = np.random.default_rng(12345).normal(size=(512, 12))
        z = features[:, ::2] - 1j * features[:, 1::2]
        target = np.ones(features.shape[0], dtype=np.complex128)
        for i in range(6):
            for j in range(i + 1, 6):
                target *= z[:, i] - z[:, j]
        target *= np.exp(-0.25 * np.sum(np.abs(z) ** 2, axis=1))
        target /= np.max(np.abs(target))
        dataset = self.root / "fitted_n6.npz"
        np.savez(dataset, features=features, psi_scaled=target)
        os.environ["LOONGFLOW_SR_DATASET"] = str(dataset)

        source = EVALUATOR._expand_initial_program(
            (HERE / "initial_programs" / "island_5.py").read_text(encoding="utf-8"), 6
        )
        context = Context(task="fitted export test", base_path=self.root)
        directory = Path(Workspace.get_executor_candidate_path(context, "0_0"))
        candidate = directory / "solution_fit.py"
        candidate.write_text(source, encoding="utf-8")
        result = EVALUATOR.evaluate(candidate)
        self.assertEqual(result["status"], "success", result)
        self.assertAlmostEqual(result["score"], 1.0)
        self.assertAlmostEqual(result["artifacts"]["fitted_parameters"]["alpha"], 0.25)
        self.assertEqual(candidate.read_text(encoding="utf-8"), source)
        (directory / "evaluation_fit.json").write_text(json.dumps(result), encoding="utf-8")

        executor = EvolveExecuteAgentFuse.__new__(EvolveExecuteAgentFuse)
        candidates = executor.load_results_for_candidate(context, 0, 0, "")
        self.assertEqual(len(candidates), 1)
        executor._write_best_results(context, candidates[0])
        exported = Path(Workspace.get_executor_best_solution_path(context)).read_text(
            encoding="utf-8"
        )
        self.assertEqual(candidate.read_text(encoding="utf-8"), exported)
        self.assertIn("# EVOLVE-BLOCK-START", exported)
        self.assertIn("# EVOLVE-BLOCK-END", exported)
        namespace = {}
        exec(compile(exported, "exported_wavefunction.py", "exec"), namespace)
        raw = namespace["wavefunction"](*z.T)
        finite_fraction = float(np.isfinite(raw).mean())
        direct_score = EVALUATOR._fidelity(raw, target) * finite_fraction
        self.assertAlmostEqual(direct_score, result["score"])

        legacy = directory / "solution_legacy.py"
        legacy.write_text(source, encoding="utf-8")
        (directory / "evaluation_legacy.json").write_text('{"score": 0.0}', encoding="utf-8")
        executor.load_results_for_candidate(context, 0, 0, "")
        self.assertEqual(legacy.read_text(encoding="utf-8"), source)

    def test_fitted_program_preserves_multiline_defaults_and_source(self) -> None:
        source = (
            "import numpy as np\n\n# EVOLVE-BLOCK-START\n"
            "def wavefunction(z1, z2,\n"
            "                 alpha=-1.0, beta=2e-3):  # 参数默认值\n"
            "    return (z1 - z2) * np.exp(-alpha * np.abs(z1 + z2) ** 2) + beta\n"
            "# EVOLVE-BLOCK-END\n"
        )
        candidate = self.root / "multiline.py"
        candidate.write_text(source, encoding="utf-8")
        exported = EVALUATOR._fitted_program(candidate, {"alpha": 0.25, "beta": -0.125})
        self.assertEqual(exported, source.replace("alpha=-1.0", "alpha=0.25").replace(
            "beta=2e-3", "beta=-0.125"
        ))
        candidate.write_text("import numpy as np\ndef wavefunction(z1, z2):\n    return z1 - z2\n")
        self.assertEqual(EVALUATOR._fitted_program(candidate, {}), candidate.read_text())

    def test_large_n_templates_fit_expression_limits(self) -> None:
        for i in range(6):
            template = (HERE / "initial_programs" / f"island_{i}.py").read_text(
                encoding="utf-8"
            )
            candidate = self.root / "n34.py"
            candidate.write_text(EVALUATOR._expand_initial_program(template, 34), encoding="utf-8")
            spec = EVALUATOR._wavefunction_spec(candidate, 34)
            EVALUATOR._Expression(
                spec["expression"], set(spec["coordinates"]) | set(spec["parameters"])
            )

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
