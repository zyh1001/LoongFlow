"""Evaluator for fixed-N FQH wavefunctions with dataset-supplied kapa.

The evolutionary program supplies a vectorized ``wavefunction()`` with numeric
defaults for fitted parameters after the required coordinates and kapa input.
Its return expression is evaluated by a small AST interpreter, so candidate
code cannot read the dataset, evaluator internals,
or the network.

This evaluator intentionally treats every finite neural-network sample as one
full-data pool.  The same complete pool is used to fit numerical parameters and
to compute the accuracy objective, with equal weight and independent global
complex alignment per kapa system. Expression-tree complexity is a separate
objective stored in a persistent Pareto archive.  Exchange antisymmetry and
angular-momentum structure are deliberately left to candidate construction and
are not measured or scored here.  There is no held-out validation or test split.
"""

from __future__ import annotations

import ast
import copy
import csv
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.optimize import least_squares


_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_DATASET_RELATIVE_PATH = Path(
    "datasets/disk_n2_Q6_kapa_100/dataset_uniform.npz"
)
DEFAULT_DATASET = next(
    (
        path
        for path in (
            _REPOSITORY_ROOT / _DEFAULT_DATASET_RELATIVE_PATH,
            _REPOSITORY_ROOT.parent / _DEFAULT_DATASET_RELATIVE_PATH,
        )
        if path.is_file()
    ),
    _REPOSITORY_ROOT / _DEFAULT_DATASET_RELATIVE_PATH,
)
MAX_PARAMETERS = 15
MAX_EXPRESSION_LENGTH = 2000
MAX_AST_NODES = 220
MAX_OPTIMIZER_EVALUATIONS = int(os.environ.get("LOONGFLOW_SR_MAX_NFEV", "120"))
MAX_BOUNDARY_EXPANSIONS = int(
    os.environ.get("LOONGFLOW_SR_MAX_BOUND_EXPANSIONS", "3")
)
BOUNDARY_EXPANSION_FACTOR = float(
    os.environ.get("LOONGFLOW_SR_BOUND_EXPANSION_FACTOR", "5.0")
)
BOUNDARY_HIT_FRACTION = float(
    os.environ.get("LOONGFLOW_SR_BOUND_HIT_FRACTION", "0.98")
)
MAX_PARAMETER_SPAN = float(
    os.environ.get("LOONGFLOW_SR_MAX_PARAMETER_SPAN", "10000.0")
)
MAX_STRUCTURAL_INTEGER = 12
PARETO_ACCURACY_EPSILON = 1e-12
DEFAULT_PARETO_ARCHIVE = Path(
    "./output-fqh-symbolic-with-kapa-as-param/pareto_archive.json"
)
_EPS = np.finfo(np.float64).tiny


class CandidateValidationError(ValueError):
    """Raised when a candidate violates the documented candidate interface."""


class _LocalExpressionInliner(ast.NodeTransformer):
    """Replace reads of local names with their previously expanded expressions."""

    def __init__(self, bindings: dict[str, ast.expr]):
        self._bindings = bindings

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self._bindings:
            return ast.copy_location(copy.deepcopy(self._bindings[node.id]), node)
        return node


def _inline_local_assignments(
    body: list[ast.stmt],
    protected_names: set[str],
    max_ast_nodes: int = MAX_AST_NODES,
) -> ast.expr:
    """Inline simple local assignments into a function's final return value."""

    if not body or not isinstance(body[-1], ast.Return):
        raise CandidateValidationError(
            "wavefunction() must end with one return statement."
        )
    if body[-1].value is None:
        raise CandidateValidationError("wavefunction() must return an expression.")

    bindings: dict[str, ast.expr] = {}
    allowed_names = protected_names | {"abs", "np"}
    for statement in body[:-1]:
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
        ):
            raise CandidateValidationError(
                "Before its final return, wavefunction() may contain only simple "
                "local assignments of the form 'name = expression'."
            )

        target = statement.targets[0].id
        if target in protected_names:
            raise CandidateValidationError(
                f"Local assignment cannot overwrite coordinate or parameter {target!r}."
            )
        if target in {"abs", "np"}:
            raise CandidateValidationError(
                f"Local assignment cannot overwrite reserved name {target!r}."
            )
        if target in bindings:
            raise CandidateValidationError(
                f"Local variable {target!r} may be assigned only once."
            )

        expanded_value = _LocalExpressionInliner(bindings).visit(
            copy.deepcopy(statement.value)
        )
        ast.fix_missing_locations(expanded_value)

        # Validate every assignment even if the final return does not use it.
        # This preserves the evaluator's strict expression whitelist.
        _Expression(ast.unparse(expanded_value), allowed_names, max_ast_nodes)
        bindings[target] = expanded_value

    expanded_return = _LocalExpressionInliner(bindings).visit(
        copy.deepcopy(body[-1].value)
    )
    ast.fix_missing_locations(expanded_return)
    return expanded_return


def _dataset_path() -> Path:
    configured = os.environ.get("LOONGFLOW_SR_DATASET") or os.environ.get(
        "FQH_SYMBOLIC_DATASET"
    )
    return Path(configured).expanduser() if configured else DEFAULT_DATASET


def _pareto_archive_path() -> Path:
    configured = os.environ.get("LOONGFLOW_SR_PARETO_ARCHIVE")
    return Path(configured).expanduser() if configured else DEFAULT_PARETO_ARCHIVE


def _pareto_csv_path() -> Path:
    configured = os.environ.get("LOONGFLOW_SR_PARETO_CSV")
    return (
        Path(configured).expanduser()
        if configured
        else _pareto_archive_path().with_suffix(".csv")
    )


def _combine(terms: list[str], operator: str, identity: str) -> str:
    """Build a balanced explicit expression for the task's seed templates."""

    if not terms:
        return identity
    if len(terms) == 1:
        return terms[0]
    middle = len(terms) // 2
    return (
        f"({_combine(terms[:middle], operator, identity)} {operator} "
        f"{_combine(terms[middle:], operator, identity)})"
    )


def _expand_initial_program(source: str, n_electrons: int) -> str:
    """Expand trusted seeds for the one fixed N of this dataset; never execute code."""

    if "def wavefunction(COORDINATES" not in source:
        return source
    coordinates = [f"z{i}" for i in range(1, n_electrons + 1)]
    pairs = [
        f"({first} - {second})"
        for index, first in enumerate(coordinates)
        for second in coordinates[index + 1:]
    ]
    replacements = {
        "COORDINATES": ", ".join(coordinates),
        "PAIR_PRODUCT": _combine(pairs, "*", "1"),
        "COORDINATE_SUM": _combine(coordinates, "+", "0"),
        "RADIUS_SUM": _combine([f"np.abs({z}) ** 2" for z in coordinates], "+", "0"),
        "PAIR_DISTANCE_SUM": _combine([f"np.abs({pair})" for pair in pairs], "+", "1"),
    }
    for name, expression in replacements.items():
        source = source.replace(name, expression)
    return source


def initialize_programs(initial_code: str, seed_codes: list[str]) -> tuple[str, list[str]]:
    """Use the existing runner hook to prepare fixed-N initial programs."""

    data = _load_evolution_data(_dataset_path())
    n_electrons = data["coordinates"].shape[1] // 2
    return (
        _expand_initial_program(initial_code, n_electrons),
        [_expand_initial_program(code, n_electrons) for code in seed_codes],
    )


def _wavefunction_spec(
    program_path: str | os.PathLike[str], n_electrons: int
) -> dict[str, Any]:
    """Read a wavefunction definition without executing candidate Python code."""

    path = Path(program_path)
    if not path.is_file():
        raise CandidateValidationError(f"Candidate file does not exist: {path}")

    try:
        source = _expand_initial_program(path.read_text(encoding="utf-8"), n_electrons)
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise CandidateValidationError(f"Cannot parse candidate file: {exc}") from exc

    top_level = list(tree.body)
    if top_level and isinstance(top_level[0], ast.Expr) and isinstance(
        top_level[0].value, ast.Constant
    ) and isinstance(top_level[0].value.value, str):
        top_level = top_level[1:]

    if (
        len(top_level) != 2
        or not isinstance(top_level[0], ast.Import)
        or len(top_level[0].names) != 1
        or top_level[0].names[0].name != "numpy"
        or top_level[0].names[0].asname != "np"
        or not isinstance(top_level[1], ast.FunctionDef)
    ):
        raise CandidateValidationError(
            "The file must contain only 'import numpy as np' and wavefunction()."
        )
    function = top_level[1]
    if function.name != "wavefunction" or function.decorator_list:
        raise CandidateValidationError(
            "The file must define an undecorated wavefunction() function."
        )
    args = function.args
    if (
        args.posonlyargs
        or args.kwonlyargs
        or args.vararg is not None
        or args.kwarg is not None
    ):
        raise CandidateValidationError(
            "wavefunction() supports positional arguments only."
        )

    coordinate_names = [f"z{index}" for index in range(1, n_electrons + 1)]
    input_names = coordinate_names + ["kapa"]
    argument_names = [argument.arg for argument in args.args]
    if argument_names[:len(input_names)] != input_names:
        raise CandidateValidationError(
            "wavefunction() must begin with required arguments "
            + ", ".join(input_names)
            + "."
        )
    parameter_names = argument_names[len(input_names):]
    if len(parameter_names) > MAX_PARAMETERS:
        raise CandidateValidationError(
            f"At most {MAX_PARAMETERS} fitted numerical parameters are allowed."
        )
    if len(args.defaults) != len(parameter_names):
        raise CandidateValidationError(
            "Coordinates and kapa must have no defaults; every fitted parameter "
            "must have a numeric default value."
        )

    clean_parameters: dict[str, float] = {}
    for name, default in zip(parameter_names, args.defaults, strict=True):
        try:
            value = ast.literal_eval(default)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise CandidateValidationError(
                f"Default value for {name!r} must be a real number."
            ) from exc
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CandidateValidationError(
                f"Default value for {name!r} must be a real number."
            )
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise CandidateValidationError(
                f"Default value for {name!r} must be finite."
            )
        clean_parameters[name] = numeric_value

    body = list(function.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(
        body[0].value, ast.Constant
    ) and isinstance(body[0].value.value, str):
        body = body[1:]
    return_expression = _inline_local_assignments(
        body,
        set(input_names) | set(parameter_names),
        max(MAX_AST_NODES, 24 * n_electrons**2),
    )
    expression = ast.unparse(return_expression).strip()
    max_length = max(MAX_EXPRESSION_LENGTH, 256 * n_electrons**2)
    if len(expression) > max_length:
        raise CandidateValidationError(
            f"Expression exceeds {max_length} characters."
        )

    return {
        "expression": expression,
        "coordinates": coordinate_names,
        "parameters": clean_parameters,
    }


def _coordinate_environment(coordinates: np.ndarray) -> dict[str, Any]:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2 or coordinates.shape[1] % 2:
        raise RuntimeError(
            "This evaluator requires features ordered as x1,y1,...,xN,yN."
        )

    n_electrons = coordinates.shape[1] // 2
    electrons = coordinates.reshape(-1, n_electrons, 2)
    x = electrons[..., 0]
    y = electrons[..., 1]
    z = x - 1j * y
    z_cm = np.mean(z, axis=1)
    r_cm_sq = np.abs(z_cm) ** 2

    environment: dict[str, Any] = {
        "Z_cm": z_cm,
        "Z_cm_bar": np.conjugate(z_cm),
        "Z_cm_conj": np.conjugate(z_cm),
        "r_cm_sq": r_cm_sq,
        "R_cm_sq": r_cm_sq,
        "r_cm": np.sqrt(r_cm_sq),
        "R_cm": np.sqrt(r_cm_sq),
        "I": 1j,
        "pi": math.pi,
        "PI": math.pi,
    }
    for index in range(n_electrons):
        number = index + 1
        radius_sq = x[:, index] ** 2 + y[:, index] ** 2
        environment.update(
            {
                f"x{number}": x[:, index],
                f"y{number}": y[:, index],
                f"z{number}": z[:, index],
                f"z{number}_bar": np.conjugate(z[:, index]),
                f"z{number}_conj": np.conjugate(z[:, index]),
                f"r{number}_sq": radius_sq,
                f"r{number}": np.sqrt(radius_sq),
            }
        )
    for first in range(n_electrons):
        for second in range(first + 1, n_electrons):
            pair = f"{first + 1}{second + 1}"
            difference = z[:, first] - z[:, second]
            distance_sq = np.abs(difference) ** 2
            environment.update(
                {
                    f"z{pair}": difference,
                    f"z{pair}_bar": np.conjugate(difference),
                    f"z{pair}_conj": np.conjugate(difference),
                    f"r{pair}_sq": distance_sq,
                    f"r{pair}": np.sqrt(distance_sq),
                    f"dot{pair}": (
                        x[:, first] * x[:, second]
                        + y[:, first] * y[:, second]
                    ),
                    f"cross{pair}": (
                        x[:, first] * y[:, second]
                        - y[:, first] * x[:, second]
                    ),
                }
            )
    return environment


_FUNCTIONS = {
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "conj": np.conjugate,
    "conjugate": np.conjugate,
    "real": np.real,
    "imag": np.imag,
    "log": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "tanh": np.tanh,
}


class _Expression:
    """A small vectorized expression interpreter with a strict whitelist."""

    def __init__(
        self, source: str, allowed_names: set[str], max_ast_nodes: int = MAX_AST_NODES
    ):
        try:
            self.tree = ast.parse(source, mode="eval")
        except SyntaxError as exc:
            raise CandidateValidationError(f"Invalid expression syntax: {exc}") from exc

        nodes = list(ast.walk(self.tree))
        if len(nodes) > max_ast_nodes:
            raise CandidateValidationError(
                f"Expression exceeds the {max_ast_nodes}-node complexity limit."
            )

        for node in nodes:
            if isinstance(node, ast.BinOp):
                if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                    raise CandidateValidationError("Unsupported binary operator.")
            elif isinstance(node, ast.UnaryOp):
                if not isinstance(node.op, (ast.UAdd, ast.USub)):
                    raise CandidateValidationError("Unsupported unary operator.")
            elif isinstance(node, ast.Call):
                if (
                    self._function_name(node.func) is None
                    or len(node.args) != 1
                    or node.keywords
                ):
                    raise CandidateValidationError(
                        "Only abs(...) and one-argument np calls to "
                        + ", ".join(sorted(_FUNCTIONS))
                        + " are allowed."
                    )
            elif isinstance(node, ast.Name):
                if node.id not in allowed_names and node.id not in {"abs", "np"}:
                    raise CandidateValidationError(
                        f"Unknown variable or function: {node.id!r}"
                    )
            elif isinstance(node, ast.Attribute):
                if (
                    not isinstance(node.value, ast.Name)
                    or node.value.id != "np"
                    or node.attr not in {*_FUNCTIONS, "pi"}
                ):
                    raise CandidateValidationError(
                        "Only whitelisted numpy functions and np.pi are allowed."
                    )
            elif isinstance(node, ast.Constant):
                self._validate_literal(node.value)
            elif not isinstance(
                node,
                (
                    ast.Expression,
                    ast.Load,
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.Div,
                    ast.Pow,
                    ast.UAdd,
                    ast.USub,
                ),
            ):
                raise CandidateValidationError(
                    f"Unsupported expression element: {type(node).__name__}"
                )

        # Match the default SymbolicRegression.jl/PySR interpretation: every
        # operator, function, variable, and literal is one expression-tree node.
        # Python's np.<function> wrapper is syntax, not an extra symbolic node.
        self.complexity = self._tree_complexity(self.tree.body)

    @staticmethod
    def _validate_literal(value: Any) -> None:
        if isinstance(value, bool):
            raise CandidateValidationError("Boolean literals are not allowed.")
        if isinstance(value, int):
            if abs(value) > MAX_STRUCTURAL_INTEGER:
                raise CandidateValidationError(
                    f"Integer literals must satisfy |n| <= {MAX_STRUCTURAL_INTEGER}; "
                    "use a fitted parameter for larger numerical constants."
                )
            return
        if isinstance(value, complex) and value == 1j:
            return
        if isinstance(value, float):
            raise CandidateValidationError(
                "Decimal literals are not allowed in the return expression; "
                "introduce a fitted parameter instead. Exact structural ratios "
                "may be written with integer literals, for example 1 / 2."
            )
        raise CandidateValidationError(
            "Only small integer structural literals, 1j, and np.pi are allowed."
        )

    @classmethod
    def _tree_complexity(cls, node: ast.AST) -> int:
        if isinstance(node, (ast.Constant, ast.Name)):
            return 1
        if isinstance(node, ast.Attribute) and node.attr == "pi":
            return 1
        if isinstance(node, ast.UnaryOp):
            return 1 + cls._tree_complexity(node.operand)
        if isinstance(node, ast.BinOp):
            return (
                1
                + cls._tree_complexity(node.left)
                + cls._tree_complexity(node.right)
            )
        if isinstance(node, ast.Call):
            return 1 + cls._tree_complexity(node.args[0])
        raise CandidateValidationError(
            f"Cannot compute complexity for {type(node).__name__}."
        )

    @staticmethod
    def _function_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name) and node.id == "abs":
            return "abs"
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "np"
            and node.attr in _FUNCTIONS
        ):
            return node.attr
        return None

    def evaluate(self, environment: dict[str, Any]) -> Any:
        with np.errstate(all="ignore"):
            return self._evaluate_node(self.tree.body, environment)

    def _evaluate_node(self, node: ast.AST, environment: dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return environment[node.id]
        if isinstance(node, ast.UnaryOp):
            value = self._evaluate_node(node.operand, environment)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = self._evaluate_node(node.left, environment)
            right = self._evaluate_node(node.right, environment)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            return left**right
        if isinstance(node, ast.Attribute) and node.attr == "pi":
            return math.pi
        if isinstance(node, ast.Call):
            function_name = self._function_name(node.func)
            if function_name is None:
                raise CandidateValidationError("Unsupported function call.")
            function = _FUNCTIONS[function_name]
            return function(self._evaluate_node(node.args[0], environment))
        raise CandidateValidationError(
            f"Unsupported expression element: {type(node).__name__}"
        )


def _parameter_values(names: list[str], vector: np.ndarray) -> dict[str, float]:
    return {
        name: float(value) for name, value in zip(names, vector, strict=True)
    }


def _as_complex_vector(value: Any, size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        array = np.full(size, array, dtype=np.complex128)
    elif array.shape == (size,):
        array = np.asarray(array, dtype=np.complex128)
    else:
        raise CandidateValidationError(
            f"Expression returned shape {array.shape}; expected a scalar or ({size},)."
        )
    try:
        return np.asarray(array, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise CandidateValidationError(
            "Expression output cannot be converted to complex numbers."
        ) from exc


def _candidate_values(
    expression: _Expression,
    coordinate_environment: dict[str, Any],
    parameter_names: list[str],
    vector: np.ndarray,
) -> np.ndarray:
    environment = dict(coordinate_environment)
    environment.update(_parameter_values(parameter_names, vector))
    value = expression.evaluate(environment)
    sample_count = len(coordinate_environment["x1"])
    return _as_complex_vector(value, sample_count)


def _aligned_prediction(
    raw: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, complex]:
    """Fit the physically irrelevant global normalization and phase safely."""

    raw = np.asarray(raw, dtype=np.complex128)
    target = np.asarray(target, dtype=np.complex128)
    finite = np.isfinite(raw.real) & np.isfinite(raw.imag)
    prediction = np.zeros_like(target)
    if not np.any(finite):
        return prediction, 0j

    raw_scale = float(np.max(np.abs(raw[finite])))
    if not math.isfinite(raw_scale) or raw_scale <= _EPS:
        return prediction, 0j

    normalized = np.zeros_like(raw)
    normalized[finite] = raw[finite] / raw_scale
    denominator = float(np.vdot(normalized[finite], normalized[finite]).real)
    if not math.isfinite(denominator) or denominator <= _EPS:
        return prediction, 0j

    normalized_scale = np.vdot(normalized[finite], target[finite]) / denominator
    prediction[finite] = normalized_scale * normalized[finite]
    return prediction, complex(normalized_scale / raw_scale)


def _normalized_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.vdot(target, target).real)
    if denominator <= _EPS:
        return float("inf")
    value = float(np.vdot(prediction - target, prediction - target).real / denominator)
    return value if math.isfinite(value) else float("inf")


def _fidelity(prediction: np.ndarray, target: np.ndarray) -> float:
    norm_prediction = float(np.vdot(prediction, prediction).real)
    norm_target = float(np.vdot(target, target).real)
    denominator = norm_prediction * norm_target
    if denominator <= _EPS or not math.isfinite(denominator):
        return 0.0
    overlap = abs(np.vdot(prediction, target)) ** 2 / denominator
    return float(np.clip(overlap, 0.0, 1.0))


def _aligned_by_kapa(
    raw: np.ndarray, target: np.ndarray, groups: dict[float, np.ndarray]
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """Remove each system's independent normalization and phase."""

    prediction = np.zeros_like(target)
    scales: dict[str, dict[str, float]] = {}
    for kapa, indices in groups.items():
        prediction[indices], scale = _aligned_prediction(raw[indices], target[indices])
        scales[str(kapa)] = {"real": float(scale.real), "imag": float(scale.imag)}
    return prediction, scales


def _fit_parameters(
    expression: _Expression,
    train_environment: dict[str, Any],
    train_target: np.ndarray,
    parameter_names: list[str],
    initial: np.ndarray,
    groups: dict[float, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    if not parameter_names:
        return initial, {
            "attempted": False,
            "success": True,
            "nfev": 0,
            "message": "No numerical parameters.",
        }

    target_rms = np.empty(train_target.size, dtype=np.float64)
    for indices in groups.values():
        rms = max(math.sqrt(float(np.mean(np.abs(train_target[indices]) ** 2))), _EPS)
        # Equal weight per system, independent of label scale and sample count.
        target_rms[indices] = rms * math.sqrt(
            len(indices) * len(groups) / train_target.size
        )

    def residual(vector: np.ndarray) -> np.ndarray:
        try:
            raw = _candidate_values(
                expression, train_environment, parameter_names, vector
            )
            prediction, _ = _aligned_by_kapa(raw, train_target, groups)
            difference = prediction - train_target
            finite = np.isfinite(raw.real) & np.isfinite(raw.imag)
            if not np.all(finite):
                difference = difference.copy()
                difference[~finite] = 10.0 * target_rms[~finite] * (1.0 + 1.0j)
            scaled = difference / target_rms
            return np.concatenate((scaled.real, scaled.imag))
        except (CandidateValidationError, ArithmeticError, FloatingPointError):
            return np.full(2 * train_target.size, 10.0, dtype=np.float64)

    initial_span = np.maximum(20.0, np.abs(initial) * 10.0 + 10.0)
    maximum_span = max(20.0, MAX_PARAMETER_SPAN)
    span = np.minimum(initial_span, maximum_span)
    expansion_limit = max(0, MAX_BOUNDARY_EXPANSIONS)
    expansion_factor = max(1.1, BOUNDARY_EXPANSION_FACTOR)
    hit_fraction = float(np.clip(BOUNDARY_HIT_FRACTION, 0.5, 1.0))

    best = initial.copy()
    initial_residual = residual(initial)
    best_cost = float(np.dot(initial_residual, initial_residual))
    start = initial.copy()
    total_nfev = 0
    expansion_count = 0
    any_success = False
    final_boundary_hit = np.zeros(len(parameter_names), dtype=bool)
    attempts: list[dict[str, Any]] = []
    information: dict[str, Any] = {
        "attempted": True,
        "success": False,
        "nfev": 0,
        "message": "Optimizer did not run.",
    }

    try:
        last_message = ""
        for attempt_index in range(expansion_limit + 1):
            result = least_squares(
                residual,
                start,
                bounds=(-span, span),
                max_nfev=MAX_OPTIMIZER_EVALUATIONS,
                ftol=1e-7,
                xtol=1e-7,
                gtol=1e-7,
                x_scale=np.maximum(np.abs(start), 1.0),
            )
            result_vector = np.asarray(result.x, dtype=np.float64)
            result_cost = float(np.dot(result.fun, result.fun))
            result_is_finite = bool(
                np.all(np.isfinite(result_vector)) and math.isfinite(result_cost)
            )
            total_nfev += int(result.nfev)
            any_success = any_success or bool(result.success)
            last_message = str(result.message)

            if result_is_finite and result_cost <= best_cost:
                best = result_vector.copy()
                best_cost = result_cost

            final_boundary_hit = (
                np.abs(result_vector) >= hit_fraction * span
                if result_is_finite
                else np.zeros(len(parameter_names), dtype=bool)
            )
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "nfev": int(result.nfev),
                    "cost": result_cost,
                    "span": {
                        name: float(value)
                        for name, value in zip(parameter_names, span, strict=True)
                    },
                    "boundary_parameters": [
                        name
                        for name, hit in zip(
                            parameter_names, final_boundary_hit, strict=True
                        )
                        if hit
                    ],
                }
            )

            expandable = final_boundary_hit & (span < maximum_span)
            if attempt_index >= expansion_limit or not np.any(expandable):
                break

            span[expandable] = np.minimum(
                span[expandable] * expansion_factor,
                maximum_span,
            )
            expansion_count += 1
            start = np.clip(result_vector, -0.999999 * span, 0.999999 * span)

        information = {
            "attempted": True,
            "success": any_success,
            "nfev": total_nfev,
            "message": last_message,
            "residual_cost": best_cost,
            "adaptive_bounds": True,
            "boundary_expansions": expansion_count,
            "boundary_hit": bool(np.any(final_boundary_hit)),
            "boundary_parameters": [
                name
                for name, hit in zip(
                    parameter_names, final_boundary_hit, strict=True
                )
                if hit
            ],
            "initial_span": {
                name: float(value)
                for name, value in zip(parameter_names, initial_span, strict=True)
            },
            "final_span": {
                name: float(value)
                for name, value in zip(parameter_names, span, strict=True)
            },
            "attempts": attempts,
        }
    except Exception as exc:  # Keep a valid initial candidate evaluable.
        information["message"] = f"Optimization failed; used initialization: {exc}"
    return best, information


_DATA_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _load_evolution_data(path: Path) -> dict[str, np.ndarray]:
    """Load every finite neural-network sample as one fitting/scoring pool."""

    cache_key = str(path.resolve())
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]
    if not path.is_file():
        raise FileNotFoundError(
            f"Symbolic-regression dataset not found: {path}. Set "
            "LOONGFLOW_SR_DATASET to override it."
        )

    with np.load(path, allow_pickle=False) as dataset:
        required = {"features"}
        missing = required.difference(dataset.files)
        if missing:
            raise KeyError("Dataset is missing: " + ", ".join(sorted(missing)))
        if "psi_scaled" not in dataset.files and not {
            "psi_real",
            "psi_imag",
        }.issubset(dataset.files):
            raise KeyError("Dataset needs psi_scaled or psi_real and psi_imag.")

        features = np.asarray(dataset["features"], dtype=np.float64)
        kapa_key = "kapa" if "kapa" in dataset.files else "kappa"
        if kapa_key not in dataset.files:
            raise KeyError("Dataset needs a 'kapa' (or 'kappa') value for every sample.")
        if np.iscomplexobj(dataset[kapa_key]):
            raise ValueError("kapa must be real-valued.")
        kapa = np.asarray(dataset[kapa_key], dtype=np.float64)
        if "psi_scaled" in dataset.files:
            target = np.asarray(dataset["psi_scaled"], dtype=np.complex128)
        else:
            target = np.asarray(dataset["psi_real"], dtype=np.float64) + 1j * np.asarray(
                dataset["psi_imag"], dtype=np.float64
            )
        finite_mask = (
            np.asarray(dataset["finite_mask"], dtype=bool)
            if "finite_mask" in dataset.files
            else np.ones(features.shape[0], dtype=bool)
        )

    if features.ndim != 2 or features.shape[1] < 2 or features.shape[1] % 2:
        raise ValueError(
            "Expected features with shape (n, 2*N) for one fixed N, "
            f"received {features.shape}."
        )
    if kapa.ndim == 0:
        kapa = np.full(features.shape[0], float(kapa), dtype=np.float64)
    if kapa.shape != (features.shape[0],):
        raise ValueError(f"kapa must be scalar or have shape ({features.shape[0]},).")
    if target.shape != (features.shape[0],):
        raise ValueError(
            f"Target shape {target.shape} does not match {features.shape[0]} rows."
        )
    if finite_mask.shape != (features.shape[0],):
        raise ValueError(
            f"finite_mask shape {finite_mask.shape} does not match the dataset."
        )
    usable = (
        finite_mask
        & np.all(np.isfinite(features), axis=1)
        & np.isfinite(target.real)
        & np.isfinite(target.imag)
        & np.isfinite(kapa)
    )
    if not np.any(usable):
        raise ValueError("The dataset contains no finite samples.")
    if np.any(kapa[usable] < 0):
        raise ValueError("kapa must be nonnegative.")

    result = {
        "coordinates": features[usable],
        "target": target[usable],
        "kapa": kapa[usable],
    }
    _DATA_CACHE[cache_key] = result
    return result


@contextmanager
def _archive_lock(archive_path: Path):
    """Serialize archive updates across evaluator subprocesses."""

    lock_path = archive_path.with_suffix(archive_path.suffix + ".lock")
    deadline = time.monotonic() + 10.0
    descriptor: int | None = None
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    while descriptor is None:
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime > 120.0
            except OSError:
                stale = False
            if stale:
                lock_path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Pareto archive lock: {lock_path}")
            time.sleep(0.05)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _candidate_identifier(
    expression: str,
    parameter_names: list[str],
) -> str:
    canonical = json.dumps(
        {"expression": expression, "parameters": parameter_names},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _pareto_front(champions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return candidates not dominated in (complexity min, accuracy max)."""

    front: list[dict[str, Any]] = []
    best_accuracy = -float("inf")
    for candidate in sorted(champions, key=lambda item: item["complexity"]):
        accuracy = float(candidate["accuracy"])
        if accuracy > best_accuracy + PARETO_ACCURACY_EPSILON:
            front.append(candidate)
            best_accuracy = accuracy
    return front


def _front_selections(front: list[dict[str, Any]]) -> dict[str, Any]:
    """Select useful representatives without collapsing the stored front."""

    if not front:
        return {}
    maximum_accuracy = max(front, key=lambda item: item["accuracy"])

    knee = maximum_accuracy
    if len(front) >= 3:
        complexities = np.asarray(
            [item["complexity"] for item in front], dtype=np.float64
        )
        accuracies = np.asarray(
            [item["accuracy"] for item in front], dtype=np.float64
        )
        complexity_span = float(np.ptp(complexities))
        accuracy_span = float(np.ptp(accuracies))
        if complexity_span > 0.0 and accuracy_span > 0.0:
            x = (complexities - complexities[0]) / complexity_span
            y = (accuracies - accuracies[0]) / accuracy_span
            x0, y0 = float(x[0]), float(y[0])
            x1, y1 = float(x[-1]), float(y[-1])
            denominator = math.hypot(y1 - y0, x1 - x0)
            distances = np.abs(
                (y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0
            ) / max(denominator, _EPS)
            knee = front[int(np.argmax(distances))]

    minimum_loss = min(max(1.0 - item["accuracy"], _EPS) for item in front)
    pysr_candidates: list[tuple[float, dict[str, Any]]] = []
    previous = front[0]
    for candidate in front[1:]:
        delta_complexity = candidate["complexity"] - previous["complexity"]
        previous_loss = max(1.0 - previous["accuracy"], _EPS)
        candidate_loss = max(1.0 - candidate["accuracy"], _EPS)
        slope = math.log(previous_loss / candidate_loss) / delta_complexity
        if candidate_loss <= 1.5 * minimum_loss:
            pysr_candidates.append((slope, candidate))
        previous = candidate
    pysr_choice = (
        max(pysr_candidates, key=lambda item: item[0])[1]
        if pysr_candidates
        else maximum_accuracy
    )

    return {
        "max_accuracy_id": maximum_accuracy["id"],
        "knee_id": knee["id"],
        "pysr_score_id": pysr_choice["id"],
    }


_PARETO_CSV_FIELDS = [
    "candidate_id",
    "original_expression",
    "score",
    "fidelity",
    "accuracy",
    "loss",
    "nmse",
    "finite_fraction",
    "complexity",
    "parameter_count",
    "fitted_parameters",
    "global_scale_real",
    "global_scale_imag",
    "global_scales",
    "pysr_score",
    "selection",
    "evaluated_at_unix",
]


def _write_pareto_csv(
    csv_path: Path,
    front: list[dict[str, Any]],
    selections: dict[str, Any],
) -> None:
    """Write a PySR-style table containing the current non-dominated front."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    selection_labels: dict[str, list[str]] = {}
    for label, candidate_id in selections.items():
        selection_labels.setdefault(str(candidate_id), []).append(label)

    pysr_scores: dict[str, float | str] = {}
    previous: dict[str, Any] | None = None
    for candidate in front:
        if previous is None:
            pysr_scores[candidate["id"]] = ""
        else:
            previous_loss = max(1.0 - float(previous["accuracy"]), _EPS)
            current_loss = max(1.0 - float(candidate["accuracy"]), _EPS)
            delta_complexity = int(candidate["complexity"]) - int(
                previous["complexity"]
            )
            pysr_scores[candidate["id"]] = (
                math.log(previous_loss / current_loss) / delta_complexity
            )
        previous = candidate

    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=_PARETO_CSV_FIELDS)
            writer.writeheader()
            for candidate in front:
                global_scale = candidate.get("global_scale", {})
                writer.writerow(
                    {
                        "candidate_id": candidate["id"],
                        "original_expression": candidate["expression"],
                        "score": candidate["accuracy"],
                        "fidelity": candidate["full_data_fidelity"],
                        "accuracy": candidate["accuracy"],
                        "loss": max(1.0 - float(candidate["accuracy"]), 0.0),
                        "nmse": candidate["full_data_nmse"],
                        "finite_fraction": candidate["finite_fraction"],
                        "complexity": candidate["complexity"],
                        "parameter_count": candidate["parameter_count"],
                        "fitted_parameters": json.dumps(
                            candidate.get("fitted_parameters", {}),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "global_scale_real": global_scale.get("real", ""),
                        "global_scale_imag": global_scale.get("imag", ""),
                        "global_scales": json.dumps(candidate.get("global_scales", {})),
                        "pysr_score": pysr_scores[candidate["id"]],
                        "selection": "|".join(
                            selection_labels.get(candidate["id"], [])
                        ),
                        "evaluated_at_unix": candidate["evaluated_at_unix"],
                    }
                )
        os.replace(temporary, csv_path)
    finally:
        temporary.unlink(missing_ok=True)


def _update_pareto_archive(
    candidate: dict[str, Any],
    dataset_path: Path,
) -> dict[str, Any]:
    """Update the best-per-complexity archive and derived Pareto front."""

    archive_path = _pareto_archive_path()
    csv_path = _pareto_csv_path()
    dataset_key = str(dataset_path.resolve())
    with _archive_lock(archive_path):
        archive: dict[str, Any] = {}
        if archive_path.is_file():
            try:
                archive = json.loads(archive_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                archive = {}
        if archive.get("schema_version") != 2 or archive.get("dataset") != dataset_key:
            archive = {
                "schema_version": 2,
                "dataset": dataset_key,
                "objectives": {
                    "accuracy": "maximize mean per-kapa(fidelity * finite_fraction)",
                    "complexity": "minimize expression-tree node count",
                },
                "complexity_champions": {},
            }

        champions = archive.setdefault("complexity_champions", {})
        bin_key = str(candidate["complexity"])
        incumbent = champions.get(bin_key)
        improved_bin = incumbent is None or candidate["accuracy"] > (
            float(incumbent["accuracy"]) + PARETO_ACCURACY_EPSILON
        )
        if improved_bin:
            champions[bin_key] = candidate

        champion_list = list(champions.values())
        front = _pareto_front(champion_list)
        selections = _front_selections(front)
        archive["pareto_front"] = front
        archive["selections"] = selections
        archive["updated_at_unix"] = time.time()

        temporary = archive_path.with_name(
            f".{archive_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(archive, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, archive_path)
        _write_pareto_csv(csv_path, front, selections)

    front_ids = {item["id"] for item in front}
    bin_champion = champions[bin_key]
    return {
        "archive_path": str(archive_path),
        "csv_path": str(csv_path),
        "improved_complexity_bin": bool(improved_bin),
        "on_pareto_front": candidate["id"] in front_ids,
        "front_size": len(front),
        "complexity_bin_best_accuracy": float(bin_champion["accuracy"]),
        "selections": selections,
        "front": [
            {
                "id": item["id"],
                "complexity": int(item["complexity"]),
                "accuracy": float(item["accuracy"]),
                "expression": item["expression"],
            }
            for item in front
        ],
    }


def _failure(
    status: str, summary: str, start_time: float, error_type: str
) -> dict[str, Any]:
    return {
        "status": status,
        "score": 0.0,
        "summary": summary,
        "metrics": {
            "validity": 0.0,
            "eval_time": float(time.perf_counter() - start_time),
        },
        "artifacts": {"error_type": error_type, "reason": summary},
    }


def evaluate(program_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Fit and score one symbolic full-wavefunction candidate."""

    start_time = time.perf_counter()
    try:
        data = _load_evolution_data(_dataset_path())

        coordinates = data["coordinates"]
        target = data["target"]
        n_electrons = coordinates.shape[1] // 2
        spec = _wavefunction_spec(program_path, n_electrons)
        coordinate_environment = _coordinate_environment(coordinates)
        coordinate_environment["kapa"] = data["kapa"]
        groups = {
            float(value): np.flatnonzero(data["kapa"] == value)
            for value in np.unique(data["kapa"])
        }

        parameter_names = list(spec["parameters"])
        reserved_names = set(coordinate_environment)
        for name in parameter_names:
            if name in reserved_names or name in _FUNCTIONS or name == "np":
                raise CandidateValidationError(
                    f"Parameter name {name!r} is reserved by the evaluator."
                )
        allowed_names = set(spec["coordinates"]) | set(parameter_names) | {"np", "kapa"}
        expression = _Expression(
            spec["expression"], allowed_names, max(MAX_AST_NODES, 24 * n_electrons**2)
        )
        initial = np.asarray(
            [spec["parameters"][name] for name in parameter_names],
            dtype=np.float64,
        )

        fitted, optimizer_information = _fit_parameters(
            expression,
            coordinate_environment,
            target,
            parameter_names,
            initial,
            groups,
        )
        raw = _candidate_values(
            expression, coordinate_environment, parameter_names, fitted
        )
        prediction, global_scales = _aligned_by_kapa(raw, target, groups)
        global_scale = next(iter(global_scales.values())) if len(groups) == 1 else {}
        finite = np.isfinite(raw.real) & np.isfinite(raw.imag)
        per_kapa = {
            str(value): {
                "samples_used": int(indices.size),
                "fidelity": _fidelity(prediction[indices], target[indices]),
                "nmse": _normalized_mse(prediction[indices], target[indices]),
                "finite_fraction": float(np.mean(finite[indices])),
            }
            for value, indices in groups.items()
        }
        finite_fraction = float(
            np.mean([item["finite_fraction"] for item in per_kapa.values()])
        )
        full_data_nmse = float(np.mean([item["nmse"] for item in per_kapa.values()]))
        full_data_fidelity = float(np.mean([item["fidelity"] for item in per_kapa.values()]))
        fitted_parameters = {
            name: float(value)
            for name, value in zip(parameter_names, fitted, strict=True)
        }

        # Accuracy is the first Pareto objective. Invalid output points count
        # directly against it, while complexity remains a separate objective.
        accuracy = float(
            np.mean(
                [item["fidelity"] * item["finite_fraction"] for item in per_kapa.values()]
            )
        )
        score = accuracy  # Stable scalar bridge required by LoongFlow.
        candidate_id = _candidate_identifier(spec["expression"], parameter_names)
        archive_candidate = {
            "id": candidate_id,
            "expression": spec["expression"],
            "complexity": int(expression.complexity),
            "accuracy": accuracy,
            "full_data_fidelity": float(full_data_fidelity),
            "full_data_nmse": float(full_data_nmse),
            "finite_fraction": finite_fraction,
            "parameter_count": len(parameter_names),
            "fitted_parameters": fitted_parameters,
            "global_scale": global_scale,
            "global_scales": global_scales,
            "per_kapa": per_kapa,
            "evaluated_at_unix": time.time(),
        }
        try:
            pareto = _update_pareto_archive(
                archive_candidate,
                _dataset_path(),
            )
        except (OSError, TimeoutError, ValueError) as exc:
            # Archive I/O must never turn a valid scientific evaluation into a
            # failed candidate. Surface the issue so it can be repaired.
            pareto = {
                "archive_path": str(_pareto_archive_path()),
                "csv_path": str(_pareto_csv_path()),
                "archive_error": f"{type(exc).__name__}: {exc}",
                "improved_complexity_bin": False,
                "on_pareto_front": False,
                "front_size": 0,
                "selections": {},
                "front": [],
            }
        elapsed = float(time.perf_counter() - start_time)

        summary = (
            f"accuracy={accuracy:.6f}; full_data_fidelity={full_data_fidelity:.6f}; "
            f"full_data_nmse={full_data_nmse:.6g}; "
            f"finite={finite_fraction:.4f}; complexity={expression.complexity}; "
            f"pareto_front={pareto['on_pareto_front']}; "
            f"front_size={pareto['front_size']}; "
            f"N={n_electrons}; kapa_systems={len(groups)}; "
            f"fitted={fitted_parameters}"
        )
        return {
            "status": "success",
            "score": score,
            "summary": summary,
            "metrics": {
                "validity": 1.0,
                "full_data_nmse": float(full_data_nmse),
                "full_data_fidelity": float(full_data_fidelity),
                "finite_fraction": finite_fraction,
                "expression_complexity": int(expression.complexity),
                "parameter_count": len(parameter_names),
                "n_electrons": n_electrons,
                "kapa_systems": len(groups),
                "accuracy": accuracy,
                "pareto_front_member": bool(pareto["on_pareto_front"]),
                "pareto_bin_improved": bool(pareto["improved_complexity_bin"]),
                "pareto_front_size": int(pareto["front_size"]),
                "optimizer_boundary_expansions": int(
                    optimizer_information.get("boundary_expansions", 0)
                ),
                "optimizer_boundary_hit": bool(
                    optimizer_information.get("boundary_hit", False)
                ),
                "eval_time": elapsed,
            },
            "artifacts": {
                "candidate_id": candidate_id,
                "expression": spec["expression"],
                "fitted_parameters": fitted_parameters,
                "global_scale": global_scale,
                "global_scales": global_scales,
                "per_kapa": per_kapa,
                "kapa_values": list(groups),
                "optimizer": optimizer_information,
                "pareto": pareto,
                "evaluation_mode": "full_data_accuracy_complexity_pareto",
                "samples_used": int(coordinates.shape[0]),
                "dataset": str(_dataset_path()),
            },
        }
    except CandidateValidationError as exc:
        return _failure(
            "validation_failed",
            f"Candidate validation failed: {exc}",
            start_time,
            type(exc).__name__,
        )
    except Exception as exc:
        return _failure(
            "execution_failed",
            f"Evaluator failed: {type(exc).__name__}: {exc}",
            start_time,
            type(exc).__name__,
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} CANDIDATE.py")
    print(json.dumps(evaluate(sys.argv[1]), indent=2, ensure_ascii=False))
