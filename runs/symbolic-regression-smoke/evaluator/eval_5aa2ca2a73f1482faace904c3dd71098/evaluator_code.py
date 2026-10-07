"""Evaluator for the two-electron FQH symbolic-regression task.

The evolutionary program supplies only a literal ``candidate_spec()``.  Its
expression is evaluated by a small AST interpreter, so candidate code cannot
read the dataset, evaluator internals, or the network.

This evaluator intentionally treats every finite neural-network sample as one
full-data pool.  The same complete pool is used to fit numerical parameters and
to compute the score.  There is no held-out validation or test split.
"""

from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
from scipy.optimize import least_squares


_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_DATASET_RELATIVE_PATH = Path(
    "datasets/disk_n2_Q6_kapa_1/dataset_50000.npz"
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
MAX_PARAMETERS = 10
MAX_EXPRESSION_LENGTH = 2000
MAX_AST_NODES = 220
MAX_OPTIMIZER_EVALUATIONS = int(os.environ.get("LOONGFLOW_SR_MAX_NFEV", "120"))
ANGULAR_MOMENTA = range(-24, 25)
ROTATION_ANGLES = (math.pi / 7.0, math.pi / 5.0)
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EPS = np.finfo(np.float64).tiny


class CandidateValidationError(ValueError):
    """Raised when a candidate violates the documented candidate interface."""


def _dataset_path() -> Path:
    configured = os.environ.get("LOONGFLOW_SR_DATASET") or os.environ.get(
        "FQH_SYMBOLIC_DATASET"
    )
    return Path(configured).expanduser() if configured else DEFAULT_DATASET


def _literal_candidate_spec(program_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a literal candidate_spec without executing candidate Python code."""

    path = Path(program_path)
    if not path.is_file():
        raise CandidateValidationError(f"Candidate file does not exist: {path}")

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise CandidateValidationError(f"Cannot parse candidate file: {exc}") from exc

    top_level = list(tree.body)
    if top_level and isinstance(top_level[0], ast.Expr) and isinstance(
        top_level[0].value, ast.Constant
    ) and isinstance(top_level[0].value.value, str):
        top_level = top_level[1:]

    if len(top_level) != 1 or not isinstance(top_level[0], ast.FunctionDef):
        raise CandidateValidationError(
            "The file must contain only the candidate_spec() function."
        )
    function = top_level[0]
    if function.name != "candidate_spec" or function.decorator_list:
        raise CandidateValidationError(
            "The file must define an undecorated candidate_spec() function."
        )
    args = function.args
    if (
        args.posonlyargs
        or args.args
        or args.kwonlyargs
        or args.vararg is not None
        or args.kwarg is not None
    ):
        raise CandidateValidationError("candidate_spec() must not accept arguments.")

    body = list(function.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(
        body[0].value, ast.Constant
    ) and isinstance(body[0].value.value, str):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        raise CandidateValidationError(
            "candidate_spec() must contain one literal return statement."
        )

    try:
        spec = ast.literal_eval(body[0].value)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise CandidateValidationError(
            "candidate_spec() must return a literal dictionary."
        ) from exc
    if not isinstance(spec, dict) or set(spec) != {"expression", "parameters"}:
        raise CandidateValidationError(
            "candidate_spec() must return exactly 'expression' and 'parameters'."
        )
    if not isinstance(spec["expression"], str) or not spec["expression"].strip():
        raise CandidateValidationError("'expression' must be a non-empty string.")
    if len(spec["expression"]) > MAX_EXPRESSION_LENGTH:
        raise CandidateValidationError(
            f"Expression exceeds {MAX_EXPRESSION_LENGTH} characters."
        )
    if not isinstance(spec["parameters"], dict):
        raise CandidateValidationError("'parameters' must be a dictionary.")
    if len(spec["parameters"]) > MAX_PARAMETERS:
        raise CandidateValidationError(
            f"At most {MAX_PARAMETERS} fitted numerical parameters are allowed."
        )

    clean_parameters: dict[str, float] = {}
    for name, value in spec["parameters"].items():
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise CandidateValidationError(f"Invalid parameter name: {name!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CandidateValidationError(
                f"Initial value for {name!r} must be a real number."
            )
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise CandidateValidationError(
                f"Initial value for {name!r} must be finite."
            )
        clean_parameters[name] = numeric_value

    return {
        "expression": spec["expression"].strip(),
        "parameters": clean_parameters,
    }


def _coordinate_environment(coordinates: np.ndarray) -> dict[str, Any]:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise RuntimeError(
            "This evaluator requires features ordered as x1,y1,x2,y2."
        )

    x1, y1, x2, y2 = coordinates.T
    z1 = x1 - 1j * y1
    z2 = x2 - 1j * y2
    z12 = z1 - z2
    z_cm = (z1 + z2) / 2.0
    r1_sq = x1 * x1 + y1 * y1
    r2_sq = x2 * x2 + y2 * y2
    r12_sq = np.abs(z12) ** 2
    r_cm_sq = np.abs(z_cm) ** 2

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "z1": z1,
        "z2": z2,
        "z1_bar": np.conjugate(z1),
        "z2_bar": np.conjugate(z2),
        "z1_conj": np.conjugate(z1),
        "z2_conj": np.conjugate(z2),
        "z12": z12,
        "z12_bar": np.conjugate(z12),
        "z12_conj": np.conjugate(z12),
        "z_rel": z12,
        "z_rel_bar": np.conjugate(z12),
        "z_rel_conj": np.conjugate(z12),
        "Z_cm": z_cm,
        "Z_cm_bar": np.conjugate(z_cm),
        "Z_cm_conj": np.conjugate(z_cm),
        "r1_sq": r1_sq,
        "r2_sq": r2_sq,
        "r12_sq": r12_sq,
        "r_rel_sq": r12_sq,
        "r1": np.sqrt(r1_sq),
        "r2": np.sqrt(r2_sq),
        "r12": np.sqrt(r12_sq),
        "r_rel": np.sqrt(r12_sq),
        "r_cm_sq": r_cm_sq,
        "R_cm_sq": r_cm_sq,
        "r_cm": np.sqrt(r_cm_sq),
        "R_cm": np.sqrt(r_cm_sq),
        "dot12": x1 * x2 + y1 * y2,
        "cross12": x1 * y2 - y1 * x2,
        "I": 1j,
        "pi": math.pi,
        "PI": math.pi,
    }


_FUNCTIONS = {
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "conj": np.conjugate,
    "real": np.real,
    "imag": np.imag,
    "log": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "tanh": np.tanh,
}


class _Expression:
    """A small vectorized expression interpreter with a strict whitelist."""

    def __init__(self, source: str, allowed_names: set[str]):
        try:
            self.tree = ast.parse(source, mode="eval")
        except SyntaxError as exc:
            raise CandidateValidationError(f"Invalid expression syntax: {exc}") from exc

        nodes = list(ast.walk(self.tree))
        if len(nodes) > MAX_AST_NODES:
            raise CandidateValidationError(
                f"Expression exceeds the {MAX_AST_NODES}-node complexity limit."
            )

        self.complexity = 1
        for node in nodes:
            if isinstance(node, ast.BinOp):
                if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                    raise CandidateValidationError("Unsupported binary operator.")
                self.complexity += 1
            elif isinstance(node, ast.UnaryOp):
                if not isinstance(node.op, (ast.UAdd, ast.USub)):
                    raise CandidateValidationError("Unsupported unary operator.")
                self.complexity += 1
            elif isinstance(node, ast.Call):
                if (
                    not isinstance(node.func, ast.Name)
                    or node.func.id not in _FUNCTIONS
                    or len(node.args) != 1
                    or node.keywords
                ):
                    raise CandidateValidationError(
                        "Only one-argument calls to "
                        + ", ".join(sorted(_FUNCTIONS))
                        + " are allowed."
                    )
                self.complexity += 1
            elif isinstance(node, ast.Name):
                if node.id not in allowed_names and node.id not in _FUNCTIONS:
                    raise CandidateValidationError(
                        f"Unknown variable or function: {node.id!r}"
                    )
            elif isinstance(node, ast.Constant):
                value = node.value
                if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
                    raise CandidateValidationError("Only numeric constants are allowed.")
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
        if isinstance(node, ast.Call):
            function = _FUNCTIONS[node.func.id]
            return function(self._evaluate_node(node.args[0], environment))
        raise CandidateValidationError(
            f"Unsupported expression element: {type(node).__name__}"
        )


def _parameter_aliases(
    names: list[str], vector: np.ndarray, reserved_names: set[str]
) -> dict[str, Any]:
    values: dict[str, Any] = {
        name: float(value) for name, value in zip(names, vector, strict=True)
    }
    for name in names:
        if not name.endswith("_re"):
            continue
        base = name[:-3]
        imaginary_name = base + "_im"
        if imaginary_name not in values:
            continue
        if base in values or base in reserved_names or base in _FUNCTIONS:
            raise CandidateValidationError(
                f"Complex parameter alias {base!r} collides with another name."
            )
        values[base] = values[name] + 1j * values[imaginary_name]
    return values


def _available_parameter_names(
    parameter_names: list[str], reserved_names: set[str]
) -> set[str]:
    dummy = np.zeros(len(parameter_names), dtype=np.float64)
    return set(_parameter_aliases(parameter_names, dummy, reserved_names))


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
    environment.update(
        _parameter_aliases(parameter_names, vector, set(coordinate_environment))
    )
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


def _fit_parameters(
    expression: _Expression,
    train_environment: dict[str, Any],
    train_target: np.ndarray,
    parameter_names: list[str],
    initial: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not parameter_names:
        return initial, {
            "attempted": False,
            "success": True,
            "nfev": 0,
            "message": "No numerical parameters.",
        }

    target_rms = math.sqrt(float(np.mean(np.abs(train_target) ** 2)))
    target_rms = max(target_rms, _EPS)

    def residual(vector: np.ndarray) -> np.ndarray:
        try:
            raw = _candidate_values(
                expression, train_environment, parameter_names, vector
            )
            prediction, _ = _aligned_prediction(raw, train_target)
            difference = prediction - train_target
            finite = np.isfinite(raw.real) & np.isfinite(raw.imag)
            if not np.all(finite):
                difference = difference.copy()
                difference[~finite] = 10.0 * target_rms * (1.0 + 1.0j)
            scaled = difference / target_rms
            return np.concatenate((scaled.real, scaled.imag))
        except (CandidateValidationError, ArithmeticError, FloatingPointError):
            return np.full(2 * train_target.size, 10.0, dtype=np.float64)

    span = np.maximum(20.0, np.abs(initial) * 10.0 + 10.0)
    best = initial.copy()
    initial_residual = residual(initial)
    best_cost = float(np.dot(initial_residual, initial_residual))
    information: dict[str, Any] = {
        "attempted": True,
        "success": False,
        "nfev": 0,
        "message": "Optimizer did not run.",
    }
    try:
        result = least_squares(
            residual,
            initial,
            bounds=(-span, span),
            max_nfev=MAX_OPTIMIZER_EVALUATIONS,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
            x_scale=np.maximum(np.abs(initial), 1.0),
        )
        result_cost = float(np.dot(result.fun, result.fun))
        if np.all(np.isfinite(result.x)) and result_cost <= best_cost:
            best = np.asarray(result.x, dtype=np.float64)
            best_cost = result_cost
        information = {
            "attempted": True,
            "success": bool(result.success),
            "nfev": int(result.nfev),
            "message": str(result.message),
            "residual_cost": best_cost,
        }
    except Exception as exc:  # Keep a valid initial candidate evaluable.
        information["message"] = f"Optimization failed; used initialization: {exc}"
    return best, information


def _relative_pair_error(first: np.ndarray, second: np.ndarray) -> float:
    finite = (
        np.isfinite(first.real)
        & np.isfinite(first.imag)
        & np.isfinite(second.real)
        & np.isfinite(second.imag)
    )
    if not np.any(finite):
        return 10.0
    a = first[finite]
    b = second[finite]
    scale = float(max(np.max(np.abs(a)), np.max(np.abs(b))))
    if not math.isfinite(scale) or scale <= _EPS:
        return 10.0
    a = a / scale
    b = b / scale
    denominator = float(np.sum(np.abs(a) ** 2 + np.abs(b) ** 2))
    if denominator <= _EPS:
        return 10.0
    error = math.sqrt(float(np.sum(np.abs(a - b) ** 2)) / denominator)
    invalid_fraction = 1.0 - float(np.mean(finite))
    return float(error + invalid_fraction)


def _exchange_error(
    expression: _Expression,
    coordinates: np.ndarray,
    raw: np.ndarray,
    parameter_names: list[str],
    fitted: np.ndarray,
) -> float:
    swapped = coordinates[:, [2, 3, 0, 1]]
    swapped_raw = _candidate_values(
        expression,
        _coordinate_environment(swapped),
        parameter_names,
        fitted,
    )
    return _relative_pair_error(swapped_raw, -raw)


def _rotate_coordinates(coordinates: np.ndarray, angle: float) -> np.ndarray:
    reshaped = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2, 2)
    x = reshaped[..., 0]
    y = reshaped[..., 1]
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotated = np.empty_like(reshaped)
    rotated[..., 0] = cosine * x - sine * y
    rotated[..., 1] = sine * x + cosine * y
    return rotated.reshape(-1, 4)


def _rotation_metrics(
    expression: _Expression,
    coordinates: np.ndarray,
    raw: np.ndarray,
    parameter_names: list[str],
    fitted: np.ndarray,
) -> tuple[int, float]:
    rotated_values = []
    for angle in ROTATION_ANGLES:
        rotated_values.append(
            _candidate_values(
                expression,
                _coordinate_environment(_rotate_coordinates(coordinates, angle)),
                parameter_names,
                fitted,
            )
        )

    best_m = 0
    best_error = float("inf")
    for angular_momentum in ANGULAR_MOMENTA:
        errors = []
        for angle, rotated in zip(ROTATION_ANGLES, rotated_values, strict=True):
            expected = np.exp(-1j * angular_momentum * angle) * raw
            errors.append(_relative_pair_error(rotated, expected))
        error = float(math.sqrt(np.mean(np.square(errors))))
        if error < best_error:
            best_m = angular_momentum
            best_error = error
    return best_m, best_error


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

    if features.ndim != 2 or features.shape[1] != 4:
        raise ValueError(
            f"Expected features with shape (n, 4), received {features.shape}."
        )
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
    )
    if not np.any(usable):
        raise ValueError("The dataset contains no finite samples.")

    result = {
        "coordinates": features[usable],
        "target": target[usable],
    }
    _DATA_CACHE[cache_key] = result
    return result


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
        spec = _literal_candidate_spec(program_path)
        data = _load_evolution_data(_dataset_path())

        coordinates = data["coordinates"]
        target = data["target"]
        coordinate_environment = _coordinate_environment(coordinates)

        parameter_names = list(spec["parameters"])
        reserved_names = set(coordinate_environment)
        for name in parameter_names:
            if name in reserved_names or name in _FUNCTIONS:
                raise CandidateValidationError(
                    f"Parameter name {name!r} is reserved by the evaluator."
                )
        allowed_names = reserved_names | _available_parameter_names(
            parameter_names, reserved_names
        )
        expression = _Expression(spec["expression"], allowed_names)
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
        )
        raw = _candidate_values(
            expression, coordinate_environment, parameter_names, fitted
        )
        prediction, global_scale = _aligned_prediction(raw, target)
        finite = np.isfinite(raw.real) & np.isfinite(raw.imag)
        finite_fraction = float(np.mean(finite))
        full_data_nmse = _normalized_mse(prediction, target)
        full_data_fidelity = _fidelity(prediction, target)

        exchange_error = _exchange_error(
            expression,
            coordinates,
            raw,
            parameter_names,
            fitted,
        )
        angular_momentum, rotation_error = _rotation_metrics(
            expression,
            coordinates,
            raw,
            parameter_names,
            fitted,
        )

        # Keep the score continuous even for poor early candidates whose NMSE
        # is slightly above one; otherwise many distinct candidates all score
        # exactly zero and provide no useful evolutionary signal.
        accuracy = math.sqrt(
            full_data_fidelity * math.exp(-min(full_data_nmse, 50.0))
        )
        physics_factor = math.exp(
            -1.5 * min(exchange_error, 20.0)
            - 1.5 * min(rotation_error, 20.0)
        )
        complexity_penalty = math.exp(
            -max(0, expression.complexity - 12) / 2000.0
            - max(0, len(parameter_names) - 6) / 1000.0
        )
        score = float(
            np.clip(
                accuracy
                * physics_factor
                * finite_fraction**2
                * complexity_penalty,
                0.0,
                1.0,
            )
        )
        elapsed = float(time.perf_counter() - start_time)

        fitted_parameters = {
            name: float(value)
            for name, value in zip(parameter_names, fitted, strict=True)
        }
        summary = (
            f"score={score:.6f}; full_data_fidelity={full_data_fidelity:.6f}; "
            f"full_data_nmse={full_data_nmse:.6g}; "
            f"exchange_error={exchange_error:.3g}; "
            f"rotation_error={rotation_error:.3g} at M={angular_momentum}; "
            f"finite={finite_fraction:.4f}; complexity={expression.complexity}; "
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
                "exchange_error": float(exchange_error),
                "rotation_error": float(rotation_error),
                "inferred_angular_momentum": int(angular_momentum),
                "finite_fraction": finite_fraction,
                "expression_complexity": int(expression.complexity),
                "parameter_count": len(parameter_names),
                "accuracy_component": float(accuracy),
                "physics_factor": float(physics_factor),
                "complexity_penalty": float(complexity_penalty),
                "eval_time": elapsed,
            },
            "artifacts": {
                "expression": spec["expression"],
                "fitted_parameters": fitted_parameters,
                "global_scale": {
                    "real": float(global_scale.real),
                    "imag": float(global_scale.imag),
                },
                "optimizer": optimizer_information,
                "evaluation_mode": "all_finite_samples_fit_and_score",
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
