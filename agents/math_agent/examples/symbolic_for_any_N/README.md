# FQH symbolic-regression task: accuracy-complexity Pareto version

This directory is a drop-in replacement for the original task files, plus a
small set of island seeds and one minimal LoongFlow compatibility patch.

## Automatic electron count

Each run uses one dataset from one fixed `(N, kappa)` system. N is inferred from
`features.shape[1] // 2`; columns must be ordered `x1,y1,...,xN,yN`.
Use the existing command directly, with no preparation step:

```bash
export LOONGFLOW_SR_DATASET="/absolute/path/to/dataset.npz"
./run_math.sh symbolic_for_any_N --background
```

With `evolve.metadata.initialize_from_dataset: true`, the math runner calls the
trusted evaluator's `initialize_programs()` at startup. It expands
`initial_program.py` and the four configured island templates in memory into ordinary
`wavefunction(z1, ..., zN, fitted_parameter=...)` programs. `COORDINATES`,
`PAIR_PRODUCT`, `COORDINATE_SUM`, `RADIUS_SUM`, and `PAIR_DISTANCE_SUM` are seed
template placeholders only; evolved candidates use explicit coordinate names
and arithmetic. No source file or prompt is rewritten. Explicit initial
programs without placeholders are preserved.

N must be positive. For N=1, empty pair products are 1 and some seeds coincide.
Expression length and AST limits scale with N squared to accommodate explicit
pair expansions. Scoring, fitting, complexity counting, and Pareto selection
are unchanged.

## What changed

- `eval_program.py` contains no exchange-antisymmetry or rotation/angular-
  momentum evaluation code, metrics, or score terms. Those requirements remain
  explicit in `task_config.yaml` and must be guaranteed algebraically by the
  candidate construction.
- The scalar score is `fidelity * finite_fraction`. Complexity is a separate
  objective and never appears as a multiplicative penalty.
- Complexity is the recursive expression-tree node count used by default in
  PySR/SymbolicRegression.jl: every operator, function, variable occurrence,
  fitted-parameter occurrence, and literal contributes one node. The Python
  spelling `np.<function>` does not add a second symbolic node.
- Fitted-parameter bounds expand adaptively when an optimum reaches a boundary.
  The number of expansions, expansion factor, hit threshold, and maximum span
  can be configured with `LOONGFLOW_SR_MAX_BOUND_EXPANSIONS`,
  `LOONGFLOW_SR_BOUND_EXPANSION_FACTOR`, `LOONGFLOW_SR_BOUND_HIT_FRACTION`, and
  `LOONGFLOW_SR_MAX_PARAMETER_SPAN`.
- Successful evaluations include `artifacts.fitted_program`, preserving the
  candidate source with its fitted parameter defaults. The fuse executor saves
  that code in candidate and best-solution files so direct calls reproduce the
  scored fidelity. Absolute normalization and phase remain in `global_scale`.
- The evaluator stores the best candidate at each exact integer complexity and
  derives the non-dominated accuracy-complexity front. The JSON archive also
  records maximum-accuracy, normalized-knee, and PySR-style log-loss-slope
  selections.
- The current Pareto front is also exported as a PySR-style CSV containing the
  original expression, score, fidelity, loss, complexity, fitted parameters, and
  selection labels.
- Decimal literals are rejected inside the return expression. Small structural
  integers (`|n| <= 12`), `1j`, and `np.pi` are allowed. Continuous constants
  must be named fitted parameters. Decimal defaults in the function signature
  remain legal because the evaluator optimizes them.
- The redundant fitted global complex coefficient was removed from
  `initial_program.py`.
- `initial_programs/` contains six genuinely different, low-complexity seed
  architectures; the current four islands use `island_0.py` through `island_3.py`.

PySR exposes operator/constant complexity controls and a hall of fame with loss
and complexity, while SymbolicRegression.jl supplies its search engine. The
relevant upstream references are:

- https://github.com/astroautomata/PySR
- https://github.com/astroautomata/SymbolicRegression.jl
- https://ai.damtp.cam.ac.uk/pysr/options/

## Multi-seed islands

The existing math-agent runner resolves `evolve.metadata.island_initial_files`
relative to `task_config.yaml`, expands the templates using the dataset's N, and
passes seed `i` to context `island_id == i`. No manual patch or preparation
command is required in this repository.

All database, migration, planner, executor, summary, and checkpoint behavior is
otherwise unchanged. The number of seed files must equal `num_islands`.

## Run

Place this directory under `agents/math_agent/examples/<task-name>/`, set the
API key and dataset path, then run the normal LoongFlow command:

```bash
export DEEPSEEK_API_KEY="..."
export LOONGFLOW_SR_DATASET="/absolute/path/to/dataset_uniform.npz"
./run_math.sh <task-name> --background
```

By default the Pareto archive is written to:

```text
./output-fqh-symbolic-regression-full-data/pareto_archive.json
./output-fqh-symbolic-regression-full-data/pareto_archive.csv
```

Set `LOONGFLOW_SR_PARETO_ARCHIVE` to override that location. A new dataset path
automatically starts a fresh archive schema so candidates from different data
sets are not mixed. The CSV path follows the JSON archive path with a `.csv`
suffix; set `LOONGFLOW_SR_PARETO_CSV` to override it independently.

## Verification

`test_eval_program.py` builds a deterministic synthetic complex-wavefunction
dataset and checks all six seeds, exact complexity counts, removal of symmetry
metrics, decimal-literal rejection, fitted-parameter defaults, and Pareto JSON
and CSV creation. It also verifies a fitted optimum that requires automatic
boundary expansion:

```bash
uv run pytest agents/math_agent/examples/symbolic_for_any_N/test_eval_program.py
```

Production evaluation requires NumPy and SciPy, as before. The test includes a
minimal optimizer stub only when SciPy is unavailable, so its structural checks
can also run in a lightweight verification environment.
