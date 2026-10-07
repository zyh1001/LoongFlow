# FQH symbolic-regression task: accuracy-complexity Pareto version

This directory is a drop-in replacement for the original task files, plus a
small set of island seeds and one minimal LoongFlow compatibility patch.

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
  architectures.

PySR exposes operator/constant complexity controls and a hall of fame with loss
and complexity, while SymbolicRegression.jl supplies its search engine. The
relevant upstream references are:

- https://github.com/astroautomata/PySR
- https://github.com/astroautomata/SymbolicRegression.jl
- https://ai.damtp.cam.ac.uk/pysr/options/

## Multi-seed islands

The current public LoongFlow math-agent runner accepts one `initial_program.py`.
Listing six files in YAML alone therefore does not create six different island
initializations. Apply the included patch once in the LoongFlow repository:

```bash
git apply /absolute/path/to/modified_symbolic_regression/loongflow_multi_seed.patch
```

The patch makes two narrow changes:

1. resolve `evolve.metadata.island_initial_files` relative to `task_config.yaml`;
2. evaluate those seeds and pass seed `i` to context `island_id == i`.

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
python -m unittest -v test_eval_program.py
```

Production evaluation requires NumPy and SciPy, as before. The test includes a
minimal optimizer stub only when SciPy is unavailable, so its structural checks
can also run in a lightweight verification environment.
