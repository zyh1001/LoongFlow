# Fixed-N FQH symbolic regression with dataset-supplied kapa

This directory is a drop-in replacement for the original task files, plus a
small set of island seeds and one minimal LoongFlow compatibility patch.

## Dataset and candidate interface

Set `LOONGFLOW_SR_DATASET` to one NPZ containing multiple kapa systems at the
same electron number. Required fields are:

| Field | Shape | Meaning |
| --- | --- | --- |
| `features` | `(samples, 2*N)` | Coordinates ordered `x1,y1,...,xN,yN`; one fixed N |
| `psi_scaled` | `(samples,)` | Complex target; alternatively `psi_real` and `psi_imag` |
| `kapa` | `(samples,)` | Real nonnegative physical input aligned with every sample |
| `finite_mask` | `(samples,)` | Optional sample mask |

`kappa` is accepted as a dataset-field alias when `kapa` is absent. A scalar
`kapa` is broadcast for a single-system control. Kapa is never inferred from
the path or optimized. Existing NPZ files without kapa must add this field.

For N=2, an evolved candidate has the interface:

```python
def wavefunction(z1, z2, kapa, alpha=1.0):
    return (z1 - z2) * np.exp(-alpha * (np.abs(z1)**2 + np.abs(z2)**2) / (1 + kapa))
```

For other fixed N, the coordinates are `z1, ..., zN`, followed by required
`kapa` and defaulted fitted coefficients. All systems share these coefficients.
The initial files use task-owned `COORDINATES`, `PAIR_PRODUCT`, `COORDINATE_SUM`,
`RADIUS_SUM` and `PAIR_DISTANCE_SUM` templates. The existing runner's
`initialize_from_dataset` hook expands them in memory for this run's N before
evolution. Candidates still use the restricted explicit-coordinate AST syntax.
The baseline length/node limits scale with N² to accommodate pair expansions;
the expression-tree complexity counting rule is unchanged.

Each kapa system has an independent global complex alignment, because separate
NN states have arbitrary normalization and phase. Parameter fitting uses equal
weight per system, and score is the mean of `fidelity * finite_fraction` over
systems. Artifacts and the JSON archive include `per_kapa` metrics and
`global_scales`; the CSV also includes the latter as JSON. Single-system results
retain `global_scale`; it is empty for multiple systems. Fitting and scoring
still use the same full-data pool, with no new train/test split.

## What changed

- `eval_program.py` contains no exchange-antisymmetry or rotation/angular-
  momentum evaluation code, metrics, or score terms. Those requirements remain
  explicit in `task_config.yaml` and must be guaranteed algebraically by the
  candidate construction.
- The scalar score is the per-kapa mean of `fidelity * finite_fraction`. Complexity is a separate
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

The current workspace runner already supports per-island initialization and
dataset-dependent template expansion. Do not reapply the legacy patch here.
For an older runner without these hooks, the included patch documents only the
per-island changes; dataset-dependent initialization is also required.

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
export LOONGFLOW_SR_DATASET="/absolute/path/to/fixed_n_multi_kapa.npz"
./run_math.sh symbolic_with_kapa_as_param --background
```

By default the Pareto archive is written to:

```text
./output-fqh-symbolic-with-kapa-as-param/pareto_archive.json
./output-fqh-symbolic-with-kapa-as-param/pareto_archive.csv
```

Set `LOONGFLOW_SR_PARETO_ARCHIVE` to override that location. A new dataset path
automatically starts a fresh archive schema so candidates from different data
sets are not mixed. Archive schema 2 also separates the new grouped scoring from
old schema-1 scores. The CSV path follows the JSON archive path with a `.csv`
suffix; set `LOONGFLOW_SR_PARETO_CSV` to override it independently.

## Verification

`test_eval_program.py` checks all six seeds, fixed-N initialization, kapa input
validation, shared coefficient recovery across independently scaled/phase-shifted
systems, AST restrictions, and the existing Pareto and adaptive-bound behavior:

```bash
uv run pytest agents/math_agent/examples/symbolic_with_kapa_as_param/test_eval_program.py -q
```

Production evaluation requires NumPy and SciPy, as before. The test includes a
minimal optimizer stub only when SciPy is unavailable, so its structural checks
can also run in a lightweight verification environment.
