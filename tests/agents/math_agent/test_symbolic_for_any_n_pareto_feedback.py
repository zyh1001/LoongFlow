from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agents.math_agent.examples.symbolic_for_any_N.eval_program import (
    PARETO_ACCURACY_EPSILON,
    _candidate_identifier,
    _update_pareto_archive,
)


def _candidate(accuracy: float, alpha: float) -> dict[str, Any]:
    expression = "(z1 - z2) * np.exp(-alpha * np.abs(z1 - z2) ** 2)"
    return {
        "id": _candidate_identifier(expression, ["alpha"]),
        "expression": expression,
        "accuracy": accuracy,
        "complexity": 13,
        "parameter_count": 1,
        "fitted_parameters": {"alpha": alpha},
        "full_data_fidelity": accuracy,
        "full_data_nmse": 1.0 - accuracy,
        "finite_fraction": 1.0,
        "global_scale": {"real": 1.0, "imag": 0.0},
        "evaluated_at_unix": 0.0,
    }


@pytest.mark.parametrize(
    ("first_accuracy", "second_accuracy", "on_front", "improved_bin"),
    [
        (1.0, 0.4091, False, False),
        (0.8, 0.8, True, False),
        (0.6, 0.9, True, True),
        (0.8, 0.8 - PARETO_ACCURACY_EPSILON / 2, True, False),
        (0.8, 0.8 - PARETO_ACCURACY_EPSILON * 2, False, False),
    ],
)
def test_pareto_feedback_compares_current_fit_accuracy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_accuracy: float,
    second_accuracy: float,
    on_front: bool,
    improved_bin: bool,
) -> None:
    archive_path = tmp_path / "pareto.json"
    monkeypatch.setenv("LOONGFLOW_SR_PARETO_ARCHIVE", str(archive_path))
    monkeypatch.setenv("LOONGFLOW_SR_PARETO_CSV", str(tmp_path / "pareto.csv"))
    dataset_path = tmp_path / "dataset.npz"
    first = _candidate(first_accuracy, alpha=0.25)
    second = _candidate(second_accuracy, alpha=10.0)
    assert first["id"] == second["id"]

    assert _update_pareto_archive(first, dataset_path)["on_pareto_front"] is True
    feedback = _update_pareto_archive(second, dataset_path)

    assert feedback["on_pareto_front"] is on_front
    assert feedback["improved_complexity_bin"] is improved_bin
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    champion = second if improved_bin else first
    assert archive["schema_version"] == 1
    assert archive["pareto_front"] == [champion]
    assert feedback["complexity_bin_best_accuracy"] == champion["accuracy"]
