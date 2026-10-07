from __future__ import annotations

import ast
from pathlib import Path

import pytest

from agents.math_agent.examples.modified_symbolic_regression.eval_program import (
    CandidateValidationError,
    _Expression,
    _wavefunction_spec,
)


def _write_candidate(tmp_path: Path, function_body: str) -> Path:
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        "import numpy as np\n\n"
        "def wavefunction(z1, z2, a=1.0):\n"
        + function_body,
        encoding="utf-8",
    )
    return candidate


def _expression_ast(source: str) -> str:
    return ast.dump(ast.parse(source, mode="eval"), include_attributes=False)


def test_wavefunction_spec_inlines_local_assignments(tmp_path: Path) -> None:
    candidate = _write_candidate(
        tmp_path,
        "    w = z1 - z2\n"
        "    r2 = np.abs(w) ** 2\n"
        "    envelope = np.exp(-a * r2)\n"
        "    return w ** 3 * envelope\n",
    )

    spec = _wavefunction_spec(candidate)
    expected = "(z1 - z2) ** 3 * np.exp(-a * np.abs(z1 - z2) ** 2)"

    assert _expression_ast(spec["expression"]) == _expression_ast(expected)
    assert _Expression(
        spec["expression"], {"z1", "z2", "a", "np"}
    ).complexity == _Expression(expected, {"z1", "z2", "a", "np"}).complexity


def test_wavefunction_spec_still_accepts_direct_return(tmp_path: Path) -> None:
    candidate = _write_candidate(
        tmp_path,
        "    return (z1 - z2) ** 3 * np.exp(-a * np.abs(z1 - z2) ** 2)\n",
    )

    spec = _wavefunction_spec(candidate)

    assert spec["parameters"] == {"a": 1.0}
    assert _expression_ast(spec["expression"]) == _expression_ast(
        "(z1 - z2) ** 3 * np.exp(-a * np.abs(z1 - z2) ** 2)"
    )


@pytest.mark.parametrize(
    ("function_body", "message"),
    [
        (
            "    z1 = z1 - z2\n"
            "    return z1\n",
            "cannot overwrite coordinate or parameter",
        ),
        (
            "    w = z1 - z2\n"
            "    w = w ** 3\n"
            "    return w\n",
            "may be assigned only once",
        ),
        (
            "    w, W = z1 - z2, z1 + z2\n"
            "    return w * W\n",
            "only simple local assignments",
        ),
        (
            "    w = open('forbidden')\n"
            "    return z1 - z2\n",
            "Only abs",
        ),
    ],
)
def test_wavefunction_spec_rejects_unsafe_or_ambiguous_assignments(
    tmp_path: Path,
    function_body: str,
    message: str,
) -> None:
    candidate = _write_candidate(tmp_path, function_body)

    with pytest.raises(CandidateValidationError, match=message):
        _wavefunction_spec(candidate)


def test_wavefunction_spec_requires_final_return(tmp_path: Path) -> None:
    candidate = _write_candidate(
        tmp_path,
        "    if np.abs(z1) > 0:\n"
        "        return z1 - z2\n"
        "    return z2 - z1\n",
    )

    with pytest.raises(CandidateValidationError, match="only simple local assignments"):
        _wavefunction_spec(candidate)
