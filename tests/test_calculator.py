"""Unit tests for the safe calculator tool."""

from __future__ import annotations

import math

import pytest

from agentic_research_agent.tools.calculator import calculator


def _calc(expr: str) -> str:
    # `calculator` is a StructuredTool; .invoke runs the underlying function.
    return calculator.invoke({"expression": expr})


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("2 ** 10", "1024"),
        ("10 / 4", "2.5"),
        ("17 % 5", "2"),
        ("17 // 5", "3"),
        ("-5 + 2", "-3"),
        ("sqrt(16)", "4"),
        ("factorial(5)", "120"),
        ("abs(-7)", "7"),
    ],
)
def test_valid_expressions(expression: str, expected: str) -> None:
    assert _calc(expression) == expected


def test_constants_and_functions() -> None:
    result = float(_calc("log10(1000)"))
    assert math.isclose(result, 3.0, abs_tol=1e-9)


def test_division_by_zero_is_handled() -> None:
    assert "division by zero" in _calc("1 / 0").lower()


def test_expression_length_is_limited() -> None:
    assert "too long" in _calc("1" * 501).lower()


def test_expression_complexity_is_limited() -> None:
    expression = " + ".join(["1"] * 100)
    assert "too complex" in _calc(expression).lower()


def test_large_exponents_are_rejected() -> None:
    assert "exponent magnitude" in _calc("2 ** 100").lower()


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",  # code injection attempt
        "open('/etc/passwd')",  # disallowed builtin
        "x + 1",  # unknown name
        "2 +",  # syntax error
        "exit()",  # disallowed call
    ],
)
def test_unsafe_or_invalid_expressions_are_rejected(expression: str) -> None:
    assert _calc(expression).lower().startswith("error")
