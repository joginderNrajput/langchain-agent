"""A safe arithmetic calculator tool.

LLMs are unreliable at exact arithmetic, so we give the agent a deterministic
calculator. Crucially we do **not** use :func:`eval` — that would let a model
(or a prompt-injected document) execute arbitrary Python. Instead we parse the
expression into an AST and evaluate only an explicit allow-list of nodes,
operators, and math functions.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable

from langchain_core.tools import tool

_MAX_EXPRESSION_LENGTH = 500
_MAX_AST_NODES = 80
_MAX_ABS_POWER_EXPONENT = 12

# Binary and unary operators we permit.
_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Named constants and functions exposed inside expressions.
_NAMES: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}
_FUNCS: dict[str, Callable[..., float]] = {
    name: getattr(math, name)
    for name in (
        "sqrt",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "log",
        "log10",
        "log2",
        "exp",
        "floor",
        "ceil",
        "factorial",
        "fabs",
        "degrees",
        "radians",
        "pow",
    )
}
_FUNCS["abs"] = abs
_FUNCS["round"] = round


def _evaluate(node: ast.AST) -> float | int:
    """Recursively evaluate an allow-listed AST node."""

    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported constant: {node.value!r}")
        # Preserve int vs float so functions like factorial() receive integers.
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in _NAMES:
            raise ValueError(f"Unknown name: {node.id!r}")
        return _NAMES[node.id]
    if isinstance(node, ast.BinOp):
        bin_op = _BIN_OPS.get(type(node.op))
        if bin_op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(float(right)) > _MAX_ABS_POWER_EXPONENT:
            raise ValueError(f"Exponent magnitude must be <= {_MAX_ABS_POWER_EXPONENT}")
        return bin_op(left, right)
    if isinstance(node, ast.UnaryOp):
        unary_op = _UNARY_OPS.get(type(node.op))
        if unary_op is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return unary_op(_evaluate(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError("Only whitelisted math functions are allowed")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported")
        args = [_evaluate(arg) for arg in node.args]
        return _FUNCS[node.func.id](*args)
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the exact result.

    Use this for any arithmetic instead of computing it yourself. Supports
    + - * / // % ** , parentheses, the constants pi/e/tau, and functions such
    as sqrt, sin, cos, tan, log, log10, exp, floor, ceil, factorial, abs, round.

    Example: "sqrt(2) * 10 + 3 ** 2".
    """

    try:
        if len(expression) > _MAX_EXPRESSION_LENGTH:
            return f"Error: expression is too long (max {_MAX_EXPRESSION_LENGTH} characters)."
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
            return "Error: expression is too complex."
        result = _evaluate(tree)
    except ZeroDivisionError:
        return "Error: division by zero."
    except (ValueError, SyntaxError, TypeError, OverflowError) as exc:
        return f"Error: could not evaluate expression ({exc})."

    # Present whole numbers cleanly (2.0 -> 2) while keeping float precision.
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)
