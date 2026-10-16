"""Restricted, safe surface-expression language: parse, validate, lower.

Surface definitions are ``(x(u, v), y(u, v), z(u, v))`` expression strings.
This module owns *syntax and safety*: it parses a string with :func:`ast.parse`
(``mode="eval"``), accepts only an explicit allowlist of nodes, and lowers the
validated AST to scalar Python source for the Numba code generator. It performs
no symbolic algebra and never evaluates a string with :func:`eval`.

Grammar (variables ``u``, ``v``; integer/float literals only)::

    expr := expr ('+'|'-'|'*'|'/'|'%'|'**') expr | ('+'|'-') expr
           | FUNC '(' expr (',' expr)* ')' | NAME | NUMBER

Supported functions are exactly :data:`SUPPORTED_FUNCTIONS`. Anything else —
attribute access, subscripts, calls, comprehensions, lambdas, comparisons,
booleans, strings, ``pi``/``e`` — is rejected with a targeted
:class:`ExpressionError`.
"""

from __future__ import annotations

import ast
from typing import Final

VARIABLES: Final[frozenset[str]] = frozenset({"u", "v"})

# name -> Numba/Python scalar printer (``abs`` is the supported builtin; the rest
# come from ``math``). The same mapping drives the generated Numba module and the
# pure-Python reference, keeping every backend on identical scalar semantics.
SCALAR_FUNCS: Final[dict[str, str]] = {
    "sqrt": "math.sqrt",
    "sin": "math.sin",
    "cos": "math.cos",
    "tan": "math.tan",
    "exp": "math.exp",
    "log": "math.log",
    "abs": "abs",
}
SUPPORTED_FUNCTIONS: Final[frozenset[str]] = frozenset(SCALAR_FUNCS)

# Division/modulo are rewritten to helpers with NumPy scalar semantics (``inf``/
# ``nan`` instead of ``ZeroDivisionError``); the helpers live in the generated
# module and in the pure-Python reference namespace.
_SAFE_DIV = "_sdiv"
_SAFE_MOD = "_smod"

_BIN_OPS: Final[tuple[type[ast.operator], ...]] = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
)
_UNARY_OPS: Final[tuple[type[ast.unaryop], ...]] = (ast.UAdd, ast.USub)


class ExpressionError(ValueError):
    """Raised for any surface expression outside the supported language."""


class ExpressionValidator:
    """Allowlist validator; raises :class:`ExpressionError` on the first offense."""

    def validate(self, tree: ast.AST) -> None:
        self._visit(tree)

    def _visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            self._visit(node.body)
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, _BIN_OPS):
                raise ExpressionError(
                    f"Unsupported operator '{type(node.op).__name__}'"
                )
            self._visit(node.left)
            self._visit(node.right)
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _UNARY_OPS):
                raise ExpressionError(
                    f"Unsupported operator '{type(node.op).__name__}'"
                )
            self._visit(node.operand)
        elif isinstance(node, ast.Call):
            self._visit_call(node)
        elif isinstance(node, ast.Name):
            if node.id not in VARIABLES:
                raise ExpressionError(f"Unknown variable '{node.id}'")
        elif isinstance(node, ast.Constant):
            self._visit_constant(node)
        else:
            raise ExpressionError(f"Unsupported syntax '{type(node).__name__}'")

    def _visit_call(self, node: ast.Call) -> None:
        if node.keywords:
            raise ExpressionError("Keyword arguments are not supported")
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                raise ExpressionError("Argument unpacking is not supported")
            self._visit(arg)
        if not isinstance(node.func, ast.Name):
            raise ExpressionError(
                f"Unsupported syntax '{type(node.func).__name__}'"
            )
        if node.func.id not in SCALAR_FUNCS:
            raise ExpressionError(f"Unsupported function '{node.func.id}'")

    def _visit_constant(self, node: ast.Constant) -> None:
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ExpressionError(f"Unsupported constant '{value!r}'")


def parse_expression(source: str) -> ast.Expression:
    """Parse and validate a surface expression into a restricted ``ast.Expression``."""
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:  # include the offending source for the UI
        raise ExpressionError(f"Invalid expression {source!r}: {exc.msg}") from exc
    ExpressionValidator().validate(tree)
    return tree


class _ScalarLower(ast.NodeTransformer):
    """Rewrite ``/`` and ``%`` to helpers and supported calls to ``math.*``."""

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Div):
            return self._wrap(node, _SAFE_DIV)
        if isinstance(node.op, ast.Mod):
            return self._wrap(node, _SAFE_MOD)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        func = node.func
        if isinstance(func, ast.Name):
            target = SCALAR_FUNCS.get(func.id)
            if target is not None and "." in target:
                mod, attr = target.split(".", 1)
                node.func = ast.Attribute(
                    value=ast.Name(id=mod, ctx=ast.Load()),
                    attr=attr,
                    ctx=ast.Load(),
                )
        return node

    @staticmethod
    def _wrap(node: ast.BinOp, name: str) -> ast.Call:
        return ast.Call(
            func=ast.Name(id=name, ctx=ast.Load()),
            args=[node.left, node.right],
            keywords=[],
        )


def scalar_source(source: str) -> str:
    """Validate ``source`` and lower it to scalar Python for the code generator."""
    tree = parse_expression(source)
    lowered = _ScalarLower().visit(tree)
    ast.fix_missing_locations(lowered)
    assert isinstance(lowered, ast.Expression)
    return ast.unparse(lowered)
