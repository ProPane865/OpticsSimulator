// Restricted, safe surface-expression language: tokenize, parse to an AST,
// evaluate.
// Grammar: variables ``u``, ``v``; numeric literals; ``+ - * / % **``, unary
// ``+``/``-``; and the unary calls ``sqrt``, ``sin``, ``cos``, ``tan``,
// ``exp``, ``log``, ``abs``. Division and modulo use NumPy scalar semantics
// (``inf``/``nan`` instead of throwing).
#pragma once

#include <memory>
#include <stdexcept>
#include <string>

namespace optics::native {

// Raised for any expression outside the supported language (mapped to the
// Python-side ``ExpressionError``).
struct ExpressionError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

enum class ExprKind {
    Num,
    VarU,
    VarV,
    Pos,
    Neg,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Call,
};

enum class FuncKind { Sqrt, Sin, Cos, Tan, Exp, Log, Abs };

struct Expr {
    ExprKind kind = ExprKind::Num;
    double num = 0.0;
    FuncKind fn = FuncKind::Sqrt;
    std::unique_ptr<Expr> lhs;
    std::unique_ptr<Expr> rhs;
};

using ExprPtr = std::unique_ptr<Expr>;

// Parse and validate a surface expression; throws ``ExpressionError``.
ExprPtr parse_expression(const std::string& source);

// Evaluate a parsed expression at ``(u, v)``.
double eval_expression(const Expr& node, double u, double v);

}  // namespace optics::native
