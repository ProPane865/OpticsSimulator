// Interpreter over the parsed expression tree, with NumPy-scalar division
// and modulo semantics.
#include <cmath>

#include "optics/expression/expression.hpp"

namespace optics::native {
namespace {

double eval_call(FuncKind fn, double x) {
    switch (fn) {
        case FuncKind::Sqrt: return std::sqrt(x);
        case FuncKind::Sin: return std::sin(x);
        case FuncKind::Cos: return std::cos(x);
        case FuncKind::Tan: return std::tan(x);
        case FuncKind::Exp: return std::exp(x);
        case FuncKind::Log: return std::log(x);
        case FuncKind::Abs: return std::fabs(x);
    }
    return NAN;
}

}  // namespace

double sdiv(double a, double b) {
    if (b == 0.0) {
        if (a == 0.0) {
            return NAN;
        }
        return std::copysign(INFINITY, a);
    }
    return a / b;
}

double smod(double a, double b) {
    if (b == 0.0) {
        return NAN;
    }
    // Python/NumPy modulo takes the sign of the divisor; C++ ``fmod`` takes the
    // sign of the dividend, so fold it over when they disagree.
    double r = std::fmod(a, b);
    if (r != 0.0 && ((r < 0.0) != (b < 0.0))) {
        r += b;
    }
    return r;
}

double eval_expression(const Expr& node, double u, double v) {
    switch (node.kind) {
        case ExprKind::Num: return node.num;
        case ExprKind::VarU: return u;
        case ExprKind::VarV: return v;
        case ExprKind::Pos: return +eval_expression(*node.lhs, u, v);
        case ExprKind::Neg: return -eval_expression(*node.lhs, u, v);
        case ExprKind::Add: return eval_expression(*node.lhs, u, v) + eval_expression(*node.rhs, u, v);
        case ExprKind::Sub: return eval_expression(*node.lhs, u, v) - eval_expression(*node.rhs, u, v);
        case ExprKind::Mul: return eval_expression(*node.lhs, u, v) * eval_expression(*node.rhs, u, v);
        case ExprKind::Div: return sdiv(eval_expression(*node.lhs, u, v), eval_expression(*node.rhs, u, v));
        case ExprKind::Mod: return smod(eval_expression(*node.lhs, u, v), eval_expression(*node.rhs, u, v));
        case ExprKind::Pow: return std::pow(eval_expression(*node.lhs, u, v), eval_expression(*node.rhs, u, v));
        case ExprKind::Call: return eval_call(node.fn, eval_expression(*node.lhs, u, v));
    }
    return NAN;
}

}  // namespace optics::native
