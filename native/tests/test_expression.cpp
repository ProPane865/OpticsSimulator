// Expression parsing semantics: operator precedence, Python/NumPy quirks
// and the rejection of anything outside the restricted language.
#include <cmath>
#include <string>

#include "optics/expression/expression.hpp"

#include "check.hpp"

void test_expression() {
    using namespace optics::native;

    auto eval_src = [](const std::string& src, double u, double v) {
        return eval_expression(*parse_expression(src), u, v);
    };

    CHECK(close(eval_src("1 + 2 * 3", 0, 0), 7.0));
    CHECK(close(eval_src("-2 ** 2", 0, 0), -4.0));   // Python: -(2**2)
    CHECK(close(eval_src("2 ** -1", 0, 0), 0.5));    // right-assoc with unary
    CHECK(close(eval_src("7 % 3", 0, 0), 1.0));
    CHECK(close(eval_src("-7 % 3", 0, 0), 2.0));     // floor-mod
    CHECK(close(eval_src("sqrt(1 - (u**2) - (v**2))", 0.5, 0.0), std::sqrt(0.75)));
    CHECK(std::isnan(eval_src("sqrt(u)", -1.0, 0.0)));
    CHECK(std::isinf(eval_src("u / (v - 0.5)", 0.3, 0.5)));
    CHECK(std::isnan(eval_src("u / (v - 0.5)", 0.0, 0.5)));
    CHECK(close(eval_src("abs(u) + exp(0) + log(1) - sin(0) - cos(0)", -2.0, 0.0), 2.0));
    CHECK(close(eval_src("(u + 1) * (v - 2)", 1.0, 3.0), 2.0));

    auto must_throw = [&](const std::string& src) {
        bool threw = false;
        try {
            parse_expression(src);
        } catch (const ExpressionError&) {
            threw = true;
        }
        CHECK(threw);
    };
    must_throw("pi");
    must_throw("e");
    must_throw("u[0]");
    must_throw("u .real");
    must_throw("open('f')");
    must_throw("1 +");
    must_throw("foo(u)");
    must_throw("(lambda: 1)()");
    must_throw("u, v");
}
