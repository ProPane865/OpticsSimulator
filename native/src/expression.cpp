// Parser/evaluator for the restricted surface-expression language.
#include "optics/expression.hpp"

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace optics::native {
namespace {

using FuncMap = std::unordered_map<std::string, FuncKind>;

const FuncMap& func_table() {
    static const FuncMap table = {
        {"sqrt", FuncKind::Sqrt}, {"sin", FuncKind::Sin}, {"cos", FuncKind::Cos},
        {"tan", FuncKind::Tan},   {"exp", FuncKind::Exp}, {"log", FuncKind::Log},
        {"abs", FuncKind::Abs},
    };
    return table;
}

enum class Tok { End, Num, Ident, Plus, Minus, Star, Slash, Percent, Pow, LParen, RParen, Comma };

struct Token {
    Tok kind = Tok::End;
    double num = 0.0;
    std::string text;
    std::size_t pos = 0;
};

bool is_ident_start(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}

bool is_ident_char(char c) { return is_ident_start(c) || (c >= '0' && c <= '9'); }

bool is_digit(char c) { return c >= '0' && c <= '9'; }

class Parser {
  public:
    explicit Parser(const std::string& source) : src_(source) { tokenize(); }

    ExprPtr parse() {
        ExprPtr expr = parse_expr();
        if (peek().kind != Tok::End) {
            fail("Unsupported syntax", peek().pos);
        }
        return expr;
    }

  private:
    const std::string& src_;
    std::vector<Token> toks_;
    std::size_t at_ = 0;

    void tokenize() {
        const std::string& s = src_;
        std::size_t i = 0;
        const std::size_t n = s.size();
        while (i < n) {
            char c = s[i];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++i;
                continue;
            }
            if (is_digit(c) || (c == '.' && i + 1 < n && is_digit(s[i + 1]))) {
                scan_number(i);
                continue;
            }
            if (is_ident_start(c)) {
                Token t;
                t.kind = Tok::Ident;
                t.pos = i;
                std::size_t start = i;
                while (i < n && is_ident_char(s[i])) {
                    ++i;
                }
                t.text = s.substr(start, i - start);
                toks_.push_back(std::move(t));
                continue;
            }
            Token t;
            t.pos = i;
            switch (c) {
                case '+': t.kind = Tok::Plus; ++i; break;
                case '-': t.kind = Tok::Minus; ++i; break;
                case '*':
                    if (i + 1 < n && s[i + 1] == '*') {
                        t.kind = Tok::Pow;
                        i += 2;
                    } else {
                        t.kind = Tok::Star;
                        ++i;
                    }
                    break;
                case '/': t.kind = Tok::Slash; ++i; break;
                case '%': t.kind = Tok::Percent; ++i; break;
                case '(': t.kind = Tok::LParen; ++i; break;
                case ')': t.kind = Tok::RParen; ++i; break;
                case ',': t.kind = Tok::Comma; ++i; break;
                default: fail("Unsupported syntax", i);
            }
            toks_.push_back(std::move(t));
        }
        Token end;
        end.kind = Tok::End;
        end.pos = n;
        toks_.push_back(end);
    }

    void scan_number(std::size_t& i) {
        Token t;
        t.kind = Tok::Num;
        t.pos = i;
        const std::string& s = src_;
        const std::size_t n = s.size();
        std::string digits;
        bool seen_dot = false;
        while (i < n) {
            char c = s[i];
            if (is_digit(c) || c == '_') {
                if (c != '_') {
                    digits.push_back(c);
                }
                ++i;
            } else if (c == '.' && !seen_dot) {
                seen_dot = true;
                digits.push_back('.');
                ++i;
            } else {
                break;
            }
        }
        if (i < n && (s[i] == 'e' || s[i] == 'E')) {
            std::string exp_part;
            exp_part.push_back('e');
            std::size_t j = i + 1;
            if (j < n && (s[j] == '+' || s[j] == '-')) {
                exp_part.push_back(s[j]);
                ++j;
            }
            if (j < n && is_digit(s[j])) {
                while (j < n && (is_digit(s[j]) || s[j] == '_')) {
                    if (s[j] != '_') {
                        exp_part.push_back(s[j]);
                    }
                    ++j;
                }
                digits += exp_part;
                i = j;
            }
        }
        if (i < n && (is_ident_start(s[i]) || s[i] == '.')) {
            fail("Unsupported constant", t.pos);
        }
        try {
            t.num = std::stod(digits);
        } catch (const std::exception&) {
            fail("Unsupported constant", t.pos);
        }
        toks_.push_back(std::move(t));
    }

    const Token& peek() const { return toks_[at_]; }
    const Token& advance() { return toks_[at_++]; }

    [[noreturn]] void fail(const std::string& what, std::size_t pos) const {
        std::string msg = what;
        msg += " at position ";
        msg += std::to_string(pos);
        throw ExpressionError(msg);
    }

    bool match(Tok kind) {
        if (peek().kind == kind) {
            ++at_;
            return true;
        }
        return false;
    }

    void expect(Tok kind, const char* what) {
        if (peek().kind != kind) {
            fail(what, peek().pos);
        }
        ++at_;
    }

    ExprPtr make(ExprKind kind, ExprPtr lhs, ExprPtr rhs) {
        auto node = std::make_unique<Expr>();
        node->kind = kind;
        node->lhs = std::move(lhs);
        node->rhs = std::move(rhs);
        return node;
    }

    ExprPtr parse_expr() {
        ExprPtr node = parse_term();
        while (peek().kind == Tok::Plus || peek().kind == Tok::Minus) {
            const ExprKind kind = peek().kind == Tok::Plus ? ExprKind::Add : ExprKind::Sub;
            advance();
            node = make(kind, std::move(node), parse_term());
        }
        return node;
    }

    ExprPtr parse_term() {
        ExprPtr node = parse_factor();
        for (;;) {
            ExprKind kind;
            switch (peek().kind) {
                case Tok::Star: kind = ExprKind::Mul; break;
                case Tok::Slash: kind = ExprKind::Div; break;
                case Tok::Percent: kind = ExprKind::Mod; break;
                default:
                    return node;
            }
            advance();
            node = make(kind, std::move(node), parse_factor());
        }
    }

    ExprPtr parse_factor() {
        if (peek().kind == Tok::Plus) {
            advance();
            auto node = std::make_unique<Expr>();
            node->kind = ExprKind::Pos;
            node->lhs = parse_factor();
            return node;
        }
        if (peek().kind == Tok::Minus) {
            advance();
            auto node = std::make_unique<Expr>();
            node->kind = ExprKind::Neg;
            node->lhs = parse_factor();
            return node;
        }
        return parse_power();
    }

    // Right-associative ``**``; the exponent may carry a unary sign (``2 ** -1``).
    ExprPtr parse_power() {
        ExprPtr base = parse_atom();
        if (peek().kind == Tok::Pow) {
            advance();
            return make(ExprKind::Pow, std::move(base), parse_factor());
        }
        return base;
    }

    ExprPtr parse_atom() {
        const Token& t = peek();
        if (t.kind == Tok::Num) {
            advance();
            auto node = std::make_unique<Expr>();
            node->kind = ExprKind::Num;
            node->num = t.num;
            return node;
        }
        if (t.kind == Tok::Ident) {
            advance();
            if (t.text == "u" || t.text == "v") {
                if (peek().kind == Tok::LParen) {
                    fail("Unknown variable '" + t.text + "'", t.pos);
                }
                auto node = std::make_unique<Expr>();
                node->kind = t.text == "u" ? ExprKind::VarU : ExprKind::VarV;
                return node;
            }
            if (peek().kind == Tok::LParen) {
                advance();
                const auto it = func_table().find(t.text);
                std::vector<ExprPtr> args;
                if (peek().kind != Tok::RParen) {
                    args.push_back(parse_expr());
                    while (match(Tok::Comma)) {
                        args.push_back(parse_expr());
                    }
                }
                expect(Tok::RParen, "expected ')'");
                if (it == func_table().end()) {
                    throw ExpressionError("Unsupported function '" + t.text + "'");
                }
                if (args.size() != 1) {
                    throw ExpressionError("Unsupported call: '" + t.text +
                                          "' takes exactly one argument");
                }
                auto node = std::make_unique<Expr>();
                node->kind = ExprKind::Call;
                node->fn = it->second;
                node->lhs = std::move(args[0]);
                return node;
            }
            throw ExpressionError("Unknown variable '" + t.text + "'");
        }
        if (t.kind == Tok::LParen) {
            advance();
            ExprPtr inner = parse_expr();
            expect(Tok::RParen, "expected ')'");
            return inner;
        }
        fail("Unsupported syntax", t.pos);
    }
};

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

ExprPtr parse_expression(const std::string& source) {
    try {
        Parser parser(source);
        return parser.parse();
    } catch (const ExpressionError&) {
        throw;
    } catch (const std::exception& exc) {
        throw ExpressionError(std::string("Invalid expression: ") + exc.what());
    }
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
