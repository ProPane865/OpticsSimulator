// Fixed-size vector/matrix math and the small-system linear algebra shared by
// the Newton paths, mirroring ``optics/numerical/linalg.py``.
#pragma once

#include <cmath>
#include <cstddef>
#include <optional>

#include "optics/constants.hpp"

namespace optics::native {

// Error raised when a vector cannot be normalized; mapped to ``ValueError``.
struct VectorError {
    const char* message;
};

struct Vec3 {
    double v[3] = {0.0, 0.0, 0.0};

    double& operator[](std::size_t i) { return v[i]; }
    double operator[](std::size_t i) const { return v[i]; }
};

inline Vec3 make_vec3(double x, double y, double z) {
    Vec3 r;
    r[0] = x;
    r[1] = y;
    r[2] = z;
    return r;
}

inline Vec3 nan3() {
    return make_vec3(NAN, NAN, NAN);
}

inline bool finite3(const Vec3& a) {
    return std::isfinite(a[0]) && std::isfinite(a[1]) && std::isfinite(a[2]);
}

inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return make_vec3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return make_vec3(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

inline Vec3 operator-(const Vec3& a) {
    return make_vec3(-a[0], -a[1], -a[2]);
}

inline Vec3 operator*(double s, const Vec3& a) {
    return make_vec3(s * a[0], s * a[1], s * a[2]);
}

inline Vec3 operator*(const Vec3& a, double s) {
    return a * s;
}

inline Vec3 operator/(const Vec3& a, double s) {
    return make_vec3(a[0] / s, a[1] / s, a[2] / s);
}

inline double dot3(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline Vec3 cross3(const Vec3& a, const Vec3& b) {
    return make_vec3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]);
}

// Euclidean norm (mirrors ``optics.numerical.linalg.norm``).
inline double norm(const Vec3& a) { return std::sqrt(dot3(a, a)); }

inline Vec3 normalize_vector(const Vec3& a) {
    double n = norm(a);
    if (!std::isfinite(n) || n == 0.0) {
        throw VectorError{"Vector must be finite and nonzero"};
    }
    return a / n;
}

struct Mat3 {
    double m[3][3] = {};

    static Mat3 identity() {
        Mat3 r;
        r.m[0][0] = r.m[1][1] = r.m[2][2] = 1.0;
        return r;
    }
};

inline Vec3 operator*(const Mat3& a, const Vec3& x) {
    return make_vec3(a.m[0][0] * x[0] + a.m[0][1] * x[1] + a.m[0][2] * x[2],
                     a.m[1][0] * x[0] + a.m[1][1] * x[1] + a.m[1][2] * x[2],
                     a.m[2][0] * x[0] + a.m[2][1] * x[1] + a.m[2][2] * x[2]);
}

inline Vec3 mul_transpose(const Mat3& a, const Vec3& x) {
    return make_vec3(a.m[0][0] * x[0] + a.m[1][0] * x[1] + a.m[2][0] * x[2],
                     a.m[0][1] * x[0] + a.m[1][1] * x[1] + a.m[2][1] * x[2],
                     a.m[0][2] * x[0] + a.m[1][2] * x[1] + a.m[2][2] * x[2]);
}

// --- Small dense systems (2x2 / 3x3) -----------------------------------------

struct SmallVec {
    int n = 0;
    double v[3] = {0.0, 0.0, 0.0};
};

struct SmallMat {
    int rows = 0;
    int cols = 0;
    double a[9] = {};  // row-major

    double& at(int r, int c) { return a[r * cols + c]; }
    double at(int r, int c) const { return a[r * cols + c]; }
};

// Whether ``det`` is large relative to the product of the column norms;
// requires a square 2x2/3x3 input.
inline bool is_nonsingular(const SmallMat& j) {
    const int n = j.rows;
    if (n != j.cols) {
        return false;
    }
    double det = 0.0;
    double scale = 0.0;
    if (n == 3) {
        const double a = j.at(0, 0), b = j.at(0, 1), c = j.at(0, 2);
        const double d = j.at(1, 0), e = j.at(1, 1), f = j.at(1, 2);
        const double g = j.at(2, 0), h = j.at(2, 1), i = j.at(2, 2);
        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        scale = std::sqrt(a * a + d * d + g * g) * std::sqrt(b * b + e * e + h * h) *
                std::sqrt(c * c + f * f + i * i);
    } else if (n == 2) {
        const double a = j.at(0, 0), b = j.at(0, 1);
        const double c = j.at(1, 0), d = j.at(1, 1);
        det = a * d - b * c;
        scale = std::sqrt(a * a + c * c) * std::sqrt(b * b + d * d);
    } else {
        return false;
    }
    if (scale <= 0.0 || !std::isfinite(scale)) {
        return false;
    }
    return std::abs(det) / scale > NEWTON_SINGULAR_REL;
}

// Gaussian elimination with partial pivoting.
inline std::optional<SmallVec> solve_small(const SmallMat& mat, const SmallVec& b) {
    const int n = mat.rows;
    SmallMat m = mat;
    SmallVec y = b;
    for (int k = 0; k < n; ++k) {
        int p = k;
        double maxv = std::abs(m.at(k, k));
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(m.at(i, k)) > maxv) {
                maxv = std::abs(m.at(i, k));
                p = i;
            }
        }
        if (!std::isfinite(maxv) || maxv <= 0.0) {
            return std::nullopt;
        }
        if (p != k) {
            for (int j = 0; j < n; ++j) {
                const double tmp = m.at(k, j);
                m.at(k, j) = m.at(p, j);
                m.at(p, j) = tmp;
            }
            const double tmp = y.v[k];
            y.v[k] = y.v[p];
            y.v[p] = tmp;
        }
        for (int i = k + 1; i < n; ++i) {
            const double f = m.at(i, k) / m.at(k, k);
            if (f != 0.0) {
                for (int j = k; j < n; ++j) {
                    m.at(i, j) -= f * m.at(k, j);
                }
                y.v[i] -= f * y.v[k];
            }
        }
    }
    SmallVec x;
    x.n = n;
    for (int k = n - 1; k >= 0; --k) {
        double s = y.v[k];
        for (int j = k + 1; j < n; ++j) {
            s -= m.at(k, j) * x.v[j];
        }
        x.v[k] = s / m.at(k, k);
    }
    for (int k = 0; k < n; ++k) {
        if (!std::isfinite(x.v[k])) {
            return std::nullopt;
        }
    }
    return x;
}

// Newton step: square nonsingular solve, else normal-equation least squares;
// ``nullopt`` when the system is unusable.
inline std::optional<SmallVec> newton_delta(const SmallMat& j, const SmallVec& f) {
    SmallVec neg_f;
    neg_f.n = f.n;
    for (int i = 0; i < f.n; ++i) {
        neg_f.v[i] = -f.v[i];
    }
    if (j.rows == j.cols && (j.rows == 2 || j.rows == 3) && is_nonsingular(j)) {
        return solve_small(j, neg_f);
    }
    if (j.cols < 2 || j.cols > 3 || j.rows > 3) {
        return std::nullopt;
    }
    SmallMat jtj;
    jtj.rows = jtj.cols = j.cols;
    for (int c1 = 0; c1 < j.cols; ++c1) {
        for (int c2 = 0; c2 < j.cols; ++c2) {
            double s = 0.0;
            for (int r = 0; r < j.rows; ++r) {
                s += j.at(r, c1) * j.at(r, c2);
            }
            jtj.at(c1, c2) = s;
        }
    }
    if (!is_nonsingular(jtj)) {
        return std::nullopt;
    }
    SmallVec rhs;
    rhs.n = j.cols;
    for (int c = 0; c < j.cols; ++c) {
        double s = 0.0;
        for (int r = 0; r < j.rows; ++r) {
            s += j.at(r, c) * neg_f.v[r];
        }
        rhs.v[c] = s;
    }
    return solve_small(jtj, rhs);
}

}  // namespace optics::native
