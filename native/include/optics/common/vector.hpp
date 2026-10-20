// Fixed-size 3D vector type and helpers shared by every native module.
#pragma once

#include <cmath>
#include <cstddef>

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

// Flattened ``(3,)`` buffer to ``Vec3``.
inline Vec3 as_vec3(const double* a) { return make_vec3(a[0], a[1], a[2]); }

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

// Euclidean norm.
inline double norm(const Vec3& a) { return std::sqrt(dot3(a, a)); }

inline Vec3 normalize_vector(const Vec3& a) {
    double n = norm(a);
    if (!std::isfinite(n) || n == 0.0) {
        throw VectorError{"Vector must be finite and nonzero"};
    }
    return a / n;
}

}  // namespace optics::native
