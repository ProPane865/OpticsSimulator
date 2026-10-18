// Fixed-size 3x3 matrix helpers and the Rodrigues placement rotation
// (mirror of ``optics/vector.py``).
#pragma once

#include "optics/common/constants.hpp"
#include "optics/common/vector.hpp"

namespace optics::native {

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

// Row-major 3x3 product.
inline Mat3 mat3_mul(const Mat3& a, const Mat3& b) {
    Mat3 r;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            r.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] + a.m[i][2] * b.m[2][j];
        }
    }
    return r;
}

// Rodrigues rotation matrix mapping the +Z axis onto ``direction``; throws
// ``VectorError`` for a non-finite or zero direction.
inline Mat3 rotation_from_z(const Vec3& direction) {
    const Vec3 z_axis = make_vec3(0.0, 0.0, 1.0);
    const Vec3 direction_norm = normalize_vector(direction);
    const Vec3 axis = cross3(z_axis, direction_norm);
    const double axis_norm = norm(axis);
    const double cos_theta = dot3(z_axis, direction_norm);
    if (axis_norm < VECTOR_PARALLEL_TOL) {
        Mat3 r;
        if (cos_theta < 0.0) {
            r = Mat3::identity();
            r.m[1][1] = -1.0;
            r.m[2][2] = -1.0;
            return r;
        }
        return Mat3::identity();
    }
    const Mat3 skew{
        {{0.0, -axis[2], axis[1]}, {axis[2], 0.0, -axis[0]}, {-axis[1], axis[0], 0.0}}};
    const Mat3 ident = Mat3::identity();
    const double s = (1.0 - cos_theta) / (axis_norm * axis_norm);
    const Mat3 skew2 = mat3_mul(skew, skew);
    Mat3 r;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            r.m[i][j] = ident.m[i][j] + skew.m[i][j] + skew2.m[i][j] * s;
        }
    }
    return r;
}

}  // namespace optics::native
