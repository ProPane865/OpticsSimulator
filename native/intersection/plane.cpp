// Ray/plane intersection kernel.
#include "optics/intersection/plane.hpp"

#include <cmath>
#include <optional>

#include "optics/common/linalg.hpp"

namespace optics::native {

std::optional<std::array<double, 3>> plane_intersect(const Vec3& ro, const Vec3& d, const Vec3& o,
                                                     const Vec3& n, double eps) {
    const double nz = n[2];
    if (std::abs(nz) <= eps || std::abs(d[0] * n[0] + d[1] * n[1] + d[2] * n[2]) <= eps) {
        return std::nullopt;
    }
    const double nxnz = n[0] / nz;
    const double nynz = n[1] / nz;
    SmallMat a;
    a.rows = a.cols = 3;
    a.at(0, 0) = -1.0;
    a.at(0, 1) = 0.0;
    a.at(0, 2) = d[0];
    a.at(1, 0) = 0.0;
    a.at(1, 1) = -1.0;
    a.at(1, 2) = d[1];
    a.at(2, 0) = nxnz;
    a.at(2, 1) = nynz;
    a.at(2, 2) = d[2];
    SmallVec b;
    b.n = 3;
    b.v[0] = -ro[0];
    b.v[1] = -ro[1];
    b.v[2] = nxnz * o[0] + nynz * o[1] + o[2] - ro[2];

    const double det = d[2] + d[1] * nynz + d[0] * nxnz;
    if (!std::isfinite(det) || std::abs(det) <= eps) {
        return std::nullopt;
    }
    std::optional<SmallVec> sol = solve_small(a, b);
    if (!sol.has_value()) {
        return std::nullopt;
    }
    const double u = sol->v[0];
    const double v = sol->v[1];
    const double t = sol->v[2];
    if (!std::isfinite(t) || t <= 0.0) {
        return std::nullopt;
    }
    if (!std::isfinite(u) || !std::isfinite(v)) {
        return std::nullopt;
    }
    return std::array<double, 3>{u, v, t};
}

}  // namespace optics::native
