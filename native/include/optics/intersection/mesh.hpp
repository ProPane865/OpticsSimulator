// Ray/mesh intersection kernel: Moller-Trumbore over flattened ``(K, 3, 3)``
// triangles.
#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "optics/common/vector.hpp"

namespace optics::native {

// Nearest forward hit with barycentric ``(u, v)`` clamped into the triangle.
struct MeshHit {
    double t;
    std::size_t face;
    double u, v;
};

std::optional<MeshHit> mesh_intersect(const Vec3& o, const Vec3& d, const std::vector<double>& tri,
                                      double det_eps, double bary_tol, double min_face_t);

}  // namespace optics::native
