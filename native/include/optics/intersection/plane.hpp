// Ray/plane intersection kernel.
#pragma once

#include <array>
#include <optional>

#include "optics/common/vector.hpp"

namespace optics::native {

// Returns ``(u, v, t)`` in plane parameters or ``nullopt``.
std::optional<std::array<double, 3>> plane_intersect(const Vec3& ro, const Vec3& d, const Vec3& o,
                                                     const Vec3& n, double eps);

}  // namespace optics::native
