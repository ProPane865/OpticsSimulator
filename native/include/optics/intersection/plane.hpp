// Ray/plane intersection kernel (mirror of ``optics/numerical/plane.py``).
#pragma once

#include <array>
#include <optional>

namespace optics::native {

// Returns ``(u, v, t)`` in plane parameters or ``nullopt``.
std::optional<std::array<double, 3>> plane_intersect(const double* ro, const double* d,
                                                     const double* o, const double* n,
                                                     double eps);

}  // namespace optics::native
