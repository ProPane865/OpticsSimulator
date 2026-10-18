// Interface normal orientation and the refraction/reflection direction
// kernels (Snell with total internal reflection).
#pragma once

#include <utility>

#include "optics/common/vector.hpp"

namespace optics::native {

Vec3 orient(const Vec3& normal, const Vec3& incident);
std::pair<Vec3, bool> refract_or_reflect(const Vec3& incident, const Vec3& normal, double n_from,
                                         double n_to);

}  // namespace optics::native
