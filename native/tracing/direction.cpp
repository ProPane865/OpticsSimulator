// Interface normal orientation and refraction/reflection directions.
#include "optics/tracing/direction.hpp"

namespace optics::native {

Vec3 orient(const Vec3& normal, const Vec3& incident) {
    if (dot3(normal, incident) < 0.0) {
        return -normal;
    }
    return normal;
}

std::pair<Vec3, bool> refract_or_reflect(const Vec3& incident, const Vec3& normal, double n_from,
                                         double n_to) {
    const double eta = n_from / n_to;
    const double cos_i = dot3(normal, incident);
    const double k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    if (k < 0.0) {
        return {normalize_vector(incident - (2.0 * dot3(normal, incident)) * normal), false};
    }
    const Vec3 dir = eta * incident + (std::sqrt(k) - eta * cos_i) * normal;
    return {normalize_vector(dir), true};
}

}  // namespace optics::native
