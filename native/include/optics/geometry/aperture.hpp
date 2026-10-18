// Surface aperture shapes. The variant is the extension point: the aperture
// membership test visits over it, and new shapes (rectangular cuts, masks)
// are added here without touching the descriptor or kernels.
#pragma once

#include <variant>

#include "optics/common/constants.hpp"
#include "optics/common/vector.hpp"

namespace optics::native {

struct NoAperture {};

// Disc in the local ``(x, y)`` plane (local-space surface points, so the
// world transform is irrelevant to membership).
struct CircularAperture {
    double radius = 0.0, cx = 0.0, cy = 0.0;
};

using Aperture = std::variant<NoAperture, CircularAperture>;

// Whether a local-space surface point lies inside the aperture. Non-finite
// points only ever fail when an aperture is present (mirroring the historical
// descriptor test's operation order and epsilon).
inline bool aperture_contains(const Aperture& aperture, const Vec3& p_local) {
    if (const auto* circle = std::get_if<CircularAperture>(&aperture)) {
        if (!finite3(p_local)) {
            return false;
        }
        const double dx = p_local[0] - circle->cx;
        const double dy = p_local[1] - circle->cy;
        return dx * dx + dy * dy <= circle->radius * circle->radius + APERTURE_EPS;
    }
    return true;
}

}  // namespace optics::native
