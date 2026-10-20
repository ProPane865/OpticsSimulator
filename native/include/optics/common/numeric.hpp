// Small shared numeric helpers (no numerical state): the seed/bounds grids all
// sample their parameters with ``numpy.linspace`` semantics, endpoint
// included; a single sample is ``[a]``.
#pragma once

#include <vector>

namespace optics::native {

inline std::vector<double> linspace(double a, double b, int n) {
    std::vector<double> out(static_cast<std::size_t>(n < 0 ? 0 : n), 0.0);
    if (n <= 0) {
        return out;
    }
    if (n == 1) {
        out[0] = a;
        return out;
    }
    const double step = (b - a) / (n - 1);
    for (int i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = a + step * i;
    }
    out[static_cast<std::size_t>(n - 1)] = b;
    return out;
}

}  // namespace optics::native
