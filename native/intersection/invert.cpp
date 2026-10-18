// Newton inversion of local ``(x, y)`` to surface parameters, from a single
// seed and from the grid-seeded nearest-sample search.
#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/intersection/solver.hpp"

namespace optics::native {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

}  // namespace

std::optional<std::pair<double, double>> solve_xy(const SurfaceDesc& desc, double x_target,
                                                  double y_target, double u, double v) {
    const double zeros[3] = {0.0, 0.0, 0.0};
    const double x0[2] = {u, v};
    std::optional<NewtonResult> res =
        newton_solve(kNewtonXY, desc, zeros, zeros, x_target, y_target, x0, 2, false);
    if (!res.has_value()) {
        return std::nullopt;
    }
    if (res->f_norm >= PARAM_TOLS[res->status]) {
        return std::nullopt;
    }
    const double uu = res->x[0];
    const double vv = res->x[1];
    if (!desc_accepts(desc, uu, vv)) {
        return std::nullopt;
    }
    return std::make_pair(uu, vv);
}

std::optional<std::pair<double, double>> parameters_at_xy(const SurfaceDesc& desc,
                                                          const SeedGrid& grid, double x,
                                                          double y) {
    const std::size_t n = grid.n();
    if (n == 0) {
        return std::nullopt;
    }
    std::vector<double> d2(n);
    bool any_fin = false;
    for (std::size_t i = 0; i < n; ++i) {
        if (grid.finite[i]) {
            const double dx = grid.pts[3 * i + 0] - x;
            const double dy = grid.pts[3 * i + 1] - y;
            d2[i] = dx * dx + dy * dy;
            any_fin = true;
        } else {
            d2[i] = kInf;
        }
    }
    if (!any_fin) {
        return std::nullopt;
    }
    std::vector<std::size_t> idx(n);
    for (std::size_t i = 0; i < n; ++i) {
        idx[i] = i;
    }
    const std::size_t k = std::min<std::size_t>(XY_SEEDS, n);
    // Mirror ``argpartition(d2, k - 1)[:k]`` followed by an ascending sort of
    // the k nearest samples.
    std::nth_element(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k) - 1, idx.end(),
                     [&](std::size_t a, std::size_t b) { return d2[a] < d2[b]; });
    std::sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k),
              [&](std::size_t a, std::size_t b) { return d2[a] < d2[b]; });
    for (std::size_t oi = 0; oi < k; ++oi) {
        const std::size_t i = idx[oi];
        if (!grid.finite[i]) {
            continue;
        }
        std::optional<std::pair<double, double>> uv =
            solve_xy(desc, x, y, grid.uf[i], grid.vf[i]);
        if (uv.has_value()) {
            return uv;
        }
    }
    return std::nullopt;
}

}  // namespace optics::native
