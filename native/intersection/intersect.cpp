// Seed selection, Newton verification and dedup for ray/surface
// intersection: the nearest forward hit over the sampled seed grid.
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/intersection/solver.hpp"

namespace optics::native {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

}  // namespace

std::optional<HitUV> newton_intersection(const SurfaceDesc& desc, const double* o, const double* d,
                                         double u, double v, double t) {
    const double x0[3] = {t, u, v};
    std::optional<NewtonResult> res =
        newton_solve(kNewtonIntersect, desc, o, d, 0.0, 0.0, x0, 3, true);
    if (!res.has_value()) {
        return std::nullopt;
    }
    if (res->f_norm >= HIT_TOLS[res->status]) {
        return std::nullopt;
    }
    const double tt = res->x[0];
    const double uu = res->x[1];
    const double vv = res->x[2];
    if (tt > 0.0 && desc_accepts(desc, uu, vv)) {
        return HitUV{tt, uu, vv};
    }
    return std::nullopt;
}

std::optional<HitUV> surface_intersect(const SurfaceDesc& desc, const SeedGrid& grid,
                                       const double* o, const double* d) {
    const std::size_t m = grid.m();
    if (m == 0) {
        return std::nullopt;
    }
    const std::vector<double>& p = grid.p_world;
    std::vector<double> ts(m);
    std::vector<double> residual(m);
    for (std::size_t i = 0; i < m; ++i) {
        const double rel0 = p[3 * i + 0] - o[0];
        const double rel1 = p[3 * i + 1] - o[1];
        const double rel2 = p[3 * i + 2] - o[2];
        const double t = rel0 * d[0] + rel1 * d[1] + rel2 * d[2];
        ts[i] = t;
        if (t > 0.0) {
            const double px = o[0] + t * d[0] - p[3 * i + 0];
            const double py = o[1] + t * d[1] - p[3 * i + 1];
            const double pz = o[2] + t * d[2] - p[3 * i + 2];
            residual[i] = std::sqrt(px * px + py * py + pz * pz);
        } else {
            residual[i] = kInf;
        }
    }

    auto count_seeds = [&](double tol) {
        std::size_t count = 0;
        for (std::size_t i = 0; i < m; ++i) {
            if (ts[i] > 0.0 && std::isfinite(residual[i]) && residual[i] <= tol) {
                ++count;
            }
        }
        return count;
    };

    double tol = SEED_TOL;
    std::size_t count = count_seeds(tol);
    if (count == 0) {
        tol = SEED_TOL_FALLBACK;
        count = count_seeds(tol);
        if (count == 0) {
            return std::nullopt;
        }
    }

    std::vector<std::size_t> idx;
    idx.reserve(count);
    for (std::size_t i = 0; i < m; ++i) {
        if (ts[i] > 0.0 && std::isfinite(residual[i]) && residual[i] <= tol) {
            idx.push_back(i);
        }
    }
    // Mirrors ``argsort(residual[idx])``; the stable sort fixes the order of
    // exact ties to ascending sample index.
    std::stable_sort(idx.begin(), idx.end(),
                     [&](std::size_t a, std::size_t b) { return residual[a] < residual[b]; });

    const std::size_t max_seeds = std::min<std::size_t>(idx.size(), MAX_SEEDS);
    std::vector<double> hit_t(max_seeds), hit_u(max_seeds), hit_v(max_seeds);
    std::size_t n_hits = 0;
    for (std::size_t k = 0; k < max_seeds; ++k) {
        const std::size_t i = idx[k];
        std::optional<HitUV> hit = newton_intersection(desc, o, d, grid.uf_valid[i], grid.vf_valid[i],
                                                       ts[i]);
        if (!hit.has_value()) {
            continue;
        }
        bool dup = false;
        for (std::size_t h = 0; h < n_hits; ++h) {
            if (std::abs(hit->u - hit_u[h]) < DEDUP_UV_TOL && std::abs(hit->v - hit_v[h]) < DEDUP_UV_TOL) {
                if (hit->t < hit_t[h]) {
                    hit_t[h] = hit->t;
                    hit_u[h] = hit->u;
                    hit_v[h] = hit->v;
                }
                dup = true;
                break;
            }
        }
        if (!dup) {
            hit_t[n_hits] = hit->t;
            hit_u[n_hits] = hit->u;
            hit_v[n_hits] = hit->v;
            ++n_hits;
        }
    }
    if (n_hits == 0) {
        return std::nullopt;
    }
    std::size_t best = 0;
    for (std::size_t h = 1; h < n_hits; ++h) {
        if (hit_t[h] < hit_t[best]) {
            best = h;
        }
    }
    return HitUV{hit_t[best], hit_u[best], hit_v[best]};
}

}  // namespace optics::native
