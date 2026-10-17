// Newton iteration and intersection/inversion kernels.
#include "optics/solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "optics/constants.hpp"

namespace optics::native {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

bool all_finite(const SmallVec& f) {
    for (int i = 0; i < f.n; ++i) {
        if (!std::isfinite(f.v[i])) {
            return false;
        }
    }
    return true;
}

double vec_norm(const SmallVec& a) {
    double s = 0.0;
    for (int i = 0; i < a.n; ++i) {
        s += a.v[i] * a.v[i];
    }
    return std::sqrt(s);
}

double max_abs(const double* x, int n) {
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
        m = std::max(m, std::abs(x[i]));
    }
    return m;
}

SmallVec residual_vec(int kind, const SurfaceDesc& desc, const double* o, const double* d,
                      double x_target, double y_target, const double* x) {
    SmallVec f;
    if (kind == kNewtonIntersect) {
        f.n = 3;
        const Vec3 w = desc_point(desc, x[1], x[2]);
        const double t = x[0];
        f.v[0] = w[0] - (o[0] + t * d[0]);
        f.v[1] = w[1] - (o[1] + t * d[1]);
        f.v[2] = w[2] - (o[2] + t * d[2]);
    } else {
        f.n = 2;
        const Vec3 p = desc_eval(desc, x[0], x[1]);
        f.v[0] = p[0] - x_target;
        f.v[1] = p[1] - y_target;
    }
    return f;
}

SmallMat jacobian_mat(int kind, const SurfaceDesc& desc, const double* o, const double* d,
                      double x_target, double y_target, const double* x) {
    SmallMat j;
    if (kind == kNewtonIntersect) {
        const auto rv = desc_derivatives(desc, x[1], x[2]);
        const Vec3 ru = rv.first;
        const Vec3 rrv = rv.second;
        const Vec3 cu = desc.rot * ru;
        const Vec3 cv = desc.rot * rrv;
        j.rows = j.cols = 3;
        j.at(0, 0) = -d[0];
        j.at(1, 0) = -d[1];
        j.at(2, 0) = -d[2];
        j.at(0, 1) = cu[0];
        j.at(1, 1) = cu[1];
        j.at(2, 1) = cu[2];
        j.at(0, 2) = cv[0];
        j.at(1, 2) = cv[1];
        j.at(2, 2) = cv[2];
    } else {
        const auto rv = desc_derivatives(desc, x[0], x[1]);
        const Vec3 ru = rv.first;
        const Vec3 rrv = rv.second;
        j.rows = j.cols = 2;
        j.at(0, 0) = ru[0];
        j.at(0, 1) = rrv[0];
        j.at(1, 0) = ru[1];
        j.at(1, 1) = rrv[1];
    }
    (void)o;
    (void)d;
    (void)x_target;
    (void)y_target;
    return j;
}

}  // namespace

std::optional<NewtonResult> newton_solve(int kind, const SurfaceDesc& desc, const double* o,
                                         const double* d, double x_target, double y_target,
                                         const double* x0, int n0, bool clamp) {
    std::vector<double> x(x0, x0 + n0);
    int status = SOLVER_EXH;
    int damped = 0;
    SmallVec f = residual_vec(kind, desc, o, d, x_target, y_target, x.data());
    for (int iter = 0; iter < MAX_NEWTON_ITER; ++iter) {
        if (!all_finite(f)) {
            return std::nullopt;
        }
        const double f_norm = vec_norm(f);
        if (f_norm < CONVERGED_F_NORM) {
            return NewtonResult{x, f_norm, SOLVER_CONV};
        }
        SmallMat j = jacobian_mat(kind, desc, o, d, x_target, y_target, x.data());
        for (int r = 0; r < j.rows; ++r) {
            for (int c = 0; c < j.cols; ++c) {
                if (!std::isfinite(j.at(r, c))) {
                    return std::nullopt;
                }
            }
        }
        std::optional<SmallVec> delta = newton_delta(j, f);
        if (!delta.has_value() || !all_finite(*delta)) {
            return std::nullopt;
        }
        if (clamp) {
            const double clamp_scale = std::max(1.0, max_abs(x.data(), n0));
            const double dn = vec_norm(*delta);
            if (dn > CLAMP_REL * clamp_scale) {
                const double s = CLAMP_REL * clamp_scale / dn;
                for (int i = 0; i < delta->n; ++i) {
                    delta->v[i] *= s;
                }
            }
        }
        double step = 1.0;
        std::vector<double> trial(static_cast<std::size_t>(n0));
        for (int i = 0; i < n0; ++i) {
            trial[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(i)] + delta->v[i];
        }
        SmallVec ft = residual_vec(kind, desc, o, d, x_target, y_target, trial.data());
        while (!all_finite(ft)) {
            step *= 0.5;
            if (step < NEWTON_MIN_STEP) {
                return std::nullopt;
            }
            for (int i = 0; i < n0; ++i) {
                trial[static_cast<std::size_t>(i)] =
                    x[static_cast<std::size_t>(i)] + step * delta->v[i];
            }
            ft = residual_vec(kind, desc, o, d, x_target, y_target, trial.data());
        }
        if (step < 1.0) {
            damped += 1;
        }
        SmallVec scaled;
        scaled.n = delta->n;
        for (int i = 0; i < delta->n; ++i) {
            scaled.v[i] = step * delta->v[i];
        }
        const double step_norm = vec_norm(scaled);
        x = trial;
        f = ft;
        if (step < NEWTON_STALL_STEP || damped >= NEWTON_MAX_DAMPED_STEPS) {
            // Pinned against the edge of the domain: further iterations crawl.
            status = SOLVER_STALL;
            break;
        }
        if (step_norm < STEP_REL_TOL * std::max(1.0, max_abs(x.data(), n0))) {
            status = SOLVER_STALL;
            break;
        }
    }
    if (!all_finite(f)) {
        return std::nullopt;
    }
    return NewtonResult{std::move(x), vec_norm(f), status};
}

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
