// Batch measurement kernels: plane power accumulation, direct and
// stack-traced plane intersection, and the fused single-trace rendering +
// all-planes pass.
#include "optics/measurement/measurement.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>

#include "optics/common/parallel.hpp"
#include "optics/common/vector.hpp"
#include "optics/intersection/plane.hpp"
#include "optics/tracing/trace.hpp"

namespace optics::native {
namespace {

// The ray that reaches the plane, read off the shared walk. Mirrors
// ``trace_stack_to_plane`` exactly: at each gap, the plane is measured there
// when no element is entered before the crossing (crossing t <= entry_t, or
// parallel with no element); otherwise the element's internal segments are
// checked (in extent and within the segment), continuing to the next gap on
// exit. Returns ``false`` when the ray never reaches the plane.
bool measure_hops_plane(const RayHops& hops, const double* po, const double* pn, double u_min,
                        double u_max, double v_min, double v_max, double eps, double* u,
                        double* v) {
    for (const RayHop& h : hops) {
        const std::optional<std::array<double, 3>> ph = plane_intersect(h.o.v, h.d.v, po, pn, eps);
        const bool enter = h.has_element && (!ph.has_value() || ph->at(2) > h.entry_t);
        if (!enter) {
            if (ph.has_value()) {
                *u = ph->at(0);
                *v = ph->at(1);
                return true;
            }
            return false;
        }
        const std::size_t m = h.pts.size();
        for (std::size_t k = 0; k + 1 < m; ++k) {
            const Vec3 seg_o = h.pts[k];
            const Vec3 diff = h.pts[k + 1] - seg_o;
            const double seg_len = norm(diff);
            if (seg_len <= 0.0) {
                continue;
            }
            const Vec3 seg_d = diff / seg_len;
            const std::optional<std::array<double, 3>> hit = plane_intersect(seg_o.v, seg_d.v, po, pn, eps);
            if (hit.has_value() && hit->at(0) >= u_min && hit->at(0) <= u_max &&
                hit->at(1) >= v_min && hit->at(1) <= v_max && hit->at(2) <= seg_len) {
                *u = hit->at(0);
                *v = hit->at(1);
                return true;
            }
        }
        if (!h.exited) {
            return false;  // absorbed inside the element before reaching the plane
        }
    }
    return false;
}

}  // namespace

double accumulate_power(double* grid, const double* u, const double* v, const double* power,
                        std::size_t n, double u_min, double u_max, double v_min, double v_max,
                        int ru, int rv, double interval_u, double interval_v) {
    double total = 0.0;
    (void)rv;
    for (std::size_t idx = 0; idx < n; ++idx) {
        const double uu = u[idx];
        const double vv = v[idx];
        const double pw = power[idx];
        if (!(std::isfinite(uu) && std::isfinite(vv) && std::isfinite(pw))) {
            continue;
        }
        if (uu < u_min || uu > u_max || vv < v_min || vv > v_max) {
            continue;
        }
        int i = static_cast<int>(std::floor((uu - u_min) / interval_u * ru));
        int j = static_cast<int>(std::floor((vv - v_min) / interval_v * rv));
        i = std::min(std::max(i, 0), ru - 1);
        j = std::min(std::max(j, 0), rv - 1);
        grid[static_cast<std::size_t>(j) * static_cast<std::size_t>(ru) +
             static_cast<std::size_t>(i)] += pw;
        total += pw;
    }
    return total;
}

void measure_direct(const double* origins, const double* dirs, const double* powers, std::size_t n,
                    const double* o, const double* nrm, double eps, double* us, double* vs,
                    double* pws) {
    run_chunks(n, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i) {
            std::optional<std::array<double, 3>> hit =
                plane_intersect(origins + 3 * i, dirs + 3 * i, o, nrm, eps);
            if (hit.has_value()) {
                us[i] = hit->at(0);
                vs[i] = hit->at(1);
                pws[i] = powers[i];
            }
        }
    });
}

void measure_stack(const TraceStack& stack, const double* origins, const double* dirs,
                   const double* powers, std::size_t n, const double* o, const double* nrm,
                   double eps, double u_min, double u_max, double v_min, double v_max, double* us,
                   double* vs, double* pws) {
    run_chunks(n, [&](std::size_t lo, std::size_t hi) {
        double out_o[3];
        double out_d[3];
        for (std::size_t i = lo; i < hi; ++i) {
            const bool found =
                trace_stack_to_plane(stack, origins + 3 * i, dirs + 3 * i, o, nrm, eps, u_min,
                                     u_max, v_min, v_max, out_o, out_d);
            if (!found) {
                continue;
            }
            std::optional<std::array<double, 3>> hit =
                plane_intersect(out_o, out_d, o, nrm, eps);
            if (hit.has_value()) {
                us[i] = hit->at(0);
                vs[i] = hit->at(1);
                pws[i] = powers[i];
            }
        }
    });
}

void measure_planes_multi(const TraceStack& stack, const double* origins, const double* dirs,
                          const double* powers, std::size_t n, std::size_t p, const double* po,
                          const double* pn, const double* u_min, const double* u_max,
                          const double* v_min, const double* v_max, double eps, double* us,
                          double* vs, double* pws) {
    run_chunks(n, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i) {
            const RayHops hops = build_hops(stack, origins + 3 * i, dirs + 3 * i);
            for (std::size_t q = 0; q < p; ++q) {
                double u = 0.0;
                double v = 0.0;
                if (measure_hops_plane(hops, po + 3 * q, pn + 3 * q, u_min[q], u_max[q], v_min[q],
                                       v_max[q], eps, &u, &v)) {
                    const std::size_t idx = q * n + i;
                    us[idx] = u;
                    vs[idx] = v;
                    pws[idx] = powers[i];
                }
            }
        }
    });
}

void trace_segments_and_planes(const TraceStack& stack, const double* origins, const double* dirs,
                               const double* powers, std::size_t n, double ray_length, std::size_t p,
                               const double* po, const double* pn, const double* u_min,
                               const double* u_max, const double* v_min, const double* v_max,
                               double eps, double* starts, double* ends, signed char* kinds,
                               std::int64_t* counts, std::size_t cap, double* us, double* vs,
                               double* pws) {
    run_chunks(n, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i) {
            const RayHops hops = build_hops(stack, origins + 3 * i, dirs + 3 * i);
            counts[i] = format_segments(hops, ray_length, starts + 3 * cap * i, ends + 3 * cap * i,
                                        kinds + cap * i, cap);
            for (std::size_t q = 0; q < p; ++q) {
                double u = 0.0;
                double v = 0.0;
                if (measure_hops_plane(hops, po + 3 * q, pn + 3 * q, u_min[q], u_max[q], v_min[q],
                                       v_max[q], eps, &u, &v)) {
                    us[q * n + i] = u;
                    vs[q * n + i] = v;
                    pws[q * n + i] = powers[i];
                }
            }
        }
    });
}

}  // namespace optics::native
