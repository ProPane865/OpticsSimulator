// Compiled surface model and descriptor helpers.
#include "optics/geometry/surface.hpp"

#include <algorithm>
#include <cmath>
#include <optional>

#include "optics/common/constants.hpp"

namespace optics::native {

namespace {

double fd_step(double lo, double hi) {
    return std::max(FD_STEP_REL * std::abs(hi - lo), FD_MIN_STEP);
}

}  // namespace

SurfaceModel::SurfaceModel(ExprPtr x, ExprPtr y, ExprPtr z, double u_min, double u_max,
                           double v_min, double v_max)
    : fx_(std::move(x)),
      fy_(std::move(y)),
      fz_(std::move(z)),
      hu_(fd_step(u_min, u_max)),
      hv_(fd_step(v_min, v_max)),
      u_min_(u_min),
      u_max_(u_max),
      v_min_(v_min),
      v_max_(v_max) {}

std::shared_ptr<const SurfaceModel> SurfaceModel::compile(const std::string& x_src,
                                                          const std::string& y_src,
                                                          const std::string& z_src, double u_min,
                                                          double u_max, double v_min,
                                                          double v_max) {
    ExprPtr x = parse_expression(x_src);
    ExprPtr y = parse_expression(y_src);
    ExprPtr z = parse_expression(z_src);
    return std::make_shared<const SurfaceModel>(std::move(x), std::move(y), std::move(z), u_min,
                                                u_max, v_min, v_max);
}

Vec3 SurfaceModel::eval_local(double u, double v) const {
    return make_vec3(eval_expression(*fx_, u, v), eval_expression(*fy_, u, v),
                     eval_expression(*fz_, u, v));
}

void SurfaceModel::eval_grid(const double* us, const double* vs, std::size_t n, double* out) const {
    for (std::size_t i = 0; i < n; ++i) {
        const Vec3 p = eval_local(us[i], vs[i]);
        out[3 * i + 0] = p[0];
        out[3 * i + 1] = p[1];
        out[3 * i + 2] = p[2];
    }
}

Vec3 SurfaceModel::partial_rt(double u, double v, double h, double lo, double hi,
                              bool along_u) const {
    double coord;
    if (along_u) {
        coord = u;
        if (lo <= u - h && u + h <= hi) {
            const Vec3 f = eval_local(u + h, v);
            const Vec3 b = eval_local(u - h, v);
            if (finite3(f) && finite3(b)) {
                return (f - b) / (2.0 * h);
            }
        }
    } else {
        coord = v;
        if (lo <= v - h && v + h <= hi) {
            const Vec3 f = eval_local(u, v + h);
            const Vec3 b = eval_local(u, v - h);
            if (finite3(f) && finite3(b)) {
                return (f - b) / (2.0 * h);
            }
        }
    }
    const Vec3 base = eval_local(u, v);
    if (!finite3(base)) {
        return nan3();
    }
    const double center = 0.5 * (lo + hi);
    const double inward = center >= coord ? 1.0 : -1.0;
    for (int s = 0; s < 2; ++s) {
        const double sign = s == 0 ? inward : -inward;
        double step = h;
        for (int t = 0; t < FD_STEP_SHRINKS; ++t) {
            const double nxt = coord + sign * step;
            if (nxt < lo || nxt > hi) {
                step *= 0.5;
                continue;
            }
            const Vec3 p = along_u ? eval_local(nxt, v) : eval_local(u, nxt);
            if (finite3(p)) {
                return (p - base) / (sign * step);
            }
            step *= 0.5;
        }
    }
    return nan3();
}

std::pair<Vec3, Vec3> SurfaceModel::derivatives(double u, double v) const {
    const Vec3 ru = partial_rt(u, v, hu_, u_min_, u_max_, true);
    const Vec3 rv = partial_rt(u, v, hv_, v_min_, v_max_, false);
    return {ru, rv};
}

bool desc_accepts(const SurfaceDesc& d, double u, double v) {
    if (u < d.u_min - d.tu || u > d.u_max + d.tu) {
        return false;
    }
    if (v < d.v_min - d.tv || v > d.v_max + d.tv) {
        return false;
    }
    if (d.ap_r >= 0.0) {
        const Vec3 p = desc_eval(d, u, v);
        if (!finite3(p)) {
            return false;
        }
        const double dx = p[0] - d.ap_cx;
        const double dy = p[1] - d.ap_cy;
        return dx * dx + dy * dy <= d.ap_r * d.ap_r + APERTURE_EPS;
    }
    return true;
}

namespace {

// One-sided finite-difference partial in direction ``(du, dv)``.
Vec3 partial_desc(const SurfaceDesc& d, double u, double v, double du, double dv) {
    const Vec3 base = desc_eval(d, u, v);
    if (!finite3(base)) {
        return nan3();
    }
    const double step = FD_STEP_REL * std::max({1.0, std::abs(u), std::abs(v)});
    for (int si = 0; si < 2; ++si) {
        const double sign = si == 0 ? 1.0 : -1.0;
        double h = step;
        for (int t = 0; t < FD_STEP_SHRINKS; ++t) {
            const Vec3 p = desc_eval(d, u + sign * du * h, v + sign * dv * h);
            if (finite3(p)) {
                return (p - base) / (sign * h);
            }
            h *= 0.5;
        }
    }
    return nan3();
}

}  // namespace

std::pair<Vec3, Vec3> desc_derivatives(const SurfaceDesc& d, double u, double v) {
    const auto analytic = desc_deriv_pair(d, u, v);
    if (finite3(analytic.first) && finite3(analytic.second)) {
        return analytic;
    }
    const double du = 0.5 * (d.u_min + d.u_max) - u;
    const double dv = 0.5 * (d.v_min + d.v_max) - v;
    if (du == 0.0 && dv == 0.0) {
        return {partial_desc(d, u, v, 1.0, 0.0), partial_desc(d, u, v, 0.0, 1.0)};
    }
    const double span = std::max({std::abs(d.u_max - d.u_min), std::abs(d.v_max - d.v_min), 1.0});
    const double length = std::hypot(du, dv);
    double frac = DERIV_INTERIOR_FRAC * span / length;
    for (int t = 0; t < DERIV_INTERIOR_TRIES; ++t) {
        if (frac >= 1.0) {
            break;
        }
        const double uu = u + du * frac;
        const double vv = v + dv * frac;
        if (desc_accepts(d, uu, vv)) {
            const auto pulled = desc_deriv_pair(d, uu, vv);
            if (finite3(pulled.first) && finite3(pulled.second)) {
                return pulled;
            }
        }
        frac *= 2.0;
    }
    return {partial_desc(d, u, v, 1.0, 0.0), partial_desc(d, u, v, 0.0, 1.0)};
}

std::optional<Vec3> surface_normal(const SurfaceDesc& desc, double u, double v) {
    if (!desc_accepts(desc, u, v)) {
        return std::nullopt;
    }
    const auto rv = desc_derivatives(desc, u, v);
    Vec3 n = -cross3(rv.first, rv.second);
    const double norm_len = norm(n);
    if (!std::isfinite(norm_len) || norm_len <= 0.0) {
        return std::nullopt;
    }
    return desc.rot * (n / norm_len);
}

}  // namespace optics::native
