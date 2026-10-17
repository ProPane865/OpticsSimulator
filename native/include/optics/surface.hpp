// Compiled surface model and the kernel surface descriptor: expression
// evaluation, finite-difference tangents (C++ mirror of
// ``optics/numerical/_codegen_runtime.py``) and descriptor-domain membership
// / derivative policy (mirror of ``optics/numerical/desc.py``).
#pragma once

#include <memory>
#include <utility>

#include "optics/expression.hpp"
#include "optics/math.hpp"

namespace optics::native {

// A parametric surface ``(x(u,v), y(u,v), z(u,v))`` with its finite-difference
// step and parameter domain; the compiled replacement for the generated
// Numba module of the Python implementation.
class SurfaceModel {
  public:
    SurfaceModel(ExprPtr x, ExprPtr y, ExprPtr z, double u_min, double u_max, double v_min,
                 double v_max);

    // Parses/validates the three expressions (throws ``ExpressionError``).
    static std::shared_ptr<const SurfaceModel> compile(const std::string& x_src,
                                                       const std::string& y_src,
                                                       const std::string& z_src, double u_min,
                                                       double u_max, double v_min, double v_max);

    Vec3 eval_local(double u, double v) const;
    void eval_grid(const double* us, const double* vs, std::size_t n, double* out) const;

    // Tangents ``(r_u, r_v)`` via the central/oriented-one-sided policy
    // (``derivatives_rt``).
    std::pair<Vec3, Vec3> derivatives(double u, double v) const;

    double hu() const { return hu_; }
    double hv() const { return hv_; }
    double u_min() const { return u_min_; }
    double u_max() const { return u_max_; }
    double v_min() const { return v_min_; }
    double v_max() const { return v_max_; }

  private:
    Vec3 partial_rt(double u, double v, double h, double lo, double hi, bool along_u) const;

    ExprPtr fx_, fy_, fz_;
    double hu_, hv_;
    double u_min_, u_max_, v_min_, v_max_;
};

// One surface flattened for the Newton/intersection kernels (the ``desc`` of
// the Python kernels: cfunc pair -> shared model, plus domain, aperture and
// transform data).
struct SurfaceDesc {
    std::shared_ptr<const SurfaceModel> model;
    double u_min = 0.0, u_max = 0.0, v_min = 0.0, v_max = 0.0;
    double tu = 0.0, tv = 0.0;
    double ap_r = -1.0, ap_cx = 0.0, ap_cy = 0.0;
    Mat3 rot = Mat3::identity();
    Vec3 trans{};
};

// Local-space point on the descriptor's surface.
inline Vec3 desc_eval(const SurfaceDesc& d, double u, double v) {
    return d.model->eval_local(u, v);
}

// World-space point on the descriptor's surface.
inline Vec3 desc_point(const SurfaceDesc& d, double u, double v) {
    return d.rot * d.model->eval_local(u, v) + d.trans;
}

// Model-level finite-difference tangents (``_deriv_pair_njit``).
inline std::pair<Vec3, Vec3> desc_deriv_pair(const SurfaceDesc& d, double u, double v) {
    return d.model->derivatives(u, v);
}

// Domain + circular-aperture membership (``accepts_njit``).
bool desc_accepts(const SurfaceDesc& d, double u, double v);

// Analytic tangents with the interior pull-back / FD fallback
// (``derivatives_njit``).
std::pair<Vec3, Vec3> desc_derivatives(const SurfaceDesc& d, double u, double v);

}  // namespace optics::native
