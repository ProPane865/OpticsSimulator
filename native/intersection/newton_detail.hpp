// Newton problem strategies for the damped solver: each problem provides its
// residual and jacobian for a fixed-dimension unknown vector, and
// ``newton_run`` drives the shared iteration. The intersection problem solves
// ``(t, u, v)`` for the ray/surface contact; the XY problem inverts local
// ``(x, y)`` back to surface parameters ``(u, v)``.
#pragma once

#include <optional>

#include "optics/common/linalg.hpp"
#include "optics/common/vector.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/solver.hpp"

namespace optics::native {

struct IntersectProblem {
    static constexpr int n = 3;

    const SurfaceDesc& desc;
    Vec3 o;
    Vec3 d;

    SmallVec residual(const double* x) const;
    SmallMat jacobian(const double* x) const;
};

struct XYProblem {
    static constexpr int n = 2;

    const SurfaceDesc& desc;
    double x_target;
    double y_target;

    SmallVec residual(const double* x) const;
    SmallMat jacobian(const double* x) const;
};

// Damped Newton iteration for ``problem.residual(x) = 0``; ``x0`` is the
// initial guess of length ``Problem::n``. The result layout and iteration
// policy are the historical ``newton_solve`` ones.
template <typename Problem>
std::optional<NewtonResult> newton_run(const Problem& problem, const double* x0, bool clamp);

extern template std::optional<NewtonResult> newton_run<IntersectProblem>(const IntersectProblem&,
                                                                          const double*, bool);
extern template std::optional<NewtonResult> newton_run<XYProblem>(const XYProblem&, const double*,
                                                                   bool);

}  // namespace optics::native
