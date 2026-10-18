// Damped Newton iteration and the ray/surface intersection and parameter
// inversion drivers (C++ mirrors of ``optics/numerical/intersect.py`` and
// ``invert.py``).
#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/grid.hpp"

namespace optics::native {

// Newton problem selector.
enum NewtonKind { kNewtonIntersect = 0, kNewtonXY = 1 };

struct NewtonResult {
    std::vector<double> x;
    double f_norm = 0.0;
    int status = SOLVER_EXH;
};

// Damped Newton iteration for ``residual(x) = 0``.
std::optional<NewtonResult> newton_solve(int kind, const SurfaceDesc& desc, const double* o,
                                         const double* d, double x_target, double y_target,
                                         const double* x0, int n0, bool clamp);

// Full ray/surface Newton solve from one seed (``_newton_intersection_kernel``).
struct HitUV {
    double t, u, v;
};

std::optional<HitUV> newton_intersection(const SurfaceDesc& desc, const double* o, const double* d,
                                         double u, double v, double t);

// Seed selection + Newton + dedup: the nearest forward hit
// (``surface_intersect_kernel``).
std::optional<HitUV> surface_intersect(const SurfaceDesc& desc, const SeedGrid& grid,
                                       const double* o, const double* d);

// Newton inversion of local ``(x, y)`` to parameters from one seed
// (``_solve_xy_kernel``).
std::optional<std::pair<double, double>> solve_xy(const SurfaceDesc& desc, double x_target,
                                                  double y_target, double u, double v);

// Grid-seeded local ``(x, y)`` inversion (``parameters_at_xy_kernel``).
std::optional<std::pair<double, double>> parameters_at_xy(const SurfaceDesc& desc,
                                                          const SeedGrid& grid, double x, double y);

}  // namespace optics::native
