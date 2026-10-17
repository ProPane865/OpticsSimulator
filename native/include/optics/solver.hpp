// Damped Newton iteration and the ray/surface intersection and parameter
// inversion drivers (C++ mirrors of ``optics/numerical/newton.py``,
// ``intersect.py`` and ``invert.py``).
#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "optics/constants.hpp"
#include "optics/grid.hpp"
#include "optics/math.hpp"
#include "optics/surface.hpp"

namespace optics::native {

// Newton problem selector (``_NEWTON_INTERSECT`` / ``_NEWTON_XY``).
enum NewtonKind { kNewtonIntersect = 0, kNewtonXY = 1 };

struct NewtonResult {
    std::vector<double> x;
    double f_norm = 0.0;
    int status = SOLVER_EXH;
};

// Damped Newton iteration for ``residual(x) = 0``; see ``_newton_njit``.
std::optional<NewtonResult> newton_solve(int kind, const SurfaceDesc& desc, const double* o,
                                         const double* d, double x_target, double y_target,
                                         const double* x0, int n0, bool clamp);

// Full ray/surface Newton solve from one seed (njit ``_newton_intersection_njit``).
struct HitUV {
    double t, u, v;
};

std::optional<HitUV> newton_intersection(const SurfaceDesc& desc, const double* o, const double* d,
                                         double u, double v, double t);

// Seed selection + Newton + dedup: the nearest forward hit (njit
// ``surface_intersect_njit``).
std::optional<HitUV> surface_intersect(const SurfaceDesc& desc, const SeedGrid& grid,
                                       const double* o, const double* d);

// Newton inversion of local ``(x, y)`` to parameters from one seed
// (njit ``_solve_xy_njit``).
std::optional<std::pair<double, double>> solve_xy(const SurfaceDesc& desc, double x_target,
                                                  double y_target, double u, double v);

// Grid-seeded local ``(x, y)`` inversion (njit ``parameters_at_xy_njit``).
std::optional<std::pair<double, double>> parameters_at_xy(const SurfaceDesc& desc,
                                                          const SeedGrid& grid, double x, double y);

}  // namespace optics::native
