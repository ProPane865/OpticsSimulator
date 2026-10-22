// Adaptive-kernel irradiance reconstruction: spreads each detector hit over
// the pixel grid with a compact Wendland C2 kernel whose per-ray radius comes
// from the local k-th nearest-neighbour spacing (the kernel side of
// ``optics.measurement_plane``).
//
// Raw ``double*`` buffers are appropriate only here, at the batch boundary:
// each view describes a contiguous NumPy-owned buffer already validated by
// the bindings.
#pragma once

#include <cstddef>

namespace optics::native {

// Reconstruct per-ray power onto a row-major ``(rv, ru)`` pixel ``grid``.
//
// Each ray gets an adaptive compact kernel radius ``h = bandwidth_scale *
// d_k`` (k-th nearest-neighbour distance among the valid hits, self
// excluded) and deposits its full power over the overlapping pixels with
// shape ``(1 - q)^4 (1 + 4q)`` for ``q = distance / h < 1``, discretely
// renormalised on the truncated pixel window (boundary rays keep their full
// power). Non-finite and out-of-extent samples are filtered before both the
// neighbour search and the deposition. Returns the total power deposited.
double reconstruct_irradiance(double* grid, const double* u, const double* v,
                              const double* power, std::size_t n, double u_min,
                              double u_max, double v_min, double v_max, int ru,
                              int rv, double interval_u, double interval_v,
                              std::size_t neighbors, double bandwidth_scale);

}  // namespace optics::native
