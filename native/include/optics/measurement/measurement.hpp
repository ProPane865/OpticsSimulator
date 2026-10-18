// Batch measurement kernels: detector-plane power accumulation, direct and
// stack-traced plane intersection, and the fused single-trace rendering +
// all-planes pass (mirrors of the measurement-plane kernels in
// ``optics/tracing/context.py``).
#pragma once

#include <cstddef>
#include <cstdint>

#include "optics/tracing/context.hpp"

namespace optics::native {

// Accumulate in-extent ray power into ``grid`` (row-major ``(rv, ru)``);
// returns the total added.
double accumulate_power(double* grid, const double* u, const double* v, const double* power,
                        std::size_t n, double u_min, double u_max, double v_min, double v_max,
                        int ru, int rv, double interval_u, double interval_v);

// Plane-intersect every ray (parallel); writes ``(u, v, power)`` per ray,
// leaving non-hits as-is.
void measure_direct(const double* origins, const double* dirs, const double* powers, std::size_t n,
                    const double* o, const double* nrm, double eps, double* us, double* vs,
                    double* pws);

// Trace rays through the stack (parallel), then hit the plane; writes
// ``(u, v, power)`` per ray, leaving misses as-is.
void measure_stack(const TraceStack& stack, const double* origins, const double* dirs,
                   const double* powers, std::size_t n, const double* o, const double* nrm,
                   double eps, double u_min, double u_max, double v_min, double v_max, double* us,
                   double* vs, double* pws);

// Trace every ray through the stack exactly once (parallel) and evaluate
// ``p`` planes off that single trace. Plane geometry is column-parallel:
// ``po``/``pn`` are ``(p, 3)`` and the range arrays are ``(p,)``; output
// ``us``/``vs``/``pws`` are ``(p, n)`` plane-major. Results match ``p``
// independent ``measure_stack`` calls (the per-plane traversal is a prefix of
// the full geometric-order walk, so refractions up to each plane are shared).
void measure_planes_multi(const TraceStack& stack, const double* origins, const double* dirs,
                          const double* powers, std::size_t n, std::size_t p, const double* po,
                          const double* pn, const double* u_min, const double* u_max,
                          const double* v_min, const double* v_max, double eps, double* us,
                          double* vs, double* pws);

// Single-trace combined path: walks every ray through the stack once and emits
// both the rendering segment buffers (byte-identical to ``trace_segments_batch``)
// and the ``(p, n)`` plane measurement buffers. ``starts``/``ends`` are
// ``(n, cap, 3)``, ``kinds`` ``(n, cap)``, ``counts`` ``(n,)``.
void trace_segments_and_planes(const TraceStack& stack, const double* origins, const double* dirs,
                               const double* powers, std::size_t n, double ray_length,
                               std::size_t p, const double* po, const double* pn,
                               const double* u_min, const double* u_max, const double* v_min,
                               const double* v_max, double eps, double* starts, double* ends,
                               signed char* kinds, std::int64_t* counts, std::size_t cap,
                               double* us, double* vs, double* pws);

}  // namespace optics::native
