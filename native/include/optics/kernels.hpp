// Ray/plane, ray/mesh and interface-direction kernels plus the element and
// stack tracing kernels and batch measurement helpers. Mirrors
// ``optics/numerical/{plane,mesh}.py``, ``optics/tracing/{direction,kernels,
// context}.py`` and the measurement-plane kernels.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "optics/grid.hpp"
#include "optics/math.hpp"
#include "optics/solver.hpp"
#include "optics/surface.hpp"

namespace optics::native {

// --- Plane (njit ``plane_intersect_kernel``) --------------------------------
// Returns ``(u, v, t)`` in plane parameters or ``nullopt``.
std::optional<std::array<double, 3>> plane_intersect(const double* ro, const double* d,
                                                     const double* o, const double* n,
                                                     double eps);

// --- Mesh (njit ``mesh_intersect_kernel``) -----------------------------------
// Moller-Trumbore over flattened ``(K, 3, 3)`` triangles; nearest forward hit
// with barycentric ``(u, v)`` clamped into the triangle.
struct MeshHit {
    double t;
    std::size_t face;
    double u, v;
};

std::optional<MeshHit> mesh_intersect(const double* o, const double* d, const std::vector<double>& tri,
                                      double det_eps, double bary_tol, double min_face_t);

// --- Direction (njit ``direction.py``) ----------------------------------------
Vec3 orient(const Vec3& normal, const Vec3& incident);
std::pair<Vec3, bool> refract_or_reflect(const Vec3& incident, const Vec3& normal, double n_from,
                                         double n_to);

// --- Tracing kernels (njit ``tracing/kernels.py``) ----------------------------
struct ElementHit {
    bool found = false;
    double t = 0.0;
    Vec3 p{};
    Vec3 n{};
};

// World-space unit normal at ``(u, v)`` (njit ``_normal_njit``).
std::optional<Vec3> surface_normal(const SurfaceDesc& desc, double u, double v);

// Nearest face hit (surfaces + sidewall) (njit ``_element_first_hit``).
ElementHit element_first_hit(const ElementContext& elem, const double* o, const double* d,
                             double min_t);

bool element_contains_point(const ElementContext& elem, const double* p);

struct ElementTrace {
    Vec3 exit_o{};
    Vec3 exit_d{};
    bool exited = false;
    int n_pts = 0;
    bool has_incident = false;
    Vec3 incident_pt{};
    std::vector<Vec3> pts;  // internal polyline
};

// Nopython ``RefractiveElement.trace`` for one closed element.
ElementTrace trace_element(const ElementContext& elem, const double* o, const double* d);

struct StackSegments {
    int count = 0;
    bool exited = false;
    Vec3 final_o{};
    Vec3 final_d{};
};

// Nopython ``trace_stack_path``; segment kinds 0/1/2 = incident/internal/exit.
// Writes up to ``cap`` segments into ``starts``/``ends``/``kinds``.
StackSegments trace_stack_segments(const TraceStack& stack, const double* o, const double* d,
                                   double ray_length, double* starts, double* ends,
                                   signed char* kinds, std::size_t cap);

// Nopython ``MeasurementPlane._trace_to_plane``; writes the reaching ray.
bool trace_stack_to_plane(const TraceStack& stack, const double* o, const double* d,
                          const double* po, const double* pn, double eps, double u_min,
                          double u_max, double v_min, double v_max, double* out_o, double* out_d);

// --- Batch measurement kernels -----------------------------------------------

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

// --- Vector helpers (``vector.py``) -------------------------------------------
// Rodrigues rotation matrix mapping the +Z axis onto ``direction``.
Mat3 rotation_from_z(const Vec3& direction);

}  // namespace optics::native
