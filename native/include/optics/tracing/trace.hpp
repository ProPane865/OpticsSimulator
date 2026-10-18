// Element and stack tracing kernels (mirror of ``optics/tracing/kernels.py``
// and the stack walk of the measurement-plane kernels): nearest-face hits,
// element containment, internal bounces, the geometric-order stack walk and
// the single-trace hop recording shared with the measurement kernels.
#pragma once

#include <cstddef>
#include <vector>

#include "optics/common/vector.hpp"
#include "optics/tracing/context.hpp"

namespace optics::native {

struct ElementHit {
    bool found = false;
    double t = 0.0;
    Vec3 p{};
    Vec3 n{};
};

// Per-trace scalar configuration.
struct TraceOptions {
    double ray_length;  // trailing-gap cap for the rendering tail segment
    double plane_eps;   // degeneracy epsilon for plane intersections
};

// Nearest face hit (surfaces + sidewall).
ElementHit element_first_hit(const Element& elem, const Vec3& o, const Vec3& d,
                             double min_t);

bool element_contains_point(const Element& elem, const Vec3& p);

struct ElementTrace {
    Vec3 exit_o{};
    Vec3 exit_d{};
    bool exited = false;
    int n_pts = 0;
    bool has_incident = false;
    Vec3 incident_pt{};
    std::vector<Vec3> pts;  // internal polyline
};

// Trace one closed element: entry, internal bounces, exit ray.
ElementTrace trace_element(const Element& elem, const Vec3& o, const Vec3& d);

struct StackSegments {
    int count = 0;
    bool exited = false;
    Vec3 final_o{};
    Vec3 final_d{};
};

// Mirrors ``trace_stack_path``; segment kinds 0/1/2 = incident/internal/exit.
// Writes up to ``cap`` segments into ``starts``/``ends``/``kinds``.
StackSegments trace_stack_segments(const TraceStack& stack, const Vec3& o, const Vec3& d,
                                   const TraceOptions& options, double* starts, double* ends,
                                   signed char* kinds, std::size_t cap);

// Traces until the plane region is reached; writes the reaching ray.
bool trace_stack_to_plane(const TraceStack& stack, const Vec3& o, const Vec3& d, const Vec3& po,
                          const Vec3& pn, const PlaneRegion& region, const TraceOptions& options,
                          Vec3& out_o, Vec3& out_d);

// --- Single geometric-order walk (shared by rendering + all planes) ----------
//
// A full geometric-order walk of the stack (the same nearest-unvisited loop
// used by ``trace_stack_segments``) is a superset of every per-plane walk
// ``trace_stack_to_plane`` performs: the refractions up to any plane are a
// prefix of this walk, so one trace serves rendering and every plane. Each hop
// records the free-travel gap (origin/direction + entry distance to the next
// element) and the entered element's internal polyline, letting both the
// rendering segment buffers and the per-plane first-crossing be derived
// without re-walking.
struct RayHop {
    Vec3 o{};
    Vec3 d{};
    Vec3 incident_pt{};
    double entry_t = 0.0;  // distance from ``o`` to the entered element
    bool has_element = false;
    bool has_incident = false;
    bool exited = false;
    std::vector<Vec3> pts;  // internal polyline of the entered element
};

using RayHops = std::vector<RayHop>;

// Walks the stack in geometric order, recording one hop per gap/element.
RayHops build_hops(const TraceStack& stack, const Vec3& o, const Vec3& d);

// Rendering segment buffers, byte-identical to ``trace_stack_segments``:
// incident + internal segments per element, then the trailing gap capped at
// ``ray_length`` (omitted when the ray is absorbed).
int format_segments(const RayHops& hops, double ray_length, double* starts, double* ends,
                    signed char* kinds, std::size_t cap);

}  // namespace optics::native
