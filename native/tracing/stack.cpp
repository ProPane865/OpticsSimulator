// Stack-level tracing: the geometric-order walk over unvisited elements,
// the rendering segment writers, the trace-to-plane driver and the
// single-trace hop recording shared with the measurement kernels.
#include <limits>
#include <utility>
#include <vector>

#include "optics/common/vector.hpp"
#include "optics/intersection/plane.hpp"
#include "optics/tracing/trace.hpp"

namespace optics::native {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Nearest unvisited element along the ray and its entry hit (handed to
// ``trace_element`` so the entry intersection is never recomputed).
std::pair<int, ElementHit> nearest_unvisited(const TraceStack& stack,
                                             std::vector<unsigned char>& visited, const Vec3& o,
                                             const Vec3& d, double t_limit) {
    int best_i = -1;
    double best_t = kInf;
    ElementHit best{};
    for (std::size_t i = 0; i < stack.size(); ++i) {
        if (!visited[i]) {
            ElementHit hit = element_first_hit(*stack.elements[i], o, d, 0.0);
            if (hit.found && hit.t < t_limit && hit.t < best_t) {
                best_t = hit.t;
                best_i = static_cast<int>(i);
                best = hit;
            }
        }
    }
    return {best_i, best};
}

}  // namespace

StackSegments trace_stack_segments(const TraceStack& stack, const Vec3& o, const Vec3& d,
                                   const TraceOptions& options, double* starts, double* ends,
                                   signed char* kinds, std::size_t cap) {
    std::vector<unsigned char> visited(stack.size(), 0);
    Vec3 cur_o = o;
    Vec3 cur_d = d;
    signed char kind = 0;
    int count = 0;
    for (;;) {
        const std::pair<int, ElementHit> nb =
            nearest_unvisited(stack, visited, cur_o, cur_d, kInf);
        if (nb.first < 0) {
            break;
        }
        visited[static_cast<std::size_t>(nb.first)] = 1;
        const ElementTrace tr = trace_element(*stack.elements[static_cast<std::size_t>(nb.first)],
                                              cur_o, cur_d, nb.second);
        if (tr.has_incident && static_cast<std::size_t>(count) < cap) {
            const Vec3 s = cur_o;
            const Vec3 e = tr.incident_pt;
            for (int i = 0; i < 3; ++i) {
                starts[3 * count + i] = s[i];
                ends[3 * count + i] = e[i];
            }
            kinds[count] = kind;
            ++count;
        }
        for (int k = 0; k + 1 < tr.n_pts; ++k) {
            if (static_cast<std::size_t>(count) >= cap) {
                break;
            }
            const double* s = tr.pts[static_cast<std::size_t>(k)].v;
            const double* e = tr.pts[static_cast<std::size_t>(k + 1)].v;
            for (int i = 0; i < 3; ++i) {
                starts[3 * count + i] = s[i];
                ends[3 * count + i] = e[i];
            }
            kinds[count] = 1;
            ++count;
        }
        if (!tr.exited) {
            return StackSegments{count, false, cur_o, cur_d};
        }
        cur_o = tr.exit_o;
        cur_d = tr.exit_d;
        kind = 2;
    }
    if (static_cast<std::size_t>(count) < cap) {
        for (int i = 0; i < 3; ++i) {
            starts[3 * count + i] = cur_o[i];
            ends[3 * count + i] = cur_o[i] + options.ray_length * cur_d[i];
        }
        kinds[count] = kind;
        ++count;
    }
    return StackSegments{count, true, cur_o, cur_d};
}

bool trace_stack_to_plane(const TraceStack& stack, const Vec3& o, const Vec3& d, const Vec3& po,
                          const Vec3& pn, const PlaneRegion& region, const TraceOptions& options,
                          Vec3& out_o, Vec3& out_d) {
    std::vector<unsigned char> visited(stack.size(), 0);
    Vec3 cur_o = o;
    Vec3 cur_d = d;
    for (;;) {
        std::optional<std::array<double, 3>> plane = plane_intersect(cur_o, cur_d, po, pn,
                                                                     options.plane_eps);
        if (!plane.has_value()) {
            break;
        }
        const std::pair<int, ElementHit> nb =
            nearest_unvisited(stack, visited, cur_o, cur_d, plane->at(2));
        if (nb.first < 0) {
            break;
        }
        visited[static_cast<std::size_t>(nb.first)] = 1;
        const ElementTrace tr = trace_element(*stack.elements[static_cast<std::size_t>(nb.first)],
                                              cur_o, cur_d, nb.second);
        for (int k = 0; k + 1 < tr.n_pts; ++k) {
            const Vec3 seg_o = tr.pts[static_cast<std::size_t>(k)];
            const Vec3 diff = tr.pts[static_cast<std::size_t>(k + 1)] - seg_o;
            const double seg_len = norm(diff);
            if (seg_len <= 0.0) {
                continue;
            }
            const Vec3 seg_d = diff / seg_len;
            std::optional<std::array<double, 3>> hit =
                plane_intersect(seg_o, seg_d, po, pn, options.plane_eps);
            if (hit.has_value() && hit->at(0) >= region.u_min && hit->at(0) <= region.u_max &&
                hit->at(1) >= region.v_min && hit->at(1) <= region.v_max &&
                hit->at(2) <= seg_len) {
                out_o = seg_o;
                out_d = seg_d;
                return true;
            }
        }
        if (!tr.exited) {
            return false;
        }
        cur_o = tr.exit_o;
        cur_d = tr.exit_d;
    }
    out_o = cur_o;
    out_d = cur_d;
    return true;
}

// --- Single-trace shared walk (rendering + all measurement planes) -----------
//
// The hop records are consumed both here (``format_segments``) and by the
// measurement kernels (first plane crossing per hop); see ``trace.hpp`` for
// the walk semantics.
RayHops build_hops(const TraceStack& stack, const Vec3& o, const Vec3& d) {
    std::vector<unsigned char> visited(stack.size(), 0);
    Vec3 cur_o = o;
    Vec3 cur_d = d;
    RayHops hops;
    for (;;) {
        const std::pair<int, ElementHit> best =
            nearest_unvisited(stack, visited, cur_o, cur_d, kInf);
        if (best.first < 0) {
            RayHop tail;
            tail.o = cur_o;
            tail.d = cur_d;
            tail.has_element = false;
            hops.push_back(std::move(tail));
            return hops;
        }
        visited[static_cast<std::size_t>(best.first)] = 1;
        const ElementTrace tr = trace_element(*stack.elements[static_cast<std::size_t>(best.first)],
                                              cur_o, cur_d, best.second);
        RayHop h;
        h.o = cur_o;
        h.d = cur_d;
        h.entry_t = best.second.t;
        h.has_element = true;
        h.has_incident = tr.has_incident;
        h.exited = tr.exited;
        if (tr.has_incident) {
            h.incident_pt = tr.incident_pt;
        }
        h.pts = tr.pts;
        hops.push_back(std::move(h));
        if (!tr.exited) {
            return hops;  // absorbed inside the element: no trailing gap
        }
        cur_o = tr.exit_o;
        cur_d = tr.exit_d;
    }
}

int format_segments(const RayHops& hops, double ray_length, double* starts, double* ends,
                    signed char* kinds, std::size_t cap) {
    std::size_t count = 0;
    signed char kind = 0;
    for (const RayHop& h : hops) {
        if (h.has_element) {
            if (h.has_incident && count < cap) {
                for (int i = 0; i < 3; ++i) {
                    starts[3 * count + i] = h.o[i];
                    ends[3 * count + i] = h.incident_pt[i];
                }
                kinds[count] = kind;
                ++count;
            }
            const std::size_t m = h.pts.size();
            for (std::size_t k = 0; k + 1 < m; ++k) {
                if (count >= cap) {
                    break;
                }
                const Vec3& s = h.pts[k];
                const Vec3& e = h.pts[k + 1];
                for (int i = 0; i < 3; ++i) {
                    starts[3 * count + i] = s[i];
                    ends[3 * count + i] = e[i];
                }
                kinds[count] = 1;
                ++count;
            }
            if (!h.exited) {
                return static_cast<int>(count);  // absorbed: stop, no trailing gap
            }
            kind = 2;
        } else {
            if (count < cap) {
                for (int i = 0; i < 3; ++i) {
                    starts[3 * count + i] = h.o[i];
                    ends[3 * count + i] = h.o[i] + ray_length * h.d[i];
                }
                kinds[count] = kind;
                ++count;
            }
            return static_cast<int>(count);
        }
    }
    return static_cast<int>(count);
}

}  // namespace optics::native
