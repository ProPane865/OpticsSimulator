// Plane/mesh/direction kernels, element and stack tracing, batch measurement.
#include "optics/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "optics/constants.hpp"
#include "optics/parallel.hpp"

namespace optics::native {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr int kMaxPathPts = MAX_INTERNAL_BOUNCES + 2;

Mat3 mat3_mul(const Mat3& a, const Mat3& b) {
    Mat3 r;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            r.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] + a.m[i][2] * b.m[2][j];
        }
    }
    return r;
}

Vec3 as_vec3(const double* a) { return make_vec3(a[0], a[1], a[2]); }

// Nearest surface hit with normal, or ``(False, inf, 0, 0)``.
ElementHit surface_hit(const SurfaceDesc& desc, const SeedGrid& grid, const double* o,
                       const double* d, double min_t) {
    std::optional<HitUV> hit = surface_intersect(desc, grid, o, d);
    if (hit.has_value() && hit->t > min_t) {
        std::optional<Vec3> normal = surface_normal(desc, hit->u, hit->v);
        if (normal.has_value()) {
            Vec3 bn = normal.value();
            Vec3 p;
            for (int i = 0; i < 3; ++i) {
                p[i] = o[i] + hit->t * d[i];
            }
            return ElementHit{true, hit->t, p, bn};
        }
    }
    return ElementHit{false, kInf, Vec3{}, Vec3{}};
}

// Local height of a surface above ``(local[0], local[1])``; NaN when the
// point does not project onto the surface.
double local_surface_z(const SurfaceDesc& desc, const SeedGrid& grid, bool graph,
                       const double* local) {
    double u = local[0];
    double v = local[1];
    if (graph) {
        if (!desc_accepts(desc, u, v)) {
            return NAN;
        }
    } else {
        std::optional<std::pair<double, double>> uv = parameters_at_xy(desc, grid, u, v);
        if (!uv.has_value()) {
            return NAN;
        }
        u = uv->first;
        v = uv->second;
    }
    const double z = desc_eval(desc, u, v)[2];
    return std::isfinite(z) ? z : NAN;
}

int nearest_unvisited(const TraceStack& stack, std::vector<unsigned char>& visited,
                      const double* o, const double* d, double t_limit) {
    int best_i = -1;
    double best_t = kInf;
    for (std::size_t i = 0; i < stack.size(); ++i) {
        if (!visited[i]) {
            ElementHit hit = element_first_hit(*stack.elements[i], o, d, 0.0);
            if (hit.found && hit.t < t_limit && hit.t < best_t) {
                best_t = hit.t;
                best_i = static_cast<int>(i);
            }
        }
    }
    return best_i;
}

// As :func:`nearest_unvisited` but also reports the chosen entry distance.
std::pair<int, double> nearest_unvisited_t(const TraceStack& stack,
                                           std::vector<unsigned char>& visited, const double* o,
                                           const double* d, double t_limit) {
    int best_i = -1;
    double best_t = kInf;
    for (std::size_t i = 0; i < stack.size(); ++i) {
        if (!visited[i]) {
            ElementHit hit = element_first_hit(*stack.elements[i], o, d, 0.0);
            if (hit.found && hit.t < t_limit && hit.t < best_t) {
                best_t = hit.t;
                best_i = static_cast<int>(i);
            }
        }
    }
    return {best_i, best_t};
}

}  // namespace

std::optional<std::array<double, 3>> plane_intersect(const double* ro, const double* d,
                                                     const double* o, const double* n,
                                                     double eps) {
    const double nz = n[2];
    if (std::abs(nz) <= eps || std::abs(d[0] * n[0] + d[1] * n[1] + d[2] * n[2]) <= eps) {
        return std::nullopt;
    }
    const double nxnz = n[0] / nz;
    const double nynz = n[1] / nz;
    SmallMat a;
    a.rows = a.cols = 3;
    a.at(0, 0) = -1.0;
    a.at(0, 1) = 0.0;
    a.at(0, 2) = d[0];
    a.at(1, 0) = 0.0;
    a.at(1, 1) = -1.0;
    a.at(1, 2) = d[1];
    a.at(2, 0) = nxnz;
    a.at(2, 1) = nynz;
    a.at(2, 2) = d[2];
    SmallVec b;
    b.n = 3;
    b.v[0] = -ro[0];
    b.v[1] = -ro[1];
    b.v[2] = nxnz * o[0] + nynz * o[1] + o[2] - ro[2];

    const double det = d[2] + d[1] * nynz + d[0] * nxnz;
    if (!std::isfinite(det) || std::abs(det) <= eps) {
        return std::nullopt;
    }
    std::optional<SmallVec> sol = solve_small(a, b);
    if (!sol.has_value()) {
        return std::nullopt;
    }
    const double u = sol->v[0];
    const double v = sol->v[1];
    const double t = sol->v[2];
    if (!std::isfinite(t) || t <= 0.0) {
        return std::nullopt;
    }
    if (!std::isfinite(u) || !std::isfinite(v)) {
        return std::nullopt;
    }
    return std::array<double, 3>{u, v, t};
}

std::optional<MeshHit> mesh_intersect(const double* o, const double* d,
                                      const std::vector<double>& tri, double det_eps,
                                      double bary_tol, double min_face_t) {
    const std::size_t k_count = tri.size() / 9;
    double best_t = kInf;
    std::size_t best_face = 0;
    double best_u = 0.0;
    double best_v = 0.0;
    bool found = false;
    for (std::size_t k = 0; k < k_count; ++k) {
        const double* base = &tri[9 * k];
        Vec3 a = as_vec3(base);
        Vec3 e1 = as_vec3(base + 3) - a;
        Vec3 e2 = as_vec3(base + 6) - a;

        const Vec3 dv = as_vec3(d);
        const Vec3 ov = as_vec3(o);
        const Vec3 h = cross3(dv, e2);
        const double det = dot3(e1, h);
        if (std::abs(det) <= det_eps) {
            continue;
        }
        const double inv_det = 1.0 / det;

        const Vec3 s = ov - a;
        const double u = dot3(s, h) * inv_det;
        const Vec3 q = cross3(s, e1);
        const double v = dot3(q, dv) * inv_det;
        const double t = dot3(e2, q) * inv_det;

        if (t > min_face_t && u >= -bary_tol && v >= -bary_tol && u + v <= 1.0 + bary_tol &&
            t < best_t) {
            best_t = t;
            best_face = k;
            best_u = u;
            best_v = v;
            found = true;
        }
    }
    if (!found) {
        return std::nullopt;
    }
    const double u_k = std::min(std::max(best_u, 0.0), 1.0);
    const double v_k = std::min(std::max(best_v, 0.0), 1.0 - u_k);
    return MeshHit{best_t, best_face, u_k, v_k};
}

Vec3 orient(const Vec3& normal, const Vec3& incident) {
    if (dot3(normal, incident) < 0.0) {
        return -normal;
    }
    return normal;
}

std::pair<Vec3, bool> refract_or_reflect(const Vec3& incident, const Vec3& normal, double n_from,
                                         double n_to) {
    const double eta = n_from / n_to;
    const double cos_i = dot3(normal, incident);
    const double k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    if (k < 0.0) {
        return {normalize_vector(incident - (2.0 * dot3(normal, incident)) * normal), false};
    }
    const Vec3 dir = eta * incident + (std::sqrt(k) - eta * cos_i) * normal;
    return {normalize_vector(dir), true};
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

ElementHit element_first_hit(const ElementContext& elem, const double* o, const double* d,
                             double min_t) {
    double best_t = kInf;
    bool found = false;
    Vec3 bp{};
    Vec3 bn{};
    ElementHit h0 = surface_hit(*elem.desc1, *elem.grid1, o, d, min_t);
    if (h0.found && h0.t < best_t) {
        best_t = h0.t;
        found = true;
        bp = h0.p;
        bn = h0.n;
    }
    ElementHit h1 = surface_hit(*elem.desc2, *elem.grid2, o, d, min_t);
    if (h1.found && h1.t < best_t) {
        best_t = h1.t;
        found = true;
        bp = h1.p;
        bn = h1.n;
    }
    if (elem.n_tris() > 0) {
        std::optional<MeshHit> wall = mesh_intersect(o, d, elem.tri, DET_SINGULAR_EPS, BARY_TOL,
                                                     MIN_FACE_T);
        if (wall.has_value() && wall->t > min_t && wall->t < best_t) {
            best_t = wall->t;
            found = true;
            for (int i = 0; i < 3; ++i) {
                bp[i] = o[i] + wall->t * d[i];
            }
            bn = as_vec3(&elem.normals[3 * wall->face]);
        }
    }
    return ElementHit{found, best_t, bp, bn};
}

bool element_contains_point(const ElementContext& elem, const double* p) {
    const SurfaceDesc& desc = *elem.desc1;
    Vec3 pv = as_vec3(p);
    for (int i = 0; i < 3; ++i) {
        pv[i] -= elem.desc1->trans[i];
    }
    const Vec3 local = mul_transpose(desc.rot, pv);
    if (!finite3(local)) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        if (local[i] < elem.lo[i] || local[i] > elem.hi[i]) {
            return false;
        }
    }
    const double z1 =
        local_surface_z(*elem.desc1, *elem.grid1, elem.graph1, local.v);
    const double z2 =
        local_surface_z(*elem.desc2, *elem.grid2, elem.graph2, local.v);
    if (std::isnan(z1) || std::isnan(z2)) {
        return false;
    }
    return std::min(z1, z2) < local[2] && local[2] < std::max(z1, z2);
}

ElementTrace trace_element(const ElementContext& elem, const double* o, const double* d) {
    const bool inside = element_contains_point(elem, o);
    ElementHit entry = element_first_hit(elem, o, d, 0.0);
    ElementTrace out;
    out.pts.assign(static_cast<std::size_t>(kMaxPathPts), Vec3{});
    if (!entry.found) {
        if (inside) {
            out.pts[0] = as_vec3(o);
            out.exit_o = as_vec3(o);
            out.exit_d = as_vec3(d);
            out.exited = true;
            out.n_pts = 1;
            out.has_incident = false;
            out.incident_pt = as_vec3(o);
            return out;
        }
        out.exit_o = as_vec3(o);
        out.exit_d = as_vec3(d);
        out.exited = false;
        out.n_pts = 0;
        out.has_incident = false;
        out.incident_pt = as_vec3(o);
        out.pts.clear();
        return out;
    }
    const double n = elem.index;
    Vec3 cur_o;
    Vec3 cur_d;
    if (inside) {
        cur_o = as_vec3(o);
        cur_d = as_vec3(d);
        out.has_incident = false;
        out.incident_pt = as_vec3(o);
        out.pts[0] = as_vec3(o);
        out.n_pts = 1;
    } else {
        out.has_incident = true;
        out.incident_pt = entry.p;
        const auto refr = refract_or_reflect(as_vec3(d), orient(entry.n, as_vec3(d)), 1.0, n);
        cur_o = entry.p + RAY_EPSILON * refr.first;
        cur_d = refr.first;
        out.pts[0] = entry.p;
        out.n_pts = 1;
    }
    for (int bounce = 0; bounce < MAX_INTERNAL_BOUNCES; ++bounce) {
        ElementHit face = element_first_hit(elem, cur_o.v, cur_d.v, MIN_HIT_T);
        if (!face.found) {
            out.exit_o = cur_o;
            out.exit_d = cur_d;
            out.exited = false;
            out.pts.resize(static_cast<std::size_t>(out.n_pts));
            return out;
        }
        out.pts[static_cast<std::size_t>(out.n_pts)] = face.p;
        out.n_pts += 1;
        const auto refr = refract_or_reflect(cur_d, orient(face.n, cur_d), n, 1.0);
        if (refr.second) {
            out.exit_o = face.p;
            out.exit_d = refr.first;
            out.exited = true;
            out.pts.resize(static_cast<std::size_t>(out.n_pts));
            return out;
        }
        cur_o = face.p + RAY_EPSILON * refr.first;
        cur_d = refr.first;
    }
    out.exit_o = cur_o;
    out.exit_d = cur_d;
    out.exited = false;
    out.pts.resize(static_cast<std::size_t>(out.n_pts));
    return out;
}

StackSegments trace_stack_segments(const TraceStack& stack, const double* o, const double* d,
                                   double ray_length, double* starts, double* ends,
                                   signed char* kinds, std::size_t cap) {
    std::vector<unsigned char> visited(stack.size(), 0);
    Vec3 cur_o = as_vec3(o);
    Vec3 cur_d = as_vec3(d);
    signed char kind = 0;
    int count = 0;
    for (;;) {
        const int best_i = nearest_unvisited(stack, visited, cur_o.v, cur_d.v, kInf);
        if (best_i < 0) {
            break;
        }
        visited[static_cast<std::size_t>(best_i)] = 1;
        const ElementTrace tr = trace_element(*stack.elements[static_cast<std::size_t>(best_i)],
                                              cur_o.v, cur_d.v);
        if (tr.has_incident && static_cast<std::size_t>(count) < cap) {
            const double* s = cur_o.v;
            const double* e = tr.incident_pt.v;
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
            ends[3 * count + i] = cur_o[i] + ray_length * cur_d[i];
        }
        kinds[count] = kind;
        ++count;
    }
    return StackSegments{count, true, cur_o, cur_d};
}

bool trace_stack_to_plane(const TraceStack& stack, const double* o, const double* d,
                          const double* po, const double* pn, double eps, double u_min,
                          double u_max, double v_min, double v_max, double* out_o, double* out_d) {
    std::vector<unsigned char> visited(stack.size(), 0);
    Vec3 cur_o = as_vec3(o);
    Vec3 cur_d = as_vec3(d);
    for (;;) {
        std::optional<std::array<double, 3>> plane =
            plane_intersect(cur_o.v, cur_d.v, po, pn, eps);
        if (!plane.has_value()) {
            break;
        }
        const int best_i = nearest_unvisited(stack, visited, cur_o.v, cur_d.v, plane->at(2));
        if (best_i < 0) {
            break;
        }
        visited[static_cast<std::size_t>(best_i)] = 1;
        const ElementTrace tr = trace_element(*stack.elements[static_cast<std::size_t>(best_i)],
                                              cur_o.v, cur_d.v);
        for (int k = 0; k + 1 < tr.n_pts; ++k) {
            const Vec3 seg_o = tr.pts[static_cast<std::size_t>(k)];
            const Vec3 diff = tr.pts[static_cast<std::size_t>(k + 1)] - seg_o;
            const double seg_len = norm(diff);
            if (seg_len <= 0.0) {
                continue;
            }
            const Vec3 seg_d = diff / seg_len;
            std::optional<std::array<double, 3>> hit =
                plane_intersect(seg_o.v, seg_d.v, po, pn, eps);
            if (hit.has_value() && hit->at(0) >= u_min && hit->at(0) <= u_max &&
                hit->at(1) >= v_min && hit->at(1) <= v_max && hit->at(2) <= seg_len) {
                for (int i = 0; i < 3; ++i) {
                    out_o[i] = seg_o[i];
                    out_d[i] = seg_d[i];
                }
                return true;
            }
        }
        if (!tr.exited) {
            return false;
        }
        cur_o = tr.exit_o;
        cur_d = tr.exit_d;
    }
    for (int i = 0; i < 3; ++i) {
        out_o[i] = cur_o[i];
        out_d[i] = cur_d[i];
    }
    return true;
}

double accumulate_power(double* grid, const double* u, const double* v, const double* power,
                        std::size_t n, double u_min, double u_max, double v_min, double v_max,
                        int ru, int rv, double interval_u, double interval_v) {
    double total = 0.0;
    (void)rv;
    for (std::size_t idx = 0; idx < n; ++idx) {
        const double uu = u[idx];
        const double vv = v[idx];
        const double pw = power[idx];
        if (!(std::isfinite(uu) && std::isfinite(vv) && std::isfinite(pw))) {
            continue;
        }
        if (uu < u_min || uu > u_max || vv < v_min || vv > v_max) {
            continue;
        }
        int i = static_cast<int>(std::floor((uu - u_min) / interval_u * ru));
        int j = static_cast<int>(std::floor((vv - v_min) / interval_v * rv));
        i = std::min(std::max(i, 0), ru - 1);
        j = std::min(std::max(j, 0), rv - 1);
        grid[static_cast<std::size_t>(j) * static_cast<std::size_t>(ru) +
             static_cast<std::size_t>(i)] += pw;
        total += pw;
    }
    return total;
}

void measure_direct(const double* origins, const double* dirs, const double* powers, std::size_t n,
                    const double* o, const double* nrm, double eps, double* us, double* vs,
                    double* pws) {
    run_chunks(n, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i) {
            std::optional<std::array<double, 3>> hit =
                plane_intersect(origins + 3 * i, dirs + 3 * i, o, nrm, eps);
            if (hit.has_value()) {
                us[i] = hit->at(0);
                vs[i] = hit->at(1);
                pws[i] = powers[i];
            }
        }
    });
}

void measure_stack(const TraceStack& stack, const double* origins, const double* dirs,
                   const double* powers, std::size_t n, const double* o, const double* nrm,
                   double eps, double u_min, double u_max, double v_min, double v_max, double* us,
                   double* vs, double* pws) {
    run_chunks(n, [&](std::size_t lo, std::size_t hi) {
        double out_o[3];
        double out_d[3];
        for (std::size_t i = lo; i < hi; ++i) {
            const bool found =
                trace_stack_to_plane(stack, origins + 3 * i, dirs + 3 * i, o, nrm, eps, u_min,
                                     u_max, v_min, v_max, out_o, out_d);
            if (!found) {
                continue;
            }
            std::optional<std::array<double, 3>> hit =
                plane_intersect(out_o, out_d, o, nrm, eps);
            if (hit.has_value()) {
                us[i] = hit->at(0);
                vs[i] = hit->at(1);
                pws[i] = powers[i];
            }
        }
    });
}

// --- Single-trace shared walk (rendering + all measurement planes) -----------
//
// A full geometric-order walk of the stack (the same nearest-unvisited loop
// used by ``trace_stack_segments``) is a superset of every per-plane walk
// ``trace_stack_to_plane`` performs: the refractions up to any plane are a
// prefix of this walk, so one trace serves rendering and every plane. Each hop
// records the free-travel gap (origin/direction + entry distance to the next
// element) and the entered element's internal polyline, letting both the
// rendering segment buffers and the per-plane first-crossing be derived
// without re-walking.
namespace {

struct MHop {
    Vec3 o{};
    Vec3 d{};
    Vec3 incident_pt{};
    double entry_t = kInf;
    bool has_element = false;
    bool has_incident = false;
    bool exited = false;
    std::vector<Vec3> pts;
};

struct RayHops {
    std::vector<MHop> hops;
};

RayHops build_hops(const TraceStack& stack, const double* o, const double* d) {
    std::vector<unsigned char> visited(stack.size(), 0);
    Vec3 cur_o = as_vec3(o);
    Vec3 cur_d = as_vec3(d);
    RayHops hops;
    for (;;) {
        const std::pair<int, double> best =
            nearest_unvisited_t(stack, visited, cur_o.v, cur_d.v, kInf);
        if (best.first < 0) {
            MHop tail;
            tail.o = cur_o;
            tail.d = cur_d;
            tail.has_element = false;
            hops.hops.push_back(std::move(tail));
            return hops;
        }
        visited[static_cast<std::size_t>(best.first)] = 1;
        const ElementTrace tr =
            trace_element(*stack.elements[static_cast<std::size_t>(best.first)], cur_o.v, cur_d.v);
        MHop h;
        h.o = cur_o;
        h.d = cur_d;
        h.entry_t = best.second;
        h.has_element = true;
        h.has_incident = tr.has_incident;
        h.exited = tr.exited;
        if (tr.has_incident) {
            h.incident_pt = tr.incident_pt;
        }
        h.pts = tr.pts;
        hops.hops.push_back(std::move(h));
        if (!tr.exited) {
            return hops;  // absorbed inside the element: no trailing gap
        }
        cur_o = tr.exit_o;
        cur_d = tr.exit_d;
    }
}

// Rendering segment buffers, byte-identical to ``trace_stack_segments``:
// incident + internal segments per element, then the trailing gap capped at
// ``ray_length`` (omitted when the ray is absorbed).
int format_segments(const RayHops& hops, double ray_length, double* starts, double* ends,
                    signed char* kinds, std::size_t cap) {
    std::size_t count = 0;
    signed char kind = 0;
    for (const MHop& h : hops.hops) {
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

// The ray that reaches the plane, read off the shared walk. Mirrors
// ``trace_stack_to_plane`` exactly: at each gap, the plane is measured there
// when no element is entered before the crossing (crossing t <= entry_t, or
// parallel with no element); otherwise the element's internal segments are
// checked (in extent and within the segment), continuing to the next gap on
// exit. Returns ``false`` when the ray never reaches the plane.
bool measure_hops_plane(const RayHops& hops, const double* po, const double* pn, double u_min,
                        double u_max, double v_min, double v_max, double eps, double* u,
                        double* v) {
    for (const MHop& h : hops.hops) {
        const std::optional<std::array<double, 3>> ph = plane_intersect(h.o.v, h.d.v, po, pn, eps);
        const bool enter = h.has_element && (!ph.has_value() || ph->at(2) > h.entry_t);
        if (!enter) {
            if (ph.has_value()) {
                *u = ph->at(0);
                *v = ph->at(1);
                return true;
            }
            return false;
        }
        const std::size_t m = h.pts.size();
        for (std::size_t k = 0; k + 1 < m; ++k) {
            const Vec3 seg_o = h.pts[k];
            const Vec3 diff = h.pts[k + 1] - seg_o;
            const double seg_len = norm(diff);
            if (seg_len <= 0.0) {
                continue;
            }
            const Vec3 seg_d = diff / seg_len;
            const std::optional<std::array<double, 3>> hit = plane_intersect(seg_o.v, seg_d.v, po, pn, eps);
            if (hit.has_value() && hit->at(0) >= u_min && hit->at(0) <= u_max &&
                hit->at(1) >= v_min && hit->at(1) <= v_max && hit->at(2) <= seg_len) {
                *u = hit->at(0);
                *v = hit->at(1);
                return true;
            }
        }
        if (!h.exited) {
            return false;  // absorbed inside the element before reaching the plane
        }
    }
    return false;
}

}  // namespace

void measure_planes_multi(const TraceStack& stack, const double* origins, const double* dirs,
                          const double* powers, std::size_t n, std::size_t p, const double* po,
                          const double* pn, const double* u_min, const double* u_max,
                          const double* v_min, const double* v_max, double eps, double* us,
                          double* vs, double* pws) {
    run_chunks(n, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i) {
            const RayHops hops = build_hops(stack, origins + 3 * i, dirs + 3 * i);
            for (std::size_t q = 0; q < p; ++q) {
                double u = 0.0;
                double v = 0.0;
                if (measure_hops_plane(hops, po + 3 * q, pn + 3 * q, u_min[q], u_max[q], v_min[q],
                                       v_max[q], eps, &u, &v)) {
                    const std::size_t idx = q * n + i;
                    us[idx] = u;
                    vs[idx] = v;
                    pws[idx] = powers[i];
                }
            }
        }
    });
}

void trace_segments_and_planes(const TraceStack& stack, const double* origins, const double* dirs,
                               const double* powers, std::size_t n, double ray_length, std::size_t p,
                               const double* po, const double* pn, const double* u_min,
                               const double* u_max, const double* v_min, const double* v_max,
                               double eps, double* starts, double* ends, signed char* kinds,
                               std::int64_t* counts, std::size_t cap, double* us, double* vs,
                               double* pws) {
    run_chunks(n, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i) {
            const RayHops hops = build_hops(stack, origins + 3 * i, dirs + 3 * i);
            counts[i] = format_segments(hops, ray_length, starts + 3 * cap * i, ends + 3 * cap * i,
                                        kinds + cap * i, cap);
            for (std::size_t q = 0; q < p; ++q) {
                double u = 0.0;
                double v = 0.0;
                if (measure_hops_plane(hops, po + 3 * q, pn + 3 * q, u_min[q], u_max[q], v_min[q],
                                       v_max[q], eps, &u, &v)) {
                    us[q * n + i] = u;
                    vs[q * n + i] = v;
                    pws[q * n + i] = powers[i];
                }
            }
        }
    });
}

Mat3 rotation_from_z(const Vec3& direction) {
    const Vec3 z_axis = make_vec3(0.0, 0.0, 1.0);
    const Vec3 direction_norm = normalize_vector(direction);
    const Vec3 axis = cross3(z_axis, direction_norm);
    const double axis_norm = norm(axis);
    const double cos_theta = dot3(z_axis, direction_norm);
    if (axis_norm < VECTOR_PARALLEL_TOL) {
        Mat3 r;
        if (cos_theta < 0.0) {
            r = Mat3::identity();
            r.m[1][1] = -1.0;
            r.m[2][2] = -1.0;
            return r;
        }
        return Mat3::identity();
    }
    const Mat3 skew{
        {{0.0, -axis[2], axis[1]}, {axis[2], 0.0, -axis[0]}, {-axis[1], axis[0], 0.0}}};
    const Mat3 ident = Mat3::identity();
    const double s = (1.0 - cos_theta) / (axis_norm * axis_norm);
    const Mat3 skew2 = mat3_mul(skew, skew);
    Mat3 r;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            r.m[i][j] = ident.m[i][j] + skew.m[i][j] + skew2.m[i][j] * s;
        }
    }
    return r;
}

}  // namespace optics::native
