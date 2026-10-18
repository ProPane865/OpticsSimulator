// Element-level tracing: nearest-face assembly over both surfaces and the
// sidewall mesh, containment testing and the internal-bounce loop for one
// closed refractive element.
#include <cmath>
#include <optional>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/matrix.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/mesh.hpp"
#include "optics/intersection/solver.hpp"
#include "optics/tracing/direction.hpp"
#include "optics/tracing/trace.hpp"

namespace optics::native {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr int kMaxPathPts = MAX_INTERNAL_BOUNCES + 2;

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

}  // namespace

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

}  // namespace optics::native
