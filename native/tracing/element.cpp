// Element-level tracing: nearest-face assembly over both surfaces and the
// sidewall mesh, containment testing and the internal-bounce loop for one
// closed refractive element, plus the construction of the persistent element.
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
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

// ``numpy.linspace(a, b, n)`` (endpoint included; a single sample is ``[a]``).
void linspace_nodes(double a, double b, int n, std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(n < 0 ? 0 : n), 0.0);
    if (n <= 0) {
        return;
    }
    if (n == 1) {
        out[0] = a;
        return;
    }
    const double step = (b - a) / (n - 1);
    for (int i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = a + step * i;
    }
    out[static_cast<std::size_t>(n - 1)] = b;
}

}  // namespace

// Accumulate one surface's local-frame sample bounds into ``lo``/``hi``.
void sample_surface_bounds(const SurfaceDesc& d, Vec3& lo, Vec3& hi) {
    std::vector<double> us;
    std::vector<double> vs;
    linspace_nodes(d.u_min, d.u_max, CONTAINMENT_BOUNDS_N, us);
    linspace_nodes(d.v_min, d.v_max, CONTAINMENT_BOUNDS_N, vs);
    const int n = CONTAINMENT_BOUNDS_N;
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            const Vec3 p =
                d.model->eval_local(us[static_cast<std::size_t>(c)], vs[static_cast<std::size_t>(r)]);
            if (!finite3(p)) {
                continue;
            }
            for (int j = 0; j < 3; ++j) {
                lo[j] = std::min(lo[j], p[j]);
                hi[j] = std::max(hi[j], p[j]);
            }
        }
    }
}

namespace {

// Nearest surface hit with normal, or ``(False, inf, 0, 0)``.
ElementHit surface_hit(const SurfaceDesc& desc, const SeedGrid& grid, const Vec3& o,
                       const Vec3& d, double min_t) {
    std::optional<HitUV> hit = surface_intersect(desc, grid, o, d);
    if (hit.has_value() && hit->t > min_t) {
        std::optional<Vec3> normal = surface_normal(desc, hit->u, hit->v);
        if (normal.has_value()) {
            Vec3 bn = normal.value();
            const Vec3 p = o + hit->t * d;
            return ElementHit{true, hit->t, p, bn};
        }
    }
    return ElementHit{false, kInf, Vec3{}, Vec3{}};
}

// Local height of a surface above ``(local[0], local[1])``; NaN when the
// point does not project onto the surface.
double local_surface_z(const SurfaceDesc& desc, const SeedGrid& grid, bool graph,
                       const Vec3& local) {
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

std::shared_ptr<const Element> Element::create(std::shared_ptr<const Surface> surface1,
                                               std::shared_ptr<const Surface> surface2,
                                               double refractive_index,
                                               std::vector<double> tri,
                                               std::vector<double> normals) {
    if (surface1 == nullptr || surface2 == nullptr) {
        throw std::invalid_argument("element requires both surfaces");
    }
    if (tri.size() % 9 != 0) {
        throw std::invalid_argument("tri: expected shape (K, 3, 3)");
    }
    if (normals.size() != tri.size() / 3) {
        throw std::invalid_argument("normals: expected shape (K, 3)");
    }
    std::shared_ptr<Element> elem(new Element());
    elem->surface1 = std::move(surface1);
    elem->surface2 = std::move(surface2);
    elem->desc1 = elem->surface1->desc_ptr();
    elem->desc2 = elem->surface2->desc_ptr();
    elem->grid1 = elem->surface1->grid_ptr();
    elem->grid2 = elem->surface2->grid_ptr();
    elem->index = refractive_index;
    for (int i = 0; i < 3; ++i) {
        elem->lo[i] = kInf;
        elem->hi[i] = -kInf;
    }
    sample_surface_bounds(*elem->desc1, elem->lo, elem->hi);
    sample_surface_bounds(*elem->desc2, elem->lo, elem->hi);
    elem->tri = std::move(tri);
    elem->normals = std::move(normals);
    elem->graph1 = elem->surface1->is_graph_over_xy();
    elem->graph2 = elem->surface2->is_graph_over_xy();
    return elem;
}

ElementHit element_first_hit(const Element& elem, const Vec3& o, const Vec3& d,
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
            bp = o + wall->t * d;
            bn = as_vec3(&elem.normals[3 * wall->face]);
        }
    }
    return ElementHit{found, best_t, bp, bn};
}

bool element_contains_point(const Element& elem, const Vec3& p) {
    const SurfaceDesc& desc = *elem.desc1;
    const Vec3 local = desc.transform.to_local(p);
    if (!finite3(local)) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        if (local[i] < elem.lo[i] || local[i] > elem.hi[i]) {
            return false;
        }
    }
    const double z1 = local_surface_z(*elem.desc1, *elem.grid1, elem.graph1, local);
    const double z2 = local_surface_z(*elem.desc2, *elem.grid2, elem.graph2, local);
    if (std::isnan(z1) || std::isnan(z2)) {
        return false;
    }
    return std::min(z1, z2) < local[2] && local[2] < std::max(z1, z2);
}

ElementTrace trace_element(const Element& elem, const Vec3& o, const Vec3& d) {
    const bool inside = element_contains_point(elem, o);
    ElementHit entry = element_first_hit(elem, o, d, 0.0);
    ElementTrace out;
    out.pts.assign(static_cast<std::size_t>(kMaxPathPts), Vec3{});
    if (!entry.found) {
        if (inside) {
            out.pts[0] = o;
            out.exit_o = o;
            out.exit_d = d;
            out.exited = true;
            out.n_pts = 1;
            out.has_incident = false;
            out.incident_pt = o;
            return out;
        }
        out.exit_o = o;
        out.exit_d = d;
        out.exited = false;
        out.n_pts = 0;
        out.has_incident = false;
        out.incident_pt = o;
        out.pts.clear();
        return out;
    }
    const double n = elem.index;
    Vec3 cur_o;
    Vec3 cur_d;
    if (inside) {
        cur_o = o;
        cur_d = d;
        out.has_incident = false;
        out.incident_pt = o;
        out.pts[0] = o;
        out.n_pts = 1;
    } else {
        out.has_incident = true;
        out.incident_pt = entry.p;
        const auto refr = refract_or_reflect(d, orient(entry.n, d), 1.0, n);
        cur_o = entry.p + RAY_EPSILON * refr.first;
        cur_d = refr.first;
        out.pts[0] = entry.p;
        out.n_pts = 1;
    }
    for (int bounce = 0; bounce < MAX_INTERNAL_BOUNCES; ++bounce) {
        ElementHit face = element_first_hit(elem, cur_o, cur_d, MIN_HIT_T);
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
