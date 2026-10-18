// Shared fixtures for the element/stack/measurement suites: a square
// parallel-plane plate element assembled field-by-field exactly the way the
// ``build_element_context`` transitional path fills ``Element`` (world-space
// sidewall triangles, local-frame AABB bounds, graph-over-xy surface flags).
#pragma once

#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/vector.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/solver.hpp"
#include "optics/tracing/context.hpp"

namespace fixture {

using optics::native::DOMAIN_TOL;
using optics::native::Element;
using optics::native::ElementRef;
using optics::native::make_vec3;
using optics::native::NoAperture;
using optics::native::SeedGrid;
using optics::native::SurfaceDesc;
using optics::native::SurfaceModel;
using optics::native::TraceStack;

// A horizontal plane patch ``z = height`` over ``[-half, half]^2``. The
// height is baked into the expression (identity transform) so both faces of
// a plate share the element-local frame, as ``RefractiveElement`` faces do.
inline std::shared_ptr<const SurfaceDesc> plate_face(double height, double half) {
    char z_src[64];
    std::snprintf(z_src, sizeof(z_src), "%g", height);
    auto desc = std::make_shared<SurfaceDesc>();
    desc->model = SurfaceModel::compile("u", "v", z_src, -half, half, -half, half);
    desc->u_min = -half;
    desc->u_max = half;
    desc->v_min = -half;
    desc->v_max = half;
    desc->tu = DOMAIN_TOL * 2.0;
    desc->tv = DOMAIN_TOL * 2.0;
    desc->aperture = NoAperture{};
    return desc;
}

// Square-cylinder sidewalls around ``[-half, half]^2`` between ``z_bot`` and
// ``z_top`` in world coordinates (2 triangles per face with outward unit
// normals), mirroring ``LensSidewall.triangles()/face_normals()``. Winding is
// irrelevant (Moller-Trumbore here is two-sided).
inline void square_walls(double half, double z_bot, double z_top, std::vector<double>& tri,
                         std::vector<double>& normals) {
    auto push = [&](double ax, double ay, double az, double bx, double by, double bz, double cx,
                    double cy, double cz, double nx, double ny, double nz) {
        const double t[9] = {ax, ay, az, bx, by, bz, cx, cy, cz};
        tri.insert(tri.end(), t, t + 9);
        const double n[3] = {nx, ny, nz};
        normals.insert(normals.end(), n, n + 3);
    };
    // -x face
    push(-half, -half, z_bot, -half, half, z_top, -half, half, z_bot, -1.0, 0.0, 0.0);
    push(-half, -half, z_bot, -half, -half, z_top, -half, half, z_top, -1.0, 0.0, 0.0);
    // +x face
    push(half, -half, z_bot, half, half, z_top, half, half, z_bot, 1.0, 0.0, 0.0);
    push(half, -half, z_bot, half, -half, z_top, half, half, z_top, 1.0, 0.0, 0.0);
    // -y face
    push(-half, -half, z_bot, half, -half, z_top, half, -half, z_bot, 0.0, -1.0, 0.0);
    push(-half, -half, z_bot, -half, -half, z_top, half, -half, z_top, 0.0, -1.0, 0.0);
    // +y face
    push(-half, half, z_bot, half, half, z_top, half, half, z_bot, 0.0, 1.0, 0.0);
    push(-half, half, z_bot, -half, half, z_top, half, half, z_top, 0.0, 1.0, 0.0);
}

// One closed plate element: two plane faces at ``z_bot``/``z_top`` (local z),
// the whole plate optionally shifted to ``trans_z`` in world space.
inline Element make_plate(double index, double half, double z_bot, double z_top,
                                 double trans_z, int grid_n, bool walls) {
    auto face_top = plate_face(z_top, half);
    auto face_bot = plate_face(z_bot, half);
    if (trans_z != 0.0) {
        const optics::native::Vec3 tz = make_vec3(0.0, 0.0, trans_z);
        auto shift = [&](const std::shared_ptr<const SurfaceDesc>& d) {
            auto copy = std::make_shared<SurfaceDesc>(*d);
            copy->transform.trans = tz;
            return copy;
        };
        face_top = shift(face_top);
        face_bot = shift(face_bot);
    }
    Element ctx;
    ctx.desc1 = face_top;
    ctx.desc2 = face_bot;
    ctx.grid1 = std::make_shared<const SeedGrid>(build_seed_grid(*face_top, grid_n));
    ctx.grid2 = std::make_shared<const SeedGrid>(build_seed_grid(*face_bot, grid_n));
    ctx.index = index;
    ctx.lo = make_vec3(-half, -half, z_bot);
    ctx.hi = make_vec3(half, half, z_top);
    ctx.graph1 = true;
    ctx.graph2 = true;
    if (walls) {
        square_walls(half, z_bot + trans_z, z_top + trans_z, ctx.tri, ctx.normals);
    }
    return ctx;
}

inline ElementRef make_plate_ptr(double index, double half, double z_bot, double z_top,
                                 double trans_z, int grid_n, bool walls) {
    return std::make_shared<const Element>(
        make_plate(index, half, z_bot, z_top, trans_z, grid_n, walls));
}

inline TraceStack make_stack(std::vector<ElementRef> elements) {
    TraceStack stack;
    stack.elements = std::move(elements);
    return stack;
}

}  // namespace fixture
