// Persistent native Surface: eager seed grid, domain/aperture membership,
// transform round-trips, graph-over-xy probe, kernel entry points and the
// aperture rim.
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/matrix.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/solver.hpp"

#include "check.hpp"

namespace {

using optics::native::make_vec3;
using optics::native::Mat3;
using optics::native::SEED_GRID_N;
using optics::native::Surface;
using optics::native::Vec3;

Surface::Options plane_opts(double half = 1.0, int grid_n = 16, double trans_z = 0.0) {
    Surface::Options o;
    o.x_src = "u";
    o.y_src = "v";
    o.z_src = "0";
    o.u_min = -half;
    o.u_max = half;
    o.v_min = -half;
    o.v_max = half;
    o.grid_n = grid_n;
    o.trans = make_vec3(0.0, 0.0, trans_z);
    return o;
}

void test_plane() {
    const auto s = Surface::create(plane_opts());
    CHECK(s->grid().p_world.size() == 3 * 16 * 16);
    CHECK(s->grid().uf_valid.size() == 16 * 16);
    CHECK(s->is_graph_over_xy());
    CHECK(s->accepts(0.0, 0.0));
    CHECK(s->accepts(1.0, 1.0));
    CHECK(!s->accepts(1.5, 0.0));

    const Vec3 p = s->eval_local(0.25, -0.5);
    CHECK(close(p[0], 0.25) && close(p[1], -0.5) && close(p[2], 0.0));

    const auto n = s->normal(0.0, 0.0);
    CHECK(n.has_value());
    CHECK(close((*n)[2], -1.0) && close((*n)[0], 0.0) && close((*n)[1], 0.0));

    const auto hit = s->intersect(make_vec3(0.0, 0.0, 5.0), make_vec3(0.0, 0.0, -1.0));
    CHECK(hit.has_value());
    if (hit.has_value()) {
        CHECK(close(hit->t, 5.0));
        CHECK(close(hit->u, 0.0) && close(hit->v, 0.0));
    }

    const auto uv = s->parameters_at_local(make_vec3(0.3, -0.2, 0.0));
    CHECK(uv.has_value());
    if (uv.has_value()) {
        CHECK(close(uv->first, 0.3) && close(uv->second, -0.2));
    }
    const auto xy = s->parameters_at_xy(0.25, -0.5);
    CHECK(xy.has_value());
    if (xy.has_value()) {
        CHECK(close(xy->first, 0.25) && close(xy->second, -0.5));
    }

    // No aperture -> rim accessors refuse.
    bool threw = false;
    try {
        s->aperture_rim(8);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    const std::vector<double> rim = s->rim(1.0, 0.0, 0.0, 8);
    CHECK(rim.size() == 24);
    for (std::size_t i = 0; i + 2 < rim.size(); i += 3) {
        const double r = std::hypot(rim[i], rim[i + 1]);
        CHECK(close(r, 1.0, 1e-9));
    }
}

void test_default_grid() {
    Surface::Options o = plane_opts();
    o.grid_n = SEED_GRID_N;
    const auto s = Surface::create(std::move(o));
    CHECK(s->grid().p_world.size() == 3 * static_cast<std::size_t>(SEED_GRID_N) * SEED_GRID_N);
}

void test_transformed() {
    Surface::Options o = plane_opts();
    // Rotation by +90 deg about x: (x, y, z) -> (x, -z, y).
    optics::native::Mat3 rot{};
    rot.m[0][0] = 1.0;
    rot.m[1][2] = -1.0;
    rot.m[2][1] = 1.0;
    o.rot = rot;
    o.trans = make_vec3(1.0, 2.0, 3.0);
    const auto s = Surface::create(std::move(o));

    const Vec3 local = make_vec3(0.1, -0.2, 0.0);
    const Vec3 world = s->to_world(local);
    CHECK(close(world[0], 1.1) && close(world[1], 2.0) && close(world[2], 2.8));
    const Vec3 back = s->to_local(world);
    CHECK(close(back[0], local[0]) && close(back[1], local[1]) && close(back[2], local[2]));

    // Normal is rotated: local (0,0,-1) -> rot * (0,0,-1) = (0, 1, 0).
    const auto n = s->normal(0.0, 0.0);
    CHECK(n.has_value());
    if (n.has_value()) {
        CHECK(close((*n)[0], 0.0) && close((*n)[1], 1.0) && close((*n)[2], 0.0));
    }
}

void test_graph_probe() {
    Surface::Options tilted = plane_opts();
    tilted.z_src = "u + v";
    CHECK(Surface::create(tilted)->is_graph_over_xy());

    Surface::Options offset = plane_opts();
    offset.x_src = "u + 0.1 * u";
    CHECK(!Surface::create(std::move(offset))->is_graph_over_xy());

    Surface::Options sphere = plane_opts();
    sphere.z_src = "1 - sqrt(25 - u * u - v * v)";
    CHECK(Surface::create(std::move(sphere))->is_graph_over_xy());
}

void test_aperture() {
    Surface::Options o = plane_opts(3.0);  // domain [-3,3]^2 covers radius 2
    o.aperture_radius = 2.0;
    const auto s = Surface::create(std::move(o));
    CHECK(s->accepts(1.9, 0.0));
    CHECK(!s->accepts(2.5, 0.0));
    CHECK(!s->normal(2.5, 0.0).has_value());

    const std::vector<double> rim = s->aperture_rim(16);
    CHECK(rim.size() == 48);
    for (std::size_t i = 0; i + 2 < rim.size(); i += 3) {
        CHECK(close(std::hypot(rim[i], rim[i + 1]), 2.0, 1e-9));
    }
}

void test_sphere_intersect() {
    Surface::Options o = plane_opts();
    o.z_src = "1 - sqrt(25 - u * u - v * v)";
    const auto s = Surface::create(std::move(o));

    const auto hit = s->intersect(make_vec3(0.0, 0.0, 10.0), make_vec3(0.0, 0.0, -1.0));
    CHECK(hit.has_value());
    if (hit.has_value()) {
        CHECK(close(hit->t, 14.0, 1e-9));
        const Vec3 p = s->point(hit->u, hit->v);
        // Bottom cap of the sphere centered at (0, 0, 1) with R = 5.
        CHECK(close(std::hypot(std::hypot(p[0], p[1]), p[2] - 1.0), 5.0, 1e-9));
    }

    // Miss (direction away from the cap).
    CHECK(!s->intersect(make_vec3(0.0, 0.0, 10.0), make_vec3(0.0, 0.0, 1.0)).has_value());
    // Off-aperture domain edge: starting inside the square, hitting past it.
    CHECK(!s->intersect(make_vec3(9.0, 9.0, 10.0), make_vec3(0.0, 0.0, -1.0)).has_value());
}

}  // namespace

void test_surface() {
    std::printf("-- surface\n");
    test_plane();
    test_default_grid();
    test_transformed();
    test_graph_probe();
    test_aperture();
    test_sphere_intersect();
}
