// Element-level kernel suite: nearest-face assembly (surfaces + sidewall),
// containment, refraction through a parallel-plane plate and total internal
// reflection.
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/vector.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/tracing/context.hpp"
#include "optics/tracing/trace.hpp"

#include "check.hpp"
#include "test_element.hpp"

namespace {

using optics::native::element_contains_point;
using optics::native::element_first_hit;
using optics::native::make_vec3;
using optics::native::norm;
using optics::native::Surface;
using optics::native::trace_element;
using optics::native::Vec3;

// sqrt(1 - 0.3^2): the incident ray below has sin(theta) = 0.3.
constexpr double kCos1 = 0.9539392014169457;

fixture::Element small_plate() {
    return fixture::make_plate(1.5, 1.0, -0.5, 0.5, 0.0, 48, true);
}

// The persistent ``Element::create`` path must derive the same AABB, graph
// flag and refractive index as the field-assembled fixture plate.
void test_create_persists_state() {
    auto face = [](double height, double half) {
        char z_src[64];
        std::snprintf(z_src, sizeof(z_src), "%g", height);
        Surface::Options o;
        o.x_src = "u";
        o.y_src = "v";
        o.z_src = z_src;
        o.u_min = -half;
        o.u_max = half;
        o.v_min = -half;
        o.v_max = half;
        o.grid_n = 48;
        return Surface::create(std::move(o));
    };
    const auto elem = optics::native::Element::create(
        face(0.5, 1.0), face(-0.5, 1.0), 1.5, std::vector<double>{}, std::vector<double>{});
    CHECK(elem->surface1 != nullptr && elem->surface2 != nullptr);
    CHECK(elem->desc1 == elem->surface1->desc_ptr());
    CHECK(elem->grid1 == elem->surface1->grid_ptr());
    CHECK(close(elem->index, 1.5));
    CHECK(close(elem->lo[0], -1.0) && close(elem->lo[1], -1.0) && close(elem->lo[2], -0.5));
    CHECK(close(elem->hi[0], 1.0) && close(elem->hi[1], 1.0) && close(elem->hi[2], 0.5));
    CHECK(elem->graph1 && elem->graph2);
    CHECK(elem->n_tris() == 0);

    // Trace matches the field-assembled fixture plate bit-for-bit.
    const optics::native::ElementHit hit =
        optics::native::element_first_hit(*elem, make_vec3(0.1, 0.2, 3.0),
                                          make_vec3(0.0, 0.0, -1.0), 0.0);
    CHECK(hit.found);
    CHECK(close(hit.t, 2.5) && close(hit.p[2], 0.5));

    // Inconsistent sidewall arrays are rejected.
    bool threw = false;
    try {
        optics::native::Element::create(face(0.5, 1.0), face(-0.5, 1.0), 1.5,
                                        std::vector<double>(10, 0.0), std::vector<double>{});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void test_first_hit() {
    const fixture::Element plate = small_plate();

    const Vec3 o = make_vec3(0.1, 0.2, 3.0);
    const Vec3 d = make_vec3(0.0, 0.0, -1.0);
    const optics::native::ElementHit hit = element_first_hit(plate, o, d, 0.0);
    CHECK(hit.found);
    CHECK(close(hit.t, 2.5));
    CHECK(close(hit.p[0], 0.1) && close(hit.p[1], 0.2) && close(hit.p[2], 0.5));
    // The normal convention is ``n = -cross(r_u, r_v)``, so a ``z = const``
    // face has world normal -z.
    CHECK(std::abs(hit.n[0]) < 1e-8 && std::abs(hit.n[1]) < 1e-8 && close(hit.n[2], -1.0));

    // Sidewall entry from the side (the plane faces are parallel to the ray).
    const Vec3 wo = make_vec3(-3.0, 0.0, 0.0);
    const Vec3 wd = make_vec3(1.0, 0.0, 0.0);
    const optics::native::ElementHit wall = element_first_hit(plate, wo, wd, 0.0);
    CHECK(wall.found);
    CHECK(close(wall.t, 2.0));
    CHECK(close(wall.p[0], -1.0) && close(wall.p[2], 0.0));
    CHECK(close(wall.n[0], -1.0) && std::abs(wall.n[1]) < 1e-12 && std::abs(wall.n[2]) < 1e-12);

    // ``min_t`` rejects the mesh hit as a whole: the kernel sees only the
    // nearest sidewall face, so a ``min_t`` past it hides the far wall too
    // (semantics the ``Element`` refactor must preserve).
    const optics::native::ElementHit far = element_first_hit(plate, wo, wd, 2.5);
    CHECK(!far.found);
}

void test_contains() {
    const fixture::Element plate = small_plate();
    CHECK(element_contains_point(plate, make_vec3(0.0, 0.1, 0.0)));
    CHECK(!element_contains_point(plate, make_vec3(0.0, 0.0, 0.6)));
    CHECK(!element_contains_point(plate, make_vec3(2.0, 0.0, 0.0)));
    CHECK(!element_contains_point(plate, make_vec3(0.0, 0.0, 0.5)));
}

void test_trace_straight() {
    const fixture::Element plate = small_plate();
    const Vec3 o = make_vec3(0.0, 0.0, 3.0);
    const Vec3 d = make_vec3(0.0, 0.0, -1.0);
    const optics::native::ElementTrace tr = trace_element(plate, o, d);
    CHECK(tr.exited);
    CHECK(tr.has_incident);
    CHECK(tr.n_pts == 2);
    CHECK(close(tr.incident_pt[2], 0.5));
    CHECK(close(tr.exit_o[2], -0.5));
    CHECK(close(tr.exit_o[0], 0.0, 1e-12) && close(tr.exit_o[1], 0.0, 1e-12));
    CHECK(close(tr.exit_d[2], -1.0, 1e-12) && std::abs(tr.exit_d[0]) < 1e-12);
}

void test_trace_oblique() {
    const fixture::Element plate = small_plate();
    const Vec3 o = make_vec3(-0.2, 0.0, 2.0);
    const Vec3 d = make_vec3(0.3, 0.0, -kCos1);
    const optics::native::ElementTrace tr = trace_element(plate, o, d);
    CHECK(tr.exited);
    CHECK(tr.n_pts == 2);
    // Snell (1.0 -> 1.5): entry point from the free path, lateral drift from
    // the internal slope ``sin/cos = 0.2 / sqrt(1 - 0.2^2)`` over ``dz = 1``.
    const double t_entry = 1.5 / kCos1;
    const double x_entry = -0.2 + 0.3 * t_entry;
    const double tan_in = 0.2 / std::sqrt(1.0 - 0.04);
    const double x_exit = x_entry + tan_in * 1.0;
    CHECK(close(tr.incident_pt[0], x_entry, 1e-8));
    CHECK(close(tr.incident_pt[2], 0.5, 1e-8));
    CHECK(close(tr.exit_o[0], x_exit, 1e-8));
    CHECK(close(tr.exit_o[2], -0.5, 1e-8));
    // A parallel-plane plate deflects the ray back onto the incident
    // direction (lateral shift only).
    CHECK(close(tr.exit_d[0], 0.3, 1e-9));
    CHECK(close(tr.exit_d[2], -kCos1, 1e-9));
    CHECK(close(norm(tr.exit_d), 1.0, 1e-12));
}

void test_trace_tir() {
    // Tall plate (no sidewalls): a 45-degree internal ray exceeds the
    // critical angle ``asin(1/1.5) ~ 41.8`` and is trapped at every face.
    const fixture::Element slab =
        fixture::make_plate(1.5, 12.0, -0.5, 0.5, 0.0, 256, false);
    const Vec3 o = make_vec3(0.0, 0.0, 0.0);
    const double s = 1.0 / std::sqrt(2.0);
    const Vec3 d = make_vec3(s, 0.0, s);  // > critical angle from the normal
    CHECK(element_contains_point(slab, o));

    const optics::native::ElementTrace tr = trace_element(slab, o, d);
    CHECK(!tr.exited);  // total internal reflection at every bounce
    CHECK(!tr.has_incident);  // the ray started inside the glass
    CHECK(tr.n_pts == optics::native::MAX_INTERNAL_BOUNCES + 1);
    for (int k = 1; k < tr.n_pts; ++k) {
        CHECK(close(std::abs(tr.pts[static_cast<std::size_t>(k)][2]), 0.5, 1e-8));
        // 45-degree zig-zag: hits at x = 0.5, 1.5, 2.5, ...
        CHECK(close(tr.pts[static_cast<std::size_t>(k)][0], k - 0.5, 1e-7));
        CHECK(close(tr.pts[static_cast<std::size_t>(k)][1], 0.0, 1e-12));
    }
}

}  // namespace

void test_element() {
    test_create_persists_state();
    test_first_hit();
    test_contains();
    test_trace_straight();
    test_trace_oblique();
    test_trace_tir();
}
