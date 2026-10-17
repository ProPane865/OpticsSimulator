// Minimal dependency-free checks for the native kernels: expression parsing
// semantics and the Newton intersection/inversion paths.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "optics/expression.hpp"
#include "optics/grid.hpp"
#include "optics/kernels.hpp"
#include "optics/solver.hpp"
#include "optics/surface.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

bool close(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) <= tol * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

void test_expression() {
    using namespace optics::native;

    auto eval_src = [](const std::string& src, double u, double v) {
        return eval_expression(*parse_expression(src), u, v);
    };

    CHECK(close(eval_src("1 + 2 * 3", 0, 0), 7.0));
    CHECK(close(eval_src("-2 ** 2", 0, 0), -4.0));   // Python: -(2**2)
    CHECK(close(eval_src("2 ** -1", 0, 0), 0.5));    // right-assoc with unary
    CHECK(close(eval_src("7 % 3", 0, 0), 1.0));
    CHECK(close(eval_src("-7 % 3", 0, 0), 2.0));     // floor-mod
    CHECK(close(eval_src("sqrt(1 - (u**2) - (v**2))", 0.5, 0.0), std::sqrt(0.75)));
    CHECK(std::isnan(eval_src("sqrt(u)", -1.0, 0.0)));
    CHECK(std::isinf(eval_src("u / (v - 0.5)", 0.3, 0.5)));
    CHECK(std::isnan(eval_src("u / (v - 0.5)", 0.0, 0.5)));
    CHECK(close(eval_src("abs(u) + exp(0) + log(1) - sin(0) - cos(0)", -2.0, 0.0), 2.0));
    CHECK(close(eval_src("(u + 1) * (v - 2)", 1.0, 3.0), 2.0));

    auto must_throw = [&](const std::string& src) {
        bool threw = false;
        try {
            parse_expression(src);
        } catch (const ExpressionError&) {
            threw = true;
        }
        CHECK(threw);
    };
    must_throw("pi");
    must_throw("e");
    must_throw("u[0]");
    must_throw("u .real");
    must_throw("open('f')");
    must_throw("1 +");
    must_throw("foo(u)");
    must_throw("(lambda: 1)()");
    must_throw("u, v");
}

void test_rotation() {
    using namespace optics::native;
    const Mat3 id = rotation_from_z(make_vec3(0.0, 0.0, 1.0));
    CHECK(close(id.m[0][0], 1.0) && close(id.m[1][1], 1.0) && close(id.m[2][2], 1.0));
    const Mat3 flip = rotation_from_z(make_vec3(0.0, 0.0, -1.0));
    CHECK(close(flip.m[0][0], 1.0) && close(flip.m[1][1], -1.0) && close(flip.m[2][2], -1.0));
    const Vec3 up = rotation_from_z(make_vec3(0.0, 0.0, 1.0)) * make_vec3(0.0, 0.0, 2.0);
    CHECK(close(up[2], 2.0) && close(up[0], 0.0) && close(up[1], 0.0));
    bool threw = false;
    try {
        normalize_vector(make_vec3(0.0, 0.0, 0.0));
    } catch (const VectorError&) {
        threw = true;
    }
    CHECK(threw);
}

// A unit-cap sphere patch ``(u, v, sqrt(1 - u^2 - v^2))`` with a hand-built
// seed grid; used to exercise Newton intersection and inversion.
struct SphereCase {
    optics::native::SurfaceDesc desc;
    optics::native::SeedGrid grid;

    SphereCase() {
        using namespace optics::native;
        desc.model = SurfaceModel::compile("u", "v", "sqrt(1 - (u**2) - (v**2))", -1.0, 1.0, -1.0,
                                           1.0);
        desc.u_min = -1.0;
        desc.u_max = 1.0;
        desc.v_min = -1.0;
        desc.v_max = 1.0;
        desc.tu = DOMAIN_TOL * 2.0;
        desc.tv = DOMAIN_TOL * 2.0;
        desc.ap_r = -1.0;
        desc.rot = Mat3::identity();
        desc.trans = make_vec3(0.0, 0.0, 0.0);

        const int n = 7;
        std::vector<double> us(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            us[static_cast<std::size_t>(i)] = -0.9 + 1.8 * i / (n - 1);
        }
        for (int a = 0; a < n; ++a) {
            for (int b = 0; b < n; ++b) {
                const double u = us[static_cast<std::size_t>(a)];
                const double v = us[static_cast<std::size_t>(b)];
                const Vec3 p = desc.model->eval_local(u, v);
                grid.uf.push_back(u);
                grid.vf.push_back(v);
                grid.pts.push_back(p[0]);
                grid.pts.push_back(p[1]);
                grid.pts.push_back(p[2]);
                grid.finite.push_back(finite3(p) ? 1 : 0);
                if (finite3(p)) {
                    grid.uf_valid.push_back(u);
                    grid.vf_valid.push_back(v);
                    grid.p_world.push_back(p[0]);
                    grid.p_world.push_back(p[1]);
                    grid.p_world.push_back(p[2]);
                }
            }
        }
    }
};

void test_intersect_and_invert() {
    SphereCase sphere;
    const double o[3] = {0.45, 0.45, 5.0};
    const double d[3] = {0.0, 0.0, -1.0};
    const auto hit = optics::native::surface_intersect(sphere.desc, sphere.grid, o, d);
    CHECK(hit.has_value());
    if (hit.has_value()) {
        const double expected_t = 5.0 - std::sqrt(1.0 - 0.45 * 0.45 - 0.45 * 0.45);
        CHECK(close(hit->t, expected_t, 1e-7));
        CHECK(close(hit->u, 0.45, 1e-7));
        CHECK(close(hit->v, 0.45, 1e-7));
    }

    const auto uv = optics::native::parameters_at_xy(sphere.desc, sphere.grid, 0.3, -0.2);
    CHECK(uv.has_value());
    if (uv.has_value()) {
        CHECK(close(uv->first, 0.3, 1e-8));
        CHECK(close(uv->second, -0.2, 1e-8));
    }

    CHECK(!optics::native::desc_accepts(sphere.desc, 2.0, 0.0));
    CHECK(optics::native::desc_accepts(sphere.desc, 0.5, 0.5));

    // The convention is ``n = -cross(r_u, r_v)`` (as in ``Surface.normal``),
    // so the upper-cap sphere pole points at -z.
    const auto normal = optics::native::surface_normal(sphere.desc, 0.0, 0.0);
    CHECK(normal.has_value());
    if (normal.has_value()) {
        CHECK(close((*normal)[2], -1.0, 1e-6));
    }
}

void test_plane_and_mesh() {
    const double ro[3] = {0.0, 0.0, 2.0};
    const double d[3] = {0.0, 0.0, -1.0};
    const double o[3] = {0.0, 0.0, 0.0};
    const double n[3] = {0.0, 0.0, 1.0};
    const auto hit = optics::native::plane_intersect(ro, d, o, n, 1e-12);
    CHECK(hit.has_value());
    if (hit.has_value()) {
        CHECK(close(hit->at(0), 0.0) && close(hit->at(1), 0.0) && close(hit->at(2), 2.0));
    }

    const std::vector<double> tri = {
        -1.0, -1.0, 1.0,   //
        1.0, -1.0, 1.0,    //
        0.0, 1.0, 1.0,
    };
    const auto mesh = optics::native::mesh_intersect(ro, d, tri, 1e-14, 1e-12, 1e-9);
    CHECK(mesh.has_value());
    if (mesh.has_value()) {
        CHECK(close(mesh->t, 1.0));
    }
}

void test_direction() {
    using namespace optics::native;
    const Vec3 inc = make_vec3(0.0, 0.0, -1.0);
    const Vec3 n = make_vec3(0.0, 0.0, 1.0);
    // ``refract_or_reflect`` expects an oriented normal (dot(n, inc) >= 0).
    const auto refr = refract_or_reflect(inc, orient(n, inc), 1.0, 1.5);
    CHECK(refr.second);
    CHECK(close(refr.first[2], -1.0));  // normal-incidence direction preserved
    const auto tir = refract_or_reflect(make_vec3(0.8, 0.0, -0.6), make_vec3(0.0, 0.0, 1.0), 1.5,
                                        1.0);
    CHECK(!tir.second);  // beyond the critical angle: total internal reflection
    CHECK(close(norm(tir.first), 1.0));
}

}  // namespace

int main() {
    test_expression();
    test_rotation();
    test_intersect_and_invert();
    test_plane_and_mesh();
    test_direction();
    if (g_failures == 0) {
        std::printf("native tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "%d native test failures\n", g_failures);
    return 1;
}
