// Measurement kernel suite: detector accumulation, direct plane intersection,
// the shared-walk multi-plane pass (analytic crossings, stack misses and
// trapped rays) and the fused rendering + planes kernel.
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/vector.hpp"
#include "optics/measurement/measurement.hpp"
#include "optics/tracing/trace.hpp"

#include "check.hpp"
#include "test_element.hpp"

namespace {

using optics::native::accumulate_power;
using optics::native::make_vec3;
using optics::native::measure_direct;
using optics::native::measure_planes_multi;
using optics::native::MeasurementOutputView;
using optics::native::PlaneBatchView;
using optics::native::RayBatchView;
using optics::native::SegmentOutputView;
using optics::native::trace_segments_and_planes;
using optics::native::TraceOptions;
using optics::native::TraceStack;
using optics::native::Vec3;

constexpr double kCos1 = 0.9539392014169457;  // sqrt(1 - 0.3^2)
constexpr double kCos2 = 0.99498743710662;    // sqrt(1 - 0.1^2)
constexpr double kSentinel = -777.0;
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr std::size_t kCap = 16;

fixture::ElementRef plate_at(double z) {
    return fixture::make_plate_ptr(1.5, 1.0, -0.5, 0.5, z, 48, true);
}

void test_accumulate() {
    double grid[4] = {0.0, 0.0, 0.0, 0.0};
    const double u[3] = {0.5, -0.5, 10.0};  // the last one is out of extent
    const double v[3] = {0.5, 0.5, 0.0};
    const double pw[3] = {1.0, 2.0, 4.0};
    const double total =
        accumulate_power(grid, u, v, pw, 3, -1.0, 1.0, -1.0, 1.0, 2, 2, 2.0, 2.0);
    CHECK(close(total, 3.0));
    CHECK(close(grid[3], 1.0));  // row-major (j, i) = (1, 1)
    CHECK(close(grid[2], 2.0));  // (1, 0)
    CHECK(close(grid[0], 0.0) && close(grid[1], 0.0));

    // Non-finite power is skipped entirely.
    const double nan_pw[1] = {kNan};
    CHECK(close(accumulate_power(grid, u, v, nan_pw, 1, -1.0, 1.0, -1.0, 1.0, 2, 2, 2.0, 2.0),
                0.0));
    CHECK(close(grid[3], 1.0));
}

void test_measure_direct() {
    const double origins[9] = {0.0, 0.0, 2.0, 5.0, 0.0, 1.0, 5.0, 1.0, 1.0};
    const double dirs[9] = {0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0};
    const double powers[3] = {1.0, 0.25, 8.0};
    const Vec3 o = make_vec3(0.0, 0.0, 0.0);
    const Vec3 nrm = make_vec3(0.0, 0.0, 1.0);
    double us[3], vs[3], pws[3];
    for (int i = 0; i < 3; ++i) {
        us[i] = vs[i] = pws[i] = kSentinel;
    }
    const RayBatchView rays{origins, dirs, powers, 3};
    const MeasurementOutputView out{us, vs, pws};
    measure_direct(rays, o, nrm, optics::native::PLANE_EPS, out);
    CHECK(close(us[0], 0.0) && close(vs[0], 0.0) && close(pws[0], 1.0));
    CHECK(close(us[1], 5.0) && close(pws[1], 0.25));
    // The parallel ray misses and leaves its entries untouched.
    CHECK(us[2] == kSentinel && vs[2] == kSentinel && pws[2] == kSentinel);
}

void test_planes_multi() {
    const TraceStack stack = fixture::make_stack({plate_at(0.0)});
    const double origins[6] = {0.0, 0.0, 4.0, -0.4, 0.0, 4.0};
    const double dirs[6] = {0.0, 0.0, -1.0, 0.3, 0.0, -kCos1};
    const double powers[2] = {1.0, 0.5};
    const double po[6] = {0.0, 0.0, 2.0, 0.0, 0.0, -3.0};
    const double pn[6] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    const double u_min[2] = {-10.0, -10.0};
    const double u_max[2] = {10.0, 10.0};
    const double v_min[2] = {-10.0, -10.0};
    const double v_max[2] = {10.0, 10.0};
    const std::size_t n = 2;
    const std::size_t p = 2;
    std::vector<double> us(p * n, kNan), vs(p * n, kNan), pws(p * n, 0.0);
    const RayBatchView rays{origins, dirs, powers, n};
    const PlaneBatchView planes{po, pn, u_min, u_max, v_min, v_max, p};
    const MeasurementOutputView out{us.data(), vs.data(), pws.data()};
    measure_planes_multi(stack, rays, planes, optics::native::PLANE_EPS, out);

    // Plane q0 (z = 2) is crossed on the free path; plane q1 (z = -3) behind
    // the plate carries the refracted shift. A ray that misses the stack
    // still hits both planes on the straight path.
    CHECK(close(us[0 * n + 0], 0.0, 1e-12) && close(pws[0 * n + 0], 1.0, 1e-15));
    CHECK(std::isfinite(us[0 * n + 1]) && close(pws[0 * n + 1], 0.5, 1e-15));
    const double free_u = -0.4 + 0.3 * (2.0 / kCos1);
    CHECK(close(us[0 * n + 1], free_u, 1e-9));
    const double x_entry = -0.4 + 0.3 * (3.5 / kCos1);
    const double x_exit = x_entry + (0.2 / std::sqrt(1.0 - 0.04)) * 1.0;
    const double back_u = x_exit + 0.3 * (2.5 / kCos1);
    CHECK(close(us[1 * n + 1], back_u, 1e-8));
    CHECK(close(us[1 * n + 0], 0.0, 1e-12));
    CHECK(std::isfinite(us[0]) && std::isfinite(us[1]) && std::isfinite(us[2]) &&
          std::isfinite(us[3]));

    // A trapped (total-internal-reflection) ray never reaches the planes:
    // every output entry stays untouched.
    const TraceStack slab = fixture::make_stack({
        fixture::make_plate_ptr(1.5, 12.0, -0.5, 0.5, 0.0, 256, false)});
    const double s = 1.0 / std::sqrt(2.0);
    const double trapped_o[3] = {0.0, 0.0, 0.0};
    const double trapped_d[3] = {s, 0.0, s};
    const double trapped_p[1] = {1.0};
    const double r_min[2] = {-10.0, -10.0};
    const double r_max[2] = {10.0, 10.0};
    for (double plane_z : {3.0, -3.0}) {
        const double po_t[3] = {0.0, 0.0, plane_z};
        const double pn_t[3] = {0.0, 0.0, 1.0};
        double tu = kSentinel;
        double tv = kSentinel;
        double tp = kSentinel;
        const RayBatchView rays_t{trapped_o, trapped_d, trapped_p, 1};
        const PlaneBatchView planes_t{po_t, pn_t, r_min, r_max, r_min, r_max, 1};
        const MeasurementOutputView out_t{&tu, &tv, &tp};
        measure_planes_multi(slab, rays_t, planes_t, optics::native::PLANE_EPS, out_t);
        CHECK(tu == kSentinel && tv == kSentinel && tp == kSentinel);
    }
}

void test_fused_kernel() {
    const TraceStack stack = fixture::make_stack({plate_at(3.0), plate_at(0.0)});
    const double origins[6] = {0.0, 0.0, 6.0, -0.2, 0.0, 6.0};
    const double dirs[6] = {0.0, 0.0, -1.0, 0.1, 0.0, -kCos2};
    const double powers[2] = {1.0, 0.5};
    const double po[3] = {0.0, 0.0, -3.0};
    const double pn[3] = {0.0, 0.0, 1.0};
    const double range_min[1] = {-10.0};
    const double range_max[1] = {10.0};
    const std::size_t n = 2;
    const std::size_t p = 1;

    std::vector<double> f_starts(n * kCap * 3), f_ends(n * kCap * 3);
    std::vector<signed char> f_kinds(n * kCap);
    std::vector<std::int64_t> f_counts(n, 0);
    std::vector<double> f_us(p * n, kNan), f_vs(p * n, kNan), f_pws(p * n, 0.0);
    const RayBatchView rays{origins, dirs, powers, n};
    const PlaneBatchView planes{po, pn, range_min, range_max, range_min, range_max, p};
    const TraceOptions options{2.0, optics::native::PLANE_EPS};
    const SegmentOutputView segs{f_starts.data(), f_ends.data(), f_kinds.data(), f_counts.data(),
                                 kCap};
    const MeasurementOutputView measurement{f_us.data(), f_vs.data(), f_pws.data()};
    trace_segments_and_planes(stack, rays, options, planes, segs, measurement);

    // Both rays traverse both plates: 5 segments each, incident-first and
    // exit-last kinds with continuous endpoints.
    CHECK(f_counts[0] == 5);
    CHECK(f_counts[1] == 5);
    for (std::size_t i = 0; i < n; ++i) {
        const std::int64_t c = f_counts[i];
        CHECK(f_kinds[kCap * i] == 0);
        CHECK(f_kinds[kCap * i + 1] == 1);
        CHECK(f_kinds[kCap * i + c - 1] == 2);
        for (std::int64_t k = 0; k + 1 < c; ++k) {
            for (int cc = 0; cc < 3; ++cc) {
                CHECK(close(f_ends[3 * (kCap * i + k) + cc],
                            f_starts[3 * (kCap * i + k + 1) + cc], 1e-12));
            }
        }
    }
    // Straight ray through both parallel plates stays on axis.
    CHECK(close(f_us[0], 0.0, 1e-12) && close(f_pws[0], 1.0, 1e-15));
    // Oblique ray: free-path z drop totals 2.5 + 2 + 2.5 across the gaps,
    // plus one lateral drift of ``tan(theta_in)`` inside each plate.
    const double tan_in = (0.1 / 1.5) / std::sqrt(1.0 - (0.1 / 1.5) * (0.1 / 1.5));
    const double u_oblique = -0.2 + 0.1 * (7.0 / kCos2) + 2.0 * tan_in;
    CHECK(close(f_us[1], u_oblique, 1e-8));
    CHECK(close(f_pws[1], 0.5, 1e-15));
}

}  // namespace

void test_measurement() {
    test_accumulate();
    test_measure_direct();
    test_planes_multi();
    test_fused_kernel();
}
