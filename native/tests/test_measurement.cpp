// Measurement kernel suite: detector accumulation, direct/stacked plane
// intersection, the shared-walk multi-plane pass matching independent
// per-plane traces, the fused rendering + planes kernel matching the direct
// walk, and absorbed (trapped) rays leaving buffers untouched.
#include <cmath>
#include <cstdint>
#include <cstring>
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
using optics::native::measure_stack;
using optics::native::MeasurementOutputView;
using optics::native::PlaneBatchView;
using optics::native::PlaneRegion;
using optics::native::RayBatchView;
using optics::native::SegmentOutputView;
using optics::native::trace_segments_and_planes;
using optics::native::trace_stack_segments;
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

const optics::native::PlaneRegion kWideRegion{-10.0, 10.0, -10.0, 10.0};

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

void test_measure_stack() {
    const TraceStack stack = fixture::make_stack({plate_at(0.0)});
    const double origins[9] = {0.0, 0.0, 3.0, -0.2, 0.0, 2.0, 5.0, 0.0, 3.0};
    const double dirs[9] = {0.0, 0.0, -1.0, 0.3, 0.0, -kCos1, 0.0, 0.0, -1.0};
    const double powers[3] = {1.0, 0.25, 2.0};
    const Vec3 po = make_vec3(0.0, 0.0, -2.0);
    const Vec3 pn = make_vec3(0.0, 0.0, 1.0);
    double us[3], vs[3], pws[3];
    for (int i = 0; i < 3; ++i) {
        us[i] = vs[i] = pws[i] = kSentinel;
    }
    const RayBatchView rays{origins, dirs, powers, 3};
    const MeasurementOutputView out{us, vs, pws};
    measure_stack(stack, rays, po, pn, optics::native::PLANE_EPS, kWideRegion, out);
    CHECK(close(us[0], 0.0, 1e-9) && close(vs[0], 0.0, 1e-12) && close(pws[0], 1.0));
    // Oblique ray: the refracted lateral shift is included in the crossing.
    const double x_entry = -0.2 + 0.3 * (1.5 / kCos1);
    const double x_exit = x_entry + (0.2 / std::sqrt(1.0 - 0.04)) * 1.0;
    const double u_expect = x_exit + 0.3 * (1.5 / kCos1);
    CHECK(close(us[1], u_expect, 1e-8));
    CHECK(close(pws[1], 0.25));
    // A ray that misses the stack still hits the plane on the straight path.
    CHECK(close(us[2], 5.0, 1e-12) && close(pws[2], 2.0));

    // A trapped (total-internal-reflection) ray never reaches either plane.
    const TraceStack slab = fixture::make_stack({
        fixture::make_plate_ptr(1.5, 12.0, -0.5, 0.5, 0.0, 256, false)});
    const double s = 1.0 / std::sqrt(2.0);
    const double trapped_o[3] = {0.0, 0.0, 0.0};
    const double trapped_d[3] = {s, 0.0, s};
    const double trapped_p[1] = {1.0};
    for (double plane_z : {3.0, -3.0}) {
        double tu = kSentinel;
        double tv = kSentinel;
        double tp = kSentinel;
        const Vec3 po_above = make_vec3(0.0, 0.0, plane_z);
        const RayBatchView rays{trapped_o, trapped_d, trapped_p, 1};
        const MeasurementOutputView out{&tu, &tv, &tp};
        measure_stack(slab, rays, po_above, pn, optics::native::PLANE_EPS, kWideRegion, out);
        CHECK(tu == kSentinel && tv == kSentinel && tp == kSentinel);
    }
}

void test_planes_multi_vs_single() {
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

    // The shared-walk results match independent per-plane ``measure_stack``
    // calls, and the analytic crossings.
    for (std::size_t q = 0; q < p; ++q) {
        double su[2], sv[2], sp[2];
        for (std::size_t i = 0; i < n; ++i) {
            su[i] = sv[i] = sp[i] = kSentinel;
        }
        const Vec3 po_q = make_vec3(po[3 * q], po[3 * q + 1], po[3 * q + 2]);
        const Vec3 pn_q = make_vec3(pn[3 * q], pn[3 * q + 1], pn[3 * q + 2]);
        const PlaneRegion region{u_min[q], u_max[q], v_min[q], v_max[q]};
        const MeasurementOutputView single_out{su, sv, sp};
        measure_stack(stack, rays, po_q, pn_q, optics::native::PLANE_EPS, region, single_out);
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t idx = q * n + i;
            const bool multi_hit = std::isfinite(us[idx]);
            const bool single_hit = su[i] != kSentinel;
            CHECK(multi_hit == single_hit);
            if (multi_hit && single_hit) {
                CHECK(close(us[idx], su[i], 1e-12));
                CHECK(close(vs[idx], sv[i], 1e-12));
                CHECK(close(pws[idx], sp[i], 1e-15));
                CHECK(close(pws[idx], powers[i], 1e-15));
            }
        }
    }
    // Plane q0 (z = 2) is crossed on the free path; plane q1 (z = -3) behind
    // the plate carries the refracted shift.
    CHECK(close(us[0 * n + 0], 0.0, 1e-12));
    const double free_u = -0.4 + 0.3 * (2.0 / kCos1);
    CHECK(close(us[0 * n + 1], free_u, 1e-9));
    const double x_entry = -0.4 + 0.3 * (3.5 / kCos1);
    const double x_exit = x_entry + (0.2 / std::sqrt(1.0 - 0.04)) * 1.0;
    const double back_u = x_exit + 0.3 * (2.5 / kCos1);
    CHECK(close(us[1 * n + 1], back_u, 1e-8));
}

void test_fused_matches_separate_paths() {
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

    // Rendering buffers are byte-identical to the direct per-ray walk.
    for (std::size_t i = 0; i < n; ++i) {
        const Vec3 o = make_vec3(origins[3 * i], origins[3 * i + 1], origins[3 * i + 2]);
        const Vec3 d = make_vec3(dirs[3 * i], dirs[3 * i + 1], dirs[3 * i + 2]);
        double s1[3 * kCap];
        double e1[3 * kCap];
        signed char k1[kCap];
        const optics::native::StackSegments seg =
            trace_stack_segments(stack, o, d, options, s1, e1, k1, kCap);
        CHECK(f_counts[i] == seg.count);
        const std::size_t c = static_cast<std::size_t>(seg.count);
        CHECK(std::memcmp(s1, f_starts.data() + 3 * kCap * i, 3 * c * sizeof(double)) == 0);
        CHECK(std::memcmp(e1, f_ends.data() + 3 * kCap * i, 3 * c * sizeof(double)) == 0);
        CHECK(std::memcmp(k1, f_kinds.data() + kCap * i, c) == 0);
    }
    // Both rays traverse both plates: 5 segments each.
    CHECK(f_counts[0] == 5);
    CHECK(f_counts[1] == 5);

    // The plane pass matches the independent multi-plane kernel exactly.
    std::vector<double> m_us(p * n, kNan), m_vs(p * n, kNan), m_pws(p * n, 0.0);
    const MeasurementOutputView multi_out{m_us.data(), m_vs.data(), m_pws.data()};
    measure_planes_multi(stack, rays, planes, optics::native::PLANE_EPS, multi_out);
    CHECK(std::memcmp(f_us.data(), m_us.data(), p * n * sizeof(double)) == 0);
    CHECK(std::memcmp(f_vs.data(), m_vs.data(), p * n * sizeof(double)) == 0);
    CHECK(std::memcmp(f_pws.data(), m_pws.data(), p * n * sizeof(double)) == 0);

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
    test_measure_stack();
    test_planes_multi_vs_single();
    test_fused_matches_separate_paths();
}
