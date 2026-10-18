// Stack-tracing entry points: single-ray segments, batched segment buffers
// and the trace-to-plane driver.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <tuple>

#include "optics/common/constants.hpp"
#include "optics/common/parallel.hpp"
#include "optics/common/vector.hpp"
#include "optics/tracing/trace.hpp"

#include "register.hpp"
#include "support.hpp"

namespace {

using optics::native::as_vec3;
using optics::native::PLANE_EPS;
using optics::native::TraceOptions;
using optics::native::TraceStack;
using namespace optics::bindings;

void py_trace_segments_batch(const TraceStack& stack, const py::array_t<double>& origins,
                             const py::array_t<double>& dirs, double ray_length,
                             const py::array_t<double>& starts, const py::array_t<double>& ends,
                             const py::array_t<std::int8_t>& kinds,
                             const py::array_t<std::int64_t>& counts) {
    const py::buffer_info o_info = origins.request();
    const py::buffer_info d_info = dirs.request();
    require(o_info.ndim == 2 && o_info.shape[1] == 3, "origins: expected shape (N, 3)");
    require(d_info.ndim == 2 && d_info.shape[1] == 3 && d_info.size == o_info.size,
            "dirs: expected shape (N, 3)");
    const std::size_t n = static_cast<std::size_t>(o_info.shape[0]);
    const py::buffer_info s_info = starts.request();
    const py::buffer_info e_info = ends.request();
    const py::buffer_info k_info = kinds.request();
    const py::buffer_info c_info = counts.request();
    require(s_info.ndim == 3 &&
                static_cast<std::size_t>(s_info.shape[0]) == n && s_info.shape[2] == 3,
            "starts: expected shape (N, cap, 3)");
    require(e_info.ndim == 3 && e_info.size == s_info.size, "ends: expected shape (N, cap, 3)");
    require(k_info.ndim == 2 && static_cast<std::size_t>(k_info.shape[0]) == n &&
                static_cast<std::size_t>(k_info.shape[1]) ==
                    static_cast<std::size_t>(s_info.shape[1]),
            "kinds: expected shape (N, cap)");
    require(c_info.ndim == 1 && static_cast<std::size_t>(c_info.size) == n,
            "counts: expected shape (N,)");
    require(!s_info.readonly && !e_info.readonly && !k_info.readonly && !c_info.readonly,
            "segment buffers must be writeable");
    const std::size_t cap = static_cast<std::size_t>(s_info.shape[1]);
    const double* od = static_cast<const double*>(o_info.ptr);
    const double* dd = static_cast<const double*>(d_info.ptr);
    double* sd = static_cast<double*>(s_info.ptr);
    double* ed = static_cast<double*>(e_info.ptr);
    signed char* kd = static_cast<signed char*>(k_info.ptr);
    std::int64_t* cd = static_cast<std::int64_t*>(c_info.ptr);
    const TraceOptions options{ray_length, PLANE_EPS};
    const py::gil_scoped_release release;
    optics::native::run_chunks(n, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i) {
            cd[i] = trace_stack_segments(stack, as_vec3(od + 3 * i), as_vec3(dd + 3 * i), options,
                                         sd + 3 * cap * i, ed + 3 * cap * i, kd + cap * i, cap)
                        .count;
        }
    });
}

py::object py_trace_stack_to_plane(const TraceStack& stack, const py::array_t<double>& o,
                                   const py::array_t<double>& d, const py::array_t<double>& po,
                                   const py::array_t<double>& pn, double eps, double u_min,
                                   double u_max, double v_min, double v_max,
                                   const py::array_t<double>& out_o,
                                   const py::array_t<double>& out_d) {
    const Vec3 ov = to_vec3(o, "origin");
    const Vec3 dv = to_vec3(d, "direction");
    const Vec3 pov = to_vec3(po, "plane origin");
    const Vec3 pnv = to_vec3(pn, "plane normal");
    double* o_out = write_doubles(out_o, "out_o");
    double* d_out = write_doubles(out_d, "out_d");
    require(out_o.request().size == 3 && out_d.request().size == 3, "out_o/out_d: expected (3,)");
    const optics::native::PlaneRegion region{u_min, u_max, v_min, v_max};
    const TraceOptions options{0.0, eps};
    optics::native::Vec3 final_o;
    optics::native::Vec3 final_d;
    const bool ok = [&] {
        const py::gil_scoped_release release;
        return trace_stack_to_plane(stack, ov, dv, pov, pnv, region, options, final_o, final_d);
    }();
    for (int i = 0; i < 3; ++i) {
        o_out[i] = final_o[i];
        d_out[i] = final_d[i];
    }
    return py::bool_(ok);
}

}  // namespace

namespace optics::bindings {

void register_tracing(py::module_& m) {
    m.def(
        "trace_stack_segments",
        [](const TraceStack& stack, const py::array_t<double>& o, const py::array_t<double>& d,
           double ray_length, const py::array_t<double>& starts, const py::array_t<double>& ends,
           const py::array_t<std::int8_t>& kinds) {
            const py::buffer_info s_info = starts.request();
            const py::buffer_info e_info = ends.request();
            const py::buffer_info k_info = kinds.request();
            require(s_info.ndim == 2 && s_info.shape[1] == 3, "starts: expected shape (cap, 3)");
            require(e_info.ndim == 2 && e_info.shape[1] == 3 && e_info.size == s_info.size,
                    "ends: expected shape (cap, 3)");
            require(k_info.ndim == 1 && static_cast<std::size_t>(k_info.size) ==
                                            static_cast<std::size_t>(s_info.shape[0]),
                    "kinds: expected shape (cap,)");
            require(!s_info.readonly && !e_info.readonly && !k_info.readonly,
                    "segment buffers must be writeable");
            const Vec3 ov = to_vec3(o, "origin");
            const Vec3 dv = to_vec3(d, "direction");
            const TraceOptions options{ray_length, PLANE_EPS};
            optics::native::StackSegments res;
            {
                const py::gil_scoped_release release;
                res = trace_stack_segments(
                    stack, ov, dv, options, static_cast<double*>(s_info.ptr),
                    static_cast<double*>(e_info.ptr), static_cast<signed char*>(k_info.ptr),
                    static_cast<std::size_t>(s_info.shape[0]));
            }
            return py::make_tuple(res.count, res.exited, np3(res.final_o), np3(res.final_d));
        },
        py::arg("stack"), py::arg("o"), py::arg("d"), py::arg("ray_length"), py::arg("starts"),
        py::arg("ends"), py::arg("kinds"));
    m.def("trace_segments_batch", &py_trace_segments_batch, py::arg("stack"), py::arg("origins"),
          py::arg("dirs"), py::arg("ray_length"), py::arg("starts"), py::arg("ends"),
          py::arg("kinds"), py::arg("counts"));
    m.def("trace_stack_to_plane", &py_trace_stack_to_plane, py::arg("stack"), py::arg("o"),
          py::arg("d"), py::arg("po"), py::arg("pn"), py::arg("eps"), py::arg("u_min"),
          py::arg("u_max"), py::arg("v_min"), py::arg("v_max"), py::arg("out_o"),
          py::arg("out_d"));
}

}  // namespace optics::bindings
