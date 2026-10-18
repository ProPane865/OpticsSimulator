// Thin GIL-releasing wrappers for the numerical kernels exposed to Python:
// Newton intersection/inversion, plane/mesh intersection and interface
// directions.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <tuple>
#include <vector>

#include "optics/geometry/surface.hpp"
#include "optics/intersection/grid.hpp"
#include "optics/intersection/mesh.hpp"
#include "optics/intersection/plane.hpp"
#include "optics/intersection/solver.hpp"
#include "optics/tracing/direction.hpp"

#include "register.hpp"
#include "support.hpp"

namespace {

using optics::native::SeedGrid;
using optics::native::SurfaceDesc;
using optics::native::Vec3;
using namespace optics::bindings;

std::tuple<double, double, double, bool> py_newton_intersection(const SurfaceDesc& desc,
                                                                const py::array_t<double>& o,
                                                                const py::array_t<double>& d,
                                                                double u, double v, double t) {
    const Vec3 ov = to_vec3(o, "origin");
    const Vec3 dv = to_vec3(d, "direction");
    const auto hit = [&] {
        const py::gil_scoped_release release;
        return optics::native::newton_intersection(desc, ov, dv, u, v, t);
    }();
    if (!hit.has_value()) {
        return {0.0, 0.0, 0.0, false};
    }
    return {hit->t, hit->u, hit->v, true};
}

std::tuple<double, double, double, bool> py_surface_intersect(const SurfaceDesc& desc,
                                                              const SeedGrid& grid,
                                                              const py::array_t<double>& o,
                                                              const py::array_t<double>& d) {
    const Vec3 ov = to_vec3(o, "origin");
    const Vec3 dv = to_vec3(d, "direction");
    const auto hit = [&] {
        const py::gil_scoped_release release;
        return optics::native::surface_intersect(desc, grid, ov, dv);
    }();
    if (!hit.has_value()) {
        return {0.0, 0.0, 0.0, false};
    }
    return {hit->t, hit->u, hit->v, true};
}

std::tuple<double, double, bool> py_solve_xy(const SurfaceDesc& desc, double x_target,
                                             double y_target, double u, double v) {
    const auto uv = [&] {
        const py::gil_scoped_release release;
        return optics::native::solve_xy(desc, x_target, y_target, u, v);
    }();
    if (!uv.has_value()) {
        return {0.0, 0.0, false};
    }
    return {uv->first, uv->second, true};
}

std::tuple<double, double, bool> py_parameters_at_xy(const SurfaceDesc& desc, const SeedGrid& grid,
                                                     double x, double y) {
    const auto uv = [&] {
        const py::gil_scoped_release release;
        return optics::native::parameters_at_xy(desc, grid, x, y);
    }();
    if (!uv.has_value()) {
        return {0.0, 0.0, false};
    }
    return {uv->first, uv->second, true};
}

std::tuple<double, double, double, bool> py_plane_intersect(const py::array_t<double>& ro,
                                                            const py::array_t<double>& d,
                                                            const py::array_t<double>& o,
                                                            const py::array_t<double>& n,
                                                            double eps) {
    const Vec3 rov = to_vec3(ro, "origin");
    const Vec3 dv = to_vec3(d, "direction");
    const Vec3 ov = to_vec3(o, "plane origin");
    const Vec3 nv = to_vec3(n, "plane normal");
    const auto hit = [&] {
        const py::gil_scoped_release release;
        return optics::native::plane_intersect(rov, dv, ov, nv, eps);
    }();
    if (!hit.has_value()) {
        return {0.0, 0.0, 0.0, false};
    }
    return {hit->at(0), hit->at(1), hit->at(2), true};
}

std::tuple<double, std::int64_t, double, double, bool> py_mesh_intersect(
    const py::array_t<double>& o, const py::array_t<double>& d, const py::array_t<double>& tri,
    double det_eps, double bary_tol, double min_face_t) {
    const py::buffer_info info = tri.request();
    require(info.ndim == 3 && info.shape[1] == 3 && info.shape[2] == 3,
            "tri: expected shape (K, 3, 3)");
    require(is_contiguous(info),
            "tri: array must be contiguous");
    const std::vector<double> tri_copy(static_cast<const double*>(info.ptr),
                                       static_cast<const double*>(info.ptr) + info.size);
    const Vec3 ov = to_vec3(o, "origin");
    const Vec3 dv = to_vec3(d, "direction");
    const auto hit = [&] {
        const py::gil_scoped_release release;
        return optics::native::mesh_intersect(ov, dv, tri_copy, det_eps, bary_tol, min_face_t);
    }();
    if (!hit.has_value()) {
        return {0.0, -1, 0.0, 0.0, false};
    }
    return {hit->t, static_cast<std::int64_t>(hit->face), hit->u, hit->v, true};
}

py::object py_orient(const py::array_t<double>& normal, const py::array_t<double>& incident) {
    return np3(optics::native::orient(to_vec3(normal, "normal"), to_vec3(incident, "incident")));
}

py::object py_refract_or_reflect(const py::array_t<double>& incident,
                                 const py::array_t<double>& normal, double n_from, double n_to) {
    const auto rv = optics::native::refract_or_reflect(to_vec3(incident, "incident"),
                                                       to_vec3(normal, "normal"), n_from, n_to);
    return py::make_tuple(np3(rv.first), rv.second);
}

std::tuple<double, double, bool> py_parameters_at(const SurfaceDesc& desc, const SeedGrid& grid,
                                                  const py::array_t<double>& point) {
    const Vec3 pv = to_vec3(point, "point");
    const auto uv = [&] {
        const py::gil_scoped_release release;
        return optics::native::parameters_at(desc, grid, pv);
    }();
    if (!uv.has_value()) {
        return {0.0, 0.0, false};
    }
    return {uv->first, uv->second, true};
}

py::array_t<double> py_build_rim(const SurfaceDesc& desc, const SeedGrid& grid, double radius,
                                 double cx, double cy, int n) {
    require(n >= 0, "n: expected a non-negative sample count");
    std::vector<double> pts;
    {
        const py::gil_scoped_release release;
        pts = optics::native::build_rim(desc, grid, radius, cx, cy, n);
    }
    const std::size_t rows = pts.size() / 3;
    py::array_t<double> out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(rows), 3},
                            std::vector<py::ssize_t>{24, 8});
    if (!pts.empty()) {
        std::memcpy(out.request().ptr, pts.data(), pts.size() * sizeof(double));
    }
    return out;
}

}  // namespace

namespace optics::bindings {

void register_kernels(py::module_& m) {
    m.def("newton_intersection", &py_newton_intersection, py::arg("desc"), py::arg("o"),
          py::arg("d"), py::arg("u"), py::arg("v"), py::arg("t"));
    m.def("surface_intersect", &py_surface_intersect, py::arg("desc"), py::arg("grid"), py::arg("o"),
          py::arg("d"));
    m.def("solve_xy", &py_solve_xy, py::arg("desc"), py::arg("x_target"), py::arg("y_target"),
          py::arg("u"), py::arg("v"));
    m.def("parameters_at_xy", &py_parameters_at_xy, py::arg("desc"), py::arg("grid"), py::arg("x"),
          py::arg("y"));
    m.def("parameters_at", &py_parameters_at, py::arg("desc"), py::arg("grid"), py::arg("point"));
    m.def("build_rim", &py_build_rim, py::arg("desc"), py::arg("grid"), py::arg("radius"),
          py::arg("cx"), py::arg("cy"), py::arg("n"));
    m.def("plane_intersect", &py_plane_intersect, py::arg("ro"), py::arg("d"), py::arg("o"),
          py::arg("n"), py::arg("eps"));
    m.def("mesh_intersect", &py_mesh_intersect, py::arg("o"), py::arg("d"), py::arg("tri"),
          py::arg("det_eps"), py::arg("bary_tol"), py::arg("min_face_t"));
    m.def("orient", &py_orient, py::arg("normal"), py::arg("incident"));
    m.def("refract_or_reflect", &py_refract_or_reflect, py::arg("incident"), py::arg("normal"),
          py::arg("n_from"), py::arg("n_to"));
}

}  // namespace optics::bindings
