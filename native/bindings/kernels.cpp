// Thin GIL-releasing wrappers for the numerical kernels exposed to Python:
// plane and mesh intersection plus the interface-direction helpers. The
// ray/surface Newton kernels run through the persistent ``NativeSurface``
// object (see ``builders.cpp``) and the element/stack kernels.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <tuple>
#include <vector>

#include "optics/intersection/mesh.hpp"
#include "optics/intersection/plane.hpp"
#include "optics/tracing/direction.hpp"

#include "register.hpp"
#include "support.hpp"

namespace {

using optics::native::Vec3;
using namespace optics::bindings;

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

}  // namespace

namespace optics::bindings {

void register_kernels(py::module_& m) {
    m.def("plane_intersect", &py_plane_intersect, py::arg("ro"), py::arg("d"), py::arg("o"),
          py::arg("n"), py::arg("eps"));
    m.def("mesh_intersect", &py_mesh_intersect, py::arg("o"), py::arg("d"), py::arg("tri"),
          py::arg("det_eps"), py::arg("bary_tol"), py::arg("min_face_t"));
    m.def("orient", &py_orient, py::arg("normal"), py::arg("incident"));
    m.def("refract_or_reflect", &py_refract_or_reflect, py::arg("incident"), py::arg("normal"),
          py::arg("n_from"), py::arg("n_to"));
}

}  // namespace optics::bindings
