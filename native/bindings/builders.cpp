// Surface/descriptor/grid/element/stack builders and the opaque handle
// classes exposed to Python.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/grid.hpp"
#include "optics/intersection/solver.hpp"
#include "optics/tracing/context.hpp"

#include "register.hpp"
#include "support.hpp"

namespace {

using optics::native::Element;
using optics::native::Mat3;
using optics::native::SeedGrid;
using optics::native::Surface;
using optics::native::SurfaceDesc;
using optics::native::SurfaceModel;
using optics::native::TraceStack;
using optics::native::Vec3;
using namespace optics::bindings;

std::shared_ptr<SurfaceModel> make_model(const std::string& x_src, const std::string& y_src,
                                         const std::string& z_src, double u_min, double u_max,
                                         double v_min, double v_max) {
    // ``compile`` hands back a const model; the freshly built object has no
    // other owners and the exposed surface is read-only.
    return std::const_pointer_cast<SurfaceModel>(
        SurfaceModel::compile(x_src, y_src, z_src, u_min, u_max, v_min, v_max));
}

std::shared_ptr<Surface> make_surface(const std::shared_ptr<SurfaceModel>& model,
                                      const std::string& x_src, const std::string& y_src,
                                      const std::string& z_src, double u_min, double u_max,
                                      double v_min, double v_max, double ap_r, double ap_cx,
                                      double ap_cy, const py::array_t<double>& rotation,
                                      const py::array_t<double>& translation, int grid_n) {
    const py::buffer_info r_info = rotation.request();
    require_rows(r_info, 2, 3, nullptr, "rotation");
    require(r_info.size == 9, "rotation: expected shape (3, 3)");
    require(is_contiguous(r_info),
            "rotation: array must be contiguous");
    const ConstDoubles t = read_doubles(translation, 1, "translation");
    Surface::Options o;
    o.x_src = x_src;
    o.y_src = y_src;
    o.z_src = z_src;
    o.u_min = u_min;
    o.u_max = u_max;
    o.v_min = v_min;
    o.v_max = v_max;
    o.aperture_radius = ap_r;
    o.aperture_cx = ap_cx;
    o.aperture_cy = ap_cy;
    const double* r = static_cast<const double*>(r_info.ptr);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            o.rot.m[i][j] = r[3 * i + j];
        }
    }
    o.trans[0] = t.data[0];
    o.trans[1] = t.data[1];
    o.trans[2] = t.data[2];
    o.grid_n = grid_n;
    // Descriptor, seed grid and the graph probe are built inside the
    // constructor (pure C++ work, GIL released).
    const std::shared_ptr<const Surface> surface = [&] {
        const py::gil_scoped_release release;
        return Surface::create_with_model(model, std::move(o));
    }();
    return std::const_pointer_cast<Surface>(surface);
}

// Persistent element from two native surface handles (AABB + graph flags are
// derived once inside ``Element::create``).
std::shared_ptr<Element> build_element(const std::shared_ptr<Surface>& surface1,
                                       const std::shared_ptr<Surface>& surface2, double index,
                                       const py::array_t<double>& tri,
                                       const py::array_t<double>& normals) {
    const py::buffer_info tri_info = tri.request();
    require(tri_info.ndim == 3 && tri_info.shape[1] == 3 && tri_info.shape[2] == 3,
            "tri: expected shape (K, 3, 3)");
    require(is_contiguous(tri_info), "tri: array must be contiguous");
    const std::size_t k = static_cast<std::size_t>(tri_info.shape[0]);
    const py::buffer_info nm_info = normals.request();
    require(nm_info.ndim == 2 && nm_info.shape[1] == 3 &&
                static_cast<std::size_t>(nm_info.shape[0]) == k,
            "normals: expected shape (K, 3)");
    require(is_contiguous(nm_info), "normals: array must be contiguous");
    std::vector<double> tri_copy(static_cast<const double*>(tri_info.ptr),
                                 static_cast<const double*>(tri_info.ptr) + tri_info.size);
    std::vector<double> nm_copy(static_cast<const double*>(nm_info.ptr),
                                static_cast<const double*>(nm_info.ptr) + nm_info.size);
    const std::shared_ptr<const Element> element = [&] {
        const py::gil_scoped_release release;
        return Element::create(surface1, surface2, index, std::move(tri_copy), std::move(nm_copy));
    }();
    return std::const_pointer_cast<Element>(element);
}

std::shared_ptr<TraceStack> build_trace_stack(
    const std::vector<std::shared_ptr<Element>>& contexts) {
    auto stack = std::make_shared<TraceStack>();
    stack->elements.reserve(contexts.size());
    for (const auto& ctx : contexts) {
        require(ctx != nullptr, "element context must not be None");
        stack->elements.push_back(ctx);
    }
    return stack;
}

}  // namespace

namespace optics::bindings {

void register_builders(py::module_& m) {
    py::class_<SurfaceModel, std::shared_ptr<SurfaceModel>>(m, "SurfaceModel")
        .def(
            "eval_local",
            [](const SurfaceModel& self, double u, double v) { return np3(self.eval_local(u, v)); },
            py::arg("u"), py::arg("v"))
        .def(
            "eval_grid",
            [](const SurfaceModel& self, const py::array_t<double>& us,
               const py::array_t<double>& vs) {
                const ConstDoubles u = read_doubles(us, 1, "us");
                const ConstDoubles v = read_doubles(vs, 1, "vs");
                require(u.size == v.size, "us/vs sizes must match");
                py::array_t<double> out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(u.size), 3},
                                        std::vector<py::ssize_t>{24, 8});
                const py::buffer_info res = out.request();
                self.eval_grid(u.data, v.data, u.size, static_cast<double*>(res.ptr));
                return out;
            },
            py::arg("us"), py::arg("vs"))
        .def(
            "derivatives_local",
            [](const SurfaceModel& self, double u, double v) {
                const auto rv = self.derivatives(u, v);
                return py::make_tuple(np3(rv.first), np3(rv.second));
            },
            py::arg("u"), py::arg("v"))
        .def_property_readonly("hu", &SurfaceModel::hu)
        .def_property_readonly("hv", &SurfaceModel::hv)
        .def_property_readonly("u_min", &SurfaceModel::u_min)
        .def_property_readonly("u_max", &SurfaceModel::u_max)
        .def_property_readonly("v_min", &SurfaceModel::v_min)
        .def_property_readonly("v_max", &SurfaceModel::v_max);

    py::class_<SurfaceDesc, std::shared_ptr<SurfaceDesc>>(m, "SurfaceDesc")
        .def("eval",
             [](const SurfaceDesc& self, double u, double v) {
                 return np3(optics::native::desc_eval(self, u, v));
             },
             py::arg("u"), py::arg("v"))
        .def("derivatives",
             [](const SurfaceDesc& self, double u, double v) {
                 const auto rv = optics::native::desc_derivatives(self, u, v);
                 return py::make_tuple(np3(rv.first), np3(rv.second));
             },
             py::arg("u"), py::arg("v"))
        .def(
            "normal",
            [](const SurfaceDesc& self, double u, double v) {
                const auto nv = optics::native::surface_normal(self, u, v);
                if (!nv.has_value()) {
                    return py::object(py::none());
                }
                return py::object(np3(*nv));
            },
            py::arg("u"), py::arg("v"))
        .def("accepts", &optics::native::desc_accepts, py::arg("u"), py::arg("v"));

    py::class_<SeedGrid, std::shared_ptr<SeedGrid>>(m, "SeedGrid");
    py::class_<Surface, std::shared_ptr<Surface>>(m, "NativeSurface")
        .def("desc", [](const Surface& self) { return self.desc_ptr(); })
        .def("grid", [](const Surface& self) { return self.grid_ptr(); })
        .def("is_graph_over_xy", &Surface::is_graph_over_xy);
    py::class_<Element, std::shared_ptr<Element>>(m, "Element");
    py::class_<TraceStack, std::shared_ptr<TraceStack>>(m, "TraceStack")
        .def("__len__", [](const TraceStack& self) { return self.size(); });

    m.def("compile_surface", &make_model, py::arg("x"), py::arg("y"), py::arg("z"),
          py::arg("u_min"), py::arg("u_max"), py::arg("v_min"), py::arg("v_max"));
    m.def("make_surface", &make_surface, py::arg("model"), py::arg("x"), py::arg("y"), py::arg("z"),
          py::arg("u_min"), py::arg("u_max"), py::arg("v_min"), py::arg("v_max"), py::arg("ap_r"),
          py::arg("ap_cx"), py::arg("ap_cy"), py::arg("rotation"), py::arg("translation"),
          py::arg("grid_n") = optics::native::SEED_GRID_N);
    m.def(
        "build_seed_grid",
        [](const SurfaceDesc& desc, int grid_n) {
            auto grid = std::make_shared<SeedGrid>();
            {
                const py::gil_scoped_release release;
                *grid = optics::native::build_seed_grid(desc, grid_n);
            }
            return grid;
        },
        py::arg("desc"), py::arg("grid_n"));
    m.def("build_element", &build_element, py::arg("surface1"), py::arg("surface2"),
          py::arg("refractive_index"), py::arg("tri"), py::arg("normals"));
    m.def("build_trace_stack", &build_trace_stack, py::arg("contexts"));
}

}  // namespace optics::bindings
