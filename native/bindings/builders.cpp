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

using optics::native::ElementContext;
using optics::native::Mat3;
using optics::native::SeedGrid;
using optics::native::SurfaceDesc;
using optics::native::SurfaceModel;
using optics::native::TraceStack;
using optics::native::Vec3;
using namespace optics::bindings;

// ``numpy.linspace(a, b, n)`` (endpoint included; a single sample is ``[a]``).
void linspace(double a, double b, int n, std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(n < 0 ? 0 : n), 0.0);
    if (n <= 0) {
        return;
    }
    if (n == 1) {
        out[0] = a;
        return;
    }
    const double step = (b - a) / (n - 1);
    for (int i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = a + step * i;
    }
    out[static_cast<std::size_t>(n - 1)] = b;
}

// Axis-aligned bounds of both surfaces sampled over their parameter domain in
// the local frame (mirror of ``RefractiveElement.local_bounds``); ``+inf``/
// ``-inf`` when no finite sample exists.
void local_bounds(const SurfaceDesc& d1, const SurfaceDesc& d2, Vec3& lo, Vec3& hi) {
    const SurfaceDesc* descs[2] = {&d1, &d2};
    for (const SurfaceDesc* d : descs) {
        std::vector<double> us;
        std::vector<double> vs;
        linspace(d->u_min, d->u_max, optics::native::CONTAINMENT_BOUNDS_N, us);
        linspace(d->v_min, d->v_max, optics::native::CONTAINMENT_BOUNDS_N, vs);
        const int n = optics::native::CONTAINMENT_BOUNDS_N;
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                const Vec3 p = d->model->eval_local(us[static_cast<std::size_t>(c)],
                                                    vs[static_cast<std::size_t>(r)]);
                if (!optics::native::finite3(p)) {
                    continue;
                }
                for (int j = 0; j < 3; ++j) {
                    lo[j] = std::min(lo[j], p[j]);
                    hi[j] = std::max(hi[j], p[j]);
                }
            }
        }
    }
}

std::shared_ptr<SurfaceModel> make_model(const std::string& x_src, const std::string& y_src,
                                         const std::string& z_src, double u_min, double u_max,
                                         double v_min, double v_max) {
    // ``compile`` hands back a const model; the freshly built object has no
    // other owners and the exposed surface is read-only.
    return std::const_pointer_cast<SurfaceModel>(
        SurfaceModel::compile(x_src, y_src, z_src, u_min, u_max, v_min, v_max));
}

std::shared_ptr<SurfaceDesc> build_surface_desc(const std::shared_ptr<SurfaceModel>& model,
                                                double u_min, double u_max, double v_min,
                                                double v_max, double tu, double tv, double ap_r,
                                                double ap_cx, double ap_cy,
                                                const py::array_t<double>& rotation,
                                                const py::array_t<double>& translation) {
    const py::buffer_info r_info = rotation.request();
    require_rows(r_info, 2, 3, nullptr, "rotation");
    require(r_info.size == 9, "rotation: expected shape (3, 3)");
    require(is_contiguous(r_info),
            "rotation: array must be contiguous");
    const ConstDoubles t = read_doubles(translation, 1, "translation");
    auto desc = std::make_shared<SurfaceDesc>();
    desc->model = model;
    desc->u_min = u_min;
    desc->u_max = u_max;
    desc->v_min = v_min;
    desc->v_max = v_max;
    desc->tu = tu;
    desc->tv = tv;
    desc->ap_r = ap_r;
    desc->ap_cx = ap_cx;
    desc->ap_cy = ap_cy;
    const double* r = static_cast<const double*>(r_info.ptr);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            desc->rot.m[i][j] = r[3 * i + j];
        }
    }
    desc->trans[0] = t.data[0];
    desc->trans[1] = t.data[1];
    desc->trans[2] = t.data[2];
    return desc;
}

std::shared_ptr<SeedGrid> build_grid(const py::array_t<double>& p_world,
                                     const py::array_t<double>& uf_valid,
                                     const py::array_t<double>& vf_valid,
                                     const py::array_t<double>& pts,
                                     const py::array_t<bool>& finite,
                                     const py::array_t<double>& uf, const py::array_t<double>& vf) {
    const py::buffer_info pw_info = p_world.request();
    std::size_t m = 0;
    require_rows(pw_info, 2, 3, &m, "p_world");
    require(is_contiguous(pw_info),
            "p_world: array must be contiguous");
    const ConstDoubles ufv = read_doubles(uf_valid, 1, "uf_valid");
    const ConstDoubles vfv = read_doubles(vf_valid, 1, "vf_valid");
    const py::buffer_info pts_info = pts.request();
    std::size_t n = 0;
    require_rows(pts_info, 2, 3, &n, "pts");
    require(is_contiguous(pts_info),
            "pts: array must be contiguous");
    const py::buffer_info fin_info = finite.request();
    const ConstDoubles uf_d = read_doubles(uf, 1, "uf");
    const ConstDoubles vf_d = read_doubles(vf, 1, "vf");
    require(ufv.size == m && vfv.size == m, "p_world rows and uf_valid/vf_valid sizes must match");
    require(fin_info.ndim == 1 && static_cast<std::size_t>(fin_info.size) == n,
            "finite: expected shape (N,)");
    require(uf_d.size == n && vf_d.size == n, "pts rows and uf/vf sizes must match");

    auto grid = std::make_shared<SeedGrid>();
    grid->p_world.assign(static_cast<const double*>(pw_info.ptr),
                         static_cast<const double*>(pw_info.ptr) + pw_info.size);
    grid->uf_valid.assign(ufv.data, ufv.data + ufv.size);
    grid->vf_valid.assign(vfv.data, vfv.data + vfv.size);
    grid->pts.assign(static_cast<const double*>(pts_info.ptr),
                     static_cast<const double*>(pts_info.ptr) + pts_info.size);
    grid->finite.assign(static_cast<const bool*>(fin_info.ptr),
                        static_cast<const bool*>(fin_info.ptr) + fin_info.size);
    grid->uf.assign(uf_d.data, uf_d.data + uf_d.size);
    grid->vf.assign(vf_d.data, vf_d.data + vf_d.size);
    return grid;
}

std::shared_ptr<ElementContext> build_element_context(
    const std::shared_ptr<SurfaceDesc>& desc1, const std::shared_ptr<SurfaceDesc>& desc2,
    const std::shared_ptr<SeedGrid>& grid1, const std::shared_ptr<SeedGrid>& grid2, double index,
    const py::array_t<double>& tri, const py::array_t<double>& normals, bool graph1, bool graph2) {
    const py::buffer_info tri_info = tri.request();
    require(tri_info.ndim == 3 && tri_info.shape[1] == 3 && tri_info.shape[2] == 3,
            "tri: expected shape (K, 3, 3)");
    require(is_contiguous(tri_info),
            "tri: array must be contiguous");
    const std::size_t k = static_cast<std::size_t>(tri_info.shape[0]);
    const py::buffer_info nm_info = normals.request();
    require(nm_info.ndim == 2 && nm_info.shape[1] == 3 &&
                static_cast<std::size_t>(nm_info.shape[0]) == k,
            "normals: expected shape (K, 3)");
    require(is_contiguous(nm_info),
            "normals: array must be contiguous");
    auto ctx = std::make_shared<ElementContext>();
    ctx->desc1 = desc1;
    ctx->desc2 = desc2;
    ctx->grid1 = grid1;
    ctx->grid2 = grid2;
    ctx->index = index;
    const double inf = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 3; ++i) {
        ctx->lo[i] = inf;
        ctx->hi[i] = -inf;
    }
    local_bounds(*desc1, *desc2, ctx->lo, ctx->hi);
    ctx->tri.assign(static_cast<const double*>(tri_info.ptr),
                    static_cast<const double*>(tri_info.ptr) + tri_info.size);
    ctx->normals.assign(static_cast<const double*>(nm_info.ptr),
                        static_cast<const double*>(nm_info.ptr) + nm_info.size);
    ctx->graph1 = graph1;
    ctx->graph2 = graph2;
    return ctx;
}

std::shared_ptr<TraceStack> build_trace_stack(
    const std::vector<std::shared_ptr<ElementContext>>& contexts) {
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
    py::class_<ElementContext, std::shared_ptr<ElementContext>>(m, "ElementContext");
    py::class_<TraceStack, std::shared_ptr<TraceStack>>(m, "TraceStack")
        .def("__len__", [](const TraceStack& self) { return self.size(); });

    m.def("compile_surface", &make_model, py::arg("x"), py::arg("y"), py::arg("z"),
          py::arg("u_min"), py::arg("u_max"), py::arg("v_min"), py::arg("v_max"));
    m.def("build_surface_desc", &build_surface_desc, py::arg("model"), py::arg("u_min"),
          py::arg("u_max"), py::arg("v_min"), py::arg("v_max"), py::arg("tu"), py::arg("tv"),
          py::arg("ap_r"), py::arg("ap_cx"), py::arg("ap_cy"), py::arg("rotation"),
          py::arg("translation"));
    m.def("build_grid", &build_grid, py::arg("p_world"), py::arg("uf_valid"), py::arg("vf_valid"),
          py::arg("pts"), py::arg("finite"), py::arg("uf"), py::arg("vf"));
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
    m.def("build_element_context", &build_element_context, py::arg("desc1"), py::arg("desc2"),
          py::arg("grid1"), py::arg("grid2"), py::arg("refractive_index"), py::arg("tri"),
          py::arg("normals"), py::arg("graph1"), py::arg("graph2"));
    m.def("build_trace_stack", &build_trace_stack, py::arg("contexts"));
}

}  // namespace optics::bindings
