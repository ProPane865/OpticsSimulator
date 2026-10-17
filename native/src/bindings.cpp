// pybind11 bindings for the native optics kernels (imported as
// ``optics._native``). Shape/dtype validation mirrors the assertions in the
// Python builders; C++ errors surface as ``ValueError`` so the Python layer
// can translate them (e.g. expression parse failures -> ``ExpressionError``).
// Buffer inspection happens with the GIL held; only the kernel computation
// runs under a scoped ``gil_scoped_release``.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "optics/expression.hpp"
#include "optics/grid.hpp"
#include "optics/kernels.hpp"
#include "optics/parallel.hpp"
#include "optics/solver.hpp"
#include "optics/surface.hpp"

namespace py = pybind11;

using optics::native::ElementContext;
using optics::native::Mat3;
using optics::native::SeedGrid;
using optics::native::SurfaceDesc;
using optics::native::SurfaceModel;
using optics::native::TraceStack;
using optics::native::Vec3;

namespace {

// --- numpy helpers -----------------------------------------------------------

void require(const bool ok, const std::string& message) {
    if (!ok) {
        throw py::value_error(message);
    }
}

struct ConstDoubles {
    const double* data;
    std::size_t size;
};

bool is_contiguous(const py::buffer_info& info) {
    py::ssize_t expected = info.itemsize;
    for (int i = info.ndim - 1; i >= 0; --i) {
        if (info.strides[i] != expected) {
            return false;
        }
        expected *= info.shape[i];
    }
    return true;
}

ConstDoubles read_doubles(const py::array_t<double>& a, int ndim, const std::string& what) {
    const py::buffer_info info = a.request();
    require(info.ndim == ndim, what + ": unexpected shape");
    require(is_contiguous(info), what + ": array must be contiguous");
    return {static_cast<const double*>(info.ptr), static_cast<std::size_t>(info.size)};
}

double* write_doubles(const py::array_t<double>& a, const std::string& what) {
    const py::buffer_info info = a.request();
    require(!info.readonly, what + ": array must be writeable");
    require(is_contiguous(info), what + ": array must be contiguous");
    return static_cast<double*>(info.ptr);
}

py::array_t<double> np3(const double* v) {
    double out[3] = {v[0], v[1], v[2]};
    return py::array_t<double>(std::vector<py::ssize_t>{3}, std::vector<py::ssize_t>{8}, out);
}

py::array_t<double> np3(const Vec3& v) { return np3(v.v); }

Vec3 to_vec3(const py::array_t<double>& a, const std::string& what) {
    const ConstDoubles d = read_doubles(a, 1, what);
    require(d.size == 3, what + ": expected shape (3,)");
    Vec3 v;
    v[0] = d.data[0];
    v[1] = d.data[1];
    v[2] = d.data[2];
    return v;
}

void require_rows(const py::buffer_info& info, int ndim, std::size_t cols, std::size_t* rows,
                  const std::string& what) {
    require(info.ndim == ndim &&
                static_cast<std::size_t>(info.shape[ndim - 1]) == cols,
            what + ": unexpected shape");
    if (rows != nullptr) {
        *rows = static_cast<std::size_t>(info.shape[0]);
    }
}

// --- surface / descriptor builders -------------------------------------------

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
    const py::array_t<double>& lo, const py::array_t<double>& hi, const py::array_t<double>& tri,
    const py::array_t<double>& normals, bool graph1, bool graph2) {
    const ConstDoubles lo_d = read_doubles(lo, 1, "lo");
    const ConstDoubles hi_d = read_doubles(hi, 1, "hi");
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
    for (int i = 0; i < 3; ++i) {
        ctx->lo[i] = lo_d.data[i];
        ctx->hi[i] = hi_d.data[i];
    }
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

// --- kernel entry points -------------------------------------------------------

std::tuple<double, double, double, bool> py_newton_intersection(const SurfaceDesc& desc,
                                                                const py::array_t<double>& o,
                                                                const py::array_t<double>& d,
                                                                double u, double v, double t) {
    const Vec3 ov = to_vec3(o, "origin");
    const Vec3 dv = to_vec3(d, "direction");
    const auto hit = [&] {
        const py::gil_scoped_release release;
        return optics::native::newton_intersection(desc, ov.v, dv.v, u, v, t);
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
        return optics::native::surface_intersect(desc, grid, ov.v, dv.v);
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
        return optics::native::plane_intersect(rov.v, dv.v, ov.v, nv.v, eps);
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
        return optics::native::mesh_intersect(ov.v, dv.v, tri_copy, det_eps, bary_tol, min_face_t);
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
    const py::gil_scoped_release release;
    optics::native::run_chunks(n, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i) {
            cd[i] = optics::native::trace_stack_segments(stack, od + 3 * i, dd + 3 * i, ray_length,
                                                         sd + 3 * cap * i, ed + 3 * cap * i,
                                                         kd + cap * i, cap)
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
    const bool ok = [&] {
        const py::gil_scoped_release release;
        return optics::native::trace_stack_to_plane(stack, ov.v, dv.v, pov.v, pnv.v, eps, u_min,
                                                    u_max, v_min, v_max, o_out, d_out);
    }();
    return py::bool_(ok);
}

double py_accumulate(const py::array_t<double>& grid, const py::array_t<double>& u,
                     const py::array_t<double>& v, const py::array_t<double>& power, double u_min,
                     double u_max, double v_min, double v_max, int ru, int rv, double interval_u,
                     double interval_v) {
    const py::buffer_info g_info = grid.request();
    require(g_info.ndim == 2 && !g_info.readonly && static_cast<int>(g_info.shape[1]) == ru &&
                static_cast<int>(g_info.shape[0]) == rv,
            "grid: expected writeable shape (rv, ru)");
    require(is_contiguous(g_info),
            "grid: array must be contiguous");
    const ConstDoubles ud = read_doubles(u, 1, "u");
    const ConstDoubles vd = read_doubles(v, 1, "v");
    const ConstDoubles pd = read_doubles(power, 1, "power");
    require(ud.size == vd.size && vd.size == pd.size, "u/v/power sizes must match");
    return optics::native::accumulate_power(static_cast<double*>(g_info.ptr), ud.data, vd.data,
                                            pd.data, ud.size, u_min, u_max, v_min, v_max, ru, rv,
                                            interval_u, interval_v);
}

void py_measure_direct(const py::array_t<double>& origins, const py::array_t<double>& dirs,
                       const py::array_t<double>& powers, const py::array_t<double>& o,
                       const py::array_t<double>& nrm, double eps, const py::array_t<double>& us,
                       const py::array_t<double>& vs, const py::array_t<double>& pws) {
    const ConstDoubles od = read_doubles(origins, 2, "origins");
    const ConstDoubles dd = read_doubles(dirs, 2, "dirs");
    const ConstDoubles pw = read_doubles(powers, 1, "powers");
    const std::size_t n = pw.size;
    require(od.size == 3 * n && dd.size == 3 * n, "origins/dirs: expected (N, 3)");
    const Vec3 ov = to_vec3(o, "plane origin");
    const Vec3 nv = to_vec3(nrm, "plane normal");
    double* ud = write_doubles(us, "us");
    double* vd = write_doubles(vs, "vs");
    double* pd = write_doubles(pws, "pws");
    const py::gil_scoped_release release;
    optics::native::measure_direct(od.data, dd.data, pw.data, n, ov.v, nv.v, eps, ud, vd, pd);
}

void py_measure_stack(const TraceStack& stack, const py::array_t<double>& origins,
                      const py::array_t<double>& dirs, const py::array_t<double>& powers,
                      const py::array_t<double>& o, const py::array_t<double>& nrm, double eps,
                      double u_min, double u_max, double v_min, double v_max,
                      const py::array_t<double>& us, const py::array_t<double>& vs,
                      const py::array_t<double>& pws) {
    const ConstDoubles od = read_doubles(origins, 2, "origins");
    const ConstDoubles dd = read_doubles(dirs, 2, "dirs");
    const ConstDoubles pw = read_doubles(powers, 1, "powers");
    const std::size_t n = pw.size;
    require(od.size == 3 * n && dd.size == 3 * n, "origins/dirs: expected (N, 3)");
    const Vec3 ov = to_vec3(o, "plane origin");
    const Vec3 nv = to_vec3(nrm, "plane normal");
    double* ud = write_doubles(us, "us");
    double* vd = write_doubles(vs, "vs");
    double* pd = write_doubles(pws, "pws");
    const py::gil_scoped_release release;
    optics::native::measure_stack(stack, od.data, dd.data, pw.data, n, ov.v, nv.v, eps, u_min,
                                  u_max, v_min, v_max, ud, vd, pd);
}

py::object py_rotation_from_z(const py::array_t<double>& direction) {
    const Vec3 d = to_vec3(direction, "direction");
    const Mat3 r = optics::native::rotation_from_z(d);
    double flat[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            flat[3 * i + j] = r.m[i][j];
        }
    }
    return py::array_t<double>(std::vector<py::ssize_t>{3, 3}, std::vector<py::ssize_t>{24, 8},
                               flat);
}

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Compiled numerical kernels for the optics simulator (C++/pybind11)";

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const optics::native::ExpressionError& exc) {
            PyErr_SetString(PyExc_ValueError, exc.what());
        } catch (const optics::native::VectorError& exc) {
            PyErr_SetString(PyExc_ValueError, exc.message);
        } catch (const py::error_already_set&) {
            // Python-originated exceptions pass through unchanged.
        } catch (const std::invalid_argument& exc) {
            PyErr_SetString(PyExc_ValueError, exc.what());
        } catch (const std::runtime_error& exc) {
            PyErr_SetString(PyExc_RuntimeError, exc.what());
        }
    });

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
    m.def("build_element_context", &build_element_context, py::arg("desc1"), py::arg("desc2"),
          py::arg("grid1"), py::arg("grid2"), py::arg("refractive_index"), py::arg("lo"),
          py::arg("hi"), py::arg("tri"), py::arg("normals"), py::arg("graph1"), py::arg("graph2"));
    m.def("build_trace_stack", &build_trace_stack, py::arg("contexts"));

    m.def("norm", [](const py::array_t<double>& v) {
        const ConstDoubles d = read_doubles(v, 1, "v");
        require(d.size == 3, "v: expected shape (3,)");
        Vec3 a;
        a[0] = d.data[0];
        a[1] = d.data[1];
        a[2] = d.data[2];
        return optics::native::norm(a);
    });
    m.def(
        "normalize",
        [](const py::array_t<double>& v) { return np3(optics::native::normalize_vector(to_vec3(v, "v"))); });
    m.def("rotation_from_z", &py_rotation_from_z, py::arg("direction"));

    m.def("newton_intersection", &py_newton_intersection, py::arg("desc"), py::arg("o"),
          py::arg("d"), py::arg("u"), py::arg("v"), py::arg("t"));
    m.def("surface_intersect", &py_surface_intersect, py::arg("desc"), py::arg("grid"), py::arg("o"),
          py::arg("d"));
    m.def("solve_xy", &py_solve_xy, py::arg("desc"), py::arg("x_target"), py::arg("y_target"),
          py::arg("u"), py::arg("v"));
    m.def("parameters_at_xy", &py_parameters_at_xy, py::arg("desc"), py::arg("grid"), py::arg("x"),
          py::arg("y"));
    m.def("plane_intersect", &py_plane_intersect, py::arg("ro"), py::arg("d"), py::arg("o"),
          py::arg("n"), py::arg("eps"));
    m.def("mesh_intersect", &py_mesh_intersect, py::arg("o"), py::arg("d"), py::arg("tri"),
          py::arg("det_eps"), py::arg("bary_tol"), py::arg("min_face_t"));
    m.def("orient", &py_orient, py::arg("normal"), py::arg("incident"));
    m.def("refract_or_reflect", &py_refract_or_reflect, py::arg("incident"), py::arg("normal"),
          py::arg("n_from"), py::arg("n_to"));

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
            optics::native::StackSegments res;
            {
                const py::gil_scoped_release release;
                res = optics::native::trace_stack_segments(
                    stack, ov.v, dv.v, ray_length, static_cast<double*>(s_info.ptr),
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

    m.def("accumulate", &py_accumulate, py::arg("grid"), py::arg("u"), py::arg("v"),
          py::arg("power"), py::arg("u_min"), py::arg("u_max"), py::arg("v_min"), py::arg("v_max"),
          py::arg("ru"), py::arg("rv"), py::arg("interval_u"), py::arg("interval_v"));
    m.def("measure_direct", &py_measure_direct, py::arg("origins"), py::arg("dirs"),
          py::arg("powers"), py::arg("o"), py::arg("n"), py::arg("eps"), py::arg("us"),
          py::arg("vs"), py::arg("pws"));
    m.def("measure_stack", &py_measure_stack, py::arg("stack"), py::arg("origins"), py::arg("dirs"),
          py::arg("powers"), py::arg("o"), py::arg("n"), py::arg("eps"), py::arg("u_min"),
          py::arg("u_max"), py::arg("v_min"), py::arg("v_max"), py::arg("us"), py::arg("vs"),
          py::arg("pws"));
}
