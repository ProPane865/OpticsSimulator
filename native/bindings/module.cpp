// ``optics._native`` module hub: the exception translation, the vector-math
// entry points and the per-area registration calls. Shape/dtype validation
// mirrors the assertions in the Python builders; C++ errors surface as
// ``ValueError`` so the Python layer can translate them (e.g. expression
// parse failures -> ``ExpressionError``).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <exception>
#include <vector>

#include "optics/common/matrix.hpp"
#include "optics/common/vector.hpp"
#include "optics/expression/expression.hpp"

#include "register.hpp"
#include "support.hpp"

namespace {

using optics::native::Mat3;
using optics::native::Vec3;
using namespace optics::bindings;

py::array_t<double> py_rotation_from_z(const py::array_t<double>& direction) {
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

    optics::bindings::register_builders(m);
    optics::bindings::register_kernels(m);
    optics::bindings::register_tracing(m);
    optics::bindings::register_measurement(m);
}
