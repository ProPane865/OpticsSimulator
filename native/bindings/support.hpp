// Shared numpy inspection/validation helpers for the ``_native`` bindings.
// Buffer checks run with the GIL held; every helper is header-inline so the
// per-area binding TUs share them without extra translation units.
#pragma once

#include <pybind11/numpy.h>

#include <cstddef>
#include <string>
#include <vector>

#include "optics/common/matrix.hpp"
#include "optics/common/vector.hpp"

namespace py = pybind11;

namespace optics::bindings {

using native::Vec3;

struct ConstDoubles {
    const double* data;
    std::size_t size;
};

inline void require(const bool ok, const std::string& message) {
    if (!ok) {
        throw py::value_error(message);
    }
}

inline bool is_contiguous(const py::buffer_info& info) {
    py::ssize_t expected = info.itemsize;
    for (int i = info.ndim - 1; i >= 0; --i) {
        if (info.strides[i] != expected) {
            return false;
        }
        expected *= info.shape[i];
    }
    return true;
}

inline ConstDoubles read_doubles(const py::array_t<double>& a, int ndim, const std::string& what) {
    const py::buffer_info info = a.request();
    require(info.ndim == ndim, what + ": unexpected shape");
    require(is_contiguous(info), what + ": array must be contiguous");
    return {static_cast<const double*>(info.ptr), static_cast<std::size_t>(info.size)};
}

inline double* write_doubles(const py::array_t<double>& a, const std::string& what) {
    const py::buffer_info info = a.request();
    require(!info.readonly, what + ": array must be writeable");
    require(is_contiguous(info), what + ": array must be contiguous");
    return static_cast<double*>(info.ptr);
}

inline double* write_matrix(const py::array_t<double>& a, std::size_t rows, std::size_t cols,
                            const std::string& what) {
    const py::buffer_info info = a.request();
    require(info.ndim == 2 && static_cast<std::size_t>(info.shape[0]) == rows &&
                static_cast<std::size_t>(info.shape[1]) == cols,
            what + ": expected shape (rows, cols)");
    require(!info.readonly, what + ": array must be writeable");
    require(is_contiguous(info), what + ": array must be contiguous");
    return static_cast<double*>(info.ptr);
}

inline py::array_t<double> np3(const double* v) {
    double out[3] = {v[0], v[1], v[2]};
    return py::array_t<double>(std::vector<py::ssize_t>{3}, std::vector<py::ssize_t>{8}, out);
}

inline py::array_t<double> np3(const Vec3& v) { return np3(v.v); }

inline Vec3 to_vec3(const py::array_t<double>& a, const std::string& what) {
    const ConstDoubles d = read_doubles(a, 1, what);
    require(d.size == 3, what + ": expected shape (3,)");
    Vec3 v;
    v[0] = d.data[0];
    v[1] = d.data[1];
    v[2] = d.data[2];
    return v;
}

inline void require_rows(const py::buffer_info& info, int ndim, std::size_t cols, std::size_t* rows,
                         const std::string& what) {
    require(info.ndim == ndim &&
                static_cast<std::size_t>(info.shape[ndim - 1]) == cols,
            what + ": unexpected shape");
    if (rows != nullptr) {
        *rows = static_cast<std::size_t>(info.shape[0]);
    }
}

}  // namespace optics::bindings
