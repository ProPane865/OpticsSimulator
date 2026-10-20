// Shared numpy inspection/validation helpers for the ``_native`` bindings.
// Buffer checks run with the GIL held; every helper is header-inline so the
// per-area binding TUs share them without extra translation units.
#pragma once

#include <pybind11/numpy.h>

#include <cstddef>
#include <cstdint>
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

// Copy a contiguous numpy double buffer into a std::vector: needed when the
// GIL is released (the Python caller may mutate/free the array meanwhile).
inline std::vector<double> copy_vector(const py::buffer_info& info) {
    return std::vector<double>(static_cast<const double*>(info.ptr),
                               static_cast<const double*>(info.ptr) + info.size);
}

// Batched rendering segment output buffers: ``(N, cap, 3)`` starts/ends,
// ``(N, cap)`` kinds and ``(N,)`` counts, validated against the ray count.
struct SegmentBuffers {
    double* starts;
    double* ends;
    signed char* kinds;
    std::int64_t* counts;
    std::size_t cap;
};

inline SegmentBuffers read_segment_buffers(const py::array_t<double>& starts,
                                           const py::array_t<double>& ends,
                                           const py::array_t<std::int8_t>& kinds,
                                           const py::array_t<std::int64_t>& counts,
                                           std::size_t n) {
    const py::buffer_info s_info = starts.request();
    const py::buffer_info e_info = ends.request();
    const py::buffer_info k_info = kinds.request();
    const py::buffer_info c_info = counts.request();
    require(s_info.ndim == 3 && static_cast<std::size_t>(s_info.shape[0]) == n &&
                s_info.shape[2] == 3,
            "starts: expected shape (N, cap, 3)");
    require(e_info.ndim == 3 && e_info.size == s_info.size, "ends: expected shape (N, cap, 3)");
    require(k_info.ndim == 2 && static_cast<std::size_t>(k_info.shape[0]) == n &&
                static_cast<std::size_t>(k_info.shape[1]) == static_cast<std::size_t>(s_info.shape[1]),
            "kinds: expected shape (N, cap)");
    require(c_info.ndim == 1 && static_cast<std::size_t>(c_info.size) == n,
            "counts: expected shape (N,)");
    require(!s_info.readonly && !e_info.readonly && !k_info.readonly && !c_info.readonly,
            "segment buffers must be writeable");
    return {static_cast<double*>(s_info.ptr), static_cast<double*>(e_info.ptr),
            static_cast<signed char*>(k_info.ptr), static_cast<std::int64_t*>(c_info.ptr),
            static_cast<std::size_t>(s_info.shape[1])};
}

}  // namespace optics::bindings
