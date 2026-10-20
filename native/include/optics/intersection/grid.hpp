// Kernel data layouts: the sampled seed grid used to seed Newton for
// intersections and parameter inversion; built once per persistent surface
// and read-only for the kernels. The element/stack layouts live in
// ``optics/tracing/context.hpp``.
#pragma once

#include <cstddef>
#include <vector>

namespace optics::native {

// Ray/surface Newton hit in surface parameters.
struct HitUV {
    double t, u, v;
};

// A sampled parameter grid used to seed Newton for intersections and
// parameter inversion. ``*_valid`` arrays hold only in-domain/aperture
// samples. Point arrays are ``(rows, 3)`` flattened row-major.
struct SeedGrid {
    std::vector<double> p_world;        // (M, 3) valid sample points in world space
    std::vector<double> uf_valid;       // (M,)
    std::vector<double> vf_valid;       // (M,)
    std::vector<double> pts;            // (N, 3) all sample points in local space
    std::vector<unsigned char> finite;  // (N,)
    std::vector<double> uf;             // (N,)
    std::vector<double> vf;             // (N,)

    std::size_t m() const { return uf_valid.size(); }
    std::size_t n() const { return uf.size(); }
};

}  // namespace optics::native
