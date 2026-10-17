// Kernel data layouts (mirror of ``optics/numerical/_kernels.py``): a sampled
// seed grid, one flattened refractive element, and a trace stack. The Python
// builders fill these from numpy arrays; the kernels read them read-only.
#pragma once

#include <memory>
#include <vector>

#include "optics/surface.hpp"

namespace optics::native {

// A sampled parameter grid used to seed Newton for intersections and
// parameter inversion. ``*_valid`` arrays hold only in-domain/aperture
// samples. Point arrays are ``(rows, 3)`` flattened row-major.
struct SeedGrid {
    std::vector<double> p_world;   // (M, 3) valid sample points in world space
    std::vector<double> uf_valid;  // (M,)
    std::vector<double> vf_valid;  // (M,)
    std::vector<double> pts;       // (N, 3) all sample points in local space
    std::vector<unsigned char> finite;  // (N,)
    std::vector<double> uf;        // (N,)
    std::vector<double> vf;        // (N,)

    std::size_t m() const { return uf_valid.size(); }
    std::size_t n() const { return uf.size(); }
};

// One closed refractive element flattened for the stack-tracing kernels.
struct ElementContext {
    std::shared_ptr<const SurfaceDesc> desc1;
    std::shared_ptr<const SurfaceDesc> desc2;
    std::shared_ptr<const SeedGrid> grid1;
    std::shared_ptr<const SeedGrid> grid2;
    double index = 1.0;
    Vec3 lo{};
    Vec3 hi{};
    std::vector<double> tri;      // (K, 3, 3) flattened row-major
    std::vector<double> normals;  // (K, 3) flattened row-major
    bool graph1 = false;
    bool graph2 = false;

    std::size_t n_tris() const { return tri.size() / 9; }
};

using ElementRef = std::shared_ptr<const ElementContext>;

// A stack of elements traced as a unit. The Python tuple layout padded this
// to fixed capacities for Numba's static typing; a native vector carries the
// real element count directly.
struct TraceStack {
    std::vector<ElementRef> elements;

    std::size_t size() const { return elements.size(); }
};

}  // namespace optics::native
