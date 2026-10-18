// Kernel data layouts (mirror of ``optics/numerical/_kernels.py``): one
// flattened refractive element and a trace stack. The Python builders fill
// these from numpy arrays; the kernels read them read-only.
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "optics/common/vector.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/intersection/grid.hpp"

namespace optics::native {

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

// A stack of elements traced as a unit. The historical tuple layout padded
// this to fixed capacities; a native vector carries the real element count
// directly.
struct TraceStack {
    std::vector<ElementRef> elements;

    std::size_t size() const { return elements.size(); }
};

}  // namespace optics::native
