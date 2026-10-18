// Rigid world transform (rotation + translation) shared by surfaces and the
// element frame they sit in. Local->world is ``rot * p + trans``; the
// rotation is orthonormal, so world->local uses the transpose (mirror of
// ``optics/primitive/transform.py``).
#pragma once

#include "optics/common/matrix.hpp"
#include "optics/common/vector.hpp"

namespace optics::native {

struct Transform {
    Mat3 rot = Mat3::identity();
    Vec3 trans{};

    Vec3 to_world(const Vec3& local) const { return rot * local + trans; }
    Vec3 to_local(const Vec3& world) const { return mul_transpose(rot, world - trans); }
    Vec3 vector_to_world(const Vec3& local) const { return rot * local; }
    Vec3 vector_to_local(const Vec3& world) const { return mul_transpose(rot, world); }
};

}  // namespace optics::native
