// Dependency-free CHECK harness shared by the native test TUs.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>

inline int g_failures = 0;

#define CHECK(cond)                                                                \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)

inline bool close(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) <= tol * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}
