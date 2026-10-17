// Chunked parallel dispatch for native kernels (mirror of
// ``optics/numerical/parallel.py``): split ``[0, n)`` across threads and
// re-raise the first chunk failure on the caller.
#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <thread>
#include <vector>

namespace optics::native {

template <typename Fn>
void run_chunks(std::size_t n, Fn&& fn) {
    if (n == 0) {
        return;
    }
    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) {
        hc = 1;
    }
    std::size_t nt = std::max<std::size_t>(1, std::min<std::size_t>(hc, n));
    const std::size_t chunk = (n + nt - 1) / nt;
    std::vector<std::exception_ptr> errors(nt);
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < nt; ++t) {
        const std::size_t lo = t * chunk;
        if (lo >= n) {
            break;
        }
        const std::size_t hi = std::min(n, lo + chunk);
        threads.emplace_back([&, t, lo, hi] {
            try {
                fn(lo, hi);
            } catch (...) {
                errors[t] = std::current_exception();
            }
        });
    }
    for (std::thread& th : threads) {
        th.join();
    }
    for (const std::exception_ptr& exc : errors) {
        if (exc) {
            std::rethrow_exception(exc);
        }
    }
}

}  // namespace optics::native
