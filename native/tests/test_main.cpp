// Native test entry point: runs every suite and reports the failure count.
#include <cstdio>

#include "check.hpp"

void test_expression();
void test_kernels();

int main() {
    test_expression();
    test_kernels();
    if (g_failures == 0) {
        std::printf("native tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "%d native test failures\n", g_failures);
    return 1;
}
