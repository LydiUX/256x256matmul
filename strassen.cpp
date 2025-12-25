#include <immintrin.h>
#include <iostream>
#include "fmadd_mat.h"
#include "strassen.h"

static inline __attribute__((always_inline)) double strassen::sum_all_squared(const double* __restrict A) {
    __m256d r0 = _mm256_setzero_pd();
    __m256d r1 = _mm256_setzero_pd();
    __m256d r2 = _mm256_setzero_pd();
    __m256d r3 = _mm256_setzero_pd();
    for (int i = 0; i < 128*128; i+=16) { // the more accumulators, the faster :D
        __m256d load_buffer_A0 = _mm256_load_pd(&A[i]);
        __m256d load_buffer_A1 = _mm256_load_pd(&A[i+4]);
        __m256d load_buffer_A2 = _mm256_load_pd(&A[i+8]);
        __m256d load_buffer_A3 = _mm256_load_pd(&A[i+12]);
        r0 = _mm256_fmadd_pd(load_buffer_A0, load_buffer_A0, r0); // r += a_ij^2
        r1 = _mm256_fmadd_pd(load_buffer_A1, load_buffer_A1, r1);
        r2 = _mm256_fmadd_pd(load_buffer_A2, load_buffer_A2, r2);
        r3 = _mm256_fmadd_pd(load_buffer_A3, load_buffer_A3, r3);
    }
    __m256d r = _mm256_add_pd(_mm256_add_pd(r0,r1),_mm256_add_pd(r2,r3));
    double t[4];
    _mm256_storeu_pd(t, r);
    return t[0]+t[1]+t[2]+t[3];
}

double strassen::multiply_matrices(double a[256][256], double b[256][256]) {
    // Strassen algorithm: Split c=axb into four 128x128 chunks c00, c01, c10, c11
    alignas(32) static double c00[128*128];
    alignas(32) static double c01[128*128];
    alignas(32) static double c10[128*128];
    alignas(32) static double c11[128*128];
    std::memset(c00, 0, sizeof(c00));
    std::memset(c01, 0, sizeof(c01));
    std::memset(c10, 0, sizeof(c10));
    std::memset(c11, 0, sizeof(c11));
    // split a and b similarly. NB these pointers require +256 to access the next row. compatible with cache_op.
    const double* a00 = &a[0][0];
    const double* a01 = &a[0][128];
    const double* a10 = &a[128][0];
    const double* a11 = &a[128][128];
    const double* b00 = &b[0][0];
    const double* b01 = &b[0][128];
    const double* b10 = &b[128][0];
    const double* b11 = &b[128][128];
    // contiguous temporary compute buffers for A and B
    alignas(32) static double buffer_a[128*128]; // regular contiguous matrix storing
    alignas(32) static double buffer_b[128*128]; // contiguous panel layout matrix
    // Strassen blocks:
    // M1 = (A00+A11)x(B00+B11) -> used in c00, c11
    cache_op::add_mat_a(a00, a11, buffer_a);
    cache_op::add_mat_b(b00, b11, buffer_b);
    cache_op::fmadd_mat(buffer_a, buffer_b, c00);
    cache_op::fmadd_mat(buffer_a, buffer_b, c11);
    // M2 = (A10+A11)xB11 -> used in c10, c11
    cache_op::add_mat_a(a10, a11, buffer_a);
    cache_op::store_mat_b(b11, buffer_b);
    cache_op::fmadd_mat(buffer_a, buffer_b, c10);
    cache_op::neg_mat(buffer_a); // -M2 = -(A10+A11)xB11
    cache_op::fmadd_mat(buffer_a, buffer_b, c11);
    // M3 = A00x(B01-B11) -> used in c01, c11
    cache_op::store_mat_a(a00, buffer_a);
    cache_op::sub_mat_b(b01, b11, buffer_b);
    cache_op::fmadd_mat(buffer_a, buffer_b, c01);
    cache_op::fmadd_mat(buffer_a, buffer_b, c11);
    // M4 = A11x(B10-B00) -> used in c00, c10
    cache_op::store_mat_a(a11, buffer_a);
    cache_op::sub_mat_b(b10, b00, buffer_b);
    cache_op::fmadd_mat(buffer_a, buffer_b, c00);
    cache_op::fmadd_mat(buffer_a, buffer_b, c10);
    // M5 = (A00+A01)xB11 -> used in c00, c01
    cache_op::add_mat_a(a00, a01, buffer_a);
    cache_op::store_mat_b(b11, buffer_b);
    cache_op::fmadd_mat(buffer_a, buffer_b, c01);
    cache_op::neg_mat(buffer_a); // -M5 = -(A00+A01)xB11
    cache_op::fmadd_mat(buffer_a, buffer_b, c00);
    // M6 = (A10-A00)x(B00+B01) -> used in c11
    cache_op::sub_mat_a(a10, a00, buffer_a);
    cache_op::add_mat_b(b00, b01, buffer_b);
    cache_op::fmadd_mat(buffer_a, buffer_b, c11);
    // M7 = (A01-A11)x(B10+B11) -> used in c00
    cache_op::sub_mat_a(a01, a11, buffer_a);
    cache_op::add_mat_b(b10, b11, buffer_b);
    cache_op::fmadd_mat(buffer_a, buffer_b, c00);
    // now sum up the squares in c00...c11
    return strassen::sum_all_squared(c00)+strassen::sum_all_squared(c01)
        +strassen::sum_all_squared(c10)+strassen::sum_all_squared(c11);
}