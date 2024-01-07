#ifndef __RAY_MATH_TOOLKIT_H
#define __RAY_MATH_TOOLKIT_H

#define OPTIMIZE_ENABLE
#define OPT001_LOOP_UNROLLING

// 64 bit architecture
// sizeof(double) -> 8 byte -> 64 bit
// #define OPT002_SIMD

#ifdef OPT002_SIMD
#include <memory.h>// for memcpy
#include <immintrin.h>
typedef double v4df __attribute__ ((vector_size (32)));
#endif//end OPT002_SIMD

#include <math.h>
#include <stdio.h>
#include <assert.h>

static inline
void normalize(double *v)
{
    double d = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    assert(d != 0.0 && "Error calculating normal");

    v[0] /= d;
    v[1] /= d;
    v[2] /= d;
}

static inline
double length(const double *v)
{
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static inline
void add_vector(const double *a, const double *b, double *out)
{
#ifndef OPTIMIZE_ENABLE
    for (int i = 0; i < 3; i++)
        out[i] = a[i] + b[i];
#else
#ifdef OPT001_LOOP_UNROLLING
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
#endif//end OPT001_LOOP_UNROLLING
#ifdef OPT002_SIMD
    v4df sa = {a[0], a[1], a[2]};
    v4df sb = {b[0], b[1], b[2]};
    v4df sout = {0, 0, 0};

    sout = sa + sb;
    memcpy(out, &sout, sizeof(double)*3);
    // __builtin_ia32_storeupd256(out, sout);
    // out = __builtin_ia32_storeupd256(out,
    //         __builtin_ia32_addsubpd256(sa, sb)
    //     );
#endif//end OPT002_SIMD
#endif//end OPTIMIZE_ENABLE
}

static inline
void subtract_vector(const double *a, const double *b, double *out)
{
#ifndef OPTIMIZE_ENABLE
    for (int i = 0; i < 3; i++)
        out[i] = a[i] - b[i];
#else
#ifdef OPT001_LOOP_UNROLLING
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
#endif//end OPT001_LOOP_UNROLLING
#ifdef OPT002_SIMD
    v4df sa = {a[0], a[1], a[2]};
    v4df sb = {b[0], b[1], b[2]};
    v4df sout = {0, 0, 0};

    sout = sa - sb;
    memcpy(out, &sout, sizeof(double)*3);
#endif//end OPT002_SIMD
#endif//end OPTIMIZE_ENABLE
}

static inline
void multiply_vectors(const double *a, const double *b, double *out)
{
    for (int i = 0; i < 3; i++)
        out[i] = a[i] * b[i];
}

static inline
void multiply_vector(const double *a, double b, double *out)
{
#ifndef OPTIMIZE_ENABLE
    for (int i = 0; i < 3; i++)
        out[i] = a[i] * b;
#else
#ifdef OPT001_LOOP_UNROLLING
    out[0] = a[0] * b;
    out[1] = a[1] * b;
    out[2] = a[2] * b;
#endif//end OPT001_LOOP_UNROLLING
#ifdef OPT002_SIMD
    v4df sa = {a[0], a[1], a[2]};
    v4df sout = {0, 0, 0};

    sout = sa * b;
    memcpy(out, &sout, sizeof(double)*3);
#endif//end OPT002_SIMD
#endif//end OPTIMIZE_ENABLE
}

static inline
void cross_product(const double *v1, const double *v2, double *out)
{
    out[0] = v1[1] * v2[2] - v1[2] * v2[1];
    out[1] = v1[2] * v2[0] - v1[0] * v2[2];
    out[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

static inline
double dot_product(const double *v1, const double *v2)
{
    double dp = 0.0;
#ifndef OPTIMIZE_ENABLE
    for (int i = 0; i < 3; i++)
        dp += v1[i] * v2[i];
#else
#ifdef OPT001_LOOP_UNROLLING
    dp = v1[0] * v2[0] +
         v1[1] * v2[1] +
         v1[2] * v2[2];
#endif//end OPT001_LOOP_UNROLLING

#ifdef OPT002_SIMD
    v4df sa = {v1[0], v1[1], v1[2]};
    v4df sb = {v2[0], v2[1], v2[2]};
    v4df sout = {0, 0, 0};

    sout = sa * sb;
    dp = sout[0] + sout[1] + sout[2];
#endif//end OPT002_SIMD
#endif//end OPTIMIZE_ENABLE
    return dp;
}

static inline
void scalar_triple_product(const double *u, const double *v, const double *w,
                           double *out)
{
    cross_product(v, w, out);
    multiply_vectors(u, out, out);
}

static inline
double scalar_triple(const double *u, const double *v, const double *w)
{
    double tmp[3];
    cross_product(w, u, tmp);
    return dot_product(v, tmp);
}

#endif
