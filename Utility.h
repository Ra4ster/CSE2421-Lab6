#ifndef UTILITY_H
#define UTILITY_H

#define _POSIX_C_SOURCE 199309L

#include "stdlib.h"
#include "time.h"


/**
 * Generates a random float value.
 * 
 * @return float r
 */
static inline float rGen()
{
    return (float)rand() / (float)RAND_MAX;
}

static inline double* rGenArray(size_t length)
{
    double *arr = (double*)malloc(length * sizeof(double));
    for (size_t i = 0; i < length; i++)
    {
        arr[i] = (double)rGen();
    }
    return arr;
}

/**
 * Returns the current time in milliseconds.
 * 
 * @return double time in milliseconds
 */
static double now_ms(void) {
struct timespec ts;
clock_gettime(1 /* CLOCK_MONOTONIC */, &ts);
return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static inline double gflops(size_t n, double t)
{
    return (2.0 * n * n * n) / (t * 1e9);
}

#endif // UTILITY_H