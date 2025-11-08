#ifndef MATRIXMATH_H
#define MATRIXMATH_H

#include "stdlib.h"
#include "math.h"
#include "immintrin.h"
#include "string.h"

/**
 * Struct of rows of x, cols of x, and cols of y.
 * Rows of y should be equal to cols of x.
 */
typedef struct {
    size_t row_x;
    size_t col_x;
    size_t col_y;
} MatrixSizes;

/**
 * Matrix multiplies 2 matrices.
 * 
 * Naive Edition.
 * 
 * @param x double* matrix 1
 * @param y double* matrix 2
 * @param sz size of a matrix
 * @param z place to return result
 * 
 */
void naive_matMult(double *x, double *y, size_t sz, double *z);

/**
 * Subtract x - y.
 * 
 * @param x double* matrix 1
 * @param x double* matrix 2
 * @param rows size_t rows
 * @param cols size_t cols
 * 
 * @return x - y
 */
double* matSub(double *x, double *y, size_t rows, size_t cols);

/**
 * Computes Frobenius error of 2 matrices.
 * 
 * @param C_test matrix to test
 * @param C_ref reference matrix
 * @param rows rows of each matrix
 * @param cols columns of each matrix
 * @return relative error
 */
double RelError(double *C_test, double *C_ref, size_t rows, size_t cols);

/**
 * Matrix multiplies 2 matrices.
 * 
 * Optimized version.
 * 
 * @param x Matrix 1
 * @param y Matrix 2
 * @param sz size of a matrix
 * @param z place to return result
 */
void optimized_matMult(double *x, double *y, size_t sz, double *z);

#endif // MATRIXMATH_H