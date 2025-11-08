#include "MatrixMath.h"

void naive_matMult(double *x, double *y, size_t n, double *z)
{
    for (int i=0; i < n; i++)
    {
        for (int j=0; j < n; j++)
        {
            for (int k=0; k < n; k++)
            {
                // z_{i,j} -> x_{i,k} * y_{k, j}
                z[i*n + j] += x[i*n + k] * y[k*n + j];
            }
        }
    }
}

/// @brief Subtracts 2 matrices of equal size
/// @param x Matrix 1
/// @param y Matrix 2
/// @param rows Rows of matrices
/// @param cols Cols of matrices
/// @return difference
inline double* matSub(double* x, double* y, size_t rows, size_t cols)
{
    double* z = (double*)malloc(sizeof(double) * rows * cols);
    for (int i=0; i < rows; i++)
    {
        for (int j=0; j < cols; j++)
        {
            z[cols*i + j] = x[cols*i + j] - y[cols*i + j];
        }
    }
    return z;
}

/// @brief Returns the norm of a matrix.
/// @param m Matrix
/// @param sz Matrix size
/// @return norm
inline double normalize(double* m, size_t sz)
{
    double sum = 0;

    for (int i=0; i < sz; i++)
    {
        sum += pow(*(m+i), 2);
    }

    return sqrt(sum);
}

double RelError(double* C_test, double* C_ref, size_t rows, size_t cols)
{
    double numerator = normalize(matSub(C_test, C_ref, rows, cols), rows*cols);
    double denominator = normalize(C_ref, rows*cols);

    return numerator / denominator;
}

void optimized_matMult(double *x, double *y, size_t n, double *z)
{
    // 1. Blocks:
    const size_t BLOCK_L3 = 1024;
    const size_t BLOCK_L2 = 256;
    const size_t BLOCK_L1 = 64;

    // L 3 CACHE
        for (size_t ii3 = 0; ii3 < n; ii3 += BLOCK_L3)
            for (size_t kk3 = 0; kk3 < n; kk3 += BLOCK_L3)
                for (size_t jj3 = 0; jj3 < n; jj3 += BLOCK_L3) // L 2
                    for (size_t ii2 = ii3; ii2 < ((ii3+BLOCK_L3)>n?n:ii3+BLOCK_L3); ii2 += BLOCK_L2)
                        for (size_t kk2 = kk3; kk2 < ((kk3+BLOCK_L3)>n?n:kk3+BLOCK_L3); kk2 += BLOCK_L2)
                            for (size_t jj2 = jj3; jj2 < ((jj3+BLOCK_L3)>n?n:jj3+BLOCK_L3); jj2 += BLOCK_L2)
                            {
                                size_t i_max2 = (ii2 + BLOCK_L2 > n) ? n : ii2 + BLOCK_L2;
                                size_t k_max2 = (kk2 + BLOCK_L2 > n) ? n : kk2 + BLOCK_L2;
                                size_t j_max2 = (jj2 + BLOCK_L2 > n) ? n : jj2 + BLOCK_L2;
                                // L 1
                                for (size_t ii1 = ii2; ii1 < i_max2; ii1 += BLOCK_L1)
                                    for (size_t kk1 = kk2; kk1 < k_max2; kk1 += BLOCK_L1)
                                        for (size_t jj1 = jj2; jj1 < j_max2; jj1 += BLOCK_L1)
                                        {
                                        size_t i_max1 = (ii1 + BLOCK_L1 > i_max2) ? i_max2 : ii1 + BLOCK_L1;
                                        size_t k_max1 = (kk1 + BLOCK_L1 > k_max2) ? k_max2 : kk1 + BLOCK_L1;
                                        size_t j_max1 = (jj1 + BLOCK_L1 > j_max2) ? j_max2 : jj1 + BLOCK_L1;

                                        for (size_t i = ii1; i < i_max1; i++)
                                            for (size_t k = kk1; k < k_max1; k++)
                                            {
                                                __m256d x_val = _mm256_set1_pd(x[i*n + k]);
                                                size_t j;
                                                for (j = jj1; j + 4 <= j_max1; j += 4)
                                                {
                                                    __m256d y_vec = _mm256_loadu_pd(&y[k*n + j]);
                                                    __m256d z_vec = _mm256_loadu_pd(&z[i*n + j]);
                                                    z_vec = _mm256_fmadd_pd(x_val, y_vec, z_vec);
                                                    _mm256_storeu_pd(&z[i*n + j], z_vec);
                                                }
                                                // Handle leftovers
                                                for (; j < j_max1; j++) z[i*n + j] += x[i*n + k] * y[k*n + j];
                                            }
                                        }
                            }
}