#include "stdio.h"
#include "cblas.h"

#include "MatrixMath.h"
#include "Utility.h"
#define n 5000

int main()
{
    /* INITIALIZE MATRICES: */
    double *x = rGenArray(n * n);
    double *y = rGenArray(n * n);

    double *z1 = calloc(n*n, sizeof(double));
    double *z2 = calloc(n*n, sizeof(double));
    double *z3 = calloc(n*n, sizeof(double));
    if (!x || !y) return 1; // Check allocation
    openblas_set_num_threads(1); // No cheating :)
    printf("Multiplying %dx%d matrices...\n", n, n);
    printf("-------------------------\n");

    /* MULTIPLY MATRIX NAIVELY: */

    double t = now_ms();
    size_t sz = n;
    // naive_matMult(x, y, sz, z1);
    t = now_ms() - t;
    printf("Time taken (naive): %f ms (Disabled! Could not run fast enough.)\n", t);
    printf("GFLOPS/sec (naive): %f\n\n", gflops(n,t/1000.0));

    /* MULTIPLY MATRIX USING BLAS: */

    t = now_ms();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n,
                1.0, x, n,
                y, n,
                0.0, z2, n);
    t = now_ms() - t;
    printf("Time taken (BLAS): %f ms\n", t);
    printf("GFLOPS/sec (BLAS): %f\n\n", gflops(n,t/1000.0));

    /* MULTIPLY MATRIX USING OPTIMIZING: */

    t = now_ms();
    optimized_matMult(x, y, sz, z3);
    t = now_ms() - t;
    printf("Time taken (optimized): %f ms\n", t);
    printf("GFLOPS/sec (optimized): %f\n\n", gflops(n,t/1000.0));

    /* NOW COMPARE ACCURACY: */

    printf("-------------------------\n");
    printf("Naive Method Passed: %s\n", RelError(z1, z2, n, n) < 1e-12 ? "True" : "False");
    printf("Optimized Method Passed: %s\n", RelError(z3, z2, n, n) < 1e-12 ? "True" : "False");

    // REMEMBER TO FREE:
    free(x); free(y);
    free(z1); free(z2); free(z3);
    return 0;
}