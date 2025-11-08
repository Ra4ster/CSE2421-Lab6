# Lab 6: CPU Matrix Multiplication in C
*By Jack C. Rose, For CSE 2241@OSU: Low Level Programming*

## Table of Contents
- [Summary](#summary)
- [Procedure](#procedure)
- [Runtime Analysis](#runtime-analysis)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run-yourself)

## Summary

This lab is a test of how well different matrix multiplication implementations work in C. The program compares 3 types of functions: a naive multiplication in "ijk" order, a [cBLAS](https://www.netlib.org/blas/cblas.h) multiplication using [dgemm](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/cblas-gemm-001.html), and a custom optimized multiplication using cache blocks, AVX2 vectorization, and an analysis of L1/L2/L3 storage capabilities.

## Procedure

> The first naive implementation was extremely simple to create; it was a simple ijk for-loop multiplication and is self-explanatory.

Flags used:
`-O3 -g -march=native`.

**Steps for optimized multiplication:**

1. I began by running the command to generate CPU instructions, to test if I had AVX2:

```cmd
My-Laptop:/mnt/c/Users/me/Projects/CSE2241/Lab6$ lscpu | grep Flags

Flags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology tsc_reliable nonstop_tsc cpuid tsc_known_freq pni pclmulqdq vmx ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves avx_vnni vnmi umip waitpkg gfni vaes vpclmulqdq rdpid movdiri movdir64b fsrm md_clear serialize flush_l1d arch_capabilities
```

2. This allowed me to come up with the initial AVX2 instructions for multiplication. Unfortunately, the storage must be unaligned since the pointers are allocated outside the method. Like so:

```c
    __m256d vx = _mm256_loadu_pd(x);
    __m256d vy = _mm256_loadu_pd(y);

    __m256d v = _mm256_mul_pd(va, vb);

    // Add: (va * vb) + va
    __m256d vz = _mm256_add_pd(_mm256_mul_pd(va, vb), va);

    double result[4];
    _mm256_storeu_pd(result, vz);
```

3. After the lecture on CAVX implementations, I found a much nicer AVX2 instruction that combines 2 others and could be implemented with the addition of the flag `-mfma`. This resulted in a 200ms reduction. See below:

```c
__m256d vz = _mm256_fmadd_pd(vx, vy, vz);
```

4. Now I began creating blocks using the L1 cache, which sped up the program by ~100ms. See here:
```c
size_t BLOCK_SIZE = 16;
    for (int ii = 0; ii < N; ii += BLOCK_SIZE)
        for (int jj = 0; jj < N; jj += BLOCK_SIZE)
            for (int kk = 0; kk < N; kk += BLOCK_SIZE)
                for (int i = ii; i < ii + BLOCK_SIZE; i++)
                    for (int k = kk; k < kk + BLOCK_SIZE; k++) {
                    __m256d vx = _mm256_loadu_pd(x);
                    __m256d vy = _mm256_loadu_pd(y);

                    __m256d v = _mm256_mul_pd(vx, vy);

                    // Add: (vx * vy) + vx
                    __m256d vz = _mm256_fmadd_pd(vx, vy, vz);

                    double result[4];
                    _mm256_storeu_pd(result, vz);
                    }
```

5. Finally, I really wanted my implementation to get under 5x the cBLAS speed, so I attempted an implementation that blocks on all caches:

```c
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
                                                    ...
```

6. Bonus! Using the cache command, I was able to see the best cache size to use:

```cmd
My-Laptop:/mnt/c/Users/me/Projects/CSE2241/Lab6$ lscpu | grep -i "cache"

L1d cache:                            480 KiB (10 instances)
L1i cache:                            320 KiB (10 instances)
L2 cache:                             12.5 MiB (10 instances)
L3 cache:                             24 MiB (1 instance)
```

This was a good amount, meaning I had enough cache space to double the blocking sizes, like so:

```c
const size_t BLOCK_L3 = 1024;
const size_t BLOCK_L2 = 256;
const size_t BLOCK_L1 = 64;
```

## Runtime Analysis

GFLOPS/Sec and time taken were measured. For GFLOPS, calculation is done in the executable and can also be done by hand using the formula:

*(for the sake of space, I did not include these)*

$$
GFLOPS/sec = \frac{2n^3}{\text{time} * 10^9}
$$

*Also: All frobelius norms compared to cBLAS returned ~0.0, meaning they were accurate.*

The executables were tested 16 times; 4 times for gcc 2000x2000 matrices, 4 times for gcc 5000x5000 matrices, 4 times for clang 2000x2000 matrices, and 4 times for clang 5000x5000 matrices. These are the output times:

| |GCC 2000x2000| CLANG 2000x2000 |GCC 5000x5000| CLANG 5000x5000 |
|-|-|-|-|-|
| <u>Run 1 Naive</u> | 21462 ms | 22724 ms | n/A | n/A |
| <u>Run 1 BLAS</u> | 277 ms| 290 ms | 4077 ms | 4074 ms |
| <u>Run 1 Optimized</u> | 989 ms | 1205 ms | 17316 ms | 18389 ms |
| <u>Run 2 Naive</u> | 22647 ms | 21644 ms | n/A | n/A |
| <u>Run 2 BLAS</u> | 282 ms | 276 ms | 3983 ms | 4073 ms |
| <u>Run 2 Optimized</u> | 974 ms | 1172 ms | 17756 ms | 18911 ms |
| <u>Run 3 Naive</u> | 21860 ms | 21885 ms | n/A | n/A |
| <u>Run 3 BLAS</u> | 283 ms | 277 ms | 4042 ms | 4114 ms |
| <u>Run 3 Optimized</u> | 1016 ms | 1189 ms | 17459 ms | 18965 ms |
| <u>Run 4 Naive</u> | 22703 ms | 19516 ms | n/A | n/A |
| <u>Run 4 BLAS</u> | 309 ms | 310 ms | 4058 ms | 4068 ms |
| <u>Run 4 Optimized</u> | 1237 ms | 1268 ms | 17517 ms | 19279 ms |

> Note: the naive implementation was unable to perform matrix math on 5000x5000 matrices; naive runtime is expected to match >60,000 ms (1 minute).

## Conclusion

All in all, the optimized gemm is certainly better than the naive version by an order of **2200%**. Unfortunately, even after using the specific cache blocks relevant to my CPU, BLAS was still able to perform **300-400%** faster than the optimized version. If I had to guess, this is due to micro-optimizations, assembly code, and further parallelization over decades of implementation testing. Nonetheless, I am still happy with my optimized function.

## How to run yourself

Executables have been added to a release on this github repo. Go to:

`Releases > [Latest] > Download`

Then, you will be able to run the executable from linux using:

```cmd
cd {path to programs}

./gccMM_5000x5000
```

> You are also able to build yourself using `make`. Switch 'all' to gcc/clang depending on your preferred compiler. Custom sizes can be tested by changing `#define N 5000`.

---

A special thank you to professor Rubao Lee! 
