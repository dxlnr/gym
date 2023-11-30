#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <immintrin.h>

void simple_gemm() {
  float D[2][2];
  D[0][0] = 1;
  D[0][1] = 2;
  D[1][0] = 3;
  D[1][1] = 4;

  float E[2][2];
  E[0][0] = 1;
  E[0][1] = 2;
  E[1][0] = 3;
  E[1][1] = 4;

  float F[2][2];

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      float sum = 0;
      for (int k = 0; k < 2; k++)
        sum += D[i][k] * E[k][j];
      F[i][j] = sum;
    }
  }
  printf("%.0f %.0f\n", F[0][0], F[0][1]);
  printf("%.0f %.0f\n", F[1][0], F[1][1]);
}

#define N 1024
#define BLOCK 8

float A[N][N] __attribute__((aligned(16)));
float B[N][N] __attribute__((aligned(16)));
float C[N][N] __attribute__((aligned(16)));

__m256* A256 = (__m256*)A;
__m256* B256 = (__m256*)B;
__m256* C256 = (__m256*)C;

int main() {
  assert(N%BLOCK == 0);

  clock_t st = clock();

  // only storage
  /* for (int i = 0; i < N; ++i) { */
  /*   for (int j = 0; j < N; ++j) { */
  /*     C[i][j] = i; */
  /*   } */
  /* } */

  for (int bi = 0; bi < N; bi += BLOCK) {
    for (int bj = 0; bj < N; bj += BLOCK) {
      for (int i = bi; i < bi + BLOCK; ++i) {
        for (int j = bj; j < bj + BLOCK; ++j) {
          float sum = 0;
          for (int k = 0; k < N; k++)
            sum += A[i][k] * B[k][j];
          C[i][j] = sum;
        }
      }
    }
  }

  clock_t et = clock();
  double dur = (double)(et - st) / CLOCKS_PER_SEC;

  printf("(naive) time: %f\n", dur);
  printf("%.2f GFLOPS.\n", 2.0*N*N*N/dur/1e9);

  clock_t st1 = clock();

  for (int bi = 0; bi < N; bi += BLOCK) {
    for (int bj = 0; bj < N; bj += BLOCK) {

      __m256 t[BLOCK];
      for (int i = 0; i < BLOCK; i++) {
        __m256 s = _mm256_setzero_ps();
        for (int k = 0; k < N; k += 8) {
          s = _mm256_fmadd_ps(A256[((bi+i)*N + k)/8], B256[(bj*N + k)/8], s);
        }
        t[i] = s;
      }

      for (int j = 0; j < BLOCK; j++) {
        C256[((bi+j)*N)/8] = t[j];
      }
    }
  }

  clock_t et1 = clock();
  double dur1 = (double)(et1 - st1) / CLOCKS_PER_SEC;

  printf("(vfma) time: %f\n", dur1);
  printf("%.2f GFLOPS.\n", 2.0*N*N*N/dur1/1e9);
  return 0;
}
