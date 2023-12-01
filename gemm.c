// clang -O2 -DTILE -march=native -mavx gemm.c -o gemm
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <immintrin.h>

void check_mm(float* C, float* CREF, int N) {
  for (int i = 0; i < N; ++i) {
    if (fabsf(C[i] - CREF[i]) > 1e-3) {
      printf("error: %d, %f != %f\n", i, C[i], CREF[i]);
      return;
    }
  } 
  printf("correct\n");
}

#define N 1024
#define BLOCK 8

float A[N*N];
float B[N*N] __attribute__((aligned(32)));
float C[N*N] __attribute__((aligned(32)));
float CREF[N*N] __attribute__((aligned(32)));

__m256* A256 = (__m256*)A;
__m256* B256 = (__m256*)B;
__m256* C256 = (__m256*)C;

int main() {
  assert(N%BLOCK == 0);

  FILE *fa = fopen("tests/mat/matA", "rb");
  fread(A, sizeof(float), N*N, fa);
  FILE *fb = fopen("tests/mat/matB", "rb");
  fread(B, sizeof(float), N*N, fb);
  FILE *fc = fopen("tests/mat/matC", "rb");
  fread(CREF, sizeof(float), N*N, fc);
  fclose(fa);
  fclose(fb);
  fclose(fc);

  clock_t st = clock();

#ifdef NAIVE
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        C[(i*N)+j] += A[(i*N)+k] * B[(k*N)+j];
      }
    }
  }

  clock_t et = clock();
  double dur = (double)(et - st) / CLOCKS_PER_SEC;

  check_mm(C, CREF, N*N);
  printf("(tiled) time: %f\n", dur);
  printf("%.2f GFLOPS. (numpy ~ 95 GFLOPS (single thread)) \n\n", 2.0*N*N*N/dur/1e9);

#elif TILE
  clock_t st2 = clock();
  
  for (int i = 0; i < N; i += BLOCK) {
    for (int j = 0; j < N; j += BLOCK) {
      for (int k = 0; k < N; k += BLOCK) {

        for (int bi = 0; bi < BLOCK; ++bi) {
          for (int bj = 0; bj < BLOCK; ++bj) {
            for (int bk = 0; bk < BLOCK; ++bk) {
              C[((i+bi)*N)+(j+bj)] += A[((i+bi)*N)+(k+bk)] * B[((k+bk)*N)+(j+bj)];
            }
          }
        }
      }
    }
  }

  clock_t et2 = clock();
  double dur2 = (double)(et2 - st2) / CLOCKS_PER_SEC;

  check_mm(C, CREF, N*N);
  printf("(tiled) time: %f\n", dur2);
  printf("%.2f GFLOPS. (numpy ~ 95 GFLOPS (single thread)) \n\n", 2.0*N*N*N/dur2/1e9);

#else
  clock_t st1 = clock();

  /* int s = 8; */

  /* for (int i = 0; i < N; i += BLOCK) { */
  /*   for (int j = 0; j < N; j += BLOCK) { */
  /*     for (int k = 0; k < N; k += BLOCK) { */

  /*       for (int bi = 0; bi < BLOCK; ++bi) { */
  /*         for (int bj = 0; bj < BLOCK; ++bj) { */
  /*           __m256 s = _mm256_setzero_ps(); */
  /*           for (int bk = 0; bk < BLOCK; bk += 8) { */
  /*             s = _mm256_fmadd_ps(A256[((bi+i)*N + k+bk)/8], B256[(bj+j)*N/8 + k+bk/8], s); */
  /*           } */
  /*           C256[((bi+i)*N + j+bj)/8] = s; */
  /*         } */
  /*       } */

  /*     } */
  /*   } */
  /* } */


  float D[8][8];
  D[0][0] = 1;
  D[0][1] = 2;
  D[0][2] = 3;
  D[0][3] = 4;
  D[0][4] = 5;
  D[0][5] = 6;
  D[0][6] = 7;
  D[0][7] = 8;
  D[1][0] = 1;
  D[1][1] = 2;
  D[1][2] = 3;
  D[1][3] = 4;
  D[1][4] = 5;
  D[1][5] = 6;
  D[1][6] = 7;
  D[1][7] = 8;
  D[2][0] = 1;
  D[2][1] = 2;
  D[2][2] = 3;
  D[2][3] = 4;
  D[2][4] = 5;
  D[2][5] = 6;
  D[2][6] = 7;
  D[2][7] = 8;
  D[3][0] = 1;
  D[3][1] = 2;
  D[3][2] = 3;
  D[3][3] = 4;
  D[3][4] = 5;
  D[3][5] = 6;
  D[3][6] = 7;
  D[3][7] = 8;
  D[4][0] = 1;
  D[4][1] = 2;
  D[4][2] = 3;
  D[4][3] = 4;
  D[4][4] = 5;
  D[4][5] = 6;
  D[4][6] = 7;
  D[4][7] = 8;
  D[5][0] = 1;
  D[5][1] = 2;
  D[5][2] = 3;
  D[5][3] = 4;
  D[5][4] = 5;
  D[5][5] = 6;
  D[5][6] = 7;
  D[5][7] = 8;
  D[6][0] = 1;
  D[6][1] = 2;
  D[6][2] = 3;
  D[6][3] = 4;
  D[6][4] = 5;
  D[6][5] = 6;
  D[6][6] = 7;
  D[6][7] = 8;
  D[7][0] = 2;
  D[7][1] = 3;
  D[7][2] = 4;
  D[7][3] = 5;
  D[7][4] = 6;
  D[7][5] = 7;
  D[7][6] = 8;
  D[7][7] = 9;

  float E[8][8];
  E[0][0] = 1;
  E[0][1] = 2;
  E[0][2] = 3;
  E[0][3] = 4;
  E[0][4] = 5;
  E[0][5] = 6;
  E[0][6] = 7;
  E[0][7] = 8;
  E[1][0] = 1;
  E[1][1] = 2;
  E[1][2] = 3;
  E[1][3] = 4;
  E[1][4] = 5;
  E[1][5] = 6;
  E[1][6] = 7;
  E[1][7] = 8;
  E[2][0] = 1;
  E[2][1] = 2;
  E[2][2] = 3;
  E[2][3] = 4;
  E[2][4] = 5;
  E[2][5] = 6;
  E[2][6] = 7;
  E[2][7] = 8;
  E[3][0] = 1;
  E[3][1] = 2;
  E[3][2] = 3;
  E[3][3] = 4;
  E[3][4] = 5;
  E[3][5] = 6;
  E[3][6] = 7;
  E[3][7] = 8;
  E[4][0] = 1;
  E[4][1] = 2;
  E[4][2] = 3;
  E[4][3] = 4;
  E[4][4] = 5;
  E[4][5] = 6;
  E[4][6] = 7;
  E[4][7] = 8;
  E[5][0] = 1;
  E[5][1] = 2;
  E[5][2] = 3;
  E[5][3] = 4;
  E[5][4] = 5;
  E[5][5] = 6;
  E[5][6] = 7;
  E[5][7] = 8;
  E[6][0] = 1;
  E[6][1] = 2;
  E[6][2] = 3;
  E[6][3] = 4;
  E[6][4] = 5;
  E[6][5] = 6;
  E[6][6] = 7;
  E[6][7] = 8;
  E[7][0] = 2;
  E[7][1] = 3;
  E[7][2] = 4;
  E[7][3] = 5;
  E[7][4] = 6;
  E[7][5] = 7;
  E[7][6] = 8;
  E[7][7] = 9;


  float TE[1][8];
  TE[0][0] = 1;
  TE[0][1] = 1;
  TE[0][2] = 1;
  TE[0][3] = 1;
  TE[0][4] = 1;
  TE[0][5] = 1;
  TE[0][6] = 1;
  TE[0][7] = 1;

  __m256* DV = (__m256*)D;
  __m256* EV = (__m256*)E;
  __m256* TEV = (__m256*)TE;

  /* float F[2][2]; */

  /* for (int i = 0; i < 2; ++i) { */
  /*   for (int j = 0; j < 2; ++j) { */
  /*     float sum = 0; */
  /*     for (int k = 0; k < 2; k++) */
  /*       sum += D[i][k] * E[k][j]; */
  /*     F[i][j] = sum; */
  /*   } */
  /* } */

  printf("%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n", DV[0][0], DV[0][1], DV[0][2], DV[0][3], DV[0][4], DV[0][5], DV[0][6], DV[0][7]);
  __m256 s = _mm256_setzero_ps();
  s = _mm256_fmadd_ps(DV[0], EV[0], s);
  printf("%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n", s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);

  float res = 0;
  for (int i = 0; i < 8; ++i) {
    res += s[i];
  }
  printf("%.0f\n", res);

  for (int i = 0; i < N; i += BLOCK) {
    for (int j = 0; j < N; j += BLOCK) {
      for (int k = 0; k < N; k += BLOCK) {
        for (int bi = 0; bi < BLOCK; ++bi) {
          for (int bj = 0; bj < BLOCK; ++bj) {
            __m256 s = _mm256_setzero_ps();
            for (int bk = 0; bk < BLOCK; bk += 8) {
              s = _mm256_fmadd_ps(A256[((bi+i)*N + k+bk)/8], B256[(bj+j)*N/8 + k+bk/8], s);
            }
            C256[((bi+i)*N + j+bj)/8] = s;
          }
        }
      }
    }
  }

  /*     __m256 t[BLOCK]; */
  /*     for (int i = 0; i < BLOCK; i++) { */
  /*       __m256 s = _mm256_setzero_ps(); */
  /*       for (int k = 0; k < N; k += 8) { */
  /*         s = _mm256_fmadd_ps(A256[((bi+i)*N + k)/8], B256[(bj*N + k)/8], s); */
  /*       } */
  /*       t[i] = s; */
  /*     } */

  /*     for (int j = 0; j < BLOCK; j++) { */
  /*       C256[((bi+j)*N)/8] = t[j]; */
  /*     } */
    /* } */
  /* } */

  clock_t et1 = clock();
  double dur1 = (double)(et1 - st1) / CLOCKS_PER_SEC;

  /* for (int i = 0; i < N; ++i) { */
  /*   for (int j = 0; j < N; ++j) { */
  /*     if (fabsf(C256[i][j] - CREF[i][j]) > 1e-5) { */
  /*       printf("error: %d %d\n", i, j); */
  /*       return 1; */
  /*     } */
  /*   } */
  /* } */ 
  /* printf("correct\n"); */

  printf("(vfma) time: %f\n", dur1);
  printf("%.2f GFLOPS.\n\n", 2.0*N*N*N/dur1/1e9);
#endif

  return 0;
}
