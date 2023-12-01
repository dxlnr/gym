// clang -O2 -DTILE -march=native -mavx gemm.c -o gemm
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <immintrin.h>

void check_mm(float* C, float* CREF, int N) 
{
  for (int i = 0; i < N; ++i) {
    if (fabsf(C[i] - CREF[i]) > 1e-3) {
      printf("error: %d, %f != %f\n", i, C[i], CREF[i]);
      return;
    }
  } 
  printf("correct\n");
}

void transpose(float *A, float *B, int N, int M)
{ 
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      B[i*N + j] = A[j*N + i]; 
    }
  }
}

#define N 1024
#define BLOCK 8

float A[N*N] __attribute__((aligned(32)));
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

  clock_t et = clock();
  double dur = (double)(et - st) / CLOCKS_PER_SEC;

  check_mm(C, CREF, N*N);
  printf("(tiled) time: %f\n", dur);
  printf("%.2f GFLOPS. (numpy ~ 95 GFLOPS (single thread)) \n\n", 2.0*N*N*N/dur/1e9);

#else

  clock_t et = clock();
  double dur = (double)(et - st) / CLOCKS_PER_SEC;

  printf("(vfma) time: %f\n", dur);
  printf("%.2f GFLOPS.\n\n", 2.0*N*N*N/dur/1e9);
#endif

  return 0;
}
