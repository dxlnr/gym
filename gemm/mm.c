// clang -O2 -DTILE -march=native -mavx -lpthread mm.c -o mm
#define _GNU_SOURCE

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <immintrin.h>
#include <unistd.h>
#include <pthread.h>
#include <stdatomic.h>

#ifndef NTHREADS
  #define NTHREADS 1
#endif

#ifndef N
  #define N 1024
#endif

float B[N*N];
float A[N*N];
float C[N*N];
float CREF[N*N];

uint64_t get_time() 
{
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  return (uint64_t)start.tv_sec*1e9 + (uint64_t)start.tv_nsec;
}

void check_mm(float* C, float* CREF, int n) 
{
  for (int i = 0; i < n; ++i) {
    if (fabsf(C[i] - CREF[i]) > 1e-3) {
      printf("error: %d, %f != %f\n", i, C[i], CREF[i]);
      return;
    }
  } 
}

void matmul_kernel(int i, int j, int k, int bsize, int n) {
  for (int bi=i; bi < bsize+i; ++bi) {
    for (int bk=k; bk < bsize+k; ++bk) {
      for (int bj=j; bj < bsize+j; ++bj) {
        C[(bi*n)+bj] += A[(bi*n)+bk] * B[(bk*n)+bj];
      }
    }
  }
}

void matmul(int n, int bsize)
{
  for (int i = 0; i < n; i+=bsize) {
    for (int j = 0; j < n; j+=bsize) {
      for (int k = 0; k < n; k+=bsize) {
        // TILING
        matmul_kernel(i,j,k,bsize, n);
      }
    }
  }
}

int main() {
  int BSIZE = 64;
  printf("Running GEMM %dx%d with %d threads\n",N,N,NTHREADS);
  assert(N%BSIZE == 0);

  FILE *fa = fopen("tests/mat/matA", "rb");
  if (fa == NULL) {
    printf("please create tests/mat/matA file using numpy. Run:\npython mm.py --save\n");
    return -1;
  }
  fread(A, sizeof(float), N*N, fa);
  FILE *fb = fopen("tests/mat/matB", "rb");
  fread(B, sizeof(float), N*N, fb);
  FILE *fc = fopen("tests/mat/matC", "rb");
  fread(CREF, sizeof(float), N*N, fc);
  fclose(fa);
  fclose(fb);
  fclose(fc);

  uint64_t st = get_time();
#if NTHREADS == 1
  matmul(N, BSIZE);
#endif
  uint64_t et = get_time();
  double dur = (double)(et - st)/1e9;
  printf("-------------------------------------------------------------------\n");
  printf("\nPerformance : %.2f GFLOPS (numpy ~ 19 GFLOPS (single thread)) \n", 2.0*N*N*N/dur/1e9);
  check_mm(C, CREF, N*N);

  return 0;
}
