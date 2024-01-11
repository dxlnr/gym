// clang -O2 -DTILE -march=native -mavx -lpthread gemm.c -o gemm
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

uint64_t get_time() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  return (uint64_t)start.tv_sec*1e9 + (uint64_t)start.tv_nsec;
}

#ifndef NTHREADS
  #define NTHREADS 1
#endif

#ifndef N
  #define N 16
#endif
#define BLOCK 8
#define BLOCK_Y 2
#define BLOCK_X 4

float B[N*N] __attribute__((aligned(64)));
float A[N*N] __attribute__((aligned(64)));
float C[N*N] __attribute__((aligned(64)));
float CREF[N*N] __attribute__((aligned(64)));

float Bf[N*N] __attribute__ ((aligned (64)));

__m256* A256 = (__m256*)A;
__m256* B256 = (__m256*)B;
__m256* C256 = (__m256*)C;
__m256 *Bfm = (__m256*)Bf;

void matmul(int ii, int iN)
{
#ifdef NAIVE
  for (int i = ii; i < iN; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        C[(i*N)+j] += A[(i*N)+k] * B[(k*N)+j];
      }
    }
  }
#elif TILE
  for (int i = ii; i < iN; i += BLOCK) {
    for (int k = 0; k < N; k += BLOCK) {
      for (int j = 0; j < N; j += BLOCK) {

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
#elif TALLS
  for (int i = ii; i < iN; ++i) {
    for (int j = 0; j < N; j += BLOCK) {
      __m256 c_row = _mm256_setzero_ps();

      for (int k = 0; k < N; ++k) {
        __m256 a = _mm256_broadcast_ss(&A[(i*N) + k]);
        __m256 b = _mm256_load_ps(&B[(k*N) + j]);
        c_row = _mm256_fmadd_ps(a, b, c_row);
      }
      _mm256_store_ps(&C[(i*N) + j], c_row);
    }
  }
#elif BTALLS
  for (int i = ii; i < iN; i += BLOCK) {
    for (int j = 0; j < N; j += BLOCK) {

      for (int bi = i; bi < i + BLOCK; ++bi) {
        for (int bj = j; bj < j + BLOCK; bj += 8) {

          __m256 c_row = {};
          for (int k = 0; k < N; ++k) {
            __m256 a = _mm256_broadcast_ss(&A[bi * N + k]);
            __m256 b = _mm256_load_ps(&B[k * N + bj]);
            c_row = _mm256_fmadd_ps(a, b, c_row);
          }
          _mm256_store_ps(&C[bi * N + bj], c_row);
        }
      } 
    }
  }
#elif TINY
  /* float BT[16*16] __attribute__((aligned(sizeof(float) * 16))); */
  /* float AT[16*16] __attribute__((aligned(sizeof(float) * 16))); */
  /* float CT[16*16] __attribute__((aligned(sizeof(float) * 16))); */

  /* float at[16*16] = {1,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7}; */
  /* float bt[16*16] = {0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7}; */
  /* float ct[16*16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; */

  float BT[8*8] __attribute__((aligned(sizeof(float) * 8)));
  float AT[8*8] __attribute__((aligned(sizeof(float) * 8)));
  float CT[8*8] __attribute__((aligned(sizeof(float) * 8)));
  float at[8*8] = {1,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7};
  float bt[8*8] = {0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7};
  float ct[8*8] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  memcpy(AT, at, sizeof(at));
  memcpy(BT, bt, sizeof(bt));
  memcpy(CT, ct, sizeof(ct));

  __m256* CM = (__m256*)CT;

  for (int k = 0; k < BLOCK; ++k) {
    __m256 acc = {};
    for (int bi = 0; bi < BLOCK; ++bi) {
      __m256 ta = _mm256_broadcast_ss(&AT[bi+ k*BLOCK]);
      __m256 b = _mm256_load_ps(&BT[bi*BLOCK]);
      acc = _mm256_fmadd_ps(ta, b, acc);
    }
    CM[k] = acc;
  }

  printf("\n");
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%.0f ", CT[i*N+ j]);
    }
    printf("\n");
  }
  printf("\n\n");
  
#else
  for (int i = ii; i < iN; i += BLOCK_X) {
    for (int j = 0; j < N; j += BLOCK*BLOCK_Y) {

      __m256 a[BLOCK_X][BLOCK_Y] = {};
      for (int k = 0; k < N; ++k) {
        for (int bi = 0; bi < BLOCK_X; ++bi) {
          __m256 ta = _mm256_broadcast_ss(&A[(i+bi)*N + k]);
          for (int bj = 0; bj < BLOCK_Y; ++bj) {
            a[bi][bj] = _mm256_fmadd_ps(ta, Bfm[((j+bj*BLOCK)*N + k*8)/8], a[bi][bj]);
          }
        }
      }
      for (int bi = 0; bi < BLOCK_X; ++bi) {
        for (int bj = 0; bj < BLOCK_Y; ++bj) {
          C256[((i+bi)*N + j + bj*BLOCK)/8] = a[bj][bi];
        }
      }
    }
  }
#endif
}

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
atomic_int tready = 0;
atomic_int tdone = 0;

void *matmul_single_thread(void *n) 
{
  cpu_set_t set;
  int k = (int)(int64_t)n;
  int lb = (int)(uint64_t)n * N/NTHREADS;
  int ub = (int)(uint64_t)(n + 1) * N/NTHREADS;

  CPU_ZERO(&set);
  CPU_SET(k, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);

  tready++;
  pthread_mutex_lock(&lock);
  pthread_mutex_unlock(&lock);

  matmul(lb, ub);
  tdone++;

  return NULL;
}

int main() {
  printf("Running GEMM %dx%d with %d threads.\n",N,N,NTHREADS);
  assert(N%BLOCK == 0);

  FILE *fa = fopen("tests/mat/matA", "rb");
  if (fa == NULL) {
    printf("please create tests/mat/matA file using numpy. Run:\npython gemm.py --save\n");
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

  // preswizzle
  for (int y = 0; y < N; y+=BLOCK) {
    for (int x = 0; x < N; x++) {
      for (int iy = 0; iy < BLOCK; iy++) {
        Bf[y*N + x*BLOCK + iy] = B[(y+iy)*N + x];
      }
    }
  }

#if NTHREADS != 1
  assert(N % NTHREADS == 0 && "N must be divisible by NTHREADS.");
  pthread_t threads[NTHREADS];

  pthread_mutex_lock(&lock);
  for (int i = 0; i < NTHREADS; i++) {
    pthread_create(&threads[i], NULL, matmul_single_thread, (void *)(uint64_t) i);
  }
  while (tready != NTHREADS) usleep(1);
#endif

  uint64_t st = get_time();
#if NTHREADS == 1
  matmul(0, N);
#else
  pthread_mutex_unlock(&lock); // start threads
  while (tdone != NTHREADS) usleep(1);
#endif
  uint64_t et = get_time();
  double dur = (double)(et - st)/1e9;

#if NTHREADS != 1
  for (int i = 0; i < NTHREADS; i++) {
    pthread_join(threads[i], NULL);
  }
#endif

  check_mm(C, CREF, N*N);
#ifdef NAIVE
  printf("(naive) time: %f\n", dur);
#elif TILE
  printf("(tiled) time: %f\n", dur);
#elif TALLS
  printf("(tall-skinny) time: %f\n", dur);
#elif BTALLS
  printf("(Btall-skinny) time: %f\n", dur);
#else
  printf("(vfma) time: %f\n", dur);
#endif
  printf("%.2f GFLOPS. (numpy ~ 95 GFLOPS (single thread)) \n\n", 2.0*N*N*N/dur/1e9);

  return 0;
}
