// nvcc gemm.c -o gemmcu

#include <stdio.h>
#include <assert.h>
#include <stdint.h>

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

uint64_t get_time() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  return (uint64_t)start.tv_sec*1e9 + (uint64_t)start.tv_nsec;
}

#define SMEM (16*16)
#define BLOCK 32

#ifndef N
  #define N 1024
#endif

float CREF[N*N] __attribute__((aligned(64)));

__global__ void matmul(float *a, float *b, float *c, int n) {
  __shared__ float AA[SMEM];
  __shared__ float BB[SMEM];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int dim = blockDim.x;
  int dk = (n + dim - 1) / dim;
  int ty = threadIdx.y, tx = threadIdx.x;
  
  float acc = 0.0;
  for (int i=0; i<dk; i++)
  {
    AA[ty * dim + tx] = a[(row * n) + (i*dim) + tx];
    BB[ty * dim + tx] = b[(i * dim * n) + (ty * n) + col];
    __syncthreads();

    for (int k=0; k<dim; ++k) 
    {
      acc += AA[ty*dim + k] * BB[k*dim + tx];
    }
    __syncthreads();
  }
  c[row*n + col] = acc;
}

int main() {
  printf("Running GEMM %dx%d on CUDA.\n",N,N);
  assert(N%BLOCK == 0);

  size_t bytes = N * N * sizeof(int);
  float *A, *B, *C;

  cudaMallocManaged(&A, bytes);
  cudaMallocManaged(&B, bytes);
  cudaMallocManaged(&C, bytes);

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
  
  int threads = 16;
  int blocks = (N + threads - 1) / threads;
  dim3 BLOCKS(blocks, blocks);
  dim3 THREADS(threads, threads);

  uint64_t st = get_time();

  matmul<<<BLOCKS, THREADS>>>(A,B,C,N);
  cudaDeviceSynchronize();

  uint64_t et = get_time();
  double dur = (double)(et - st)/1e9;

  check_mm(C, CREF, N*N);
  printf("%.2f GFLOPS. (numpy ~ 95 GFLOPS (single thread)) \n\n", 2.0*N*N*N/dur/1e9);

  return 0;
}
