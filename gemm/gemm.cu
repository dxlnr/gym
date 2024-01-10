include <stdio.h>

__global__ void matmul(float *A, float *B, float *C, int N) {}

#ifndef N
  #define N 1024
#endif

int main() {
  printf("Running GEMM %dx%d on CUDA.\n",N,N);
  assert(N%BLOCK == 0);

  size_t bytes = N * N * sizeof(int);
  int *A, *B, *C;

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
  
  // cudaMalloc(&A, bytes);
  // cudaMalloc(&B, bytes);
  // cudaMalloc(&C, bytes);
  return 0;
}
