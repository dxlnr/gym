#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define N 96

float A[N*N];
float B[N*N];
float C[N*N];

int main() {

  for (int i = 0; i < N*N; i++) {
    A[i] = (float)rand()/(float)(RAND_MAX/1.0);
    B[i] = (float)rand()/(float)(RAND_MAX/1.0);
  }

  clock_t st = clock();

  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++)
      for (int j = 0; j < N; j++)
        C[i*N+j] += A[i*N+k] * B[k*N+j];
  }

  clock_t et = clock();
  double dur = (double)(et - st) / CLOCKS_PER_SEC;

  printf("time: %f\n", dur);
  printf("%f GFLOPS.\n", 2.0*N*N*N/dur/1e9);
  return 0;
}
