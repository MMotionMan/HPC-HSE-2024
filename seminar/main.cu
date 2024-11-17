#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#define N (1024)
#define THREADS_PER_BLOCK 512
__global__ void get_f_x_gpu (float *dA)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = 2.0f * 3.1415926f * (float) idx / (float) N;
    dA[idx] = sinf(exp(x));
}

int main (int argc, char *argv[])
{
    float *hA, *dA;
    hA = (float*) malloc (N * sizeof(float));
    cudaMalloc((void**) &dA, N * sizeof(float));
    get_f_x_gpu
    <<< N/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( dA );
    cudaMemcpy(hA, dA, N * sizeof(float), cudaMemcpyDeviceToHost);
    free(hA);
    cudaFree(dA);
    return 0;
}
