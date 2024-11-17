#include <iostream>
#include <cstdlib>
#include <omp.h>

#define N 1024

void generateRandomMatrix(float* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
	matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    } 
}

void matrixMultiplyOpenMP(float* A, float* B, float* C, int n) {
    #pragma omp target data map(to: A[0:n*n], B[0:n*n]) map(from: C[0:n*n])
    {
	#pragma omp target teams distribute parallel for collapse(2)
	for (int i = 0; i < n; i++) {
	    for (int j = 0; j < n; j++) {
	        float sum = 0.0f;
		for (int k = 0; k < n; k++) {
		    sum += A[i * n + k] * B[k * n + j];
		}
		C[i * n + j] = sum;
	    }
	}
    }
}

int main() {
    int matrixSize = N * N;

    float* A = new float[matrixSize];
    float* B = new float[matrixSize];
    float* C = new float[matrixSize];

    generateRandomMatrix(A, N);
    generateRandomMatrix(B, N);

    double start = omp_get_wtime();
    matrixMultiplyOpenMP(A, B, C, N);
    double end = omp_get_wtime();

    std::cout << "Результат умножения матриц(частично)" << std::endl;
    for (int i = 0; i < 4; i++) {
	for (int j = 0; j < 4; j++) {
	    std::cout << C[i * N + j] << " ";
	}
	std::cout << std::endl;
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
