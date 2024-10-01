#include <iostream>
#include "mkl.h"
#include <omp.h>
#include <chrono>

void create_random_matrix(double *matrix, int height, int width)
{
    for (int i = 0; i != height; ++i)
        for (int j = 0; j != width; ++j)
            matrix[i * width + j] = i + j + 1;
}

void init_zeros_matrix(double *matrix, int height, int width)
{
    for (int i = 0; i != height; ++i)
        for (int j = 0; j != width; ++j)
            matrix[i * width + j] = 0;
}

int main()
{
    int size_count = 3;
    int N[size_count] = {500, 1000, 1500};
    // int m = 1000;
    // int n = 1000;
    // int k = 1000;
    
    for (int matr_size = 0; matr_size != size_count; ++matr_size) {
	int m = N[matr_size];
        int n = N[matr_size];
        int k = N[matr_size];
	
	double* A = (double*)mkl_malloc(m * k * sizeof(double), 64);
        double* B = (double*)mkl_malloc(k * n * sizeof(double), 64);
        double* C = (double*)mkl_malloc(m * n * sizeof(double), 64);	
	
		

        auto start_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> serial_duration;

        std::cout << "\n" << "matrix size: " << N[matr_size] << std::endl;
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, m, B, k, 0.0, C, m);
	auto end_time = std::chrono::high_resolution_clock::now();        
        serial_duration = end_time - start_time; 
        std::cout << "mkl cblass dgemm time: " << serial_duration.count() << std::endl;

        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
    }
}
