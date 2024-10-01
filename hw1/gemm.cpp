#include <iostream>
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

void print_matrix(double *matrix, int height, int width)
{
    for (int i = 0; i != height; ++i) {
        for (int j = 0; j != width; ++j) {
            std::cout << matrix[i * width + j] << "\t";
        }
        std::cout << "\n";
    }
}

void dgemm(int M, int N, int K, double *A, double *B, double *C)
{
    double sum = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            sum = 0;
            for (int l = 0; l < K; ++l) {
                sum += A[l * M + i] * B[j * K + l];
                
            }
            C[j * M + i] = sum;
        }
    }
}

void dgemm_parallel_and_red(int M, int N, int K, double *A, double *B, double *C)
{
    double sum = 0;
#pragma omp parallel for
    for (int i = 0; i < M; ++i) {
#pragma omp parallel for
        for (int j = 0; j < N; ++j) {
            sum = 0;
#pragma omp parallel for reduction(+:sum)
            for (int l = 0; l < K; ++l) {
                sum += A[l * M + i] * B[j * K + l];
                
            }
            C[j * M + i] = sum;
        }
    }
}

void dgemm_one_parallel(int M, int N, int K, double *A, double *B, double *C)
{
    double sum = 0;
#pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            sum = 0;
            for (int l = 0; l < K; ++l) {
                sum += A[l * M + i] * B[j * K + l];
                
            }
            C[j * M + i] = sum;
        }
    }
}

void all_parallel(int M, int N, int K, double *A, double *B, double *C)
{
    double sum = 0;
#pragma omp parallel for
    for (int i = 0; i < M; ++i) {
#pragma omp parallel for
        for (int j = 0; j < N; ++j) {
            sum = 0;
#pragma omp parallel for
            for (int l = 0; l < K; ++l) {
                sum += A[l * M + i] * B[j * K + l];
                
            }
            C[j * M + i] = sum;
        }
    }
}

void dgemm_with_collapse(int M, int N, int K, double *A, double *B, double *C)
{
    double sum = 0;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            sum = 0;
            for (int l = 0; l < K; ++l) {
                sum += A[l * M + i] * B[j * K + l];
                
            }
            C[j * M + i] = sum;
        }
    }
}

int main()
{
    int threads_count = 3;
    int size_count = 3;
    int N[3] = {500, 1000, 1500};
    int P[size_count] = {1, 2, 4};

    // int m = 1000;
    // int n = 1000;
    // int k = 1000;

    double *A;
    double *B;
    double *C;
    
    for (int matr_size = 0; matr_size != size_count; ++matr_size) {
        int m = N[matr_size];
        int n = N[matr_size];
        int k = N[matr_size];

        A = new double[m * k];
        B = new double[k * n];
        C = new double[m * n];

        std::cout << "\n" << "matrix size: " << N[matr_size] << std::endl;
        for (int threads_num = 0; threads_num != threads_count; ++threads_num) {
            omp_set_num_threads(P[threads_num]);

            std::cout << "\n" << "threads num: " << P[threads_num] << std::endl;

            create_random_matrix(A, m, k);
            create_random_matrix(B, k, n);
            init_zeros_matrix(C, m, n);
            // print_matrix(A, m, k);
            // print_matrix(B, k, n);
            auto start_time = std::chrono::high_resolution_clock::now(); 
            dgemm(m, n, k, A, B, C);
            auto end_time = std::chrono::high_resolution_clock::now(); 
            std::chrono::duration<double> serial_duration = end_time - start_time; 
            std::cout << "dgemm time: " << serial_duration.count() << std::endl;

            start_time = std::chrono::high_resolution_clock::now(); 
            dgemm_parallel_and_red(m, n, k, A, B, C);
            end_time = std::chrono::high_resolution_clock::now(); 
            serial_duration = end_time - start_time; 
            std::cout << "dgemm_openmp_with_red time: " << serial_duration.count() << std::endl;
            
            start_time = std::chrono::high_resolution_clock::now(); 
            dgemm_one_parallel(m, n, k, A, B, C);
            end_time = std::chrono::high_resolution_clock::now(); 
            serial_duration = end_time - start_time; 
            std::cout << "dgemm_one_parallel time: " << serial_duration.count() << std::endl;

            start_time = std::chrono::high_resolution_clock::now(); 
            dgemm_parallel_and_red(m, n, k, A, B, C);
            end_time = std::chrono::high_resolution_clock::now(); 
            serial_duration = end_time - start_time; 
            std::cout << "dgemm_all_parallel time: " << serial_duration.count() << std::endl;

            start_time = std::chrono::high_resolution_clock::now(); 
            dgemm_with_collapse(m, n, k, A, B, C);
            end_time = std::chrono::high_resolution_clock::now(); 
            serial_duration = end_time - start_time; 
            std::cout << "dgemm_with_collapse time: " << serial_duration.count() << std::endl;
        }
        delete[] A;
        delete[] B;
        delete[] C;
    }

    

}