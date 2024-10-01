#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

void blas_dgemm(size_t M, size_t N, size_t K,
    const std::vector<std::vector<double> >& A,
    const std::vector<std::vector<double> >& B,
    std::vector<std::vector<double> >& C)
{
    for (size_t i = 0; i != M; ++i) {
        for (size_t j = 0; j != M; ++j) {
            for (size_t k = 0; k != M; ++k) {
                C[i][j] += A[i][k] * B[j][k];
            }
        }
    }
}

void blas_dgemm_parallel_and_red(size_t M, size_t N, size_t K,
    const std::vector<std::vector<double> >& A,
    const std::vector<std::vector<double> >& B,
    std::vector<std::vector<double> >& C)
{
    double sum = 0;
#pragma omp parallel for
    for (size_t i = 0; i != M; ++i) {
#pragma omp parallel for
        for (size_t j = 0; j != N; ++j) {
            sum = 0;
#pragma omp parallel for reduction(+:sum)
            for (size_t k = 0; k != K; ++k) {
                sum += A[i][k] * B[j][k];
            }
            C[i][j] = sum;
        }
    }
}

void blas_dgemm_one_parallel(size_t M, size_t N, size_t K,
    const std::vector<std::vector<double> >& A,
    const std::vector<std::vector<double> >& B,
    std::vector<std::vector<double> >& C)
{
    double sum = 0;
#pragma omp parallel for
    for (size_t i = 0; i != M; ++i) {
        for (size_t j = 0; j != N; ++j) {
            sum = 0;
            for (size_t k = 0; k != K; ++k) {
                sum += A[i][k] * B[j][k];
            }
            C[i][j] = sum;
        }
    }
}

void blas_dgemm_all_parallel(size_t M, size_t N, size_t K,
    const std::vector<std::vector<double> >& A,
    const std::vector<std::vector<double> >& B,
    std::vector<std::vector<double> >& C)
{
    double sum = 0;
#pragma omp parallel for
    for (size_t i = 0; i != M; ++i) {
#pragma omp parallel for
        for (size_t j = 0; j != N; ++j) {
            sum = 0;
#pragma omp parallel for
            for (size_t k = 0; k != K; ++k) {
                sum += A[i][k] * B[j][k];
            }
            C[i][j] = sum;
        }
    }
}

void blas_dgemm_with_collapse(size_t M, size_t N, size_t K,
    const std::vector<std::vector<double> >& A,
    const std::vector<std::vector<double> >& B,
    std::vector<std::vector<double> >& C)
{
    double sum = 0;
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i != M; ++i) {
        for (size_t j = 0; j != N; ++j) {
            sum = 0;
            for (size_t k = 0; k != K; ++k) {
                sum += A[i][k] * B[j][k];
            }
            C[i][j] = sum;
        }
    }
}

void print(const std::vector<std::vector<double> >& A, int m) {
    for (size_t i = 0; i != m; ++i) {
        for (size_t j = 0; j != m; ++j) {
            std::cout << A[i][j] << "\t";
        }
        std::cout << "\n";
    }
}

int main()
{
    size_t m, n, k;
    std::cin >> m >> n >> k;
    std::vector<std::vector<double> > A(m, std::vector<double>(n));
    std::vector<std::vector<double> > B(n, std::vector<double>(k));
    std::vector<std::vector<double> > C(m, std::vector<double>(k));
    for (size_t i = 0; i != m; ++i) {
        for (size_t j = 0; j != n; ++j) {
            A[i][j] = i + j;
        }
    }

    for (size_t i = 0; i != n; ++i) {
        for(size_t j = 0; j != k; ++j) {
            B[j][i] = i + i;
        }
    }
    auto start_time = std::chrono::high_resolution_clock::now(); 
    blas_dgemm(m, n, k, A, B, C);
    auto end_time = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double> serial_duration = end_time - start_time; 
    std::cout << "blas_dgemm time: " << serial_duration.count() << std::endl;

    start_time = std::chrono::high_resolution_clock::now(); 
    blas_dgemm_parallel_and_red(m, n, k, A, B, C);
    end_time = std::chrono::high_resolution_clock::now(); 
    serial_duration = end_time - start_time; 
    std::cout << "blas_dgemm_openmp_with_red time: " << serial_duration.count() << std::endl;
    
    start_time = std::chrono::high_resolution_clock::now(); 
    blas_dgemm_one_parallel(m, n, k, A, B, C);
    end_time = std::chrono::high_resolution_clock::now(); 
    serial_duration = end_time - start_time; 
    std::cout << "blas_dgemm_one_parallel time: " << serial_duration.count() << std::endl;

    start_time = std::chrono::high_resolution_clock::now(); 
    blas_dgemm_parallel_and_red(m, n, k, A, B, C);
    end_time = std::chrono::high_resolution_clock::now(); 
    serial_duration = end_time - start_time; 
    std::cout << "blas_dgemm_all_parallel time: " << serial_duration.count() << std::endl;

    start_time = std::chrono::high_resolution_clock::now(); 
    blas_dgemm_with_collapse(m, n, k, A, B, C);
    end_time = std::chrono::high_resolution_clock::now(); 
    serial_duration = end_time - start_time; 
    std::cout << "blas_dgemm_with_collapse time: " << serial_duration.count() << std::endl;
}