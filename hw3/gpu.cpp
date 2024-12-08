#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

__global__ void heat_transfer_kernel(double* d_current, double* d_next, int size, double coef) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < size - 1) {
        d_next[i] = d_current[i] + coef * (d_current[i + 1] - 2 * d_current[i] + d_current[i - 1]);
    }
}

void compute_on_gpu(double* current, double* next, int local_size, int steps, double coef) {
    double *d_current, *d_next;

    cudaMalloc(&d_current, local_size * sizeof(double));
    cudaMalloc(&d_next, local_size * sizeof(double));

    cudaMemcpy(d_current, current, local_size * sizeof(double), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks = (local_size + threads_per_block - 1) / threads_per_block;

    for (int step = 0; step < steps; ++step) {
        heat_transfer_kernel<<<blocks, threads_per_block>>>(d_current, d_next, local_size, coef);

        std::swap(d_current, d_next);
    }

    cudaMemcpy(current, d_current, local_size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_current);
    cudaFree(d_next);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            std::cerr << "This program requires exactly 2 MPI processes.\n";
        }
        MPI_Finalize();
        return -1;
    }

    double T = 0.1;
    double k = 1.0;
    double h = 0.02;
    double tau = 0.0002;
    int num_points = 50;

    int steps = static_cast<int>(T / tau);
    double coef = k * tau / (h * h);

    int local_size = num_points / 2 + (rank == 1 ? num_points % 2 : 0);
    std::vector<double> current(local_size + 2, 1.0);
    std::vector<double> next(local_size + 2, 0.0);

    if (rank == 0) current[0] = 0.0;
    if (rank == 1) current[local_size + 1] = 0.0;

    compute_on_gpu(current.data(), next.data(), local_size + 2, steps, coef);

    if (rank == 0) {
        MPI_Send(&current[local_size], 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&current[local_size + 1], 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
        MPI_Recv(&current[0], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&current[1], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        std::vector<double> result(num_points + 2, 0.0);
        for (int i = 0; i <= local_size; ++i) {
            result[i] = current[i];
        }

        MPI_Recv(&result[local_size + 1], num_points - local_size, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::cout << "Final temperature distribution:\n";
        for (int i = 1; i <= num_points; ++i) {
            std::cout << std::fixed << std::setprecision(5) << result[i] << " ";
        }
        std::cout << std::endl;
    } else if (rank == 1) {
        MPI_Send(&current[1], local_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
