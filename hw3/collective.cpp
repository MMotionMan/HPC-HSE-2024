#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

double* initialize_rod(int num_points, int rank, int size) {
    int part_size = num_points / size + (rank < num_points % size ? 1 : 0);
    double* part = new double[part_size + 2];

    for (int i = 0; i < part_size + 2; ++i) {
        if (rank == 0 && i == 0){
            part[i] == 0.0;
        } else if (rank == size - 1 && i == part_size + 1) {
            part[i] = 0.0;
        } else {
            part[i] = 1.0;
        }
    }
    return part;
}

void output_result(double *rod, int num_points, int rank) {
    if (rank == 0) {
        std::cout << "Final temperature distribution: " << std::endl;
        for (int i = 0; i < num_points; ++i) {
            std::cout << std::fixed << std::setprecision(5) << rod[i] << " ";
        }
    }
}

double* compute(double T, double k, double h, double tau, int num_points) {
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    int part_size = num_points / size;
    int remainder = num_points % size;

    int offset = 0;
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = part_size + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int local_size = sendcounts[rank];
    double* local_part = new double[local_size + 2];
    double* local_next = new double[local_size + 2];

    double* rod = nullptr;
    if (rank == 0) {
        rod = new double[num_points + 2];
        rod[0] = 0.0;
        rod[num_points + 1] = 0.0;
        for (int i = 1; i <= num_points; ++i) {
            rod[i] = 1.0;
        }
    }

    MPI_Scatterv(rod + 1, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_part + 1, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    local_part[0] = (rank == 0) ? 0.0 : 0.0;
    local_part[local_size + 1] = (rank == size - 1) ? 0.0 : 0.0;

    int steps = static_cast<int>(T / tau);
    double coef = k * tau / (h * h);

    for (int t = 0; t < steps; ++t) {
        for (int i = 1; i <= local_size; ++i) {
            local_next[i] = local_part[i] + coef * (local_part[i + 1] - 2 * local_part[i] + local_part[i - 1]);
        }

        std::swap(local_part, local_next);
    }

    MPI_Gatherv(local_part + 1, local_size, MPI_DOUBLE,
                rod + 1, sendcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    delete[] local_part;
    delete[] local_next;

    if (rank == 0) {
        return rod;
    } else {
        return nullptr;
    }
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);

    int rank, size;
    double start, end;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double T = 0.1;
    double k = 1.0;
    double h = 0.02;
    double tau = 0.0002;
    int num_points[3]{10000, 25000, 50000};

    for (auto N : num_points) {
        start = MPI_Wtime();
        double* rod = compute(T, k, h, tau, N);
        end = MPI_Wtime();
        if (rank == 0){
            std::cout << "Runtime for " << N << "nodes = " << end - start << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
