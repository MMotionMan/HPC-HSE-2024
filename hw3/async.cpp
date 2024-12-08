#include <mpi.h>
#include <iostream>
#include <cmath>
#include <iomanip>
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

    int part_size = num_points / size + (rank < num_points % size ? 1 : 0);
    int start_idx = rank * (num_points / size) + std::min(rank, num_points % size);

    double* part = new double[part_size + 2];
    double* part_next = new double[part_size + 2];
    double* rod = nullptr;

    if (rank == 0) {
        rod = new double[num_points + 2];
        rod[0] = 0.0;
        rod[num_points + 1] = 0.0;

        for (int i = 1; i <= num_points; ++i) {
            rod[i] = 0.0;
        }

        for (int j = 1; j < size; ++j){
            int offset = j * (num_points / size) + std::min(j, num_points % size);
            int size_to_send = num_points / size + (j < num_points % size ? 1 : 0);
            MPI_Send(&rod[offset], size_to_send + 2, MPI_DOUBLE, j, 0, MPI_COMM_WORLD);
        }

        for (int i = 0; i < part_size + 2; ++i) {
            part[i] = rod[start_idx + i];
        }
    } else {
        MPI_Recv(part, part_size + 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int steps = (int)(T / tau);
    double coef = k * tau / (h * h);

    for (int t = 0; t < steps; ++t) {
        MPI_Request requests[4];
        int request_count = 0;

        if (rank > 0) {
            MPI_Irecv(&part[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[request_count++]);
        }
        if (rank < size - 1) {
            MPI_Irecv(&part[part_size + 1], 1, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &requests[request_count++]);
        }

        if (rank > 0) {
            MPI_Isend(&part[1], 1, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &requests[request_count++]);
        }
        if (rank < size - 1) {
            MPI_Isend(&part[part_size], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[request_count++]);
        }

        MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

        for (int i = 1; i <= part_size; ++i) {
            part_next[i] = part[i] + coef * (part[i + 1] - 2 * part[i] + part[i - 1]);
        }

        std::swap(part, part_next);
    }

    if (rank == 0) {
        for (int i = 1; i <= part_size; ++i) {
            rod[start_idx + i - 1] = part[i];
        }

        for (int j = 1; j < size; ++j) {
            int offset = j * (num_points / size) + std::min(j, num_points % size);
            int size_to_recv = num_points / size + (j < num_points % size ? 1 : 0);
            MPI_Recv(&rod[offset], size_to_recv, MPI_DOUBLE, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }       
    } else {
        MPI_Send(&part[1], part_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    delete[] part;
    delete[] part_next;

    return rod;
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
            //output_result(rod, num_points, rank);
        }
    }
    MPI_Finalize();
    return 0;
}
