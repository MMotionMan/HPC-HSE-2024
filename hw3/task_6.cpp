#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include "mkl.h"

void solve_heat_equation_2d(int nx, int ny, double T, double alpha, double h, double tau) {
    int size = nx * ny;
    double coef = alpha * tau / (h * h);

    std::vector<int> ia(size + 1, 0);
    std::vector<int> ja;
    std::vector<double> a;

    int index = 0;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            int idx = i * ny + j;
            ia[idx] = index + 1;

            if (i > 0) {
                ja.push_back((i - 1) * ny + j + 1);
                a.push_back(-coef);
                ++index;
            }

            if (j > 0) {
                ja.push_back(i * ny + (j - 1) + 1);
                a.push_back(-coef);
                ++index;
            }

            ja.push_back(idx + 1);
            a.push_back(1 + 4 * coef);
            ++index;

            if (j < ny - 1) {
                ja.push_back(i * ny + (j + 1) + 1);
                a.push_back(-coef);
                ++index;
            }

            if (i < nx - 1) {
                ja.push_back((i + 1) * ny + j + 1);
                a.push_back(-coef);
                ++index;
            }
        }
    }
    ia[size] = index + 1;

    std::vector<double> u(size, 0.0);
    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            u[i * ny + j] = 1.0;
        }
    }

    void* pt[64] = { nullptr };
    MKL_INT maxfct = 1, mnum = 1, mtype = 11, phase = 13, nrhs = 1, msglvl = 0, error = 0;
    MKL_INT n = size;
    std::vector<MKL_INT> iparm(64, 0);
    iparm[0] = 1;
    iparm[1] = 2;

    int time_steps = static_cast<int>(T / tau);
    for (int t = 0; t < time_steps; ++t) {
        phase = 11;
        pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, a.data(), ia.data(), ja.data(),
                nullptr, &nrhs, iparm.data(), &msglvl, nullptr, nullptr, &error);
        if (error != 0) {
            std::cerr << "PARDISO analysis error: " << error << std::endl;
            break;
        }

        phase = 33;
        std::vector<double> b = u;
        pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, a.data(), ia.data(), ja.data(),
                nullptr, &nrhs, iparm.data(), &msglvl, b.data(), u.data(), &error);
        if (error != 0) {
            std::cerr << "PARDISO solve error: " << error << std::endl;
            break;
        }
    }

    phase = -1;
    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, nullptr, ia.data(), ja.data(),
            nullptr, &nrhs, iparm.data(), &msglvl, nullptr, nullptr, &error);

    std::cout << "Final temperature distribution:\n";
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            std::cout << std::fixed << std::setprecision(2) << u[i * ny + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int nx = 10, ny = 10;
    double T = 0.1;
    double alpha = 1.0;
    double h = 0.1;
    double tau = 0.0001;

    solve_heat_equation_2d(nx, ny, T, alpha, h, tau);

    return 0;
}

