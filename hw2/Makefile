all: test_cuda

test_openmp: build_omp
	srun -G 1 -N 1 -A proj_1593 run_openmp.out

nvidia_nsight: build
	srun --gres=gpu:1 -G 1 -N 1 -A proj_1593 ncu --log-file main_report.ncu --target-processes all ./main.out

test_cuda: build
	srun -G 1 -N 1 -A proj_1593  main.out

build_omp: openmp.cpp
	g++ -fopenmp openmp.cpp -o run_openmp.out

build: main.cu
	nvcc -lcublas -lineinfo main.cu -o main.out

clean:
	rm *.out
