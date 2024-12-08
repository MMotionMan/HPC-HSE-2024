all: test_main

test_main: build_async build_collective build_parallel
	srun -N 2 --ntasks-per-node=1 -A proj_1593 ./async_realization.out
	srun -N 2 --ntasks-per-node=1 -A proj_1593 ./collective_realization.out
	srun -N 2 --ntasks-per-node=1 -A proj_1593 ./parallel_realization.out

build_async: async.cpp
	mpicxx async.cpp -o async_realization.out

build_collective: collective.cpp
	mpicxx collective.cpp -o collective_realization.out

build_parallel: parallel.cpp
	mpicxx parallel.cpp -o parallel_realization.out

clean:
	rm *.out
