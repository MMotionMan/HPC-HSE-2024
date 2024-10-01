CC=g++
FLAGS=-fopenmp
BLASS=-lopenblas
MKL=-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm -ldl

all: test_omp

test_omp: build_blass
	./main_arr.o

test_mkl: build_mkl
	./main_mkl_arr.o

build_blass: gemm_with_blass.cpp
	$(CC) $(FLAGS) $(BLASS) gemm_with_blass.cpp -o main_arr.o

build_mkl: gemm_with_mkl.cpp
	$(CC) $(FLAGS) $(MKL) gemm_with_mkl.cpp -o main_mkl_arr.o

clean:
	rm -rf *.o hw1
	rm hw1
