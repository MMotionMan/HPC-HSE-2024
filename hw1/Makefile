CC=g++
FLAGS=-fopenmp
BLASS=-lopenblas
MKLROOT=/opt/intel/mkl
MKL=-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
MKL_INCLUDE=-I$(MKLROOT)/include
MKL_LINK=-L$(MKLROOT)/lib/intel64

all: test_omp

test_omp: build_blass
	./main_arr.o

test_mkl: build_mkl
	./main_mkl_arr.o

build_blass: gemm_with_blass.cpp
	$(CC) $(FLAGS) $(BLASS) gemm_with_blass.cpp -o main_arr.o

build_mkl: gemm_with_mkl.cpp
	$(CC) $(FLAGS) $(MKL_INCLUDE) $(MKL_LINK) $(MKL) gemm_with_mkl.cpp -o main_mkl_arr.o

clean:
	rm -rf *.o hw1
	rm hw1
