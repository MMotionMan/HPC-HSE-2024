CC=g++
FLAGS=-lopenblas -fopenmp

all: test_omp

test_omp: build
	./main_arr.o

build: gemm.cpp
	$(CC) $(FLAGS) gemm.cpp -o main_arr.o

clean:
	rm -rf *.o hw1
	rm hw1