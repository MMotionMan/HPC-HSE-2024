CC=g++-14
FLAGS=-std=c++20 -fopenmp

all: main_arr

main_arr: main_arr.o
	$(CC) $(FLAGS) main_arr.cpp -o main_arr.o
	./main_arr.o
clean:
	rm -rf *.o hw1
	rm hw1