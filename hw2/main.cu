#include <iostream>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 16
#define BLOCK_SIZE_WITH_PADDING (BLOCK_SIZE + 1)
const int N = 1024;

void generateMatrixVector(std::vector<float>& mat, int size) {
    for (int i = 0; i < size * size; ++i)
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
}

void generateMatrixArray(float* mat, int size) {
    for (int i = 0; i < size * size; ++i) {
	mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

__global__ void matrixMulKernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matrixMulKernelAsync(const float* A, const float* B, float* C, int N, int blockSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < blockSize && col < blockSize) {
	int globalRow = blockIdx.z * blockSize + row;
	int globalCol = blockIdx.z * blockSize + col;

	if (globalRow < N && globalCol < N) {
	    float sum = 0.0f;
	    for (int k = 0; k < N; ++k) {
		sum += A[globalRow * N + k] * B[k * N + globalCol];
	    }
	    C[globalRow * N + globalCol] = sum;
	}
    }
}

void matrixMultiplyOpenMP(float* A, float* B, float* C, int n) {
    #pragma omp target data map(to: A[0:n*n], B[0:n*n]) map(from: C[0:n*n])
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int k = 0; k < n; k++) {
                    sum += A[i * n + k] * B[k * n + j];
                }
                C[i * n + j] = sum;
            }
        }
    }
}

__global__ void matrixMulSharedKernel(const float* A, const float* B, float* C, int N) {
    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int blockIdx = 0; blockIdx < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++blockIdx)  {
	if (row < N && (blockIdx * BLOCK_SIZE + threadIdx.x) < N) {
	    sharedA[threadIdx.y][threadIdx.x] = A[row * N + blockIdx * BLOCK_SIZE + threadIdx.x];
	} else {
	    sharedA[threadIdx.y][threadIdx.x] = 0.0f;
	}

	if (col < N && (blockIdx * BLOCK_SIZE + threadIdx.y) < N) {
	    sharedB[threadIdx.y][threadIdx.x] = B[(blockIdx * BLOCK_SIZE + threadIdx.y) * N + col];
	} else {
	    sharedB[threadIdx.y][threadIdx.x] = 0.0f;
	}

	__syncthreads();

	for (int k = 0; k < BLOCK_SIZE; ++k) {
	    sum += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
	}

	__syncthreads();

	if (row < N && col < N) {
	    C[row * N +col] = sum;
	}
    }
}

__global__ void matrixMulSharedKernelWithPadding(const float* A, const float* B, float* C, int N) {
    __shared__ float sharedA[BLOCK_SIZE_WITH_PADDING][BLOCK_SIZE_WITH_PADDING];
    __shared__ float sharedB[BLOCK_SIZE_WITH_PADDING][BLOCK_SIZE_WITH_PADDING];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int blockIdx = 0; blockIdx < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++blockIdx)  {
        if (row < N && (blockIdx * BLOCK_SIZE + threadIdx.x) < N) {
            sharedA[threadIdx.y][threadIdx.x] = A[row * N + blockIdx * BLOCK_SIZE + threadIdx.x];
        } else {
            sharedA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && (blockIdx * BLOCK_SIZE + threadIdx.y) < N) {
            sharedB[threadIdx.y][threadIdx.x] = B[(blockIdx * BLOCK_SIZE + threadIdx.y) * N + col];
        } else {
            sharedB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
        }

        __syncthreads();

        if (row < N && col < N) {
            C[row * N +col] = sum;
        }
    }
}

void measureExecutionTime(float* d_A, float* d_B, float* d_C, int N, bool usePadding) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaMemset(d_C, 0, N * N * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    if (usePadding) {
        matrixMulSharedKernelWithPadding<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    } else {
	matrixMulSharedKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Use padding = " << usePadding << "time = " << milliseconds << std::endl;
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
	std::cerr << "CUDA ERROR: " << msg << " - " << cudaGetErrorString(err) << std::endl;
	exit(EXIT_FAILURE);
    }
}

void baseRealization() {
    const int size = N * N;
    const size_t bytes = size * sizeof(float);

    std::vector<float> h_A(size), h_B(size), h_C(size);

    generateMatrixVector(h_A, N);
    generateMatrixVector(h_B, N);

    float *d_A, *d_B, *d_C;

    checkCudaError(cudaMalloc(&d_A, bytes), "Allocating d_A");
    checkCudaError(cudaMalloc(&d_B, bytes), "Allocating d_B");
    checkCudaError(cudaMalloc(&d_C, bytes), "Allocating d_C");

    checkCudaError(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice), "Copying A to d_A");
    checkCudaError(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice), "Copying B to d_B");

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    checkCudaError(cudaGetLastError(), "Kernel execution");

    checkCudaError(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost), "Copying C to host");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "C[0][0] = " << h_C[0] << std::endl;
}

void pinnedMemoryRealization() {
    const int N = 1024;
    size_t bytes = N * N * sizeof(float);

    float *h_A, *h_B, *h_C;
    cudaHostAlloc(&h_A, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_B, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_C, bytes, cudaHostAllocDefault);
    
    srand(time(NULL));

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    matrixMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    float sum = 0.0f;
    for (int i = 0; i < N * N; i++) {
        sum += h_C[i];
    }
    printf("Сумма элементов результирующей матрицы: %f\n", sum);

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void unifiedMemoryRealization() {
    const int N = 1024; // Размер матриц NxN
    size_t bytes = N * N * sizeof(float);

    srand(time(NULL));

    float *A, *B, *C;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    generateMatrixArray(A, N);
    generateMatrixArray(B, N);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    matrixMulKernel<<<gridSize, blockSize>>>(A, B, C, N);

    cudaDeviceSynchronize();

    float sum = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        sum += C[i];
    }
    printf("Сумма всех элементов результирующей матрицы: %f\n", sum);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

void cudaStreamsRealization() {
    const int N = 1024;    
    const int blockSize = 256;
    const int numStreams = 4;
    size_t bytes = N * N * sizeof(float);

    srand(time(NULL));

    float *h_A, *h_B, *h_C;
    cudaHostAlloc(&h_A, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_B, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_C, bytes, cudaHostAllocDefault);

    generateMatrixArray(h_A, N);
    generateMatrixArray(h_B, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 blockDim(16, 16);
    dim3 gridDim((blockSize + blockDim.x - 1) / blockDim.x, (blockSize + blockDim.y - 1) / blockDim.y, 1);

    for (int streamIdx = 0; streamIdx < numStreams; ++streamIdx) {
        int offset = streamIdx * blockSize * blockSize;
        size_t blockBytes = blockSize * blockSize * sizeof(float);

        cudaMemcpyAsync(&d_A[offset], &h_A[offset], blockBytes, cudaMemcpyHostToDevice, streams[streamIdx]);

        matrixMulKernelAsync<<<gridDim, blockDim, 0, streams[streamIdx]>>>(d_A, d_B, d_C, N, blockSize);

        cudaMemcpyAsync(&h_C[offset], &d_C[offset], blockBytes, cudaMemcpyDeviceToHost, streams[streamIdx]);
    }

    for (int i = 0; i < numStreams; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        sum += h_C[i];
    }
    printf("Сумма всех элементов результирующей матрицы: %f\n", sum);
    
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    for (int i = 0; i < numStreams; ++i) {
        cudaStreamDestroy(streams[i]);
    }
}

void sharedMemoryRealization() {
    const int N = 1024;
    size_t bytes = N * N * sizeof(float);

    srand(time(NULL));

    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);

    generateMatrixArray(h_A, N);
    generateMatrixArray(h_B, N);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    matrixMulSharedKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        sum += h_C[i];
    }
    printf("Сумма всех элементов результирующей матрицы: %f\n", sum);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

void sharedMemoryOptimRealization() {
    const int N = 1024;
    size_t bytes = N * N * sizeof(float);
    srand(time(NULL));
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);

    generateMatrixArray(h_A, N);
    generateMatrixArray(h_B, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    measureExecutionTime(d_A, d_B, d_C, N, true);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);

    generateMatrixArray(h_A, N);
    generateMatrixArray(h_B, N);
    
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes); 
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    measureExecutionTime(d_A, d_B, d_C, N, false);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);    
}

void mulWithCuBLASS(){
    int bytes = N * N * sizeof(float);

    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    generateMatrixArray(h_A, N);
    generateMatrixArray(h_B, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_A, N,
                d_B, N,
                &beta,
                d_C, N);
    
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        sum += h_C[i];
    }
    std::cout << "Сумма всех элементов результирующей матрицы: " << sum << std::endl;    

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

int main() {
    baseRealization();
    pinnedMemoryRealization();
    unifiedMemoryRealization();
    cudaStreamsRealization();
    sharedMemoryRealization();
    sharedMemoryOptimRealization();
    mulWithCuBLASS();
    return 0;
}
