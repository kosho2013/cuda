#include <iostream>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

using namespace nvcuda;

const int M = 16;  // Tile dimensions for Tensor Cores (16x16x16)
const int N = 16;
const int K = 16;
const int TILE_DIM = 1024; // GEMM size

__global__ void tensorCoreGemm1024x1024x1024(half *a, half *b, float *c, int TILE_DIM) {
    __shared__ half shared_a[M * K];
    __shared__ half shared_b[K * N];

    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    int tileRow = blockIdx.y * blockDim.y + threadIdx.y;
    int tileCol = blockIdx.x * blockDim.x + threadIdx.x;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int tileIdx = 0; tileIdx < TILE_DIM / K; ++tileIdx) {
        shared_a[threadIdx.y * K + threadIdx.x] = a[(tileRow * TILE_DIM) + (tileIdx * K + threadIdx.x)];
        shared_b[threadIdx.y * K + threadIdx.x] = b[(tileIdx * K + threadIdx.y) * TILE_DIM + tileCol];
        __syncthreads();

        wmma::load_matrix_sync(a_frag, &shared_a[0], K);
        wmma::load_matrix_sync(b_frag, &shared_b[0], N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    wmma::store_matrix_sync(&c[tileRow * TILE_DIM + tileCol], c_frag, TILE_DIM, wmma::mem_row_major);
}

int main() {
    const int MATRIX_SIZE_A = TILE_DIM * TILE_DIM * sizeof(half);
    const int MATRIX_SIZE_B = TILE_DIM * TILE_DIM * sizeof(half);
    const int MATRIX_SIZE_C = TILE_DIM * TILE_DIM * sizeof(float);

    half *a, *b;
    float *c;

    cudaMalloc(&a, MATRIX_SIZE_A);
    cudaMalloc(&b, MATRIX_SIZE_B);
    cudaMalloc(&c, MATRIX_SIZE_C);

    // Host-side arrays for initialization and verification
    half *host_a = new half[TILE_DIM * TILE_DIM];
    half *host_b = new half[TILE_DIM * TILE_DIM];
    float *host_c = new float[TILE_DIM * TILE_DIM];

    for (int i = 0; i < TILE_DIM * TILE_DIM; ++i) {
        host_a[i] = __float2half(1.0f);  // Fill with 1s
        host_b[i] = __float2half(1.0f);  // Fill with 1s
    }

    // Copy initialized arrays to device
    cudaMemcpy(a, host_a, MATRIX_SIZE_A, cudaMemcpyHostToDevice);
    cudaMemcpy(b, host_b, MATRIX_SIZE_B, cudaMemcpyHostToDevice);

    dim3 grid(TILE_DIM / M, TILE_DIM / N);
    dim3 block(M, N);

    // Initialize CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start);

    // Launch the GEMM kernel
    tensorCoreGemm1024x1024x1024<<<grid, block>>>(a, b, c, TILE_DIM);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Convert time to nanoseconds
    float nanoseconds = milliseconds * 1e6;

    // Calculate FLOPS
    long long int num_operations = 2LL * TILE_DIM * TILE_DIM * TILE_DIM;
    float tflops = (num_operations / (nanoseconds / 1e9)) / 1e12; // TFLOPS

    std::cout << "Time: " << nanoseconds << " ns" << std::endl;
    std::cout << "Achieved TFLOPS: " << tflops << std::endl;

    // Copy result matrix C back to host for verification
    cudaMemcpy(host_c, c, MATRIX_SIZE_C, cudaMemcpyDeviceToHost);

    // Verify correctness by checking if each element in C is 1024
    bool correct = true;
    float expected_value = 1024.0f;
    for (int i = 0; i < TILE_DIM * TILE_DIM; ++i) {
        if (abs(host_c[i] - expected_value) > 1e-3) {  // Allow small floating-point tolerance
            correct = false;
            std::cout << "Mismatch at index " << i << ": " << host_c[i] << " != " << expected_value << std::endl;
            break;
        }
    }
    if (correct) {
        std::cout << "Matrix multiplication result is correct!" << std::endl;
    } else {
        std::cout << "Matrix multiplication result is incorrect." << std::endl;
    }

    // Cleanup
    delete[] host_a;
    delete[] host_b;
    delete[] host_c;
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
