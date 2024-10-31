#include <iostream>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

const int M = 16; // warp size, must be multiple of 16
const int N = 16;
const int K = 16;

__global__ void tensorCoreGemm(half *a, half *b, float *c, int M, int N, int K) {
    // Define fragments
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    // Load the inputs
    wmma::load_matrix_sync(a_frag, a, M);
    wmma::load_matrix_sync(b_frag, b, K);

    // Initialize output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Perform matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the output
    wmma::store_matrix_sync(c, c_frag, N, wmma::mem_row_major);
}

int main() {
    // Define matrix sizes
    const int SIZE = M * N;
    const int MATRIX_SIZE = SIZE * sizeof(half);

    // Allocate device memory
    half *a, *b;
    float *c;
    cudaMalloc(&a, MATRIX_SIZE);
    cudaMalloc(&b, MATRIX_SIZE);
    cudaMalloc(&c, SIZE * sizeof(float));

    // Initialize data (left out for brevity)
    // Fill `a` and `b` with values on host and copy to device

    // Launch kernel
    dim3 grid(1);
    dim3 block(32, 32); // A block per warp
    tensorCoreGemm<<<grid, block>>>(a, b, c, M, N, K);

    // Synchronize and cleanup
    cudaDeviceSynchronize();
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
