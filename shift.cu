#include <cuda_runtime.h>
#include <iostream>

__global__ void shiftLeftKernel(int *arr, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n - 1) {
        arr[idx] = arr[idx + 1];
    } else if (idx == n - 1) {
        // Handle the last element separately
        arr[idx] = arr[0];
    }
}

int main()
{
    const int n = 20;
    int* arr, d_arr;
    int size = n * sizeof(int);

    for (int i = 0; i < n; i++)
    {
        arr[i] = i;
    }

    cudaMalloc(&d_arr, size);
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    shiftLeftKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, n);

    cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);


    std::cout << "Shifted array: ";
    for (int i = 0; i < n; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
