#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void shift(int *a, int *b, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        b[i] = a[i];
    }
}


int main() {
    int n = 10;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = n * sizeof(int);

    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    add<<<1, n>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
        printf("%d + %d = %d\n", a[i], b[i], c[i]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);






    int n = 10;
    int *a, *b;
    int *d_a, *d_b;
    int size = n * sizeof(int);


    a = (int *)malloc(size);
    b = (int *)malloc(size);

    for (int i = 0; i < n; i++)
    {
        a[i] = i;
    }

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    shift<<<1, n>>>(d_a, d_b, n);

    cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
    {
        printf(a[i], b[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    free(a);
    free(b);


    return 0;
}