#include <stdio.h>

__global__
void hello_kernel() {
    printf("hello world from cuda thread %d\n", int(threadIdx.x));
}

int main(void) {
    hello_kernel<<<1, 32>>>();
    //cudaDeviceSynchronize();
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

    double* ptr;

    auto size = 10 * sizeof(double);
    double *ptr_host = (double*)malloc(size);


    cudaMalloc(&ptr, size);

    cudaFree(ptr);

    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

    return 0;
}

