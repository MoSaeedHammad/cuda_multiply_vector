#include <stdio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include<iostream>
#include<cstdlib>
#include<cmath>

//#define DEBUG

__global__ void kernel(int* a, int* b, int* c, int* d, const int N, const int M) {
    int ThreadIndex = threadIdx.x + blockIdx.x * blockDim.x;
#if defined(DEBUG)
    printf("ThreadIndex: %d\n", ThreadIndex);
#endif
    if (ThreadIndex < (N * M)) {
        d[ThreadIndex] = a[ThreadIndex % N] * b[ThreadIndex % M] + c[ThreadIndex];
    }
}

// debuging functions
void init_array(int* a, const int N);
void init_mat(int* a, const int N, const int M);
void print_array(int* a, const int N, char* d);
void print_mat(int* a, const int N, const int M, char* d);
int calculate_error(int* a, int* b, const int N, const int M);
void cpu_calculate_array(int* a, int* b, int* c, int* d, const int N, const int M);


void cpu_calculate_array(int* a, int* b, int* c, int* d, const int N, const int M) {
    for (int iterationIndex = 0; iterationIndex < N * M; iterationIndex++)
    {
        d[iterationIndex] = a[iterationIndex % N] * b[iterationIndex % M] + c[iterationIndex];
#if defined(DEBUG)
        printf("[ Indices id:%d, aId:%d, bId:%d ] Inputs a:%d, b:%d, c:%d >> Output d:%d\n", iterationIndex, iterationIndex % N, iterationIndex % M, a[iterationIndex % N], b[iterationIndex % M], c[iterationIndex], d[iterationIndex]);
#endif
    }
}

int main(void) {
    srand(time(NULL));

    int* a, * b, * c, * d, * cpu_output;
    int* dev_a, * dev_b, * dev_c, * dev_d;
    cudaEvent_t start, end;
    float time = 0.0;

    int N = 4;
    int M = 3;

    a = (int*)malloc(sizeof(int) * N);
    b = (int*)malloc(sizeof(int) * M);
    c = (int*)malloc(sizeof(int) * N * M);
    d = (int*)malloc(sizeof(int) * N * M);
    cpu_output = (int*)malloc(sizeof(int) * N * M);

    init_array(a, N);
    init_array(b, M);
    init_mat(c, N, M);

#ifdef DEBUG
    printf("<<<<<<<<<< initial data:\n");
    print_array(a, N, "a-vector");
    print_array(b, M, "b-vector");
    print_mat(c, N, M, "c-matrix");
#endif // !DEBUG


    cudaMalloc((void**)&dev_a, sizeof(int) * N);
    cudaMalloc((void**)&dev_b, sizeof(int) * M);
    cudaMalloc((void**)&dev_c, sizeof(int) * N * M);
    cudaMalloc((void**)&dev_d, sizeof(int) * N * M);

    cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(int) * M, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, sizeof(int) * N * M, cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&end);

#ifdef DEBUG
    printf("\n\nRunning Kernel...\n\n");
#endif // DEBUG

    cudaEventRecord(start);
    kernel << <N * M / 256 + 1, 256 >> > (dev_a, dev_b, dev_c, dev_d, N, M);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);

    cudaMemcpy(d, dev_d, sizeof(int) * N * M, cudaMemcpyDeviceToHost);

    printf("Gpu Return with Error Code: %s\n", cudaGetErrorString(cudaGetLastError()));
    printf("\tGPU Time Elapsed: %f ms\n", time);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_d);

    cpu_calculate_array(a, b, c, cpu_output, N, M);

#ifdef DEBUG
    printf(">>>>>>>>>> Gpu final data:\n");
    print_mat(d, N, M, "d-matrix");

    printf(">>>>>>>>>> Cpu final data:\n");
    print_mat(cpu_output, N, M, "cpu-matrix");
#endif // DEBUG
    auto cpu_start = std::chrono::high_resolution_clock::now();
    int error = calculate_error(d, cpu_output, N, M);
    auto cpu_done = std::chrono::high_resolution_clock::now();
    printf("\tCPU Time Elapsed: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(cpu_done - cpu_start).count() / 1000.0);

    printf("Deviation between Cpu and Gpu: %d\n",error);

    free(a);
    free(b);
    free(c);
    free(d);
    free(cpu_output);
    return 0;
};

void init_array(int* a, const int N) {
    int i;
    for (i = 0; i < N; i++)
    {
        a[i] = rand() % 4 + 1;
    }
}
void init_mat(int* a, const int N, const int M) {
    int i, j;
    for (i = 0; i < N * M; i++)
    {
        a[i] = rand() % 4 + 1;
    }
}
void print_array(int* a, const int N, char* d) {
    int i;
    for (i = 0; i < N; i++)
        printf("\n%s[%d]: %d", d, i, a[i]);
    printf("\n");
}
void print_mat(int* a, const int N, const int M, char* d) {
    int i;
    for (i = 0; i < N*M; i++) {
        if (i % M == 0)
        {
            printf("\n%s[%d]:", d, i);
        }
        printf("\t%d", a[i]);
    }
    printf("\n");
}


int calculate_error(int* a, int* b, const int N, const int M)
{
    int i, error_result = 0;
    for (i = 0; i < N * M; i++)
    {
        if (abs(a[i] - b[i]) > 0)
        {
            printf("\tIndex %d has error InputA=%d , InputB=%d\n", i, a[i], b[i]);
        }
        error_result += abs(a[i] - b[i]);
    }
    return error_result;
}

