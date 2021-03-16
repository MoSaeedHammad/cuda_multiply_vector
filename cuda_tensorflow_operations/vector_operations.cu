#include <stdio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include<iostream>
#include<cstdlib>
#include<cmath>

//#define DEBUG

__global__ void kernel(int* a_InputArray, int* b_InputArray, int* AddMatrix, int* ResultMatrix, const int MatrixRows, const int MatrixColums) {
    int ThreadIndex = threadIdx.x + blockIdx.x * blockDim.x;
#if defined(DEBUG)
    printf("ThreadIndex: %d\n", ThreadIndex);
#endif
    if (ThreadIndex < (MatrixRows * MatrixColums)) {
        ResultMatrix[ThreadIndex] = a_InputArray[ThreadIndex % MatrixRows] * b_InputArray[ThreadIndex % MatrixColums] + AddMatrix[ThreadIndex];
    }
}

// debuging functions
void init_array(int* pArray, const int MatrixRows);
void init_mat(int* pMatrix, const int MatrixRows, const int MatrixColums);
void print_array(int* pArray, const int MatrixRows, char* MatrixName);
void print_mat(int* pMatrix, const int MatrixRows, const int MatrixColums, char* MatrixName);
int calculate_error(int* p_a_Matrix, int* p_b_Matrix, const int MatrixRows, const int MatrixColums);
void cpu_calculate_array(int* a_InputArray, int* b_InputArray, int* AddMatrix, int* ResultMatrix, const int MatrixRows, const int MatrixColums);


void cpu_calculate_array(int* a_InputArray, int* b_InputArray, int* AddMatrix, int* ResultMatrix, const int MatrixRows, const int MatrixColums) {
    for (int LoopIndex = 0; LoopIndex < MatrixRows * MatrixColums; LoopIndex++)
    {
        ResultMatrix[LoopIndex] = a_InputArray[LoopIndex % MatrixRows] * b_InputArray[LoopIndex % MatrixColums] + AddMatrix[LoopIndex];
#if defined(DEBUG)
        printf("[ Indices id:%d, aId:%d, bId:%d ] Inputs a:%d, b:%d, c:%d >> Output d:%d\n", LoopIndex, LoopIndex % MatrixRows, LoopIndex % MatrixColums, a_InputArray[LoopIndex % MatrixRows], b_InputArray[LoopIndex % MatrixColums], AddMatrix[LoopIndex], ResultMatrix[LoopIndex]);
#endif
    }
}

int main(void) {
    srand(time(NULL));

    int* h_a_InputArray, * h_b_InputArray, * h_c_AddMatrix, * h_d_GpuResultMatrix, * d_CpuResultMatrix;
    int* dev_a_InputArray, * dev_b_InputArray, * dev_c_AddMatrix, * dev_d_GpuResultMatrix;
    cudaEvent_t start, end;
    float time = 0.0;

    int MatrixRows = 512;
    int MatrixColums = 512;

    h_a_InputArray = (int*)malloc(sizeof(int) * MatrixRows);
    h_b_InputArray = (int*)malloc(sizeof(int) * MatrixColums);
    h_c_AddMatrix = (int*)malloc(sizeof(int) * MatrixRows * MatrixColums);
    h_d_GpuResultMatrix = (int*)malloc(sizeof(int) * MatrixRows * MatrixColums);
    d_CpuResultMatrix = (int*)malloc(sizeof(int) * MatrixRows * MatrixColums);

    init_array(h_a_InputArray, MatrixRows);
    init_array(h_b_InputArray, MatrixColums);
    init_mat(h_c_AddMatrix, MatrixRows, MatrixColums);

#ifdef DEBUG
    printf("<<<<<<<<<< initial data:\n");
    print_array(h_a_InputArray, MatrixRows, "a-vector");
    print_array(h_b_InputArray, MatrixColums, "b-vector");
    print_mat(h_c_AddMatrix, MatrixRows, MatrixColums, "c-matrix");
#endif // !DEBUG


    cudaMalloc((void**)&dev_a_InputArray, sizeof(int) * MatrixRows);
    cudaMalloc((void**)&dev_b_InputArray, sizeof(int) * MatrixColums);
    cudaMalloc((void**)&dev_c_AddMatrix, sizeof(int) * MatrixRows * MatrixColums);
    cudaMalloc((void**)&dev_d_GpuResultMatrix, sizeof(int) * MatrixRows * MatrixColums);

    cudaMemcpy(dev_a_InputArray, h_a_InputArray, sizeof(int) * MatrixRows, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b_InputArray, h_b_InputArray, sizeof(int) * MatrixColums, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c_AddMatrix, h_c_AddMatrix, sizeof(int) * MatrixRows * MatrixColums, cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&end);

#ifdef DEBUG
    printf("\n\nRunning Kernel...\n\n");
#endif // DEBUG

    cudaEventRecord(start);
    kernel << <MatrixRows * MatrixColums / 256 + 1, 256 >> > (dev_a_InputArray, dev_b_InputArray, dev_c_AddMatrix, dev_d_GpuResultMatrix, MatrixRows, MatrixColums);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);

    cudaMemcpy(h_d_GpuResultMatrix, dev_d_GpuResultMatrix, sizeof(int) * MatrixRows * MatrixColums, cudaMemcpyDeviceToHost);

    printf("Gpu Return with Error Code: %s\n", cudaGetErrorString(cudaGetLastError()));
    printf("\tGPU Time Elapsed: %f ms\n", time);

    cudaFree(dev_a_InputArray);
    cudaFree(dev_b_InputArray);
    cudaFree(dev_c_AddMatrix);
    cudaFree(dev_d_GpuResultMatrix);

    cpu_calculate_array(h_a_InputArray, h_b_InputArray, h_c_AddMatrix, d_CpuResultMatrix, MatrixRows, MatrixColums);

#ifdef DEBUG
    printf(">>>>>>>>>> Gpu final data:\n");
    print_mat(h_d_GpuResultMatrix, MatrixRows, MatrixColums, "d-matrix");

    printf(">>>>>>>>>> Cpu final data:\n");
    print_mat(d_CpuResultMatrix, MatrixRows, MatrixColums, "cpu-matrix");
#endif // DEBUG
    auto cpu_start = std::chrono::high_resolution_clock::now();
    int error = calculate_error(h_d_GpuResultMatrix, d_CpuResultMatrix, MatrixRows, MatrixColums);
    auto cpu_done = std::chrono::high_resolution_clock::now();
    printf("\tCPU Time Elapsed: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(cpu_done - cpu_start).count() / 1000.0);

    printf("Deviation between Cpu and Gpu: %d\n",error);

    free(h_a_InputArray);
    free(h_b_InputArray);
    free(h_c_AddMatrix);
    free(h_d_GpuResultMatrix);
    free(d_CpuResultMatrix);
    return 0;
};

void init_array(int* pArray, const int ArraySize) {
    int LoopIndex;
    for (LoopIndex = 0; LoopIndex < ArraySize; LoopIndex++)
    {
        pArray[LoopIndex] = rand() % 4 + 1;
    }
}
void init_mat(int* pMatrix, const int MatrixRows, const int MatrixColums) {
    int LoopIndex;
    for (LoopIndex = 0; LoopIndex < MatrixRows * MatrixColums; LoopIndex++)
    {
        pMatrix[LoopIndex] = rand() % 4 + 1;
    }
}
void print_array(int* pArray, const int ArraySize, char* ArrayName) {
    int LoopIndex;
    for (LoopIndex = 0; LoopIndex < ArraySize; LoopIndex++)
        printf("\n%s[%d]: %d", ArrayName, LoopIndex, pArray[LoopIndex]);
    printf("\n");
}
void print_mat(int* pMatrix, const int MatrixRows, const int MatrixColums, char* MatrixName) {
    int LoopIndex;
    for (LoopIndex = 0; LoopIndex < MatrixRows*MatrixColums; LoopIndex++) {
        if (LoopIndex % MatrixColums == 0)
        {
            printf("\n%s[%d]:", MatrixName, LoopIndex);
        }
        printf("\t%d", pMatrix[LoopIndex]);
    }
    printf("\n");
}


int calculate_error(int* p_a_Matrix, int* p_b_Matrix, const int MatrixRows, const int MatrixColums)
{
    int LoopIndex, error_result = 0;
    for (LoopIndex = 0; LoopIndex < MatrixRows * MatrixColums; LoopIndex++)
    {
        if (abs(p_a_Matrix[LoopIndex] - p_b_Matrix[LoopIndex]) > 0)
        {
            printf("\tIndex %d has error InputA=%d , InputB=%d\n", LoopIndex, p_a_Matrix[LoopIndex], p_b_Matrix[LoopIndex]);
        }
        error_result += abs(p_a_Matrix[LoopIndex] - p_b_Matrix[LoopIndex]);
    }
    return error_result;
}

