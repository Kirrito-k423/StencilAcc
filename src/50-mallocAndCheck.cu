#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
/* #include <thrust/universal_vector.h> */
#include <chrono>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

#define ProblemSize 14
#define TPBX 32
#define TPBY 32
#define RAD 1 // radius of the stencil

__device__ bool isResultRight = true;
__device__ int xborderNum=(1 << ProblemSize)/TPBX-1;
__device__ int yborderNum=(1 << ProblemSize)/TPBY-1;

__global__ void stencil(int row_num, int col_num, int *arr_data, int *result) {
    if (blockIdx.x==0||blockIdx.x==xborderNum||blockIdx.y==0||blockIdx.y==yborderNum){
        auto global_row = blockIdx.x * blockDim.x + threadIdx.x;
        auto global_col = blockIdx.y * blockDim.y + threadIdx.y;
        auto index = global_row * col_num + global_col;

        auto data0 = arr_data[index];
        // up
        auto data1 = arr_data[(global_row + row_num - 1) % row_num * col_num + global_col];
        // down
        auto data2 = arr_data[(global_row + 1) % row_num * col_num + global_col];
        // left
        auto data3 = arr_data[global_row * col_num + (global_col + col_num - 1) % col_num ];
        // right
        auto data4 = arr_data[global_row * col_num + (global_col + 1) % col_num];

        result[index] = data1 + data2 + data3 + data4 - 7 * data0;
    }else{
        auto global_row = blockIdx.x * blockDim.x + threadIdx.x; //hyq没写错，x是行号，因为后面是在实际运行时是相邻近的
        auto global_col = blockIdx.y * blockDim.y + threadIdx.y;
        auto global_block_size = col_num * TPBX;
        auto index = global_row * col_num + global_col;

        extern __shared__ int sdata[];
        
        auto local_row = threadIdx.x + RAD;
        auto local_col = threadIdx.y + RAD;
        auto local_col_num = TPBY + 2 * RAD;
        auto block_size = TPBX * local_col_num;
        auto local_index = local_row * local_col_num + local_col;

        // Regular cells
        sdata[local_index] = arr_data[index];
        // up down Halo cells
        if(threadIdx.x < RAD){
            sdata[local_index - local_col_num] = arr_data[index - col_num];
            sdata[local_index + block_size] = arr_data[index + global_block_size];
        }
        // left right Halo cells
        if(threadIdx.y < RAD){
            sdata[local_index - RAD] = arr_data[index - RAD];
            sdata[local_index + TPBX] = arr_data[index + TPBX];
        }
        __syncthreads();

        result[index] = sdata[local_index - local_col_num]
                    +   sdata[local_index + local_col_num] 
                    +   sdata[local_index - 1]
                    +   sdata[local_index + 1] 
                - 7 *   sdata[local_index];
    }
}

__global__ void initArr(int row_num, int col_num, int *arr_data){
    auto globalx = blockIdx.x * blockDim.x + threadIdx.x;
    auto globaly = blockIdx.y * blockDim.y + threadIdx.y;
    auto index = globaly * col_num + globalx;
    arr_data[index] = 1; 
}

__global__ void checkResult(int row_num, int col_num, int *result_data){
    auto globalx = blockIdx.x * blockDim.x + threadIdx.x;
    auto globaly = blockIdx.y * blockDim.y + threadIdx.y;
    auto index = globaly * col_num + globalx;
    if(result_data[index]!=-3){
        if(isResultRight == true){
            isResultRight = false;
        }
    }
}

__host__ void debugPrintCudaMatrix(int row_num, int col_num, int *inputCuda){
    int *tmpPrint = (int *)malloc(sizeof(int) * row_num * col_num);
    if(cudaSuccess != cudaMemcpy(tmpPrint , inputCuda, sizeof(int) * row_num * col_num, cudaMemcpyDeviceToHost)) 
        printf("cudaMemcpy Wrong in %s",__func__);
    for(int i=0; i<row_num; i++){
        for(int j=0; j<col_num; j++){
            printf("%d ",tmpPrint[i*col_num+j]);
        }
        printf("\n");
    }
}

int main() {
    int row_num = 1 << ProblemSize;
    int col_num = 1 << ProblemSize;

    int *arr;
    int *result;
    cudaMalloc(&arr, sizeof(int) * row_num * col_num);
    cudaMalloc(&result, sizeof(int) * row_num * col_num);

    initArr<<<dim3(row_num / TPBX, col_num / TPBY, 1), dim3(TPBX, TPBY, 1)>>>(row_num, col_num, arr);
    cudaDeviceSynchronize();
    if(ProblemSize <= 6)
        debugPrintCudaMatrix(row_num, col_num, arr);

    auto begin_millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    const size_t smemSize = (TPBX + 2 * RAD) * (TPBY + 2 * RAD) * sizeof(int);
    stencil<<<dim3(row_num / TPBX, col_num / TPBY, 1), dim3(TPBX, TPBY, 1), smemSize>>>(row_num, col_num, arr, result);
    cudaDeviceSynchronize(); 
    auto end_millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    printf("%ld\n", end_millis - begin_millis);

    checkResult<<<dim3(row_num / TPBX, col_num / TPBY, 1), dim3(TPBX, TPBY, 1)>>>(row_num, col_num, result);
    cudaDeviceSynchronize(); 
    if(ProblemSize <= 6)
        debugPrintCudaMatrix(row_num, col_num, result);
    bool isResultRightHost;
    // cudaMemcpy(&isResultRightHost, &isResultRight, sizeof(isResultRightHost), cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&isResultRightHost, isResultRight, sizeof(bool), 0, cudaMemcpyDeviceToHost);
    if(isResultRightHost){
        printf("Right Answer!!!\n");
    }else{
        printf("Ops!!! wrong Answer~\n");
    }
    return 0;
}

