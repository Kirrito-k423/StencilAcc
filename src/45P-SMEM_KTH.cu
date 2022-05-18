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

#define TPBX 32
#define TPBY 32
#define RAD 1 // radius of the stencil


__global__ void stencil(int row_num, int col_num, int *arr_data, int *result) {
    
    if (blockIdx.x==0||blockIdx.x==511||blockIdx.y==0||blockIdx.y==511){
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

        result[index] = data1 + data2 + data3 + data4 - 4 * data0;
    }else{
        auto global_row = blockIdx.x * blockDim.x + threadIdx.x; //hyq写反了，我也懒得改了记得x是行号就行
        auto global_col = blockIdx.y * blockDim.y + threadIdx.y;
        auto global_block_size = col_num * TPBX;
        auto index = global_row * col_num + global_col;

        extern __shared__ int sdata[];
        
        auto local_row = threadIdx.x + RAD;
        auto local_col = threadIdx.y + RAD;
        auto local_col_num = TPBY + 2 * RAD;
        auto block_size = TPBX * TPBY;
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

        result[index] = sdata[local_index-local_col_num] + sdata[local_index+local_col_num] 
                    + sdata[local_index-1] + sdata[local_index+1] - 4 * sdata[local_index];
    }
}

int main() {
    int row_num = 1 << 14;
    int col_num = 1 << 14;

    int *arr;
    int *result;
    cudaMallocManaged(&arr, sizeof(int) * row_num * col_num);
    cudaMallocManaged(&result, sizeof(int) * row_num * col_num);

    for (int index = 0; index < row_num * col_num; ++index) {
        arr[index] = rand() % 1024 - 512;
    }

    cudaDeviceSynchronize();
    auto begin_millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

    cudaDeviceSynchronize();
    const size_t smemSize = (TPBX + 2 * RAD) * (TPBY + 2 * RAD) * sizeof(int);
    stencil<<<dim3(row_num / TPBX, col_num / TPBY, 1), dim3(TPBX, TPBY, 1), smemSize>>>(row_num, col_num, arr, result);

    cudaDeviceSynchronize(); 
    auto end_millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    printf("%ld\n", end_millis - begin_millis);
    cudaDeviceSynchronize();
    return 0;
}

