#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
/* #include <thrust/universal_vector.h> */
#include <chrono>
#include <nvToolsExt.h>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

// debug 5 1 8 4 1 1
// #define ProblemSize 5
// #define TPBX 1
// #define TPBY 8
// #define scaleX 4    
// #define scaleY 1
// #define RAD 1 // radius of the stencil

// 必须是2的倍数
#define ProblemSize 14
#define TPBX 1
#define TPBY 512
#define scaleX 16 // max = 16
#define scaleY 1
#define RAD 1 // radius of the stencil

__device__ bool isResultRight = true;
__device__ int xborderNum=(1 << ProblemSize)/TPBX/scaleX-1;
__device__ int yborderNum=(1 << ProblemSize)/TPBY/scaleY-1;

__global__ void stencil(int row_num, int col_num, int *arr_data, int *result) {
    if (blockIdx.x==0||blockIdx.x==xborderNum||blockIdx.y==0||blockIdx.y==yborderNum){
        auto idxInB = threadIdx.x * TPBY + threadIdx.y;
        auto scale_row = blockIdx.x * blockDim.x * scaleX;
        auto scale_col = blockIdx.y * blockDim.y * scaleY;
        auto block_scale_index = scale_row * col_num + scale_col;
        auto thread_scale_index = block_scale_index + idxInB ;

        //假设Block内线程数正好等于SMEM的列数- 2 * RAD
        // TPBY * TPBX = local_scale_col_num - 2 * RAD = TPBY * scaleY
        // TPBX = scaleY
        auto local_scale_col_num = TPBY * scaleY + 2 * RAD;
        auto local_scale_index = 1 * local_scale_col_num + 1 + idxInB; 

        extern __shared__ int sdata[];

        // up down Regular
        if(blockIdx.x==0){
            //up 
            sdata[1+idxInB]=arr_data[(row_num-1)*col_num+scale_col+idxInB];
            auto Regular_local_index = local_scale_index;
            auto Regular_global_index = thread_scale_index;
            //Regular cells + down
            for(int i = 0 ; i <= scaleX; i++ ){
                sdata[Regular_local_index]=arr_data[Regular_global_index];
                Regular_global_index += col_num;
                Regular_local_index += local_scale_col_num;
            }
        }else if(blockIdx.x==xborderNum){
            // start from up
            auto Regular_local_index = local_scale_index - local_scale_col_num;
            auto Regular_global_index = thread_scale_index - col_num;
            // up + Regular cells 
            for(int i = 0 ; i <= scaleX; i++ ){// for-loop end before down line
                sdata[Regular_local_index]=arr_data[Regular_global_index];
                Regular_global_index += col_num;
                Regular_local_index += local_scale_col_num;
            }
            sdata[Regular_local_index]=arr_data[scale_col+idxInB];
        }else{
            // start from up
            auto Regular_local_index = local_scale_index - local_scale_col_num;
            auto Regular_global_index = thread_scale_index - col_num;
            // up + Regular cells + down
            for(int i = 0 ; i <= scaleX + 1; i++ ){// for-loop end in down line
                sdata[Regular_local_index]=arr_data[Regular_global_index];
                Regular_global_index += col_num;
                Regular_local_index += local_scale_col_num;
            }
        }
        

        //left right cells
        if(idxInB < 2 * scaleX)
            if(blockIdx.y==0){
                auto LR_col = idxInB%2;
                auto LR_col_offset = LR_col * (local_scale_col_num - 1);
                auto LR_row = idxInB/2;
                auto LR_local_index = (LR_row + 1) * local_scale_col_num + LR_col_offset ;
                auto LR_global_index = block_scale_index - 1 + LR_row * col_num + LR_col_offset + (1-LR_col)*(col_num);
                sdata[LR_local_index]=arr_data[LR_global_index];
            }
            else if(blockIdx.y==yborderNum){
                auto LR_col = idxInB%2;
                auto LR_col_offset = LR_col * (local_scale_col_num - 1);
                auto LR_row = idxInB/2;
                auto LR_local_index = (LR_row + 1) * local_scale_col_num + LR_col_offset ;
                auto LR_global_index = block_scale_index - 1 + LR_row * col_num + LR_col_offset - (LR_col * (col_num));
                sdata[LR_local_index]=arr_data[LR_global_index];
            }
            else{
                auto LR_col = idxInB%2;
                auto LR_col_offset = LR_col * (local_scale_col_num - 1);
                auto LR_row = idxInB/2;
                auto LR_local_index = (LR_row + 1) * local_scale_col_num + LR_col_offset ;
                auto LR_global_index = block_scale_index - 1 + LR_row * col_num + LR_col_offset ;
                sdata[LR_local_index]=arr_data[LR_global_index];
            }
        
        __syncthreads();

        auto Regular_local_index = local_scale_index;
        auto Regular_global_index = thread_scale_index;
        // calculate
        for(int i = 0 ; i < scaleX; i++ ){
            result[Regular_global_index] =  sdata[Regular_local_index - local_scale_col_num]
                                        +   sdata[Regular_local_index - 1]
                                    - 7 *   sdata[Regular_local_index]                                       
                                        +   sdata[Regular_local_index + 1] 
                                        +   sdata[Regular_local_index + local_scale_col_num] ;
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
        }
    }else{
        auto idxInB = threadIdx.x * TPBY + threadIdx.y;
        auto scale_row = blockIdx.x * blockDim.x * scaleX;
        auto scale_col = blockIdx.y * blockDim.y * scaleY;
        auto block_scale_index = scale_row * col_num + scale_col;
        auto thread_scale_index = block_scale_index + idxInB ;

        //假设Block内线程数正好等于SMEM的列数- 2 * RAD
        // TPBY * TPBX = local_scale_col_num - 2 * RAD
        auto local_scale_col_num = TPBY * scaleY + 2 * RAD;
        auto local_scale_index = 1 * local_scale_col_num + 1 + idxInB; 

        extern __shared__ int sdata[];

        // start from up
        auto Regular_local_index = local_scale_index - local_scale_col_num;
        auto Regular_global_index = thread_scale_index - col_num;
        // up + Regular cells + down
        // for(int i = 0 ; i <= scaleX + 1; i++ ){// for-loop end in down line
        sdata[Regular_local_index]=arr_data[Regular_global_index];
        Regular_global_index += col_num;
        Regular_local_index += local_scale_col_num;
        sdata[Regular_local_index]=arr_data[Regular_global_index];
        Regular_global_index += col_num;
        Regular_local_index += local_scale_col_num;
        sdata[Regular_local_index]=arr_data[Regular_global_index];
        Regular_global_index += col_num;
        Regular_local_index += local_scale_col_num;
        //scaleX=1 At least, 3 load
        if(scaleX>=2){
            sdata[Regular_local_index]=arr_data[Regular_global_index];
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
        }
        if(scaleX>=4){
            sdata[Regular_local_index]=arr_data[Regular_global_index];
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            sdata[Regular_local_index]=arr_data[Regular_global_index];
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
        }
        if(scaleX>=8){
            sdata[Regular_local_index]=arr_data[Regular_global_index];
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            sdata[Regular_local_index]=arr_data[Regular_global_index];
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            sdata[Regular_local_index]=arr_data[Regular_global_index];
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            sdata[Regular_local_index]=arr_data[Regular_global_index];
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
        }
        if(scaleX>=16){
            sdata[Regular_local_index]=arr_data[Regular_global_index];
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            sdata[Regular_local_index]=arr_data[Regular_global_index];
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            sdata[Regular_local_index]=arr_data[Regular_global_index];
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            sdata[Regular_local_index]=arr_data[Regular_global_index];
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            sdata[Regular_local_index]=arr_data[Regular_global_index];
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            sdata[Regular_local_index]=arr_data[Regular_global_index];
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            sdata[Regular_local_index]=arr_data[Regular_global_index];
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            sdata[Regular_local_index]=arr_data[Regular_global_index];
            // Regular_global_index += col_num;
            // Regular_local_index += local_scale_col_num;
        }
        


        //left right cells
        if(idxInB < 2 * scaleX){
            auto LR_col = idxInB%2;
            auto LR_col_offset = LR_col * (local_scale_col_num - 1);
            auto LR_row = idxInB/2;
            auto LR_local_index = (LR_row + 1) * local_scale_col_num + LR_col_offset ;
            auto LR_global_index = block_scale_index - 1 + LR_row * col_num + LR_col_offset ;
            sdata[LR_local_index]=arr_data[LR_global_index];
        }
        
        __syncthreads();

        Regular_local_index = local_scale_index;
        Regular_global_index = thread_scale_index;
        // calculate
        // for(int i = 0 ; i < scaleX; i++ ){
        result[Regular_global_index] =  sdata[Regular_local_index - local_scale_col_num]
                                    +   sdata[Regular_local_index - 1]
                                - 7 *   sdata[Regular_local_index]                                       
                                    +   sdata[Regular_local_index + 1] 
                                    +   sdata[Regular_local_index + local_scale_col_num] ;
        Regular_global_index += col_num;
        Regular_local_index += local_scale_col_num; 
        if(scaleX >= 2){
            result[Regular_global_index] =  sdata[Regular_local_index - local_scale_col_num]
                        +   sdata[Regular_local_index - 1]
                    - 7 *   sdata[Regular_local_index]                                       
                        +   sdata[Regular_local_index + 1] 
                        +   sdata[Regular_local_index + local_scale_col_num] ;
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
        }
        if(scaleX >= 4){
            result[Regular_global_index] =  sdata[Regular_local_index - local_scale_col_num]
                        +   sdata[Regular_local_index - 1]
                    - 7 *   sdata[Regular_local_index]                                       
                        +   sdata[Regular_local_index + 1] 
                        +   sdata[Regular_local_index + local_scale_col_num] ;
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            result[Regular_global_index] =  sdata[Regular_local_index - local_scale_col_num]
                        +   sdata[Regular_local_index - 1]
                    - 7 *   sdata[Regular_local_index]                                       
                        +   sdata[Regular_local_index + 1] 
                        +   sdata[Regular_local_index + local_scale_col_num] ;
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
        }  
        if(scaleX >= 8){
            result[Regular_global_index] =  sdata[Regular_local_index - local_scale_col_num]
                        +   sdata[Regular_local_index - 1]
                    - 7 *   sdata[Regular_local_index]                                       
                        +   sdata[Regular_local_index + 1] 
                        +   sdata[Regular_local_index + local_scale_col_num] ;
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            result[Regular_global_index] =  sdata[Regular_local_index - local_scale_col_num]
                        +   sdata[Regular_local_index - 1]
                    - 7 *   sdata[Regular_local_index]                                       
                        +   sdata[Regular_local_index + 1] 
                        +   sdata[Regular_local_index + local_scale_col_num] ;
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            result[Regular_global_index] =  sdata[Regular_local_index - local_scale_col_num]
                        +   sdata[Regular_local_index - 1]
                    - 7 *   sdata[Regular_local_index]                                       
                        +   sdata[Regular_local_index + 1] 
                        +   sdata[Regular_local_index + local_scale_col_num] ;
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            result[Regular_global_index] =  sdata[Regular_local_index - local_scale_col_num]
                        +   sdata[Regular_local_index - 1]
                    - 7 *   sdata[Regular_local_index]                                       
                        +   sdata[Regular_local_index + 1] 
                        +   sdata[Regular_local_index + local_scale_col_num] ;
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
        }  
        if(scaleX >= 16){
            result[Regular_global_index] =  sdata[Regular_local_index - local_scale_col_num]
                        +   sdata[Regular_local_index - 1]
                    - 7 *   sdata[Regular_local_index]                                       
                        +   sdata[Regular_local_index + 1] 
                        +   sdata[Regular_local_index + local_scale_col_num] ;
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            result[Regular_global_index] =  sdata[Regular_local_index - local_scale_col_num]
                        +   sdata[Regular_local_index - 1]
                    - 7 *   sdata[Regular_local_index]                                       
                        +   sdata[Regular_local_index + 1] 
                        +   sdata[Regular_local_index + local_scale_col_num] ;
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            result[Regular_global_index] =  sdata[Regular_local_index - local_scale_col_num]
                        +   sdata[Regular_local_index - 1]
                    - 7 *   sdata[Regular_local_index]                                       
                        +   sdata[Regular_local_index + 1] 
                        +   sdata[Regular_local_index + local_scale_col_num] ;
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            result[Regular_global_index] =  sdata[Regular_local_index - local_scale_col_num]
                        +   sdata[Regular_local_index - 1]
                    - 7 *   sdata[Regular_local_index]                                       
                        +   sdata[Regular_local_index + 1] 
                        +   sdata[Regular_local_index + local_scale_col_num] ;
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            result[Regular_global_index] =  sdata[Regular_local_index - local_scale_col_num]
                        +   sdata[Regular_local_index - 1]
                    - 7 *   sdata[Regular_local_index]                                       
                        +   sdata[Regular_local_index + 1] 
                        +   sdata[Regular_local_index + local_scale_col_num] ;
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            result[Regular_global_index] =  sdata[Regular_local_index - local_scale_col_num]
                        +   sdata[Regular_local_index - 1]
                    - 7 *   sdata[Regular_local_index]                                       
                        +   sdata[Regular_local_index + 1] 
                        +   sdata[Regular_local_index + local_scale_col_num] ;
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            result[Regular_global_index] =  sdata[Regular_local_index - local_scale_col_num]
                        +   sdata[Regular_local_index - 1]
                    - 7 *   sdata[Regular_local_index]                                       
                        +   sdata[Regular_local_index + 1] 
                        +   sdata[Regular_local_index + local_scale_col_num] ;
            Regular_global_index += col_num;
            Regular_local_index += local_scale_col_num;
            result[Regular_global_index] =  sdata[Regular_local_index - local_scale_col_num]
                        +   sdata[Regular_local_index - 1]
                    - 7 *   sdata[Regular_local_index]                                       
                        +   sdata[Regular_local_index + 1] 
                        +   sdata[Regular_local_index + local_scale_col_num] ;
            // Regular_global_index += col_num;
            // Regular_local_index += local_scale_col_num;
        }  
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

    nvtxRangePush("initArr");
    initArr<<<dim3(row_num / TPBX, col_num / TPBY, 1), dim3(TPBX, TPBY, 1)>>>(row_num, col_num, arr);
    cudaDeviceSynchronize();
    nvtxRangePop();
    if(ProblemSize <= 6)
        debugPrintCudaMatrix(row_num, col_num, arr);

    auto begin_millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    const size_t smemSize = (TPBX * scaleX + 2 * RAD) * (TPBY * scaleY + 2 * RAD) * sizeof(int);
    nvtxRangePush("stencil");
    stencil<<<dim3(row_num / TPBX / scaleX , col_num / TPBY / scaleY , 1), dim3(TPBX, TPBY, 1), smemSize>>>(row_num, col_num, arr, result);
    cudaDeviceSynchronize(); 
    nvtxRangePop();
    auto end_millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    printf("%ld\n", end_millis - begin_millis);

    nvtxRangePush("checkResult");
    checkResult<<<dim3(row_num / TPBX, col_num / TPBY, 1), dim3(TPBX, TPBY, 1)>>>(row_num, col_num, result);
    cudaDeviceSynchronize(); 
    nvtxRangePop();
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

