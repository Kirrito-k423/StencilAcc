
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/#entry-content-comments

__global__ void device_copy_scalar_kernel(int* d_in, int* d_out, int N) { 
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) { 
      d_out[i] = d_in[i]; 
    } 
  } 
  
  void device_copy_scalar(int* d_in, int* d_out, int N) 
  { 
    int threads = 128; 
    int blocks = min((N + threads-1) / threads, MAX_BLOCKS);  
    device_copy_scalar_kernel<<<blocks, threads>>>(d_in, d_out, N); 
  }

  