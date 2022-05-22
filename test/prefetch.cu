// https://developer.nvidia.com/zh-cn/blog/boosting-application-performance-with-gpu-memory-prefetching/

//baseline
for (i=threadIdx.x; i<imax; i+= BLOCKDIMX) { 
    double locvar = arr[i]; 
    <lots of instructions using locvar, for example, transcendentals>
}

//scalar batched (4 registers)
double v0, v1, v2, v3;
for (i=threadIdx.x, ctr=0; i<imax; i+= BLOCKDIMX, ctr++) { 
    ctr_mod = ctr%4; 
    if (ctr_mod==0) { // only fill the buffer each 4th iteration 
        v0=arr[i+0* BLOCKDIMX]; 
        v1=arr[i+1* BLOCKDIMX]; 
        v2=arr[i+2* BLOCKDIMX]; 
        v3=arr[i+3* BLOCKDIMX]; 
    } 
    switch (ctr_mod) { // pull one value out of the prefetched batch 
        case 0: locvar = v0; break; 
        case 1: locvar = v1; break; 
        case 2: locvar = v2; break; 
        case 3: locvar = v3; break; 
    } 
    <lots of instructions using locvar, for example, transcendentals>
}

//smem batched
#define vsmem(index) v[index+PDIST*threadIdx.x]
//为了避免bank conflict
#define vsmem(index) v[index+(PDIST+PADDING)*threadIdx.x]
__shared__ double v[PDIST* BLOCKDIMX];
for (i=threadIdx.x, ctr=0; i<imax; i+= BLOCKDIMX, ctr++) { 
    ctr_mod = ctr%PDIST; 
    if (ctr_mod==0) { 
        for (k=0; k<PDIST; ++k) 
            vsmem(k) = arr[i+k* BLOCKDIMX]; 
        } 
    locvar = vsmem(ctr_mod); 
    <more instructions using locvar, for example, transcendentals>
}

//scalar rolling
__shared__ double v[PDIST* BLOCKDIMX];
for (k=0; k<PDIST; ++k) 
    vsmem(k) = arr[threadIdx.x+k* BLOCKDIMX];
for (i=threadIdx.x, ctr=0; i<imax; i+= BLOCKDIMX, ctr++) { 
    ctr_mod= ctr%PDIST; 
    locvar = vsmem(ctr_mod); 
    if ( i<imax-PDIST* BLOCKDIMX) 
        vsmem(ctr_mod) = arr[i+PDIST* BLOCKDIMX]; 
    <more instructions using locvar, for example, transcendentals>
}

//scalar rolling async (通过pipeline 流水线异步实现)
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#pipeline-interface
#include <cuda_pipeline_primitives.h>
__shared__ double v[PDIST* BLOCKDIMX];
for (k=0; k<PDIST; ++k) { // fill the prefetch buffer asynchronously 
    __pipeline_memcpy_async(&vsmem(k), &arr[threadIdx.x+k* BLOCKDIMX], 8); 
    __pipeline_commit();
}
for (i=threadIdx.x, ctr=0; i<imax; i+= BLOCKDIMX, ctr++) { 
    __pipeline_wait_prior(PDIST-1); //wait on needed prefetch value 
    ctr_mod= ctr%PDIST; 
    locvar = vsmem(ctr_mod); 
    if ( i<imax-PDIST* BLOCKDIMX) { // prefetch one new value 
        __pipeline_memcpy_async(&vsmem(ctr_mod), &arr[i+PDIST* BLOCKDIMX], 8);
        __pipeline_commit(); 
    } 
    <more instructions using locvar, for example, transcendentals>
}