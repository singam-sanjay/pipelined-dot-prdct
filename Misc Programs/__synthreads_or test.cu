#include<iostream>
#include<cstdio>
#include<cuda.h>

__global__ void kernel1()
{
  int lvl = 0;
  __shared__ int sm[1];
  if( threadIdx.x==(blockDim.x-1) )
  {
    lvl = threadIdx.x;
    sm[0] = lvl;
  }
  printf("block:%i thread:%i lvl b4:%i\n",blockIdx.x,threadIdx.x,lvl);
  //lvl = __syncthreads_or(lvl); : This returned 1(non-zero)
  /* Output
  block:0 thread:0 lvl b4:0
  block:0 thread:1 lvl b4:0
  block:0 thread:2 lvl b4:0
  block:0 thread:3 lvl b4:0
  block:0 thread:4 lvl b4:0
  block:0 thread:5 lvl b4:0
  block:0 thread:6 lvl b4:6
  block:0 thread:0 lvl aftr:1
  block:0 thread:1 lvl aftr:1
  block:0 thread:2 lvl aftr:1
  block:0 thread:3 lvl aftr:1
  block:0 thread:4 lvl aftr:1
  block:0 thread:5 lvl aftr:1
  block:0 thread:6 lvl aftr:1
  */
  __syncthreads();
  lvl = sm[0];
  printf("block:%i thread:%i lvl aftr:%i\n",blockIdx.x,threadIdx.x,lvl);
  /* Output
  block:0 thread:0 lvl b4:0
  block:0 thread:1 lvl b4:0
  block:0 thread:2 lvl b4:0
  block:0 thread:3 lvl b4:0
  block:0 thread:4 lvl b4:0
  block:0 thread:5 lvl b4:0
  block:0 thread:6 lvl b4:6
  block:0 thread:0 lvl aftr:6
  block:0 thread:1 lvl aftr:6
  block:0 thread:2 lvl aftr:6
  block:0 thread:3 lvl aftr:6
  block:0 thread:4 lvl aftr:6
  block:0 thread:5 lvl aftr:6
  block:0 thread:6 lvl aftr:6
 */
}

int main()
{
  cudaSetDevice(7);
  kernel1<<<1,7>>>();
  cudaDeviceSynchronize();
}
