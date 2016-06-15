#include<iostream>
#include<cstdio>
#include<cuda.h>

__global__ void kernel1(int N)
{
  int lvl=0, c1, c2;
  __shared__ int max_lvl;
  if( threadIdx.x<N )
  {
    c1 = c2 = threadIdx.x;
    //printf("blk:%i thrd:%i c1:%i c2:%i",blockIdx.x, threadIdx.x, c1, c2);
  }
  else
  {
    int N_lvl = N, prev_lvl_1st, curr_lvl_1st = 0, nxt_lvl_1st = N;
    do
    {
      prev_lvl_1st = curr_lvl_1st;
      curr_lvl_1st = nxt_lvl_1st;
      N_lvl = (N_lvl/2) + (N_lvl%2); // N_lvl = ceil(N/2) = N/2 + N%2 = (N>>1) + N&1
      nxt_lvl_1st = curr_lvl_1st + N_lvl;
      ++lvl;
    }while( !(curr_lvl_1st<=threadIdx.x && threadIdx.x<nxt_lvl_1st) );
    c1 = prev_lvl_1st + ((threadIdx.x-curr_lvl_1st)*2);
    if( (c1+1)<nxt_lvl_1st )
    {
      c2 = c1+1;
    }
    else
    {
      c2 = blockDim.x-1;
    }
    if( threadIdx.x==(blockDim.x-1) )
    {
      max_lvl = lvl;
    }
  }
  __syncthreads();
  printf("blk:%i thrd:%i idle_init:%i idle_end:%i\n",blockIdx.x, threadIdx.x, lvl, max_lvl-lvl);
}

/*
Output
[sanjay@rn-gpu Misc Programs]$ ./a.out
blk:0 thrd:32 idle_init:1 idle_end:4
blk:0 thrd:33 idle_init:1 idle_end:4
blk:0 thrd:34 idle_init:1 idle_end:4
blk:0 thrd:35 idle_init:1 idle_end:4
blk:0 thrd:36 idle_init:1 idle_end:4
blk:0 thrd:37 idle_init:1 idle_end:4
blk:0 thrd:38 idle_init:1 idle_end:4
blk:0 thrd:39 idle_init:1 idle_end:4
blk:0 thrd:40 idle_init:1 idle_end:4
blk:0 thrd:41 idle_init:1 idle_end:4
blk:0 thrd:42 idle_init:1 idle_end:4
blk:0 thrd:43 idle_init:1 idle_end:4
blk:0 thrd:44 idle_init:1 idle_end:4
blk:0 thrd:45 idle_init:1 idle_end:4
blk:0 thrd:46 idle_init:1 idle_end:4
blk:0 thrd:47 idle_init:1 idle_end:4
blk:0 thrd:48 idle_init:2 idle_end:3
blk:0 thrd:49 idle_init:2 idle_end:3
blk:0 thrd:50 idle_init:2 idle_end:3
blk:0 thrd:51 idle_init:2 idle_end:3
blk:0 thrd:52 idle_init:2 idle_end:3
blk:0 thrd:53 idle_init:2 idle_end:3
blk:0 thrd:54 idle_init:2 idle_end:3
blk:0 thrd:55 idle_init:2 idle_end:3
blk:0 thrd:56 idle_init:3 idle_end:2
blk:0 thrd:57 idle_init:3 idle_end:2
blk:0 thrd:58 idle_init:3 idle_end:2
blk:0 thrd:59 idle_init:3 idle_end:2
blk:0 thrd:60 idle_init:4 idle_end:1
blk:0 thrd:61 idle_init:4 idle_end:1
blk:0 thrd:62 idle_init:5 idle_end:0
blk:0 thrd:0 idle_init:0 idle_end:5
blk:0 thrd:1 idle_init:0 idle_end:5
blk:0 thrd:2 idle_init:0 idle_end:5
blk:0 thrd:3 idle_init:0 idle_end:5
blk:0 thrd:4 idle_init:0 idle_end:5
blk:0 thrd:5 idle_init:0 idle_end:5
blk:0 thrd:6 idle_init:0 idle_end:5
blk:0 thrd:7 idle_init:0 idle_end:5
blk:0 thrd:8 idle_init:0 idle_end:5
blk:0 thrd:9 idle_init:0 idle_end:5
blk:0 thrd:10 idle_init:0 idle_end:5
blk:0 thrd:11 idle_init:0 idle_end:5
blk:0 thrd:12 idle_init:0 idle_end:5
blk:0 thrd:13 idle_init:0 idle_end:5
blk:0 thrd:14 idle_init:0 idle_end:5
blk:0 thrd:15 idle_init:0 idle_end:5
blk:0 thrd:16 idle_init:0 idle_end:5
blk:0 thrd:17 idle_init:0 idle_end:5
blk:0 thrd:18 idle_init:0 idle_end:5
blk:0 thrd:19 idle_init:0 idle_end:5
blk:0 thrd:20 idle_init:0 idle_end:5
blk:0 thrd:21 idle_init:0 idle_end:5
blk:0 thrd:22 idle_init:0 idle_end:5
blk:0 thrd:23 idle_init:0 idle_end:5
blk:0 thrd:24 idle_init:0 idle_end:5
blk:0 thrd:25 idle_init:0 idle_end:5
blk:0 thrd:26 idle_init:0 idle_end:5
blk:0 thrd:27 idle_init:0 idle_end:5
blk:0 thrd:28 idle_init:0 idle_end:5
blk:0 thrd:29 idle_init:0 idle_end:5
blk:0 thrd:30 idle_init:0 idle_end:5
blk:0 thrd:31 idle_init:0 idle_end:5
*/

__global__ void kernel2()
{

}

using namespace std;

int num_threads_in_tree(int N)
{
  int sum = N;
  do
  {
    N = (N>>1) + (N&1);
    sum += N;
  }while(N>1);
  return sum;
}

int main()
{
  int N = 32;
  cudaSetDevice(7);
  kernel1<<<1,num_threads_in_tree(N)>>>(N);
  cudaDeviceSynchronize();
}
