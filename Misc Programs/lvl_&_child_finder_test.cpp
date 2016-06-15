#include<iostream>
#include<cstdio>
#include<omp.h>

int kernel1_CPU(int N,int blockIdx_x, int threadIdx_x, int blockDim_x)
{
  int lvl=0, c1, c2;
  
  if( threadIdx_x<N )
  {
    c1 = c2 = threadIdx_x;
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
    }while( !(curr_lvl_1st<=threadIdx_x && threadIdx_x<nxt_lvl_1st) );
    c1 = prev_lvl_1st + ((threadIdx_x-curr_lvl_1st)*2);
    if( (c1+1)<nxt_lvl_1st )
    {
      c2 = c1+1;
    }
    else
    {
      c2 = blockDim_x-1;
    }
  }
  printf("blk:%i thrd:%i lvl:%i\n",blockIdx_x, threadIdx_x, lvl);
  return lvl;
  //fflush(stdout);
}

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
  int N = 5;
  /*cudaSetDevice(7);
  kernel1<<<1,num_threads_in_tree(N)>>>(N);
  cudaDeviceSynchronize();*/
  int iter, max_iter;
  iter = 0;
  max_iter = num_threads_in_tree(N);
  int lvl_arr[max_iter];
  volatile int max_lvl;
  #pragma omp parallel for
  for( iter=0 ; iter<max_iter ; ++iter )
  {
    int lvl;
    lvl_arr[iter] = lvl = kernel1_CPU(N,0,iter,max_iter);
    if( iter==(max_iter-1) )
    {
      max_lvl = lvl;
    }
  }
  #pragma omp parallel for
  for( iter=0 ; iter<max_iter ; ++iter )
  {
    printf("blk:%i thread:%i runs %i idle iterations\n", 0, iter, max_lvl-lvl_arr[iter] );
  }
}
