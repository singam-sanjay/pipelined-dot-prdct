#include"cublas_v2.h"

__global__ sub_kernel( size_t N, TYPE *vec, TYPE *ref )
{
	__shared__ TYPE sub[THREADS_PER_BLOCK] = {};
	size_t idx = ((size_t)blockIdx.x)*THREADS_PER_BLOCK + threadIdx.x;
	if( idx >= N )
		return;
	TYPE ref_elem;

	ref_elem = ref[idx];
	sub[threadIdx.x] = vec[idx] - ref;
	
	__syncthreads();

	if( threadIdx.x != 0 )
		return;
	
	ref += sub[1]+sub[2]+sub[3]+sub[4]+sub[5]+sub[6]+sub[7]+sub[8]+sub[9]+sub[10]+sub[11]+sub[12]+sub[13]+sub[14]+sub[15];
	vec[idx] = ref;
}
