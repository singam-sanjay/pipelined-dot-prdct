/* CUDA COMPUTE CAPABILITY 3.0 */
#define MAX_THREADS_PER_MP (2048)
#define MAX_BLOCKS_PER_MP  (16)
#define MAX_THREADS_PER_BLOCK (1024)
/* -------------------------- */

#define THREADS_PER_BLOCK ( 128 )       //( MAX_THREADS_PER_MP/MAX_BLOCKS_PER_MP )
#define NUMBER_OF_BLOCKS ( (rows-1)/THREADS_PER_BLOCK + 1 )

__global__ void sub_kernel( size_t N, TYPE *vec, TYPE *ref )
{
	__shared__ TYPE sub[THREADS_PER_BLOCK];
	TYPE sub_result;
	size_t idx = ((size_t)blockIdx.x)*THREADS_PER_BLOCK + threadIdx.x;
	if( idx < N )
	{
		sub_result = vec[idx] - ref[idx];
		sub[threadIdx.x] = ( sub_result *= sub_result);
	}

	__syncthreads();

	if( threadIdx.x != 0 )
		return;
	
	size_t maxIdx = ( (idx+THREADS_PER_BLOCK)<N ? (idx+THREADS_PER_BLOCK) : N ), iter=idx+1;
	while( iter<maxIdx )
	{
		sub_result += sub[iter];
	}
	vec[idx] = sub_result;
}
