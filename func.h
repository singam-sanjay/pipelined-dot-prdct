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
	size_t idx = ((size_t)blockIdx.x)*THREADS_PER_BLOCK + threadIdx.x;
	if( idx >= N )
	{
		sub[threadIdx.x] = ZERO_OF_TYPE;
		return;
	}
	TYPE sub_result;

	sub_result = vec[idx] - ref[idx];
	sub[threadIdx.x] = ( sub_result *= sub_result);
	
	__syncthreads();

	if( threadIdx.x != 0 )
		return;
	
	sub_result += (((sub[1]+sub[2])+(sub[3]+sub[4]))+((sub[5]+sub[6])+(sub[7]+sub[8])))+(((sub[9]+sub[10])+(sub[11]+sub[12]))+((sub[13]+sub[14])+sub[15]));
	vec[idx] = sub_result;
}
