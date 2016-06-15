/* kernels and cuBLAS call */
__constant__ TYPE alpha = -1.0, beta = 1.0;
TYPE *gpu_addr_alpha,*gpu_addr_beta;

const char* __cuBLAS_error_string( cublasStatus_t stat )
{
  switch( stat )
  {
    case CUBLAS_STATUS_SUCCESS          : return "the operation completed successfully";
    case CUBLAS_STATUS_NOT_INITIALIZED  : return "the library was not initialized";
    case CUBLAS_STATUS_ARCH_MISMATCH    : return "the device does not support double-precision";
    case CUBLAS_STATUS_EXECUTION_FAILED : return "CUBLAS_STATUS_EXECUTION_FAILED";
    default                             : return "<Unknown Error>";
  }
}

void setup_cuBLAS_func_env()
{
	cudaError_t err;
	err = cudaGetSymbolAddress( (void**)&gpu_addr_alpha , alpha );
	if( err!=cudaSuccess )
	{
		err_sstr << "Error while getting GPU address of alpha::" << cudaGetErrorString(err) << '\n';
		throw_str_excptn();
	}
	err = cudaGetSymbolAddress( (void**)&gpu_addr_beta , beta );
	if( err!=cudaSuccess )
	{
		err_sstr << "Error while getting GPU address of beta::" << cudaGetErrorString(err) << '\n';
		throw_str_excptn();
	}
	cublasStatus_t stat;
	stat = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
	if( stat != CUBLAS_STATUS_SUCCESS )
	{
		err_sstr << "cublasSetPointerMode::" << __cuBLAS_error_string(stat) << ".\n";
		throw_str_excptn();
	}
}

void seq()
{
  cublasStatus_t stat;
  for( int row_num = 0 ; row_num<k ; ++row_num )
  {
    stat = cublasDaxpy( handle, N, gpu_addr_alpha, gpu_vec, 1, gpu_wrk_mat+N*row_num, 1 );
    if( stat!=CUBLAS_STATUS_SUCCESS )
    {
      err_sstr << __func__ << "::cublasDaxpy::" << __cuBLAS_error_string(stat);
      throw_str_excptn();
    }
    stat = cublasDnrm2( handle, N, gpu_wrk_mat+N*row_num, 1, gpu_res+row_num );
    if( stat!=CUBLAS_STATUS_SUCCESS )
    {
      err_sstr << __func__ << "::cublasDnrm2::" << __cuBLAS_error_string(stat);
      throw_str_excptn();
    }
  }
}

void par_OpenMP()
{
  cublasStatus_t stat;
  #pragma omp parallel for private(stat)
  for( int row_num = 0 ; row_num<k ; ++row_num )
  {
    stat = cublasDaxpy( handle, N, gpu_addr_alpha, gpu_vec, 1, gpu_wrk_mat+N*row_num, 1 );
    if( stat!=CUBLAS_STATUS_SUCCESS )
    {
      err_sstr << __func__ << "::cublasDaxpy::" << __cuBLAS_error_string(stat);
      throw_str_excptn();
    }
    stat = cublasDnrm2( handle, N, gpu_wrk_mat+N*row_num, 1, gpu_res+row_num );
    if( stat!=CUBLAS_STATUS_SUCCESS )
    {
      err_sstr << __func__ << "::cublasDnrm2::" << __cuBLAS_error_string(stat);
      throw_str_excptn();
    }
  }
}
/*
__global__ void dyn_par_kernel( cublasHandle_t handle, int N, TYPE *gpu_addr_alpha, TYPE *gpu_vec, TYPE *gpu_wrk_mat, TYPE *gpu_res )
{
  int row_num = blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;
  if( row_num>=N )
  {
    return;
  }
  cublasStatus_t stat;
  stat = cublasDaxpy( handle, N, gpu_addr_alpha, gpu_vec, 1, gpu_wrk_mat+N*row_num, 1 );
  if( stat!=CUBLAS_STATUS_SUCCESS )
  {
    // Globally visible data / Write error message to string in global memory
    printf("%s::%i::cublasDaxpy::error",__func__,row_num);
    asm("trap;");
  }
  stat = cublasDnrm2( handle, N, gpu_wrk_mat+N*row_num, 1, gpu_res+row_num );
  if( stat!=CUBLAS_STATUS_SUCCESS )
  {
    // Globally visible data / Write error message to string in global memory
    printf("%s::%i::cublasDnrm2::error",__func__,row_num);
    asm("trap;");
  }
}

void par_dyn_parll()
{
  #define NUMBER_OF_BLOCKS ( (k-1)/THREADS_PER_BLOCK + 1 )
  dyn_par_kernel<<<NUMBER_OF_BLOCKS,THREADS_PER_BLOCK>>>(handle,N,gpu_addr_alpha,gpu_vec,gpu_wrk_mat,gpu_res);
  cudaError_t err = cudaGetLastError();
  if( err!=cudaSuccess )
  {
    err_sstr << __func__ << cudaGetErrorString(err);
    throw_str_excptn();
  }
}
*/

__global__ void pipeline_kernel( int N, TYPE *gpu_mat, TYPE *gpu_res )
{
  extern __shared__ TYPE sw_cache[];
  TYPE *src,*dest;
  int c1,c2,lvl=0; // Perf chk : Try unsigned char lvl instead of int lvl to 1.reduce total size of registers
  unsigned char class_of_thread = '\0';
  if( threadIdx.x<N )
  {
    src = gpu_mat;
    dest = sw_cache;
    c1 = threadIdx.x;
    //class_of_thread =
  }
  else
  {
    int N_lvl = N, prev_lvl_1st, curr_lvl_1st = 0, nxt_lvl_1st = N+1;
    do
    {
      prev_lvl_1st = curr_lvl_1st;
      curr_lvl_1st = nxt_lvl_1st;
      N_lvl = (N_lvl/2) + (N_lvl%2); // N_lvl = ceil(N/2) = N/2 + N%2 = (N>>1) + N&1
      /* Perf chk
      1. Try
        a. (N>>1) instead of (N/2)
        b. (N&1) instead of (N%2)
        c. ceil instead of all this arith magic
      2. Use explicit temporary register "temp_int" , and see if  1.performance improves and  2.number of registers is reduced
      */
      nxt_lvl_1st = curr_lvl_1st + N_lvl;
      ++lvl;
    }while( threadIdx.x<nxt_lvl_1st );
    c1 = prev_lvl_1st + ((threadIdx.x-curr_lvl_1st)*2);
    // Perf chk :                                  ^^ try ((...blah...)<<1) instead of ((...blah...)*2)
    if( (c1+1)<nxt_lvl_1st )
    {
      c2 = c1+1; //using c2 as c1+1 since avoiding 1.extra addition including reg access.  and  2.repetetive addition
    }
    else
    {
      c2 = blockDim.x-1; //assuming the last thread's __shared__ spot is a 'source of zeros', sw_cache[blockDim.x-1]==0, since right child doesn't exist.
    }
    /* Perf chk : avoid temp reg for c1+1
    ++c1;
    if( c1>=nxt_lvl_1st )
    {
      c2 = c1;
    }else{...}
    --c1;
    */
    src = sw_cache;
    if( threadIdx.x == blockDim.x-1 )
    {
      dest = gpu_res;
      sw_cache[threadIdx.x] = 0; //init 'source of zeros' only once to avoid 1.redundant writes  and  2.serialised writes to same memory bank
      /* Perf chk
      Instead of treating Intermediate Threads(IT) in the same way as reducing threads (by having the "source of zeros" in place of the missing right child),
      Try treating them differently : 1 Big(#threads) Slow warp Vs 1 Smaller(#threads) warp + few threads diverege and execute simpler instructions
      expected effect : slower run, because 1 Big warp and 1 Smaller warp would take the same time.
      */
      //class_of_thread =
    }
    else
    {
      dest = sw_cache;
      //class_of_thread =
    }
  }

  TYPE src1, src2;
  while(true)
  {
    __syncthreads();
    src1 = src[c1];
    src2 = src[c2];
    if( class_of_thread==1 )
    {
      src1 -= src2;
      src1 *= src1;
    }
    else if( class_of_thread&2 )
    {
      src1 += src2;
    }

    __syncthreads();
    if( class_of_thread&1 )
    {
      dest[threadIdx.x] = src1;
    }
    else if( class_of_thread==2 )
    {
      dest[iter] = src1;
    }

    ++iter;
    if( iter==N )
    {
      class_of_thread = 0;
    }
    else if( iter==0 )
    {
      class_of_thread>>=2;
    }
    else if( !class_of_thread )
    {
      --wait;
      if( !(wait) )
      {
        return;
      }
    }

  }
}

void pipelined()
{

}
