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

__device__ void dyn_par_kernel( cublasHandle_t handle, int N, TYPE *gpu_addr_alpha, TYPE *gpu_vec, TYPE *gpu_wrk_mat, TYPE *gpu_res )
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
  dyn_par_kernel<<<NUMBER_OF_BLOCKS,THREADS_PER_BLOCK>>>(handle,N,gpu_addr_alpha,gpu_wrk_mat,gpu_res);
  cudaError_t err = cudaGetLastError();
  if( err!=cudaSuccess )
  {
    err_sstr << __func__ << cudaGetErrorString(err);
    throw_str_excptn();
  }
}
