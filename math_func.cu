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
  #if DEBUG
  cublasStatus_t stat;
  std::cerr << __func__ << "()" << std::endl;
  #endif
  for( int row_num = 0 ; row_num<k ; ++row_num )
  {
    #ifdef DEBUG
    //char var_name[] = "gpu_wrk_mat+N*row_num";
    stat = cublasDaxpy( handle, N, gpu_addr_alpha, gpu_vec, 1, gpu_wrk_mat+N*row_num, 1 );
    if( stat!=CUBLAS_STATUS_SUCCESS )
    {
      err_sstr << __func__ << "::cublasDaxpy::" << __cuBLAS_error_string(stat);
      throw_str_excptn();
    }
    //sprintf(var_name,"gpu_wrk_mat+N*%i",row_num);
    //print_gpu_var(gpu_wrk_mat+N*row_num,var_name,1,N);
    stat = cublasDnrm2( handle, N, gpu_wrk_mat+N*row_num, 1, gpu_res+row_num );
    //sprintf(var_name,"gpu_res+%i",row_num);
    //print_gpu_var(gpu_res+row_num,var_name,1,1);
    if( stat!=CUBLAS_STATUS_SUCCESS )
    {
      err_sstr << __func__ << "::cublasDnrm2::" << __cuBLAS_error_string(stat);
      throw_str_excptn();
    }
    #else
    (void)cublasDaxpy( handle, N, gpu_addr_alpha, gpu_vec, 1, gpu_wrk_mat+N*row_num, 1 );
    (void)cublasDnrm2( handle, N, gpu_wrk_mat+N*row_num, 1, gpu_res+row_num );
    #endif
  }
  #ifdef DEBUG
  std::cerr << "completed" << __func__ << "()" << std::endl;
  #endif
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

#define CLASS_LEAF (0b01)
#define CLASS_ROOT (0b10)
#define CLASS_INTM (0b11)
#define is_CLASS_LEAF (class_of_thread==CLASS_LEAF)
#define is_CLASS_ROOT (class_of_thread==CLASS_ROOT)
#define have_2nd_MEM_OPRND (class_of_thread&2)
#define save_to_SHRD_MEM (class_of_thread&1)
#define save_to_GLBL_MEM (is_CLASS_ROOT)
#define is_ACTIVE (class_of_thread&3)
#define just_became_ACTIVE (iter==0)
#define IDLE_and_DONE (class_of_thread==0)
#define wait_is_over (!(wait))

__global__ void pipeline_kernel( int N, int k, TYPE *gpu_vec, TYPE *gpu_mat, TYPE *gpu_res )
{
  extern __shared__ TYPE sw_cache[];
  TYPE *src1,*src2,op1,op2,*dest;
  int iter,c1,c2,lvl=0,max_lvl,wait; // Perf chk : Try unsigned char lvl instead of int lvl to 1.reduce total size of registers
  unsigned char class_of_thread = '\0';
  if( threadIdx.x<N )
  {
    src1 = gpu_mat;
    src2 = gpu_vec;
    dest = sw_cache;
    c1 = c2 = threadIdx.x;
    op2 = gpu_vec[c2]; //load the reference vector.
    class_of_thread = CLASS_LEAF;
  }
  else
  {
    int N_lvl = N, prev_lvl_1st, curr_lvl_1st = 0, nxt_lvl_1st = N;
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
    }while( (curr_lvl_1st>threadIdx.x) || (threadIdx.x>=nxt_lvl_1st) );//!(curr_lvl_1st<=threadIdx.x && threadIdx.x<nxt_lvl_1st)
    //Perf chk :                       ^^ once the 1st cond fails, the 2nd immediately fails
    c1 = prev_lvl_1st + ((threadIdx.x-curr_lvl_1st)*2);
    // Perf chk :                                  ^^ try ((...blah...)<<1) instead of ((...blah...)*2)
    if( (c1+1)<curr_lvl_1st )
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
    src1 = src2 = sw_cache;
    if( threadIdx.x == blockDim.x-1 )
    {
      dest = gpu_res;
      sw_cache[threadIdx.x] = lvl;
      /* Perf chk
      Instead of treating Intermediate Threads(IT) in the same way as reducing threads (by having the "source of zeros" in place of the missing right child),
      Try treating them differently : 1 Big(#threads) Slow warp Vs 1 Smaller(#threads) warp + few threads diverege and execute simpler instructions
      expected effect : slower run, because 1 Big warp and 1 Smaller warp would take the same time.
      */
      class_of_thread = CLASS_ROOT;
    }
    else
    {
      dest = sw_cache;
      class_of_thread = CLASS_INTM;
    }
    class_of_thread <<= 2;//These threads are idle, initially
  }

  iter = -lvl;
  __syncthreads(); //waiting for max_lvl
  max_lvl = sw_cache[blockDim.x-1];
  __syncthreads(); //Waiting to for all to read max_lvl
  wait = max_lvl-lvl;
  if( threadIdx.x==blockDim.x-1)
  {
    sw_cache[threadIdx.x] = 0; //init 'source of zeros' only once to avoid 1.redundant writes  and  2.serialised writes to same memory bank
    //The 1st __syncthreads in while would ensure this write be visible to all threads
  }

  /*Perf chk :
  src1 += c1;//Eliminate repeated addition of offset in src1[c1]
  src2 += c2;//Eliminate repeated addition of offset in src2[c2]*/
  while(true)
  {
    __syncthreads();
    op1 = src1[c1];
    if( have_2nd_MEM_OPRND )
    {
      op2 = src2[c2];
      op1 += op2;
    }
    else if( is_CLASS_LEAF )
    {
      op1 -= op2;
      op1 *= op1;
      src1 += N;//Go to next vector
    }
    /*Perf chk : exchange and observe if perf improvement
      Possible reason : else if executed first, we could execute the other branch of the warp while one branch waits for data from shared memory*/

    __syncthreads();
    if( save_to_SHRD_MEM )
    {
      dest[threadIdx.x] = op1;
    }
    else if( save_to_GLBL_MEM )
    {
      dest[iter] = sqrt(op1);
    }

    ++iter;
    if( is_ACTIVE )//Changed control structure since active threads were going through all the conditions
    {
      if( iter==k )
      {
        class_of_thread = 0;
        //Idle threads keep accessing memory, to prevent that direct then to some safe source (that doesn't cost much)
        src1 = sw_cache;
        // Now all Idle threads would be hitting the same location, so BROADCAST :)
        c1 = blockDim.x-1;
      }
    }
    else if( just_became_ACTIVE )
    {
      class_of_thread>>=2;
    }
    else if( IDLE_and_DONE )
    {
      if( wait_is_over )
      {
        break;
      }
      --wait; //tID==blockDim.x-1 doesn't have to wait at all
    }

  }
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

void pipelined()
{

}
