/* Alloc and Free functions */

#include <new> // For std::bad_alloc ba

TYPE* handled_new( size_t bytes, const char *var_name, int line )
{
	static TYPE* ptr;
	try
	{
		ptr = new TYPE[bytes];
	}
	catch(std::bad_alloc ba)
	{
		err_sstr << "alloc_mem_CPU:" << line << ": Error while alloc mem for " << var_name << " : " << ba.what() << '\n';
		throw_str_excptn(); // Immediately raise an exception as soon as we're unable to allocate memory,
                        // since the program cannot progress without required memory.
	}
	return ptr;
}

bool handled_delete( TYPE *loc, const char *var_name, int line )
{
  /*
    Returns true if delete fails
  */
	try
	{
		delete[] loc;
	}
	catch(std::bad_alloc ba)
	{
		err_sstr << "free_mem_CPU:" << line << ": Error while freeing mem of "<< var_name << " : " << ba.what() << '\n';
    return true;
    // Do not raise an exception even if delete fails, since the program is now given a chance to free other allocations
	}
  return false;
}

void alloc_mem_CPU()
{
  #define alloc_CPU_MACRO(var,size) { var = handled_new( sizeof(TYPE)*size,#var,__LINE__); }
  alloc_CPU_MACRO(cpu_vec,max_N);
  alloc_CPU_MACRO(cpu_mat,max_k*max_N);
  #ifdef DEBUG
  alloc_CPU_MACRO(cpu_res,max_k);
  #endif
  #undef alloc_CPU_MACRO
}

void free_mem_CPU()
{
  bool got_error = false;
  #define free_CPU_MACRO(var) { got_error |= handled_delete(var,#var,__LINE__); }
  free_CPU_MACRO(cpu_vec);
  free_CPU_MACRO(cpu_mat);
  #ifdef DEBUG
  free_CPU_MACRO(cpu_res);
  #endif
  #undef free_CPU_MACRO
  if(got_error)throw_str_excptn();
}

TYPE* handled_cudaMalloc( size_t bytes, const char* var_name, int line )
{
	static TYPE* d_ptr;
	switch( cudaMalloc(&d_ptr,bytes) )
	{
		case cudaSuccess :break;
		case cudaErrorMemoryAllocation: err_sstr << "alloc_mem_GPU():" << line << ": Failed to allocate " << var_name << '\n';
						throw_str_excptn();
		default: err_sstr << "alloc_mem_GPU():" << line << ": Received unknown error while allocating " << var_name << '\n';
			 			throw_str_excptn();
	}
	return d_ptr;
}

bool handled_cudaFree( TYPE* d_ptr, const char* var_name, int line )
{
	switch( cudaFree(d_ptr) )
	{
		case cudaSuccess:break;
		case cudaErrorInvalidDevicePointer:err_sstr << "free_mem_GPU():" << line << ":Invalid pointer passed while freeing " << var_name << '\n';
						   return true;
		case cudaErrorInitializationError:err_sstr << "free_mem_GPU():" << line << ":Failed to init CUDA runtime and driver while freeing "<< var_name << '\n';
						  return true;
	}
	return false;
}

void alloc_mem_GPU()
{
	#define alloc_GPU_MACRO(var,size) { var = handled_cudaMalloc( sizeof(TYPE)*size,#var,__LINE__); }
	alloc_GPU_MACRO(gpu_vec,max_N);
	alloc_GPU_MACRO(gpu_wrk_mat,max_k*max_N);
	alloc_GPU_MACRO(gpu_rep_mat,max_k*max_N);
	alloc_GPU_MACRO(gpu_res,max_k);
	#undef alloc_GPU_MACRO
}

void free_mem_GPU()
{
	bool got_error = false;
	#define free_GPU_MACRO(var) { got_error |= handled_cudaFree(var,#var,__LINE__); }
	free_GPU_MACRO(gpu_vec);
	free_GPU_MACRO(gpu_wrk_mat);
	free_GPU_MACRO(gpu_rep_mat);
	free_GPU_MACRO(gpu_res);
	#undef free_GPU_MACRO
	if(got_error)throw_str_excptn();
}
