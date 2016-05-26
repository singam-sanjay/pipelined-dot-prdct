/* Load and Write Functions */
#include <fstream>

void __ld_data_frm_binfile( const char *file_name, size_t offset, const char *var_name, TYPE *ptr, size_t num, int line )
{
  try
  {
    std::ifstream f( file_name, std::ifstream::binary );
    if( !f.is_open() )
    {
      err_sstr << "ld__frm_file_to_CPU::Unable to open " << file_name << '\n';
      throw_str_excptn();
    }
    f.exceptions( std::ifstream::failbit | std::ifstream::badbit | std::ifstream::eofbit );
    f.seekg( offset, std::ios_base::beg );
    f.read( (char*)ptr, sizeof(TYPE)*num );
    f.close();
  }
  catch( std::ios_base::failure &fail )
  {
    err_sstr << "ld__frm_file_to_CPU::" << line << "::failure::" << file_name << "::" << fail.what() << '\n';
    throw_str_excptn();
  }
}

void ld__frm_file_to_CPU( const char * dataset, const char * computed_result = NULL )
{
  #define ld_cpu_MACRO(file,offset,var,size) { __ld_data_frm_binfile(file,offset,#var,var,size,__LINE__); }
  ld_cpu_MACRO(dataset, 0,cpu_vec,1*N);
  ld_cpu_MACRO(dataset,+N,cpu_mat,k*N);
  #ifdef DEBUG
  ld_cpu_MACRO(computed_result,0,cpu_res,k*N);
  #endif
  #undef ld_cpu_MACRO
}

#include "cublas_v2.h"

void __ld_CPU_to_GPU( TYPE* d_vec, TYPE* vec, size_t bytes, const char* var_name, const char* d_var_name )
{
	cudaError_t stat;
	stat = cudaMemcpy( d_vec, vec, bytes, cudaMemcpyHostToDevice  );
	if( stat==cudaSuccess )
	{
    return;
  }
	err_sstr << "ld__frm_CPU_to_GPU::" << var_name << "->" << d_var_name << "::";
  switch( stat )
  {
    case cudaErrorInvalidValue : err_sstr << "parameters passed to the API call is not within an acceptable range of values.\n";break;
    case cudaErrorInvalidDevicePointer : err_sstr << "at least one device pointer passed to the API call is not a valid device pointer\n";break;
    case cudaErrorInvalidMemcpyDirection 	 : err_sstr << "direction of the memcpy passed to the API call is not one of the types specified by cudaMemcpyKind\n";break;
  }
  throw_str_excptn();
}

void ld__frm_CPU_to_GPU()
{
  #define ld_GPU_dat_MACRO( d_var,var,bytes ) { __ld_CPU_to_GPU( d_var,var,bytes,#var,#d_var ); }
  ld_GPU_dat_MACRO( gpu_vec,cpu_vec,N );
  ld_GPU_dat_MACRO( gpu_rep_mat,cpu_mat,(k*N) );
  #undef ld_GPU_dat_MACRO
}

void rp__frm_rplca_to_wrkspc_on_GPU()
{
  cudaError_t status;
  status = cudaMemcpy(gpu_wrk_mat,gpu_rep_mat,k*N,cudaMemcpyDeviceToDevice);
  if( status==cudaSuccess )
  {
    return;
  }
  err_sstr << __func__ << "::" ;
  switch( status )
  {
    case cudaErrorInvalidValue : err_sstr << "parameters passed to the API call is not within an acceptable range of values.\n";break;
    case cudaErrorInvalidDevicePointer : err_sstr << "at least one device pointer passed to the API call is not a valid device pointer\n";break;
    case cudaErrorInvalidMemcpyDirection 	 : err_sstr << "direction of the memcpy passed to the API call is not one of the types specified by cudaMemcpyKind\n";break;
  }
  throw_str_excptn();
}

#ifdef DEBUG
void wb__to_CPU_frm_GPU()
{
  cublasStatus_t status;
  status = cublasGetVector( size_vec,sizeof(TYPE),gpu_res,1,cpu_res,1 );
  if( status!=CUBLAS_STATUS_SUCCESS )
  {
    err_sstr << __func__ << "::gpu_res->cpu_res::";
    switch( status )
    {
      case CUBLAS_STATUS_NOT_INITIALIZED: err_sstr << "The library was not initialized.\n";break;
      case CUBLAS_STATUS_INVALID_VALUE: err_sstr << "The parameters incx , incy , elemSize<=0\n";break;
      case CUBLAS_STATUS_MAPPING_ERROR: err_sstr << "There was an error accessing GPU memory\n";break;
      default:			  err_sstr << "Got something else\n";break;
    }
    throw_str_excptn();
  }
}

void wb__to_file_frm_CPU()
{
  /*
    Not writing any supporing function for this since only 1 variable has to written back
    */
  try
  {
    ofstream f( file_name, std::ifstream::binary );
    if( !f.is_open() )
    {
      err_sstr << __func__ << "::Unable to open " << file_name << '\n';
      throw_str_excptn();
    }
    f.exceptions( std::ifstream::failbit | std::ifstream::badbit | std::ifstream::eofbit );
    f.write( ptr, sizeof(TYPE)*num );
    f.close();
  }
  catch( std::ios_base::failure &fail )
  {
    err_sstr << __func__ << "::failure::" << file_name << "::" << fail.what() << '\n';
    throw_str_excptn();
  }
}
#endif
