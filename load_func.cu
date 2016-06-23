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
    f.seekg( offset*sizeof(TYPE), std::ios_base::beg );
    f.read( (char*)ptr, sizeof(TYPE)*num );
    f.close();
  }
  catch( std::ios_base::failure &fail )
  {
    err_sstr << "ld__frm_file_to_CPU::" << line << "::failure::" << file_name << "::" << fail.what() << '\n';
    throw_str_excptn();
  }
}

void ld__frm_file_to_CPU( const char * dataset )
{
  #define ld_cpu_MACRO(file,offset,var,size) { __ld_data_frm_binfile(file,offset,#var,var,size,__LINE__); }
  ld_cpu_MACRO(dataset, 0,cpu_vec,1*N);
  ld_cpu_MACRO(dataset,+N,cpu_mat,k*N);
  #undef ld_cpu_MACRO
}

void __cudaMemcpy_wrapper( TYPE* &dest, TYPE *&src, int num_bytes, enum cudaMemcpyKind kind, const char *caller_func_name, const char *dest_name, const char *src_name )
{
  static cudaError_t stat;
  stat = cudaMemcpy( dest, src, num_bytes, kind );
  if( stat==cudaSuccess )
  {
    #ifdef DEBUG
    std::cerr << caller_func_name << ":(" << num_bytes << "B)" << src_name << "->" << dest_name << std::endl;
    #endif
    return;
  }
  err_sstr << caller_func_name << "::" << src_name << "->" << dest_name << "::";
  switch( stat )
  {
    case cudaErrorInvalidValue             : err_sstr << "parameters passed to the API call is not within an acceptable range of values.\n";break;
    case cudaErrorInvalidDevicePointer     : err_sstr << "at least one device pointer passed to the API call is not a valid device pointer\n";break;
    case cudaErrorInvalidMemcpyDirection 	 : err_sstr << "direction of the memcpy passed to the API call is not one of the types specified by cudaMemcpyKind\n";break;
    default                                : err_sstr << "Unknown error.\n";break;
  }
  throw_str_excptn();
}

void __ld_CPU_to_GPU( TYPE* d_vec, TYPE* vec, int bytes, const char* var_name, const char* d_var_name )
{
  __cudaMemcpy_wrapper( d_vec, vec, bytes, cudaMemcpyHostToDevice, "ld__frm_CPU_to_GPU", d_var_name, var_name );
}

void ld__frm_CPU_to_GPU()
{
  #define ld_GPU_dat_MACRO( d_var,var,bytes ) { __ld_CPU_to_GPU( d_var,var,bytes,#var,#d_var ); }
  ld_GPU_dat_MACRO( gpu_vec,cpu_vec,sizeof(TYPE)*N );
  ld_GPU_dat_MACRO( gpu_rep_mat,cpu_mat,(sizeof(TYPE)*k*N) );
  #undef ld_GPU_dat_MACRO
}

void rp__frm_rplca_to_wrkspc_on_GPU()
{
  __cudaMemcpy_wrapper( gpu_wrk_mat, gpu_rep_mat, sizeof(TYPE)*k*N, cudaMemcpyDeviceToDevice, __func__, "gpu_wrk_mat", "gpu_rep_mat");
}

#ifdef DEBUG
void wb__to_CPU_frm_GPU()
{
  __cudaMemcpy_wrapper( cpu_res, gpu_res, sizeof(TYPE)*k, cudaMemcpyDeviceToHost, __func__, "cpu_res", "gpu_res" );
}

void wb__to_file_frm_CPU( const char *results )
{
  /*
    Not writing any supporing function for this since only 1 variable has to written back
    */
  try
  {
    std::ofstream f( results, std::ifstream::binary );
    if( !f.is_open() )
    {
      err_sstr << __func__ << "::Unable to open " << results << '\n';
      throw_str_excptn();
    }
    f.exceptions( std::ifstream::failbit | std::ifstream::badbit | std::ifstream::eofbit );
    f.write( (char*)cpu_res, sizeof(TYPE)*k );
    f.close();
  }
  catch( std::ios_base::failure &fail )
  {
    err_sstr << __func__ << "::failure::" << results << "::" << fail.what() << '\n';
    throw_str_excptn();
  }
}
#endif
