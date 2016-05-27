/* Miscellaneous helper functions */

void cmdln_usage_help()
{
  std::cerr << "Usage : ./a.out full_RxC.bin[or]sp_RxC.bin [FILES]\n"
          "1 or more binary files with\n"
          "names of the form full_RxC.bin or sp_RxC.bin, where\n"
          "R (int)(>2) indicates the total number of roes including the feature vector and\n"
          "C (int)(>1) indicates the dimension of the vectors.\n";
}


#include <cstdio>

bool extract_RandC_from_fname( const char *f_name, int *ptr_R, int *ptr_C )
{
  int R,C;
  if( sscanf(f_name,"full_%ix%i.bin",&R,&C)!=2 || sscanf(f_name,"sp_%ix%i.bin",&R,&C)!=2 )
  {
    return false;
  }
  *ptr_R = R;
  *ptr_C = C;
  return true;
}

void verify_cmdln_args( char *files[], int num_files )
{
  FILE *f;fpos_t size;
  int R, C;
  bool got_error = false;
  for( int f_iter = 0 ; f_iter < num_files ; ++f_iter )
  {
    f = fopen( files[f_iter], "rb" );
    if( f==NULL )
    {
      got_error = true;
      err_sstr << __func__ << "::failed to open " << files[f_iter] << '\n';
      continue;
    }
    if( fseek( f, 0, SEEK_END )!=0 )
    {
      got_error = true;
      err_sstr << __func__ << "::fseek failed to process " << files[f_iter] << '\n';
      goto clse;
    }
    if( fgetpos(f,&size)!=0 )
    {
      got_error = true;
      err_sstr << __func__ << "::fgetpos failed to process " << files[f_iter] << '\n';
      goto clse;
    }
    if( extract_RandC_from_fname(files[f_iter],&R,&C)==false ) // 2 since 2 arguments need to be assigned
    {
      got_error = true;
      err_sstr << __func__ << "::sscanf failed to parse " << files[f_iter] << '\n';
      goto clse;
    }
    if( R<2 || C<1 )
    {
      got_error = true;
      err_sstr << __func__ << "::R<2 or C<1 in " << files[f_iter] << '\n';
      goto clse;
    }
    if( size.__pos != sizeof(TYPE)*R*C )
    {
      got_error = true;
      err_sstr << __func__ << "::size mimatch::expected " << sizeof(TYPE)*R*C << "B with each element " << sizeof(TYPE) << "B ::got " << size.__pos << "B\n";
    }
    if( !got_error )
    {
      max_k = max(R-1,max_k);
      max_N = max(  C,max_N);
    }
clse:fclose(f);
  }
  if( got_error )
  {
    throw_str_excptn();
  }
}

#include "cublas_v2.h"
void cub_init()
{
	 cublasStatus_t status;
   status = cublasCreate( &handle );
	 if( status==CUBLAS_STATUS_SUCCESS )
	 {
		 return;
	 }
   /* failure */
   err_sstr << __func__ << ": ";
	 switch( status )
	 {
		 case CUBLAS_STATUS_NOT_INITIALIZED: err_sstr << "The CUDA TM Runtime initialization failed\n";break;
		 case CUBLAS_STATUS_ALLOC_FAILED:    err_sstr << "The resources could not be allocated\n";break;
		 default:                            err_sstr << "Got something else\n";break;
	 }
	 throw_str_excptn();
}

void cub_wrapup()
{
	 cublasStatus_t status;
   status = cublasDestroy( handle );
	 if( status==CUBLAS_STATUS_SUCCESS )
	 {
		 return;
	 }
   /* failure */
   err_sstr << __func__ << ": ";
	 switch( status )
	 {
		 case CUBLAS_STATUS_NOT_INITIALIZED: err_sstr << "The library was not initialized\n";break;
		 default:                            err_sstr << "Got something else\n";
	 }
	 throw_str_excptn();
}
