/* Miscellaneous helper functions */

void cmdln_usage_help()
{
  cerr << "Usage : ./a.out full_RxC.bin[or]sp_RxC.bin [FILES]\n"
          "1 or more binary files with\n"
          "names of the form full_RxC.bin or sp_RxC.bin, where\n"
          "R (int) indicates the total number of roes including the feature vector and\n"
          "C (int) indicates the dimension of the vectors.\n"
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
  itn R, C;
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
      err_sstr << __func__ << "::sscanf failed to parse " << files[f_ter] << '\n';
      goto clse;
    }
    if( size != R*C*sizeof(TYPE) )
    {
      got_error = true;
      err_sstr << __func__ << "::size mimatch::expected " << sizeof(TYPE)*R*C << "B with each element " << sizeof(BYTE) << "B ::got " << size << "B\n";
    }
clse:fclose(f);
  }
  if( got_error )
  {
    throw_str_excptn();
  }
}

void cub_init()
{
	 cublasStatus_t status;
         status = cublasCreate( &handle );
	 if( status!=CUBLAS_STATUS_SUCCESS )
	 {
		 err_str << __func__ << ": ";
	 }
	 else
	 {
		 return;
	 }
	 switch( status )
	 {
		 case CUBLAS_STATUS_NOT_INITIALIZED: err_str << "The CUDA TM Runtime initialization failed\n";break;
		 case CUBLAS_STATUS_ALLOC_FAILED: err_str << "The resources could not be allocated\n";break;
		 default:                         err_str << "Got something else\n";break;
	 }
	 throw_str_excptn();
}

void cub_wrapup()
{
	 cublasStatus_t status;
         status = cublasDestroy( handle );
	 if( status!=CUBLAS_STATUS_SUCCESS )
	 {
		 err_str << __func__ << ": ";
	 }
	 else
	 {
		 return;
	 }
	 switch( status )
	 {
		 case CUBLAS_STATUS_NOT_INITIALIZED: err_str << "The library was not initialized\n";
		 default:                         err_str << "Got something else\n";
	 }
	 throw_str_excptn();
}
