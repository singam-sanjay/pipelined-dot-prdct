/* Miscellaneous helper functions */

void rst_err_sstr()
{
  err_sstr.str("");
  err_sstr.clear();
}

void cmdln_usage_help()
{
  std::cerr << "Usage : ./a.out full_RxC.bin[or]sp_RxC.bin [FILES]\n"
          "1 or more binary files with\n"
          "names of the form full_RxC.bin or sp_RxC.bin, where\n"
          "R (int)(>2) indicates the total number of roes including the feature vector and\n"
          "C (int)(>1) indicates the dimension of the vectors.\n";
}

void cuda_set_device( int dev_num )
{
  cudaError_t stat;
  stat = cudaSetDevice( dev_num );
  if( stat==cudaSuccess )
  {
    return;
  }
  err_sstr << __func__ << "::";
  switch( stat )
  {
    case cudaErrorInvalidDevice      : err_sstr << "Invalid device ordinal.\n";break;
    case cudaErrorSetOnActiveProcess : err_sstr << "cudaErrorSetOnActiveProcess\n";break;
    default                          : err_sstr << "Unknown error.\n";break;
  }
  throw_str_excptn();
}
#include <cstdio>
#include <libgen.h>

bool extract_RandC_from_fname( char *f_name, int *ptr_R, int *ptr_C )
{
  int R,C;
  const char *base_name = basename(f_name);
  if( (sscanf(base_name,"full_%ix%i.bin",&R,&C)==2) || (sscanf(base_name,"sp_%ix%i.bin",&R,&C)==2) )
  {
    *ptr_R = R;
    *ptr_C = C;
    return true;
  }
  return false;
}

void verify_cmdln_args( char *files[], const int num_files )
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
    if( size.__pos != sizeof(TYPE)*R*C ) // Warning : size.__pos is long (signed), sizeof(TYPE)*R*C is size_t (unsigned, uint{64,32}_t)
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

void cudaEventInit()
{
  cudaError_t stat;
  stat = cudaEventCreate(&start);
  if( stat!=cudaSuccess )
  {
    err_sstr << __func__ << ":" << cudaGetErrorString(stat) << ":failed creating cudaEvent_t start\n";
    throw_str_excptn();
  }
  stat = cudaEventCreate(&stop);
  if( stat!=cudaSuccess )
  {
    err_sstr << __func__ << ":" << cudaGetErrorString(stat) << ":failed creating cudaEvent_t stop\n";
    throw_str_excptn();
  }
}

void tick(const char __func[])
{
  static cudaError_t stat;
  stat = cudaEventRecord(start,0);
  if( stat!=cudaSuccess )
  {
    err_sstr << __func__ << ":" << cudaGetErrorString(stat) << ": failed to start timer in " << __func << '\n';
    throw_str_excptn();
  }
}

void tock(const char __func[])
{
  static cudaError_t stat;
  static float time_elapsed;
  stat = cudaEventRecord(stop,0);
  if( stat!=cudaSuccess )
  {
    err_sstr << __func__ << ":" << cudaGetErrorString(stat) << ": failed to stop timer in " << __func << '\n';
    throw_str_excptn();
  }
  stat = cudaEventSynchronize(stop);
  if( stat!=cudaSuccess )
  {
    err_sstr << __func__ << ":" << cudaGetErrorString(stat) << ": failed to synchronise timer in " << __func << '\n';
    throw_str_excptn();
  }
  stat = cudaEventElapsedTime(&time_elapsed,start,stop);
  if( stat!=cudaSuccess )
  {
    err_sstr << __func__ << ":" << cudaGetErrorString(stat) << ": failed to measure time in " << __func << '\n';
    throw_str_excptn();
  }
  std::cout << __func << ' ' << time_elapsed << "ms\n";
}

void cudaEventDest()
{
  cudaError_t stat;
  stat = cudaEventDestroy(start);
  if( stat!=cudaSuccess )
  {
    err_sstr << __func__ << ":" << cudaGetErrorString(stat) << ":failed creating cudaEvent_t start\n";
    throw_str_excptn();
  }
  stat = cudaEventDestroy(stop);
  if( stat!=cudaSuccess )
  {
    err_sstr << __func__ << ":" << cudaGetErrorString(stat) << ":failed creating cudaEvent_t stop\n";
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

#ifdef DEBUG
void print_cpu_var(TYPE* cpu_mat, const char var_name[], int k, int N)
{
  int i,j;
  fprintf(stderr,"%s:\n",var_name);
  for( i=0 ; i<k ; ++i )
  {
    for( j=0 ; j<N ; ++j )
    {
      fprintf(stderr,PRINTF_FORMAT_STRING" ",cpu_mat[j]);
    }
    printf("\n");cpu_mat += N;
  }
  fflush(stderr);
}

__global__ void print_gpu_mat_kern( TYPE* gpu_mat, int k, int N )
{
	if( blockIdx.x!=0 || threadIdx.x!=0 )
	return;
	int i,j;
  for( i=0 ; i<k ; ++i )
	{
		for( j=0 ; j<N ; ++j )
		{
			printf(PRINTF_FORMAT_STRING" ",gpu_mat[j]);
		}
		printf("\n");gpu_mat += N;
	}
}

void print_gpu_var(TYPE* gpu_mat, const char var_name[], int k, int N)
{
  printf("%s:\n",var_name);
	print_gpu_mat_kern<<<1,1>>>(gpu_mat,k,N);
  cudaDeviceSynchronize();
}
#endif
