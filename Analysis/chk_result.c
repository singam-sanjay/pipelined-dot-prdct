#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <stdbool.h>
#include <math.h>
#include "../env_decl.h"

bool extract_RandC_from_fname( char *f_name, int *ptr_R, int *ptr_C )
{
  int R,C;
  const char *base_name = basename(f_name);
  if( (sscanf(base_name,"ans_full_%ix%i.bin",&R,&C)==2) || (sscanf(base_name,"ans_sp_%ix%i.bin",&R,&C)==2) ||  (sscanf(base_name,"res_sp_%ix%i.bin",&R,&C)==2) || (sscanf(base_name,"res_full_%ix%i.bin",&R,&C)==2) )
  {
    *ptr_R = R;
    *ptr_C = C;
    return false;
  }
  printf("Unable to parse %s.\n",f_name);
  return true;
}

bool chk_size_inconcistency( int R, int C, char *file )
{
  fpos_t size;
  bool got_error = false;
  FILE *f = fopen( file, "rb" );
  if( f==NULL )
  {
    got_error = true;
    printf("%s::failed to open %s.\n",__func__,file);
    goto clse;
  }
  if( fseek( f, 0, SEEK_END )!=0 )
  {
    got_error = true;
    printf("%s::fseek failed to process %s.\n",__func__,file);
    goto clse;
  }
  if( fgetpos(f,&size)!=0 )
  {
    got_error = true;
    printf("%s::fseek failed to process %s.\n",__func__,file);
    goto clse;
  }
  if( R<2 || C<1 )
  {
    got_error = true;
    printf("%s::R<2 or C<1 in %s.\n",__func__,file);
    goto clse;
  }
  R = R-1;
  if( size.__pos != sizeof(TYPE)*R ) // Warning : size.__pos is long (signed), sizeof(TYPE)*R*C is size_t (unsigned, uint{64,32}_t)
  {
    got_error = true;
    printf("%s::size mimatch::expected %zuB::got %li::%s.\n",__func__,sizeof(TYPE)*R,size.__pos,file);
  }
  clse:
  fclose(f);
  return got_error;
}

void file_chk( int *R1, int *C1, char *file1, int *R2, int *C2, char *file2 )
{
	if( extract_RandC_from_fname(file1,R1,C1) || chk_size_inconcistency(*R1,*C1,file1) )
	{
		printf("Unable to process the 1st file.\n");
		exit(EXIT_FAILURE);
	}
  --(*R1);
	if( extract_RandC_from_fname(file2,R2,C2) || chk_size_inconcistency(*R2,*C2,file2) )
	{
		printf("Unable to process the 2nd file.\n");
		exit(EXIT_FAILURE);
	}
  --(*R2);
	if( *R1!=*R2 || *C1!=*C2 )
	{
		printf("Dimension mismatch::1st is %ix%i::2nd is %ix%i.\n",*R1,*C1,*R2,*C2);
		exit(EXIT_FAILURE);
	}
}

bool fread_handled( FILE *f, TYPE *ip, int f_num, int R )
{
	size_t ret;
	if( (ret=fread(ip,sizeof(TYPE),R,f))!=R )
	{
		printf("fread::file%i::ret=%zu::",f_num,ret);
		if( feof(f) )
		{
			printf("EOF.\n");
		}
		else if( ferror(f) )
		{
			printf("Error.\n");
		}
		else
		{
			printf("Unknown error.\n");
		}
    return true;
	}
  return false;
}

#define close_all_f() {fclose(f1);fclose(f2);}
#define free_all_ip() {free(ip1);free(ip2);}

int main( int argc, char* argv[] )
{
  if( argc!=3 )
  {
    fprintf(stderr,"Usage : ./a.out file1 file2.\n"
                  "file1 and file2 are of the format {ans,res}_{full,sp}_RxC.bin.\n");
    return EXIT_FAILURE;
  }

  int R1,C1,R2,C2;

  file_chk( &R1,&C1,argv[1],  &R2,&C2,argv[2] );

  FILE *f1,*f2;
  f1 = fopen(argv[1],"rb");
  f2 = fopen(argv[2],"rb");


  TYPE *ip1,*ip2;
  ip1 = (TYPE*)malloc( sizeof(TYPE)*R1 );
  ip2 = (TYPE*)malloc( sizeof(TYPE)*R2 );
  if( ip1==NULL || ip2==NULL )
  {
	  printf("Unable to allocate memory for ip1 and ip2.\n");
	  close_all_f();
	  return EXIT_FAILURE;
  }

  if( fread_handled( f1,ip1,1,R1 ) || fread_handled( f2,ip2,2,R2 ) )
  {
	  close_all_f();
	  free_all_ip();
	  return EXIT_FAILURE;
  }

  register int iter;
  register TYPE (*abs_f)(TYPE);
  #if defined _TYPE_double_
    abs_f = fabs;
  #elif defined _TYPE_float_
    abs_f = fabsf;
  #endif

  for( iter=0 ; iter<R1 ; ++iter )
  {
	  if( abs_f(ip1[iter]-ip2[iter]) > 1.0e-12 )
	  {
		  printf("Error @%i.\n",iter);
		  close_all_f();
		  free_all_ip();
		  return EXIT_FAILURE;
	  }
  }

  close_all_f();
  free_all_ip();
  return EXIT_SUCCESS;
}
