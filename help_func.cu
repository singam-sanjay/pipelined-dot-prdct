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
void extrct_and_verify_cmdln_args( char *files[], int num_files )
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
    if( sscanf(files[f_iter],"full_%ix%i.bin",&R,&C)!=2 || sscanf(files[f_iter],"sp_%ix%i.bin",&R,&C)!=2 ) // 2 since 2 arguments need to be assigned
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
