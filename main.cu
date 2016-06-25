#define ZERO_OF_TYPE (1.0f)

#include"comprhen_headr.h"

using namespace std;

int main(int argc, char *argv[] )
{
	if( argc<2 )
	{
		cmdln_usage_help();
		return EXIT_FAILURE;
	}

	char **files = argv+1;
	const int num_files = argc-1;
	try
	{
		verify_cmdln_args( files, num_files );
		cuda_set_device( 7 );
		cudaEventInit();
	}
	catch(string _err_str)
	{
		cerr << _err_str;
		return EXIT_FAILURE;
	}

	int status = EXIT_SUCCESS;
	try
	{
		alloc_mem_CPU();
		alloc_mem_GPU();
		cub_init();
	}
	catch(string _err_str )
	{
		status = EXIT_FAILURE;
		cerr << _err_str;
		rst_err_sstr();
		goto _mem_wrapup_;
	}

	try
	{
		setup_cuBLAS_func_env();
	}
	catch( string _err_str )
	{
		status = EXIT_FAILURE;
		cerr << _err_str;
		rst_err_sstr();
		goto _cub_wrapup_;
	}

	for( int f_iter=0 ; f_iter<num_files ; ++f_iter )
	{
		try
		{
			printf("%s\n",basename(files[f_iter]));
			extract_RandC_from_fname(files[f_iter],&k,&N);
			k = k-1;// Need to exclude 'feature' vector
			ld__frm_file_to_CPU(files[f_iter]);
			//print_cpu_var(cpu_mat,"cpu_mat",k,N);
			//print_cpu_var(cpu_vec,"cpu_vec",1,N);
			ld__frm_CPU_to_GPU();
			/* Math functions */
			rp__frm_rplca_to_wrkspc_on_GPU();
			////print_gpu_var(gpu_wrk_mat,"gpu_wrk_mat",k,N);
			seq();
			////print_gpu_var(gpu_res,"gpu_res",1,k);
			rp__frm_rplca_to_wrkspc_on_GPU();
			pipelined();
			/* Optinal write back of results for verification */
			#ifdef DEBUG
			wb__to_CPU_frm_GPU();
			char res_file[strlen(files[f_iter])+4];
			sprintf(res_file,"%s/res_%s",dirname(files[f_iter]),basename(files[f_iter]));
			wb__to_file_frm_CPU(res_file);
			#endif
		}
		catch( string err_str )
		{
			status = EXIT_FAILURE;
			cerr << err_str << "\nFailed to process " << files[f_iter] << ".\n";
			rst_err_sstr();
			continue;
		}
	}

_cub_wrapup_:
	try
	{
		cub_wrapup();
	}
	catch( string err_str )
	{
		status = EXIT_FAILURE;
		cerr << err_str;
		rst_err_sstr();
	}

_mem_wrapup_:
	try
	{
		free_mem_GPU();
	}
	catch( string err_str )
	{
		status = EXIT_FAILURE;
		cerr << err_str;
		rst_err_sstr();
		status = EXIT_FAILURE;
	}


	try
	{
		free_mem_CPU();
	}
	catch( string err_str )
	{
		status = EXIT_FAILURE;
		cerr << err_str;
		rst_err_sstr();
		status = EXIT_FAILURE;
	}

	try
	{
		cudaEventDest();
	}
	catch( string err_str )
	{
		status = EXIT_FAILURE;
		cerr << err_str;
		rst_err_sstr();
		status = EXIT_FAILURE;
	}

	return status;
}
