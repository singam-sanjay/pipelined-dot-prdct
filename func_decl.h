/* Helper functions */

void cmdln_usage_help(); // Displays command line usage
#define throw_str_excptn() {throw err_sstr.str();} // Throws a string containing a cumulative exceptional behaviour inside a function
void rst_err_sstr();     // Resets err_sstr to an empty string to make it ready for a different function.
void verify_cmdln_args( char *files[], int num_files ); // Add description
bool extract_RandC_from_fname( char *f_name, int *ptr_R, int *ptr_C ); // Add description

/* Alloc and Free functions */

void alloc_mem_CPU(); //Add a description if deemed necessary
void free_mem_CPU();
void alloc_mem_GPU();
void free_mem_GPU();

/* Load and Write Functions */

void ld__frm_file_to_CPU();
void ld__frm_CPU_to_GPU();
void rp__frm_rplca_to_wrkspc_on_GPU();
#ifdef DEBUG
void wb__to_CPU_frm_GPU();
void wb__to_file_frm_CPU();
#endif

/* cuBLAS Paperwork */

void cub_init();
void cub_wrapup();

/* Math helper functions */

void setup_cuBLAS_func_env();

/* Math functions */

void seq();
void par_OpenMP();
void par_dyn_parll();
void pipelined();
