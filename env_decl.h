/* DESC */
typedef double TYPE;
#define SCANF_FORMAT_STRING ("%lf")
#define PRINTF_FORMAT_STRING (SCANF_FORMAT_STRING)

//#if __cplusplus > 199711L // Make this work at a later time.
	#define nullptr (NULL)
//#endif


/* Figuring out the environment */
#if __x86_64__ || _WIN64 //Probably add other 64-bit architectures for which we have nvcc
	#define ENV64
#else
	#define ENV32
#endif

/* CUDA Compute Capability 3.5 */
#define MAX_THREADS_PER_MP (2048)
#define MAX_BLOCKS_PER_MP  (16)
#define MAX_THREADS_PER_BLOCK (1024)
/*-----------------------------*/
#ifdef __CUDA_ARCH__
	#if __CUDA_ARCH__ == 350
	#else
		//Stop Compilation, untill you've built a CC architecture agnostic program
	#endif
#else
	//Stop Compilation
#endif
