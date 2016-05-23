/* DESC */
typedef double TYPE;
SCANF_FORMAT_STRING "%lf"
PRINTF_FORMAT_STRING SCANF_FORMAT_STRING

/* Figuring out the environment */
#if __x86_64__ || _WIN64 //Probably add other 64-bit architectures for which we have nvcc
	#define ENV64
#else
	#define ENV32
#endif

/* CUDA Compute Capability 3.5 */
#ifdef __CUDA_ARCH__
	#if __CUDA_ARCH__ == 350
	#else
		//Stop Compilation, untill you've built a CC architecture agnostic program
	#endif
#else
	//Stop Compilation
#endif
