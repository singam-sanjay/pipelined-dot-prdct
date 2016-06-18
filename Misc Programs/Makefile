SED_OUTPUT_pipelined_kernel_chk.c: Makefile pipelined_kernel_chk.c
	cat pipelined_kernel_chk.c | sed -e 's/__syncthreads();/#pragma omp barrier/g' > SED_OUTPUT_pipelined_kernel_chk.c
pipelined_kernel_chk: Makefile SED_OUTPUT_pipelined_kernel_chk.c
	g++ SED_OUTPUT_pipelined_kernel_chk.c -o pipelined_kernel_chk -fopenmp -g -Wall
