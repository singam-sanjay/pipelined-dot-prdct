comprhen_headr.h: env_decl.h var_decl.h func_decl.h help_func.cu alloc_func.cu load_func.cu math_func.cu
	touch comprhen_headr.h
main.cu: comprhen_headr.h
	touch main.cu
./prgm: main.cu
	nvcc -lcublas main.cu -Xcompiler "-Wall -O3 -fopenmp" -o prgm
./prgm-DEBUG: main.cu
	nvcc -DDEBUG -lcublas -g main.cu -Xcompiler "-Wall -g -fopenmp" -o prgm
