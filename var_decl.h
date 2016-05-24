/* Variable declarations */ // Choose a better description

#include<sstream>
std::stringstream err_sstr;

/* cuBLAS Paperwork */
#include"cublas_v2.h"
cublasHandle_t handle;


/* CPU variables */
TYPE *cpu_vec = nullptr, // Vector in RAM
     *cpu_mat = nullptr; // Matrix in RAM
#ifdef DEBUG
TYPE *cpu_res = nullptr; // Result in the CPU
#endif

/* GPU variables */
TYPE *gpu_vec = nullptr,     // Vector on the GPU
     *gpu_wrk_mat = nullptr, // Matrix on the GPU, worked/operated upon
     *gpu_rep_mat = nullptr, // Matrix on the GPU, replica of cpu_mat, used to initialise gpu_wrk_mat before every operation
     *gpu_res = nullptr;     // Result on the GPU

