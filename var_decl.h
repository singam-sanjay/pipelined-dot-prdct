/* Variable declarations */ // Choose a better description

/* CPU variables */
TYPE *cpu_vec = nullptr, // Vector in RAM
     *cpu_mat = nullptr; // Matrix in RAM

/* GPU variables */
TYPE *gpu_vec = nullptr,     // Vector on the GPU
     *gpu_wrk_mat = nullptr, // Matrix on the GPU, worked/operated upon
     *gpu_rep_mat = nullptr; // Matrix on the GPU, replica of cpu_mat, used to initialise gpu_wrk_mat before every operation

