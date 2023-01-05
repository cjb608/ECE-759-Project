#ifndef PARALLEL_INPAINTING_CUH
#define PARALLEL_INPAINTING_CUH

#include <cstddef>
#include <string>

__host__ void parallel_inpainting(const std::string input_path, const std::string output_path, const std::string mask_path, 
	const float dt, const float epsilon, const int iterations, 
	const int aniso_diff_iters, const int aniso_diff_every_n_iterations);

#endif