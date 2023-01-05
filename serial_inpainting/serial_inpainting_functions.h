#ifndef SERIAL_INPAINTING_FUNCTIONS_H
#define SERIAL_INPAINTING_FUNCTIONS_H

#include <cstddef>

// Computes the convolution of a input array and a predefined kernel
// input is the 1d array to be convolved with the kernel (stored in row-major order)
// input_row_dim is the row dimension of the input array
// input_col_dim is the column dimension of the input array
// kernel is the predefined kernel used for the convolution operation (stored in row-major order)
// kernel_dim is the dimension of the kernel that will be used in the convolution operation
// output is the 1d array that stores the convolution results (stored in row-major order)
// Padding rules:
// f[i,j] = 0, if image_row_dim <= i <= 0 or image_col_dim <= j <= 0
// f[i,j] = 0, if image_row_dim <= i <= 0 and image_col_dim <= j <= 0
void serial_convolution(const float* input, const std::size_t input_row_dim, const std::size_t input_col_dim,
                        const float* kernel, const std::size_t kernel_dim, float* output);

// Performs n iterations of anisotropic diffusion on the input
// input is the 1d array to be convolved with the kernel (stored in row-major order)
// input_row_dim is the row dimension of the input array
// input_col_dim is the column dimension of the input array
// kci and kcj are the central difference kernels
// kii and kjj are the gradient kernels
// kernel_dim is the dimension of the kernels
// dt is the rate of change used in the update
// epsilon is a parameter to ensure there is no division by 0
// gepsilon is an array that stores the dilated mask - mask
// iterations is the number of iterations of anisotropic diffusion to complete
void serial_anisotropic_diffusion(float* input, const std::size_t input_row_dim, const std::size_t input_col_dim,
                                  const float* kci, const float* kcj, const float* kii, const float* kjj,
                                  const std::size_t kernel_dim, const float dt, const float epsilon,
                                  const float* gepsilon, const std::size_t iterations);

#endif
