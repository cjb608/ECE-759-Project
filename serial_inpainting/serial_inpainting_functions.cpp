#include "serial_inpainting_functions.h"

void serial_convolution(const float* input, const std::size_t input_row_dim, const std::size_t input_col_dim,
                        const float* kernel, const std::size_t kernel_dim, float* output)
{
    // loop over all the indices in the input array
    for (std::size_t i = 0; i < input_row_dim; i++)
    {
        for (std::size_t j = 0; j < input_col_dim; j++)
        {
            // initialize the output
            output[i * input_col_dim + j] = 0;
            // loop over all the indices in the kernel array
            for (std::size_t x = 0; x < kernel_dim; x++)
            {
                for (std::size_t y = 0; y < kernel_dim; y++)
                {
                    // check to see if the kernel over-layed on the input at position [i, j] is in bounds
                    if ((i + x - (kernel_dim - 1) / 2 >= 0 && i + x - (kernel_dim - 1) / 2 < input_row_dim) &&
                        (j + y - (kernel_dim - 1) / 2 >= 0 && j + y - (kernel_dim - 1) / 2 < input_col_dim))
                    {
                        // compute the convolution
                        output[i * input_col_dim + j] += kernel[x * kernel_dim + y] *
                                input[((i + x - (kernel_dim - 1) / 2) * input_col_dim + (j + y - (kernel_dim - 1) / 2))];
                    }
                }
            }
        }
    }
}

void serial_anisotropic_diffusion(float* input, const std::size_t input_row_dim, const std::size_t input_col_dim,
                           const float* kci, const float* kcj, const float* kii, const float* kjj,
                           const std::size_t kernel_dim, const float dt, const float epsilon,
                           const float* gepsilon, const std::size_t iterations)
{
    // create arrays for the convolution operations
    float* input_dx = new float[input_row_dim * input_col_dim];
    float* input_dy = new float[input_row_dim * input_col_dim];
    float* input_dxx = new float[input_row_dim * input_col_dim];
    float* input_dyy = new float[input_row_dim * input_col_dim];
    float* input_dxy = new float[input_row_dim * input_col_dim];

    // compute the necessary convolutions
    serial_convolution(input, input_row_dim, input_col_dim, kci, kernel_dim, input_dx);
    serial_convolution(input, input_row_dim, input_col_dim, kcj, kernel_dim, input_dy);
    serial_convolution(input, input_row_dim, input_col_dim, kii, kernel_dim, input_dxx);
    serial_convolution(input, input_row_dim, input_col_dim, kjj, kernel_dim, input_dyy);
    serial_convolution(input_dx, input_row_dim, input_col_dim, kcj, kernel_dim, input_dxy);

    // perform n iterations of anisotropic diffusion
    for (std::size_t iter = 0; iter < iterations; iter++)
    {
        // loop over all the indices of the input array
        for (std::size_t i = 0; i < input_row_dim; i++)
        {
            for (std::size_t j = 0; j < input_col_dim; j++)
            {
                // variable for the current index
                std::size_t idx = i * input_col_dim + j;
                // compute the anisotropic diffusion upadte
                input[idx] += dt * gepsilon[idx] * (input_dyy[idx] * input_dx[idx] * input_dx[idx] +
                                                    input_dxx[idx] * input_dy[idx] * input_dy[idx] -
                                                    2 * input_dx[idx] * input_dy[idx] * input_dxy[idx]) /
                                                            (input_dx[idx] * input_dx[idx] + input_dy[idx] * input_dy[idx] + epsilon);
            }
        }
    }

    // free the memory
    delete[] input_dx;
    delete[] input_dy;
    delete[] input_dxx;
    delete[] input_dyy;
    delete[] input_dxy;
}