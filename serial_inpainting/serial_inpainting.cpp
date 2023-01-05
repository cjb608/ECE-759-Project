#include "serial_inpainting.h"
#include "csv_functions.h"
#include "serial_inpainting_functions.h"
#include <iostream>
#include <algorithm>
#include <math.h>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

void serial_inpainting(const std::string input_path, const std::string output_path, const std::string mask_path,
                       const float dt, const float epsilon, const int iterations,
                       const int aniso_diff_iters, const int aniso_diff_every_n_iters)
{
    // Timing variables
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> ms;
    double total_time = 0;

    // create and initialize the kernels
    std::size_t kernel_rad = 1;
    std::size_t kernel_dim = kernel_rad * 2 + 1;
    float* kfi = new float[kernel_dim * kernel_dim];
    kfi[0] = 0; kfi[1] = 0; kfi[2] = 0;
    kfi[3] = 0; kfi[4] = -1; kfi[5] = 0;
    kfi[6] = 0; kfi[7] = 1; kfi[8] = 0;
    float* kfj = new float[kernel_dim * kernel_dim];
    kfj[0] = 0; kfj[1] = 0; kfj[2] = 0;
    kfj[3] = 0; kfj[4] = -1; kfj[5] = 1;
    kfj[6] = 0; kfj[7] = 0; kfj[8] = 0;
    float* kbi = new float[kernel_dim * kernel_dim];
    kbi[0] = 0; kbi[1] = -1; kbi[2] = 0;
    kbi[3] = 0; kbi[4] = 1; kbi[5] = 0;
    kbi[6] = 0; kbi[7] = 0; kbi[8] = 0;
    float* kbj = new float[kernel_dim * kernel_dim];
    kbj[0] = 0; kbj[1] = 0; kbj[2] = 0;
    kbj[3] = -1; kbj[4] = 1; kbj[5] = 0;
    kbj[6] = 0; kbj[7] = 0; kbj[8] = 0;
    float* kci = new float[kernel_dim * kernel_dim];
    kci[0] = 0; kci[1] = -0.5; kci[2] = 0;
    kci[3] = 0; kci[4] = 0; kci[5] = 0;
    kci[6] = 0; kci[7] = 0.5; kci[8] = 0;
    float* kcj = new float[kernel_dim * kernel_dim];
    kcj[0] = 0; kcj[1] = 0; kcj[2] = 0;
    kcj[3] = -0.5; kcj[4] = 0; kcj[5] = 0.5;
    kcj[6] = 0; kcj[7] = 0; kcj[8] = 0;
    float* klap = new float[kernel_dim * kernel_dim];
    klap[0] = 0; klap[1] = 1; klap[2] = 0;
    klap[3] = 1; klap[4] = -4; klap[5] = 1;
    klap[6] = 0; klap[7] = 1; klap[8] = 0;
    float* kii = new float[kernel_dim * kernel_dim];
    kii[0] = 0; kii[1] = 1; kii[2] = 0;
    kii[3] = 0; kii[4] = -2; kii[5] = 0;
    kii[6] = 0; kii[7] = 1; kii[8] = 0;
    float* kjj = new float[kernel_dim * kernel_dim];
    kjj[0] = 0; kjj[1] = 0; kjj[2] = 0;
    kjj[3] = 1; kjj[4] = -2; kjj[5] = 1;
    kjj[6] = 0; kjj[7] = 0; kjj[8] = 0;

    // read in the input csv and store it in a 1d array
    std::vector<std::vector<std::string>> img_vec;
    img_vec = read_csv(input_path);
    std::size_t img_row_dim = img_vec.size();
    std::size_t img_col_dim = img_vec[0].size();
    float* img = new float[img_row_dim * img_col_dim];
    vec_string_to_arr_float(img_vec, img_row_dim, img_col_dim, img);

    // read in the mask csv and store it in a 1d array
    std::vector<std::vector<std::string>> mask_vec;
    mask_vec = read_csv(mask_path);
    std::size_t mask_row_dim = mask_vec.size();
    std::size_t mask_col_dim = mask_vec[0].size();
    float* mask = new float[mask_row_dim * mask_col_dim];
    vec_string_to_arr_float(mask_vec, mask_row_dim, mask_col_dim, mask);

    // define the gepsilon array and copy the mask contents to it
    float* gepsilon = new float[mask_row_dim * mask_col_dim];
    vec_string_to_arr_float(mask_vec, mask_row_dim, mask_col_dim, gepsilon);

    // create arrays for the inpainting process
    float* laplacian = new float[img_row_dim * img_col_dim];
    float* grad_i = new float[img_row_dim * img_col_dim];
    float* grad_j = new float[img_row_dim * img_col_dim];
    float* fwd_diff_i = new float[img_row_dim * img_col_dim];
    float* bwd_diff_i = new float[img_row_dim * img_col_dim];
    float* fwd_diff_j = new float[img_row_dim * img_col_dim];
    float* bwd_diff_j = new float[img_row_dim * img_col_dim];

    float zero = 0;

    // do an initial iteration of aniostropic diffusion
    serial_anisotropic_diffusion(img, img_row_dim, img_col_dim,
                                 kci, kcj, kii, kjj,
                                 kernel_dim, dt, epsilon,
                                 gepsilon, 1);

    // loop for n iterations of inpainting
    for (std::size_t i = 0; i < iterations; i++)
    {
        // Get the starting timestamp
        start = high_resolution_clock::now();

        // compute the necessary convolutions
        serial_convolution(img, img_row_dim, img_col_dim, klap, kernel_dim, laplacian);
        serial_convolution(img, img_row_dim, img_col_dim, kci, kernel_dim, grad_i);
        serial_convolution(img, img_row_dim, img_col_dim, kcj, kernel_dim, grad_j);
        serial_convolution(img, img_row_dim, img_col_dim, kfi, kernel_dim, fwd_diff_i);
        serial_convolution(img, img_row_dim, img_col_dim, kbi, kernel_dim, bwd_diff_i);
        serial_convolution(img, img_row_dim, img_col_dim, kfj, kernel_dim, fwd_diff_j);
        serial_convolution(img, img_row_dim, img_col_dim, kbj, kernel_dim, bwd_diff_j);

        // loop over all indices in the image
        for (std::size_t x = 0; x < img_row_dim; x++)
        {
            for (std::size_t y = 0; y < img_col_dim; y++)
            {
                std::size_t idx = x * img_col_dim + y;
                // check if we are at an index that requires inpainting and if the index not on the bounary
                if (mask[idx] == 1 && x - 1 >= 0 && x + 1 < img_row_dim && y - 1 >= 0 && y + 1 < img_col_dim)
                {
                    // calculate the delta laplacians
                    float delta_laplacian_i =
                            laplacian[(x + 1) * img_col_dim + y] - laplacian[(x - 1) * img_col_dim + y];
                    float delta_laplacian_j =
                            laplacian[x * img_col_dim + (y + 1)] - laplacian[x * img_col_dim + (y - 1)];

                    // calculate the normal
                    float normal_den = sqrt(grad_i[idx] * grad_i[idx] + grad_j[idx] * grad_j[idx] + epsilon);
                    float normal_i = grad_i[idx] / normal_den;
                    float normal_j = grad_j[idx] / normal_den;

                    // calculate beta
                    float beta = delta_laplacian_i * -1 * normal_j + delta_laplacian_j * normal_i;

                    // calculate the slope limited norm of the gradient
                    if (beta > 0)
                    {
                        fwd_diff_i[idx] = std::max(fwd_diff_i[idx], zero);
                        bwd_diff_i[idx] = std::min(bwd_diff_i[idx], zero);
                        fwd_diff_j[idx] = std::max(fwd_diff_j[idx], zero);
                        bwd_diff_j[idx] = std::min(bwd_diff_j[idx], zero);
                    }
                    else
                    {
                        fwd_diff_i[idx] = std::min(fwd_diff_i[idx], zero);
                        bwd_diff_i[idx] = std::max(bwd_diff_i[idx], zero);
                        fwd_diff_j[idx] = std::min(fwd_diff_j[idx], zero);
                        bwd_diff_j[idx] = std::max(bwd_diff_j[idx], zero);
                    }

                    float slope_limiter = sqrt(fwd_diff_i[idx] * fwd_diff_i[idx] + bwd_diff_i[idx] * bwd_diff_i[idx] +
                                               fwd_diff_j[idx] * fwd_diff_j[idx] + bwd_diff_j[idx] * bwd_diff_j[idx]);

                    // update the pixel value
                    img[idx] = img[idx] + dt * beta * slope_limiter;
                }

            }
        }

        // perform anisotropic diffusion
        if (i % aniso_diff_every_n_iters == 0)
        {
            serial_anisotropic_diffusion(img, img_row_dim, img_col_dim, kci, kcj, kii, kjj,
                                         kernel_dim, dt, epsilon,
                                         gepsilon, aniso_diff_iters);
        }

        // Get the ending timestamp
        end = high_resolution_clock::now();

        // Convert the calculated duration to a double using the standard library
        ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        total_time += ms.count();

        std::cout << "iteration " << i + 1 << " duration: " << ms.count() << "ms" << "\n";
    }

    std::cout << "Total inpainting time for " << iterations << " iterations: " << total_time / 1000 << "s" << "\n";
    std::cout << "Average inpainting time per iteration: " << total_time / iterations << "ms" << "\n";
    write_csv(output_path, img, img_row_dim, img_col_dim);

    // free the memory
    delete[] kfi;
    delete[] kfj;
    delete[] kbi;
    delete[] kbj;
    delete[] kci;
    delete[] kcj;
    delete[] klap;
    delete[] kii;
    delete[] kjj;
    delete[] img;
    delete[] mask;
    delete[] gepsilon;
    delete[] laplacian;
    delete[] grad_i;
    delete[] grad_j;
    delete[] fwd_diff_i;
    delete[] bwd_diff_i;
    delete[] fwd_diff_j;
    delete[] bwd_diff_j;
}


