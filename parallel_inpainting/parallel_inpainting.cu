#include "parallel_inpainting.cuh"
#include "csv_functions.cuh"
#include "parallel_inpainting_functions.cuh"
#include <iostream>
#include <algorithm>
#include <math.h>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

__host__ void parallel_inpainting(const std::string input_path, const std::string output_path, const std::string mask_path,
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

    // create the device arrays
    float* d_img = NULL;
    float* d_kfi = NULL;
    float* d_kfj = NULL;
    float* d_kbi = NULL;
    float* d_kbj = NULL;
    float* d_kci = NULL;
    float* d_kcj = NULL;
    float* d_klap = NULL;
    float* d_kii = NULL;
    float* d_kjj = NULL;
    float* d_laplacian = NULL;
    float* d_grad_i = NULL;
    float* d_grad_j = NULL;
    float* d_fwd_diff_i = NULL;
    float* d_bwd_diff_i = NULL;
    float* d_fwd_diff_j = NULL;
    float* d_bwd_diff_j = NULL;
    cudaMalloc((void**)&d_img, sizeof(float) * img_row_dim * img_col_dim);
    cudaMalloc((void**)&d_kfi, sizeof(float) * kernel_dim * kernel_dim);
    cudaMalloc((void**)&d_kfj, sizeof(float) * kernel_dim * kernel_dim);
    cudaMalloc((void**)&d_kbi, sizeof(float) * kernel_dim * kernel_dim);
    cudaMalloc((void**)&d_kbj, sizeof(float) * kernel_dim * kernel_dim);
    cudaMalloc((void**)&d_kci, sizeof(float) * kernel_dim * kernel_dim);
    cudaMalloc((void**)&d_kcj, sizeof(float) * kernel_dim * kernel_dim);
    cudaMalloc((void**)&d_klap, sizeof(float) * kernel_dim * kernel_dim);
    cudaMalloc((void**)&d_kii, sizeof(float) * kernel_dim * kernel_dim);
    cudaMalloc((void**)&d_kjj, sizeof(float) * kernel_dim * kernel_dim);
    cudaMalloc((void**)&d_laplacian, sizeof(float) * img_row_dim * img_col_dim);
    cudaMalloc((void**)&d_grad_i, sizeof(float) * img_row_dim * img_col_dim);
    cudaMalloc((void**)&d_grad_j, sizeof(float) * img_row_dim * img_col_dim);
    cudaMalloc((void**)&d_fwd_diff_i, sizeof(float) * img_row_dim * img_col_dim);
    cudaMalloc((void**)&d_bwd_diff_i, sizeof(float) * img_row_dim * img_col_dim);
    cudaMalloc((void**)&d_fwd_diff_j, sizeof(float)* img_row_dim* img_col_dim);
    cudaMalloc((void**)&d_bwd_diff_j, sizeof(float)* img_row_dim* img_col_dim);

    // copy the host kernels to the device
    cudaMemcpy(d_kfi, kfi, sizeof(float)* kernel_dim* kernel_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kfj, kfj, sizeof(float)* kernel_dim* kernel_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kfi, kbi, sizeof(float)* kernel_dim* kernel_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kfi, kfj, sizeof(float)* kernel_dim* kernel_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kfi, kci, sizeof(float)* kernel_dim* kernel_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kfi, kcj, sizeof(float)* kernel_dim* kernel_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_klap, klap, sizeof(float)* kernel_dim* kernel_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kii, kii, sizeof(float)* kernel_dim* kernel_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kjj, kjj, sizeof(float)* kernel_dim* kernel_dim, cudaMemcpyHostToDevice);

    // define the tile size, block size
    int tile_width = 30;
    int block_size = tile_width + kernel_dim - 1;

    // define the block and grid dimensions
    dim3 block_dim(block_size, block_size);
    dim3 grid_dim((img_row_dim - 1) / tile_width + 1, (img_col_dim - 1) / tile_width + 1);

    float zero = 0;

    // do an initial iteration of aniostropic diffusion
    parallel_anisotropic_diffusion(img, img_row_dim, img_col_dim, 
        kci, kcj, kii, kjj,
        kernel_dim, dt, epsilon,
        gepsilon, 1);

    // loop for n iterations of inpainting
    for (std::size_t i = 0; i < iterations; i++)
    {
        // Get the starting timestamp
        start = high_resolution_clock::now();


        // copy the current img arrays to the device
        cudaMemcpy(d_img, img, sizeof(float) * img_row_dim * img_col_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(d_laplacian, laplacian, sizeof(float) * img_row_dim * img_col_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(d_grad_i, grad_i, sizeof(float) * img_row_dim * img_col_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(d_grad_j, grad_j, sizeof(float) * img_row_dim * img_col_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(d_fwd_diff_i, fwd_diff_i, sizeof(float) * img_row_dim * img_col_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(d_bwd_diff_i, bwd_diff_i, sizeof(float) * img_row_dim * img_col_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(d_fwd_diff_j, fwd_diff_j, sizeof(float) * img_row_dim * img_col_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(d_bwd_diff_j, bwd_diff_j, sizeof(float) * img_row_dim * img_col_dim, cudaMemcpyHostToDevice);

        parallel_convolution << <grid_dim, block_dim, (sizeof(float) * block_size * block_size + sizeof(float) * kernel_dim * kernel_dim) >> > (d_img, img_row_dim, img_col_dim, d_klap, kernel_dim, d_laplacian);
        cudaDeviceSynchronize();
        cudaMemcpy(laplacian, d_laplacian, sizeof(float) * img_row_dim * img_col_dim, cudaMemcpyDeviceToHost);
        parallel_convolution << <grid_dim, block_dim, (sizeof(float) * block_size * block_size + sizeof(float) * kernel_dim * kernel_dim) >> > (d_img, img_row_dim, img_col_dim, d_kci, kernel_dim, d_grad_i);
        cudaDeviceSynchronize();
        cudaMemcpy(grad_i, d_grad_i, sizeof(float) * img_row_dim * img_col_dim, cudaMemcpyDeviceToHost);
        parallel_convolution << <grid_dim, block_dim, (sizeof(float) * block_size * block_size + sizeof(float) * kernel_dim * kernel_dim) >> > (d_img, img_row_dim, img_col_dim, d_kcj, kernel_dim, d_grad_j);
        cudaDeviceSynchronize();
        cudaMemcpy(grad_j, d_grad_j, sizeof(float) * img_row_dim * img_col_dim, cudaMemcpyDeviceToHost);
        parallel_convolution << <grid_dim, block_dim, (sizeof(float) * block_size * block_size + sizeof(float) * kernel_dim * kernel_dim) >> > (d_img, img_row_dim, img_col_dim, d_kfi, kernel_dim, d_fwd_diff_i);
        cudaDeviceSynchronize();
        cudaMemcpy(fwd_diff_i, d_fwd_diff_i, sizeof(float) * img_row_dim * img_col_dim, cudaMemcpyDeviceToHost);
        parallel_convolution << <grid_dim, block_dim, (sizeof(float) * block_size * block_size + sizeof(float) * kernel_dim * kernel_dim) >> > (d_img, img_row_dim, img_col_dim, d_kbi, kernel_dim, d_bwd_diff_i);
        cudaDeviceSynchronize();
        cudaMemcpy(bwd_diff_i, d_bwd_diff_i, sizeof(float) * img_row_dim * img_col_dim, cudaMemcpyDeviceToHost);
        parallel_convolution << <grid_dim, block_dim, (sizeof(float) * block_size * block_size + sizeof(float) * kernel_dim * kernel_dim) >> > (d_img, img_row_dim, img_col_dim, d_kfj, kernel_dim, d_fwd_diff_j);
        cudaDeviceSynchronize();
        cudaMemcpy(fwd_diff_j, d_fwd_diff_j, sizeof(float) * img_row_dim * img_col_dim, cudaMemcpyDeviceToHost);
        parallel_convolution << <grid_dim, block_dim, (sizeof(float) * block_size * block_size + sizeof(float) * kernel_dim * kernel_dim) >> > (d_img, img_row_dim, img_col_dim, d_kbj, kernel_dim, d_bwd_diff_j);
        cudaDeviceSynchronize();
        cudaMemcpy(bwd_diff_j, d_bwd_diff_j, sizeof(float) * img_row_dim * img_col_dim, cudaMemcpyDeviceToHost);

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
            parallel_anisotropic_diffusion(img, img_row_dim, img_col_dim, kci, kcj, kii, kjj,
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

    // free the device memory
    cudaFree(d_img);
    cudaFree(d_kfi);
    cudaFree(d_kfj);
    cudaFree(d_kbi);
    cudaFree(d_kbj);
    cudaFree(d_kci);
    cudaFree(d_kcj);
    cudaFree(d_klap);
    cudaFree(d_kii);
    cudaFree(d_kjj);
    cudaFree(d_laplacian);
    cudaFree(d_grad_i);
    cudaFree(d_grad_j);
    cudaFree(d_fwd_diff_i);
    cudaFree(d_bwd_diff_i);
    cudaFree(d_fwd_diff_j);
    cudaFree(d_bwd_diff_j);

    // free the host memory
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