#include "parallel_inpainting_functions.cuh"

__global__ void parallel_convolution(const float* input, const std::size_t input_row_dim, const std::size_t input_col_dim, 
    const float* kernel, const std::size_t kernel_dim, float* output)
{
    // dynamically allocate the shared memory
    extern __shared__ float s[];
    float* s_input = s;
    float* s_kernel = (float*)&s_input[blockDim.x * blockDim.y];

    // define indexing variables
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tile_row_dim = blockDim.y - kernel_dim + 1;
    int tile_col_dim = blockDim.y - kernel_dim + 1;
    int row_o = blockIdx.y * tile_row_dim + ty;
    int col_o = blockIdx.x * tile_col_dim + tx;
    int row_i = row_o - kernel_dim / 2;
    int col_i = col_o - kernel_dim / 2;

    // load the kernel into shared memory
    if (ty * blockDim.x + ty < kernel_dim * kernel_dim)
    {
        s_kernel[ty * blockDim.x + tx] = kernel[ty * blockDim.x + tx];
    }

    // load the input + padding into shared memory
    if ((row_i >= 0) && (row_i < input_row_dim) && (col_i >= 0) && (col_i < input_col_dim))
    {
        s_input[ty * blockDim.x + tx] = input[row_i * input_col_dim + col_i];
    }
    else
    {
        s_input[ty * blockDim.x + tx] = 0;
    }

    // wait until all threads have loaded their data into shared memory
    __syncthreads();

    // initialize the themp variable to store the convolution sum
    float temp = 0;

    // check if the current thread is in the orignal tile dimensions
    if (ty < tile_row_dim && tx < tile_col_dim)
    {
        // loop over the kernel and cumpute the convolution
        for (int i = 0; i < kernel_dim; i++)
        {
            for (int j = 0; j < kernel_dim; j++)
            {
                temp += s_kernel[i * kernel_dim + j] * s_input[(ty + i) * blockDim.x + (tx + j)];
            }
        }
    }

    // check if the output idx is within the orignal input dimensions
    if (row_o < input_row_dim && col_o < input_col_dim)
    {
        // check if the current thread is in the orignal tile dimensions
        if (ty < tile_row_dim && tx < tile_col_dim)
        {
            // write the convolution result to the output array
            output[row_o * input_col_dim + col_o] = temp;
        }
    }

}

__host__ void parallel_anisotropic_diffusion(float* input, const std::size_t input_row_dim, const std::size_t input_col_dim,
    const float* kci, const float* kcj, const float* kii, const float* kjj,
    const std::size_t kernel_dim, const float dt, const float epsilon,
    const float* gepsilon, const std::size_t iterations)
{
    // create host arrays to store the convolution operations
    float* input_dx = new float[input_row_dim * input_col_dim];
    float* input_dy = new float[input_row_dim * input_col_dim];
    float* input_dxx = new float[input_row_dim * input_col_dim];
    float* input_dyy = new float[input_row_dim * input_col_dim];
    float* input_dxy = new float[input_row_dim * input_col_dim];

    // create device arrays to compute the convolution operations
    float* d_input = NULL;
    float* d_kci = NULL;
    float* d_kcj = NULL;
    float* d_kii = NULL;
    float* d_kjj = NULL;
    float* d_input_dx = NULL;
    float* d_input_dy = NULL;
    float* d_input_dxx = NULL;
    float* d_input_dyy = NULL;
    float* d_input_dxy = NULL;
    cudaMalloc((void **)&d_input, sizeof(float) * input_row_dim * input_col_dim);
    cudaMalloc((void **)&d_kci, sizeof(float) * kernel_dim * kernel_dim);
    cudaMalloc((void **)&d_kcj, sizeof(float) * kernel_dim * kernel_dim);
    cudaMalloc((void **)&d_kii, sizeof(float) * kernel_dim * kernel_dim);
    cudaMalloc((void **)&d_kjj, sizeof(float) * kernel_dim * kernel_dim);
    cudaMalloc((void **)&d_input_dx, sizeof(float) * input_row_dim * input_col_dim);
    cudaMalloc((void**)&d_input_dy, sizeof(float) * input_row_dim * input_col_dim);
    cudaMalloc((void**)&d_input_dxx, sizeof(float) * input_row_dim * input_col_dim);
    cudaMalloc((void**)&d_input_dyy, sizeof(float) * input_row_dim * input_col_dim);
    cudaMalloc((void**)&d_input_dxy, sizeof(float) * input_row_dim * input_col_dim);

    // copy data to the device
    cudaMemcpy(d_input, input, sizeof(float) * input_row_dim * input_col_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kci, kci, sizeof(float) * kernel_dim * kernel_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kcj, kcj, sizeof(float) * kernel_dim * kernel_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kii, kii, sizeof(float) * kernel_dim * kernel_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kjj, kjj, sizeof(float) * kernel_dim * kernel_dim, cudaMemcpyHostToDevice);
    
    // define the tile size, block size
    int tile_width = 30;
    int block_size = tile_width + kernel_dim - 1;

    // define the block and grid dimensions
    dim3 block_dim(block_size, block_size);
    dim3 grid_dim((input_row_dim - 1) / tile_width + 1, (input_col_dim - 1) / tile_width + 1);

    // comput the necessary convolutions
    parallel_convolution << <grid_dim, block_dim, (sizeof(float) * block_size * block_size + sizeof(float) * kernel_dim * kernel_dim) >> > (d_input, input_row_dim, input_col_dim, d_kci, kernel_dim, d_input_dx);
    cudaDeviceSynchronize();
    cudaMemcpy(input_dx, d_input_dx, sizeof(float) * input_row_dim * input_col_dim, cudaMemcpyDeviceToHost);
    parallel_convolution << <grid_dim, block_dim, (sizeof(float) * block_size * block_size + sizeof(float) * kernel_dim * kernel_dim) >> > (d_input, input_row_dim, input_col_dim, d_kcj, kernel_dim, d_input_dy);
    cudaDeviceSynchronize();
    cudaMemcpy(input_dy, d_input_dy, sizeof(float) * input_row_dim * input_col_dim, cudaMemcpyDeviceToHost);
    parallel_convolution << <grid_dim, block_dim, (sizeof(float) * block_size * block_size + sizeof(float) * kernel_dim * kernel_dim) >> > (d_input, input_row_dim, input_col_dim, d_kii, kernel_dim, d_input_dxx);
    cudaDeviceSynchronize();
    cudaMemcpy(input_dxx, d_input_dxx, sizeof(float) * input_row_dim * input_col_dim, cudaMemcpyDeviceToHost);
    parallel_convolution << <grid_dim, block_dim, (sizeof(float) * block_size * block_size + sizeof(float) * kernel_dim * kernel_dim) >> > (d_input, input_row_dim, input_col_dim, d_kjj, kernel_dim, d_input_dyy);
    cudaDeviceSynchronize();
    cudaMemcpy(input_dyy, d_input_dyy, sizeof(float) * input_row_dim * input_col_dim, cudaMemcpyDeviceToHost);
    parallel_convolution << <grid_dim, block_dim, (sizeof(float) * block_size * block_size + sizeof(float) * kernel_dim * kernel_dim) >> > (d_input_dx, input_row_dim, input_col_dim, d_kcj, kernel_dim, d_input_dxy);
    cudaDeviceSynchronize();
    cudaMemcpy(input_dxy, d_input_dxy, sizeof(float) * input_row_dim * input_col_dim, cudaMemcpyDeviceToHost);

    // perform n iternations of anisotropic diffusion
    for (std::size_t iter = 0; iter < iterations; iter++)
    {
        // loop over all the indices of the input array
        for (std::size_t i = 0; i < input_row_dim; i++)
        {
            for (std::size_t j = 0; j < input_col_dim; j++)
            {
                // variable for the current index
                std::size_t idx = i * input_col_dim + j;
                // compute the anisotropic update
                input[idx] += dt * gepsilon[idx] * (input_dyy[idx] * input_dx[idx] * input_dx[idx] +
                    input_dxx[idx] * input_dy[idx] * input_dy[idx] -
                    2 * input_dx[idx] * input_dy[idx] * input_dxy[idx]) / 
                    (input_dx[idx] * input_dx[idx] + input_dy[idx] * input_dy[idx] + epsilon);
            }
        }
    }

    // free the device memory
    cudaFree(d_input);
    cudaFree(d_kci);
    cudaFree(d_kcj);
    cudaFree(d_kii);
    cudaFree(d_kjj);
    cudaFree(d_input_dx);
    cudaFree(d_input_dy);
    cudaFree(d_input_dxx);
    cudaFree(d_input_dyy);
    cudaFree(d_input_dxy);

    // free the host memory
    delete[] input_dx;
    delete[] input_dy;
    delete[] input_dxx;
    delete[] input_dyy;
    delete[] input_dxy;
}