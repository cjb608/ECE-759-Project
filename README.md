# Image Inpainting

## Introduction

This notebook contains a base C++ implementation and an optimized C++ implementation of the image inpainting algorithm proposed by Bertalmío et al. in "Image Inpainting" [1]. The base implementation processes each pixel sequentially on the CPU where as the optimized implementation processes the pixels in parallel on the GPU.

## Base Implementation

The code for the CPU/serial implementation is located in the serial_inpainting sub-directory. The code can be run on either a local machine or euler. If running on euler an example slurm script has also been included. Test data has also been provided in the test_images sub-directory and an explanation of what is include in the test data can be found below.

In order to compile the code please use the following command:
g++ csv_functions.cpp serial_inpainting_functions.cpp serial_inpainting.cpp serial_inpainting_main.cpp -Wall -O3 -std=c++17 -o serial_inpainting

After compiling in order to run the code please use the following command:
./serial_inpainting test_image_data.csv test_output_data.csv test_mask_data.csv dt epsilon iterations iterations_of_diff diff_every_n_iterations

•	test_image_data.csv is the csv file containing the test image data that can be downloaded from the test_images sub-directory
•	test_output_data.csv is the name of the file that will be created to store the final inpainted image data
•	test_mask_data.csv is the csv file containing the mask data (which pixels to inpaint) that can be downloaded from test_images sub-directory
•	dt is the rate of change for the inpainting updates (default=0.1)
•	epsilon is a small value used to avoid division by zero (default=1e-10)
•	iterations is the total number of iterations of inpainting to run (input/mask dependent)
•	iterations_of_diff is the number of iterations of anisotropic diffusion to run
•	diff_every_n_iterations is how often to perform anisotropic diffusion

## Optimized Implementation

The code for the GPU/parallel implementation is located in the parallel_inpainting sub-directory. The code can be run on either a local machine with an NVIDIA GPU or euler. If running on euler an example slurm script has also been included. Test data has also been provided in the test_images sub-directory and an explanation of what is include in the test data can be found below.

In order to compile the code please use the following command:
nvcc csv_functions.cu parallel_inpainting_functions.cu parallel_inpainting.cu parallel_inpainting_main.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o parallel_inpainting

After compiling in order to run the code please use the following command:
./parallel_inpainting test_image_data.csv test_output_data.csv test_mask_data.csv dt epsilon iterations iterations_of_diff diff_every_n_iterations

•	test_image_data.csv is the csv file containing the test image data that can be downloaded from the test_images sub-directory
•	test_output_data.csv is the name of the file that will be created to store the final inpainted image data
•	test_mask_data.csv is the csv file containing the mask data (which pixels to inpaint) that can be downloaded from test_images sub-directory
•	dt is the rate of change for the inpainting updates (default=0.1)
•	epsilon is a small value used to avoid division by zero (default=1e-10)
•	iterations is the total number of iterations of inpainting to run (input/mask dependent)
•	iterations_of_diff is the number of iterations of anisotropic diffusion to run
•	diff_every_n_iterations is how often to perform anisotropic diffusion


## Test Data

Test data has been included in the test_images sub-directory of the GitLab repo linked in the abstract. The test data includes seven different test cases. Each test case includes the input image as a png, the input image data as a csv, the input mask as a png, and the input mask data as a csv. The png files are provided for visuals only, the codebase uses the csv files as inputs.

The corresponding files for each test case are as files:

1.	test_image_64.png, test_image_data_64.csv, test_mask_64.png, test_mask_data_64.csv
2.	test_image_128.png, test_image_data_128.csv, test_mask_128.png, test_mask_data_128.csv
3.	test_image_256.png, test_image_data_256.csv, test_mask_256.png, test_mask_data_256.csv
4.	test_image_512.png, test_image_data_512.csv, test_mask_512.png, test_mask_data_512.csv
5.	test_image_1024.png, test_image_data_1024.csv, test_mask_1024.png, test_mask_data_1024.csv
6.	test_synthetic_image.png, test_synthetic_image_data.png, test_synthetic_mask.png, test_synthetic_mask_data.csv
7.	test_color_image.png, test_color_image_data_channel1.csv, test_color_image_data_channel2.csv, test_color_image_data_channel3.csv, test_color_mask.png, test_color_mask_data.csv

The seventh test case includes a color image, each RGB channel has been separated into its own csv file as the algorithm currently only takes single channel inputs.

## Results

A scaling analysis that compares the base implementation against the optimized implemenation using the first five test cases can be seen below.

| Image Size | Number of Inpainted Pixels | Serial Average Iteration Time (ms) | Parallel Average Iteration Time (ms) |
| ---------- | -------------------------- | ---------------------------------- | ------------------------------------ |
| 64x64      | 384                        | 0.407557                           | 0.182746                             |
| 128x128    | 1536                       | 1.700380                           | 0.385397                             |
| 256x256    | 6144                       | 7.820990                           | 1.1108                               |
| 512x512    | 24576                      | 29.094200                          | 4.085500                             |
| 1024x1024  | 98304                      | 152.935000                         | 11.751200                            |

## References

[1] Bertalmío, Marcelo & Sapiro, Guillermo & Caselles, Vicent & Ballester, C.. (2000). Image inpainting. Proceedings of the ACM SIGGRAPH Conference on Computer Graphics. 417-424.
