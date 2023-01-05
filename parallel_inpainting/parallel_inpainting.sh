#!/usr/bin/env zsh

#SBATCH -p wacc
#SBATCH -J parallel_inpainting
#SBATCH -o parallel_inpainting.out 
#SBATCH -t 0-00:30:00
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

module load nvidia/cuda/11.6.0 gcc/9.4.0

nvcc csv_functions.cu parallel_inpainting_functions.cu parallel_inpainting.cu parallel_inpainting_main.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o parallel_inpainting

./parallel_inpainting test_image_1024.csv test_output_1024.csv test_mask_1024.csv 0.1 0.0001 10000 2 15

