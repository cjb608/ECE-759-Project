#!/usr/bin/env zsh

#SBATCH -p wacc
#SBATCH -J serial_inpainting
#SBATCH -o serial_inpainting.out 
#SBATCH -t 0-00:30:00

cd $SLURM_SUBMIT_DIR

g++ csv_functions.cpp serial_inpainting_functions.cpp serial_inpainting.cpp serial_inpainting_main.cpp -Wall -O3 -std=c++17 -o serial_inpainting

./serial_inpainting test_image_1024.csv test_output_1024.csv test_mask_1024.csv 0.1 0.0001 10000 2 15
