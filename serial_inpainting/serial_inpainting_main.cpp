#include "serial_inpainting.h"
#include <iostream>

int main(int argc, char *argv[])
{
    std::string input_img_path(argv[1]);
    std::string output_img_path(argv[2]);
    std::string mask_path(argv[3]);

    float dt = std::atof(argv[4]);
    float epsilon = std::atof(argv[5]);
    std::size_t iterations = std::atoi(argv[6]);
    std::size_t aniso_diff_iters = std::atoi(argv[7]);
    std::size_t aniso_diff_every_n_iters = std::atoi(argv[8]);

    serial_inpainting(input_img_path, output_img_path, mask_path,
                      dt, epsilon, iterations,
                      aniso_diff_iters, aniso_diff_every_n_iters);

    return 0;
}
