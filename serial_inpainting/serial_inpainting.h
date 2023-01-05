#ifndef SERIAL_INPAINTING_H
#define SERIAL_INPAINTING_H

#include <cstddef>
#include <string>

void serial_inpainting(const std::string input_path, const std::string output_path, const std::string mask_path,
                       const float dt, const float epsilon, const int iterations,
                       const int aniso_diff_iters, const int aniso_diff_every_n_iters);

#endif
