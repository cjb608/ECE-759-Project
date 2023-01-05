#ifndef CSV_FUNCTIONS_CUH
#define CSV_FUNCTIONS_CUH

#include <cstddef>
#include <vector>
#include <string>

// Reads in a csv file and returns it as a 2d vector
// path is the file path to the csv file
__host__ std::vector<std::vector<std::string>> read_csv(const std::string path);

// Writes an array to a csv file
// path is the file path for where to the csv file
// out_data is the 1d array of data to write to the csv file (stored in row-major order)
// out_data_row_dim is the row dimension of out_data
// out_data_col_dim is the column dimension of out_data
__host__ void write_csv(const std::string path, const float* out_data,
    const std::size_t out_data_row_dim, const std::size_t out_data_col_dim);

// Converts a 2d vector of string to a 1d array of floats
// in_data_vec is the 2d vector where the csv data is stored
// image_row_dim is the row dimensions of the input image stored in the csv
// image_col_dim is the column dimensions of the input image stored in the csv
// image is a 1d array that stores the result of the conversion (stored in row-major order)
__host__ void vec_string_to_arr_float(std::vector<std::vector<std::string>> in_data_vec,
    const std::size_t in_data_row_dim, const std::size_t in_data_col_dim,
    float* in_data);

#endif
