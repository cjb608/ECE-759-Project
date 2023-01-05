#include "csv_functions.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

std::vector<std::vector<std::string>> read_csv(const std::string path)
{
    // open the csv file specified at path
    std::ifstream in_file(path);
    // create a 2d vector of string to store the csv data
    std::vector<std::vector<std::string>> in_data;
    // create a string to read a single row of the csv data
    std::string line_data;

    // check if the file can be opened
    // return with error code -1 if file cannot be opened
    if (!in_file.is_open())
    {
        std::cout << "File cannot be opened!" << "\n";
        exit(-1);
    }

    // loop for all lines in the csv file
    while (std::getline(in_file, line_data))
    {
        // create a stream for the csv row data
        std::stringstream lineStream(line_data);
        // create a string for the csv cell data
        std::string cell_data;
        // create a 1d vector to store the csv row data
        std::vector<std::string> in_data_row;

        // Loop over all the cells in the row
        while (std::getline(lineStream, cell_data, ','))
        {
            // write the cell data to the row data vector
            in_data_row.push_back(cell_data);
        }

        // write the row data to the next row in the 2d csv vector
        in_data.push_back(in_data_row);
    }

    // return the csv data in a 2d vector
    return in_data;
}

void write_csv(const std::string path, const float* out_data,
               const std::size_t out_data_row_dim, const std::size_t out_data_col_dim)
{
    // create/open a csv file at the specified path
    std::ofstream out_file(path);

    // check to see if the file can be created/opened
    // return with error code -1 if file cannot be opened
    if (!out_file.is_open())
    {
        std::cout << "File cannot be created/opened!" << "\n";
        exit(-1);
    }

    // loop over the array to be stored in the csv file
    for (std::size_t i = 0; i < out_data_row_dim; i++)
    {
        for (std::size_t j = 0; j < out_data_col_dim; j++)
        {
            // write the value to the output csv file
            out_file << out_data[i * out_data_col_dim + j];
            // if we are on the same row separate entries with a comma
            if (j < out_data_col_dim - 1)
            {
                out_file << ",";
            }
            // if we are on a new row create a new row
            else
            {
                out_file << "\n";
            }
        }
    }

    // close the output csv file
    out_file.close();
}

void vec_string_to_arr_float(const std::vector<std::vector<std::string>> in_data_vec,
                             const std::size_t in_data_row_dim, const std::size_t in_data_col_dim,
                             float* in_data)
{
    // loop over all the indices in the 2d input vector
    for (std::size_t i = 0; i < in_data_row_dim; i++)
    {
        for (std::size_t j = 0; j < in_data_col_dim; j++)
        {
            // convert the string value into a float and store it in the 1d array
            std::string temp_str = in_data_vec[i][j];
            in_data[i * in_data_col_dim + j] = std::atof(temp_str.data());
        }

    }
}

