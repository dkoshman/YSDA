#include <vector>


void swap_columns(std::vector<std::vector<int>>& matrix, size_t i, size_t j){
    std::vector<int> v;

    v.resize(matrix.size());
    for (size_t k = 0; k < v.size(); ++k){
        v[k] = matrix[k][i];
    }
    for (size_t k = 0; k < v.size(); ++k){
        matrix[k][i] = matrix[k][j];
        matrix[k][j] = v[k];
    }
}
