#include <vector>

std::pair<size_t, size_t> max_element(const std::vector<std::vector<int>>& matrix){
    std::pair<size_t, size_t> res = {0, 0};
    int max = matrix[0][0];

    for (size_t i = 0; i < matrix.size(); ++i){
        for (size_t j = 0; j < matrix[i].size(); ++j){
            if (matrix[i][j] > max){
                max = matrix[i][j];
                res = {i, j};
            }
        }
    }
    return res;
}
