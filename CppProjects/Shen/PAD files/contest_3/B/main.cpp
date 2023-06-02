#include <vector>


std::vector<std::vector<int>> transpose(const std::vector<std::vector<int>>& matrix){
    std::vector<std::vector<int>> r;
    size_t n = matrix.size();
    size_t m = matrix[0].size();

    r.resize(m);
    for (size_t i = 0; i < m; ++i){
        r[i].resize(n);
    }
    for (size_t i = 0; i < m; ++i){
        for (size_t j = 0; j < n; ++j){
            r[i][j] = matrix[j][i];
        }
    }
    return r;
}

