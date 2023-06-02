#include <vector>

void reverse(std::vector<int>& numbers){
    std::vector<int> v = numbers;
    for (size_t i = 0; i < v.size(); ++i){
        numbers[i] = v[v.size() - 1 - i];
    }
}

