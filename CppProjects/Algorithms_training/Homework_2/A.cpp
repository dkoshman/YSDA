//https://contest.yandex.ru/contest/28736/problems/
#include <algorithm>
#include <iostream>
#include <vector>

int main() {
    int n_elements, n_transformations;
    std::cin >> n_elements >> n_transformations;
    std::vector<int64_t> vector(n_elements);
    for (auto& i : vector) {
        std::cin >> i;
    }
    std::cout << *std::max_element(vector.begin(), vector.end()) -
                     *std::min_element(vector.begin(), vector.end())
              << std::endl;
    return 0;
}
