#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <functional>

int main() {
    std::vector<std::function<int(int)>> vector;
    vector.reserve(10);
    vector.push_back([](int x) { return x; });
    std::cout << vector[0](10);
    std::cout << '\n' << (-12 % 7);
    int64_t a = 7;
    int64_t b = 13;
    std::vector<int32_t> vector1{1,2,3};
//    for (int64_t i = 1; i < 1'000'000'000; ++i) {
//        assert(a / b == (a * i) / (b * i));
//    }
    std::cout << '\n' << (1. / 100'000 + 1. / 1.1) * 100'000;
    std::cout << '\n' << (1 / 2 == 17 * 12378 / (34 * 12378));
    std::cout << '\n' << (-1 % 3 + 3) % 3;
    std::cout << '\n' << vector1[-1];
    return 0;
}