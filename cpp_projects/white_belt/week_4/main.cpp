#include <iostream>
#include <sstream>


int main() {
    std::stringstream stream("");
    int n;
    if (stream >> n) {
        std::cout << "Hello, World!" << n << std::endl;
    }
    enum class num {red, blue, green = 6, orange};
    std::cout << static_cast<int>(num::orange) << std::endl;
    std::cout << static_cast<int>(num::red) << std::endl;
    num a = static_cast<num>(7);
    assert(a == num::orange);

    enum access_t { read = 1, write = 3, exec = 3, what }; // enumerators: 1, 2, 4 range: 0..7
    access_t rwe = static_cast<access_t>(7);
    access_t rwe2 = static_cast<access_t>(100);
    assert(write == what);
    std::cout << static_cast<int>(what) << std::endl;
    assert(((rwe & read) == read) && (rwe & write) && (rwe & exec));
    return 0;
}
