#include <assert.h>
#include <iostream>
#include <sstream>
#include <vector>

using std::vector;

int ComputeSmallestLength(std::istream& in = std::cin) {
    int n_shreds, max = 0, sum = 0;
    in >> n_shreds;
    for (int i = 0; i < n_shreds; ++i) {
        int shred;
        in >> shred;
        if (shred > max) {
            max = shred;
        }
        sum += shred;
    }
    if (sum - max < max) {
        return max - (sum - max);
    } else {
        return sum;
    }
}

void Test() {
    std::stringstream stream;
    stream << "4 1 5 2 1";
    assert(ComputeSmallestLength(stream) == 1);
    stream.clear();
    stream << "4 5 12 4 3";
    assert(ComputeSmallestLength(stream) == 24);
    std::cout << "OK\n";
}

int main() {
    Test();
//    std::cout << ComputeSmallestLength() << std::endl;
    return 0;
}
