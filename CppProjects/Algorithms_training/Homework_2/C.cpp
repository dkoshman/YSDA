#include <array>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <vector>

using std::vector;

int ComputePerimeter(std::istream& in = std::cin) {
    vector<vector<bool>> board(10);
    for (auto& line : board) {
        line.resize(10);
    }
    int n_cut_cells, perimeter = 0;
    in >> n_cut_cells;
    for (int i = 0; i < n_cut_cells; ++i) {
        int x, y;
        in >> x >> y;
        board[x][y] = true;
        for (auto [x_offset, y_offset] : {
                 std::array{-1, 0},
                 {0, -1},
                 {0, 1},
                 {1, 0},
             }) {
            perimeter += board[x + x_offset][y + y_offset] ? -1 : 1;
        }
    }
    return perimeter;
}

void Test() {
    std::stringstream stream;
    stream << "1 3 5";
    assert(ComputePerimeter(stream) == 4);
    stream.clear();
    stream << "3\n1 1\n1 2\n2 1";
    assert(ComputePerimeter(stream) == 8);
    stream.clear();
    stream << "3\n8 8\n1 1\n1 8";
    assert(ComputePerimeter(stream) == 12);
    stream.clear();
}

int main() {
    Test();

    std::cout << ComputePerimeter() << std::endl;
    return 0;
}
