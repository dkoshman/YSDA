#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

int main() {
    std::ifstream input("input.txt");
    int rows, columns;
    input >> rows >> columns;
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            int value;
            input >> value;
            input.ignore(1);
            std::cout << std::setw(10) << value;
            if (column + 1 != columns) {
                std::cout << ' ';
            }
        }
        std::cout << '\n';
    }
    return 0;
}