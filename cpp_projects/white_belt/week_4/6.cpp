#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

int main() {
    std::ifstream input("input.txt");
    while (input) {
        double line;
        input >> line;
        if (input) {
            std::cout << std::setprecision(3) << std::fixed << line << '\n';
        }
    }
    return 0;
}