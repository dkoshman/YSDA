#include <iostream>
#include <fstream>
#include <string>

int main() {
    std::ifstream input("input.txt");
    std::ofstream output("output.txt");
    if (input) {
        std::string line;
        while (getline(input, line)) {
            output << line << '\n';
        }
    }
    return 0;
}