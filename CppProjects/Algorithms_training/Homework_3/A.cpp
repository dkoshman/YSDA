#include <iostream>
#include <set>
#include <sstream>
#include <vector>

std::vector<int> GenerateAnswers(std::istream &in = std::cin) {
    int n = 0;
    in >> n;
    in.ignore(1);
    std::vector<bool> possible_nums(n, true);
    int available_numbers = n;
    std::string line;
    std::getline(in, line);
    while (line != "HELP") {
        std::stringstream stream(line);
        std::vector<bool> guess_no = possible_nums, guess_yes(n);
        int number = 0, guess_yes_size = 0;
        while (stream >> number) {
            --number;
            if (guess_no[number]) {
                guess_no[number] = false;
                guess_yes[number] = true;
                ++guess_yes_size;
            }
        }
        if (guess_yes_size * 2 <= available_numbers) {
            std::cout << "NO\n";
            possible_nums = guess_no;
            available_numbers -= guess_yes_size;
        } else {
            std::cout << "YES\n";
            possible_nums = guess_yes;
            available_numbers = guess_yes_size;
        }
        std::getline(in, line);
    }
    std::vector<int> result;
    for (int i = 0; i < n; ++i) {
        if (possible_nums[i]) {
            result.push_back(i + 1);
        }
    }
    return result;
}

void Test() {
    std::stringstream stream;
    stream << "16\n"
              "1 2 3 4 5 6 7 8\n"
              "9 10 11 12\n"
              "13 14\n"
              "16\n"
              "HELP\n";
//    assert(GenerateAnswers(stream) == std::vector<int>{15});
}

int main() {
    Test();
    for (auto i : GenerateAnswers()) {
        std::cout << i << ' ';
    }
    std::cout << std::endl;

    return 0;
}
