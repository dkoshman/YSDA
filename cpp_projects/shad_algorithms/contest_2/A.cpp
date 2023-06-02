// https://contest.yandex.ru/contest/29057/problems/

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

enum class PointerAction { increase_left_pointer, increase_right_pointer };

class InputType {
public:
    explicit InputType(std::istream &in) {
        in >> n_numbers;
        numbers.resize(n_numbers);
        for (auto &number : numbers) {
            in >> number;
        }

        in >> n_pointer_actions;
        pointer_actions.resize(n_pointer_actions);
        for (auto &action : pointer_actions) {
            char left_or_right = '\0';
            in >> left_or_right;
            action = left_or_right == 'L' ? PointerAction::increase_left_pointer
                                          : PointerAction::increase_right_pointer;
        }
    }

    int32_t n_numbers = 0;
    int32_t n_pointer_actions = 0;
    std::vector<int32_t> numbers;
    std::vector<PointerAction> pointer_actions;
};

class OutputType {
public:
    OutputType() = default;

    explicit OutputType(const std::string &string) {
        std::stringstream ss{string};
        int32_t item = 0;
        while (ss >> item) {
            max_in_a_sliding_window.push_back(item);
        }
    }

    std::ostream &Write(std::ostream &out) const {
        for (auto item : max_in_a_sliding_window) {
            out << item << ' ';
        }
        return out;
    }

    bool operator==(const OutputType &other) const {
        if (max_in_a_sliding_window.size() != other.max_in_a_sliding_window.size()) {
            return false;
        }
        for (size_t i = 0; i < max_in_a_sliding_window.size(); ++i) {
            if (max_in_a_sliding_window[i] != other.max_in_a_sliding_window[i]) {
                return false;
            }
        }
        return true;
    }

    std::vector<int32_t> max_in_a_sliding_window;
};

std::ostream &operator<<(std::ostream &os, OutputType const &output) {
    return output.Write(os);
}

class WrongAnswerException : public std::exception {
public:
    explicit WrongAnswerException(std::string const &message) : message{message.data()} {
    }

    [[nodiscard]] const char *what() const noexcept override {
        return message;
    }

    const char *message;
};

OutputType Solve(InputType input) {
    OutputType output;
    output.max_in_a_sliding_window.reserve(input.pointer_actions.size());
    std::vector<int32_t> max_heap;
    std::vector<int32_t> to_lazily_pop;
    auto left = input.numbers.begin();
    auto right = input.numbers.begin();
    max_heap.push_back(*left);

    for (auto pointer_action : input.pointer_actions) {
        if (pointer_action == PointerAction::increase_left_pointer) {
            to_lazily_pop.push_back(*left++);
            std::push_heap(to_lazily_pop.begin(), to_lazily_pop.end());
            while (not to_lazily_pop.empty() and to_lazily_pop.front() == max_heap.front()) {
                std::pop_heap(to_lazily_pop.begin(), to_lazily_pop.end());
                to_lazily_pop.pop_back();
                std::pop_heap(max_heap.begin(), max_heap.end());
                max_heap.pop_back();
            }
        } else {
            max_heap.push_back(*++right);
            std::push_heap(max_heap.begin(), max_heap.end());
        }
        output.max_in_a_sliding_window.push_back(max_heap.front());
    }

    return output;
}

void Check(const std::string &test_case, const std::string &expected) {
    std::stringstream input_stream{test_case};
    auto input = InputType{input_stream};
    auto output = Solve(input);
    if (not(output == OutputType{expected})) {
        std::stringstream ss;
        ss << "\nExpected:\n" << expected << "\nReceived:" << output << "\n";
        throw WrongAnswerException{ss.str()};
    }
}

void Test() {
    Check(
        "10\n"
        "1 4 2 3 5 8 6 7 9 10\n"
        "12\n"
        "R R L R R R L L L R L L\n",
        "4 4 4 4 5 8 8 8 8 8 8 6");
    Check(
        "5\n"
        "1 2 3 2 1\n"
        "8\n"
        "R R R L R L L L\n",
        "2 3 3 3 3 3 2 1");
    std::cout << "OK\n";
}

void SetUp() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
}

int main(int argc, char *argv[]) {
    SetUp();
    if (argc > 1 && std::strcmp(argv[1], "test") == 0) {
        Test();
    } else {
        std::cout << Solve(InputType{std::cin});
    }
    return 0;
}
