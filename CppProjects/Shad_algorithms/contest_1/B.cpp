#include <assert.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct InputType {
    std::string value;
    InputType(const std::string &value) : value{value} {
    }
    InputType(std::istream &in = std::cin) {
        std::getline(in, value);
    }
};

struct OutputType {
    int32_t answers;
    std::string ToString() const {
        return answers == -1 ? "CORRECT" : std::to_string(answers);
    }
};

void SetUp() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
}

bool inline AreMatchingBrackets(char left, char right) {
    return (left == '(' and right == ')') or (left == '[' and right == ']') or
           (left == '{' and right == '}');
}

bool inline IsOpeningBracket(char bracket) {
    return bracket == '(' or bracket == '[' or bracket == '{';
}

OutputType Solve(const InputType &input) {
    std::vector<char> opening_brackets_stack;
    int prefix_size_extendable_to_valid = 0;
    for (auto bracket : input.value) {
        if (IsOpeningBracket(bracket)) {
            opening_brackets_stack.push_back(bracket);
        } else if (opening_brackets_stack.empty() or
                   not AreMatchingBrackets(opening_brackets_stack.back(), bracket)) {
            return {prefix_size_extendable_to_valid};
        } else {
            opening_brackets_stack.pop_back();
        }
        ++prefix_size_extendable_to_valid;
    }
    if (opening_brackets_stack.empty()) {
        return {-1};
    }
    return {prefix_size_extendable_to_valid};
}

std::string StripWhitespace(const std::string &line) {
    const char *white_space = " \t\v\r\n";
    std::size_t start = line.find_first_not_of(white_space);
    std::size_t end = line.find_last_not_of(white_space);
    return start > end ? std::string() : line.substr(start, end - start + 1);
}

void Check(const std::string &test_case, const std::string &expected) {
    auto output = Solve(InputType{test_case});
    auto output_trimmed = StripWhitespace(output.ToString());
    auto expected_trimmed = StripWhitespace(expected);
    if (output_trimmed != expected_trimmed) {
        std::cerr << "Check failed, expected:\n\"";
        std::cerr << expected_trimmed << "\"\nReceived:\n\"";
        std::cerr << output_trimmed << "\"\n";
        assert(false);
    }
}

void Test() {
    Check("(())", "CORRECT");
    Check("([)]", "2");
    Check("(([{", "4");
    Check("(([{}]))", "CORRECT");
    Check("([}])", "2");
    Check("}]", "0");
    Check("({}]", "3");
    Check("(([{}])[)]", "8");
    std::cout << "OK\n";
}

int main(int argc, char *argv[]) {
    SetUp();
    if (argc > 1 && std::strcmp(argv[1], "test") == 0) {
        Test();
    } else {
        std::cout << Solve(InputType()).ToString() << std::endl;
    }
    return 0;
}
