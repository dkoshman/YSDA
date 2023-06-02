#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

struct Request {
    int32_t non_decreasing_array_id = 0;
    int32_t non_increasing_array_id = 0;
};

using Matrix = std::vector<std::vector<int32_t>>;

class InputType {
public:
    explicit InputType(std::istream &in) {
        in >> non_decreasing_n >> non_increasing_n >> array_size;
        ReadMatrix(in, non_decreasing_matrix, non_decreasing_n, array_size);
        ReadMatrix(in, non_increasing_matrix, non_increasing_n, array_size);
        in >> requests_n;
        requests.resize(requests_n);
        for (auto &[i, j] : requests) {
            in >> i >> j;
            --i;
            --j;
        }
    }

    void ReadMatrix(std::istream &in, Matrix &matrix, size_t n_rows, size_t n_columns) const {
        matrix.resize(n_rows, std::vector<int32_t>(n_columns));
        for (auto &row : matrix) {
            for (auto &i : row) {
                in >> i;
            }
        }
    }

    size_t non_decreasing_n = 0;
    size_t non_increasing_n = 0;
    size_t array_size = 0;
    size_t requests_n = 0;
    Matrix non_decreasing_matrix;
    Matrix non_increasing_matrix;
    std::vector<Request> requests;
};

class OutputType {
public:
    OutputType() = default;

    explicit OutputType(const std::string &string) {
        std::stringstream ss{string};
        int32_t item = 0;
        while (ss >> item) {
            answers.push_back(--item);
        }
    }

    std::ostream &Write(std::ostream &out) const {
        for (auto item : answers) {
            out << item + 1 << '\n';
        }
        return out;
    }

    std::vector<int32_t> answers;
};

std::ostream &operator<<(std::ostream &os, OutputType const &output) {
    return output.Write(os);
}

class WrongAnswerException : public std::exception {
public:
    explicit WrongAnswerException(std::string const &message) : message{message.data()} {
    }

    const char *what() const noexcept {
        return message;
    }

    const char *message;
};

void ThrowIfAnswerIsIncorrect(const InputType &input, const OutputType &output) {
    if (input.requests.size() != output.answers.size()) {
        throw WrongAnswerException{"Incorrect number of answers."};
    }

    for (size_t request_id = 0; request_id < input.requests_n; ++request_id) {
        auto non_decreasing =
            input.non_decreasing_matrix[input.requests[request_id].non_decreasing_array_id];
        auto non_increasing =
            input.non_increasing_matrix[input.requests[request_id].non_increasing_array_id];
        auto answer = output.answers[request_id];
        auto output_min_max = std::max(non_decreasing[answer], non_increasing[answer]);

        for (size_t index = 0; index < input.array_size; ++index) {
            if (std::max(non_decreasing[index], non_increasing[index]) < output_min_max) {
                std::stringstream ss;
                ss << "\nGiven arrays:\n";
                for (auto j : non_decreasing) {
                    ss << j << ' ';
                }
                ss << "\nand\n";
                for (auto j : non_increasing) {
                    ss << j << ' ';
                }
                ss << "\nReceived:\n" << answer << "\nWhich is not the minimum max.";
                throw WrongAnswerException{ss.str()};
            }
        }
    }
}

int FindPositionWithMinimalValueBetweenSortedArrays(const std::vector<int> &non_decreasing,
                                                    const std::vector<int> &non_increasing) {
    size_t left = 0;
    size_t right = non_decreasing.size() - 1;
    while (left + 1 < right) {
        int middle = (left + right) / 2;
        if (non_decreasing[middle] < non_increasing[middle]) {
            left = middle;
        } else {
            right = middle;
        }
    }
    if (std::max(non_decreasing[left], non_increasing[left]) <=
        std::max(non_decreasing[right], non_increasing[right])) {
        return left;
    } else {
        return right;
    }
}

OutputType Solve(InputType input) {
    OutputType result;
    result.answers.reserve(input.requests.size());
    for (auto [i, j] : input.requests) {
        result.answers.push_back(FindPositionWithMinimalValueBetweenSortedArrays(
            input.non_decreasing_matrix[i], input.non_increasing_matrix[j]));
    }
    return result;
}

void Check(const std::string &test_case, const std::string &expected) {
    std::stringstream input_stream{test_case};
    auto input = InputType{input_stream};
    auto output = Solve(input);
    ThrowIfAnswerIsIncorrect(input, OutputType{expected});
    ThrowIfAnswerIsIncorrect(input, output);
}

void Test() {
    Check(
        "4 3 5\n"
        "1 2 3 4 5\n"
        "1 1 1 1 1\n"
        "0 99999 99999 99999 99999\n"
        "0 0 0 0 99999\n"
        "5 4 3 2 1\n"
        "99999 99999 99999 0 0\n"
        "99999 99999 0 0 0\n"
        "12\n"
        "1 1\n"
        "1 2\n"
        "1 3\n"
        "2 1\n"
        "2 2\n"
        "2 3\n"
        "3 1\n"
        "3 2\n"
        "3 3\n"
        "4 1\n"
        "4 2\n"
        "4 3\n",
        "3\n"
        "4\n"
        "3\n"
        "5\n"
        "4\n"
        "3\n"
        "1\n"
        "4\n"
        "3\n"
        "4\n"
        "4\n"
        "4\n");

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
