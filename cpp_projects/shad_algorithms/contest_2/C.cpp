#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

using Matrix = std::vector<std::vector<int32_t>>;

class InputType {
public:
    InputType() = default;

    explicit InputType(std::istream &in) {
        in >> n_rows >> n_cols;
        matrix.resize(n_rows);
        for (auto &row : matrix) {
            row.resize(n_cols);
            for (auto &element : row) {
                in >> element;
            }
        }
    }

    int32_t n_rows = 0;
    int32_t n_cols = 0;
    Matrix matrix;
};

class OutputType {
public:
    OutputType() = default;

    explicit OutputType(const std::string &string) {
        std::stringstream ss{string};
        int32_t item = 0;
        while (ss >> item) {
            merged_and_sorted_values.push_back(item);
        }
    }

    std::ostream &Write(std::ostream &out) const {
        for (auto item : merged_and_sorted_values) {
            out << item << ' ';
        }
        return out;
    }

    bool operator==(const OutputType &other) const {
        if (merged_and_sorted_values.size() != other.merged_and_sorted_values.size()) {
            return false;
        }
        for (size_t i = 0; i < merged_and_sorted_values.size(); ++i) {
            if (merged_and_sorted_values[i] != other.merged_and_sorted_values[i]) {
                return false;
            }
        }
        return true;
    }

    std::vector<int32_t> merged_and_sorted_values;
};

std::ostream &operator<<(std::ostream &os, OutputType const &output) {
    return output.Write(os);
}

struct ValueAndParentPointer {
    int32_t value;
    std::vector<int32_t>::const_iterator parent_it;
    std::vector<int32_t>::const_iterator parent_end;
};

class ComparatorByValueGreater {
public:
    bool operator()(const ValueAndParentPointer &lhv, const ValueAndParentPointer &rhv) const {
        return lhv.value > rhv.value;
    }
};

OutputType Solve(InputType input) {
    const auto &matrix = input.matrix;
    std::vector<ValueAndParentPointer> min_heap;

    for (const auto &vector : matrix) {
        min_heap.push_back({vector.front(), vector.begin(), vector.end()});
    }

    std::make_heap(min_heap.begin(), min_heap.end(), ComparatorByValueGreater());
    OutputType output;
    output.merged_and_sorted_values.reserve(matrix.size() * matrix.front().size());

    while (not min_heap.empty()) {
        std::pop_heap(min_heap.begin(), min_heap.end(), ComparatorByValueGreater());
        auto heap_min = min_heap.back();
        min_heap.pop_back();
        output.merged_and_sorted_values.push_back(heap_min.value);
        if (++heap_min.parent_it != heap_min.parent_end) {
            heap_min.value = *heap_min.parent_it;
            min_heap.push_back(heap_min);
            std::push_heap(min_heap.begin(), min_heap.end(), ComparatorByValueGreater());
        }
    }

    return output;
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

void ThrowIfAnswerIsIncorrect(const InputType &input, const OutputType &output,
                              const OutputType &expected) {
    if (not(output == expected)) {
        std::stringstream ss;
        ss << "\nExpected:\n" << expected << "\nReceived:" << output << "\n";
        throw WrongAnswerException{ss.str()};
    }
}

void Check(const std::string &test_case, const std::string &expected) {
    std::stringstream input_stream{test_case};
    auto input = InputType{input_stream};
    auto output = Solve(input);
    ThrowIfAnswerIsIncorrect(input, output, OutputType{expected});
}

void Test() {
    Check(
        "2 5\n"
        "1 3 5 7 9\n"
        "2 4 6 8 10\n",
        "1 2 3 4 5 6 7 8 9 10");
    Check(
        "4 2\n"
        "1 4\n"
        "2 8\n"
        "3 7\n"
        "5 6\n",
        "1 2 3 4 5 6 7 8");

    std::random_device random_device;
    std::mt19937 generator(random_device());

    for (int test_case_id = 1; test_case_id < 1 << 10; test_case_id <<= 1) {

        std::uniform_int_distribution<> distribution(-(1 << 30), 1 << 30);

        InputType input;
        input.matrix.resize(test_case_id);
        OutputType expected;

        for (auto &vector : input.matrix) {
            vector.resize(test_case_id);
            for (auto &element : vector) {
                element = distribution(generator);
            }
            std::sort(vector.begin(), vector.end());
            expected.merged_and_sorted_values.insert(expected.merged_and_sorted_values.end(),
                                                     vector.begin(), vector.end());
        }

        std::sort(expected.merged_and_sorted_values.begin(),
                  expected.merged_and_sorted_values.end());
        ThrowIfAnswerIsIncorrect(input, Solve(input), OutputType{expected});
    }

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
