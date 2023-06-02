#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <random>
#include <vector>
#include <utility>

class RandomNumberGenerator {
public:
    explicit RandomNumberGenerator(unsigned a_parameter, unsigned b_parameter)
        : a_parameter_{a_parameter}, b_parameter_{b_parameter} {
    }

    unsigned NextRand24() {
        current_ = current_ * a_parameter_ + b_parameter_;
        return current_ >> 8;
    }
    unsigned NextRand32() {
        unsigned first = NextRand24();
        unsigned second = NextRand24();
        return (first << 8) ^ second;
    }

private:
    unsigned current_ = 0;
    unsigned a_parameter_ = 0;
    unsigned b_parameter_ = 0;
};

class InputType {
public:
    explicit InputType(std::istream &in) {
        in >> n_elements >> a_parameter >> b_parameter;
    }

    std::vector<unsigned> GenerateNumbers() const {
        std::vector<unsigned> numbers;
        numbers.resize(n_elements);
        RandomNumberGenerator generator(a_parameter, b_parameter);
        for (auto &item : numbers) {
            item = generator.NextRand32();
        }
        return numbers;
    }

    int32_t n_elements = 0;
    unsigned a_parameter = 0;
    unsigned b_parameter = 0;
};

class OutputType {
public:
    OutputType() = default;

    explicit OutputType(int64_t min_total_distance_to_all_points)
        : min_total_distance_to_all_points{min_total_distance_to_all_points} {
    }

    explicit OutputType(const std::string &string) {
        std::stringstream ss{string};
        ss >> min_total_distance_to_all_points;
    }

    std::ostream &Write(std::ostream &out) const {
        out << min_total_distance_to_all_points;
        return out;
    }

    bool operator==(const OutputType &other) const {
        return min_total_distance_to_all_points == other.min_total_distance_to_all_points;
    }

    int64_t min_total_distance_to_all_points = 0;
};

std::ostream &operator<<(std::ostream &os, OutputType const &output) {
    return output.Write(os);
}

template <class Iterator>
Iterator PickPivot(Iterator begin, Iterator end) {
    Iterator mid = begin + (end - begin) / 2;
    --end;
    if (*mid < *begin) {
        std::swap(mid, begin);
    }
    if (*mid < *end) {
        return mid;
    }
    if (*begin < *end) {
        return end;
    }
    return begin;
}

template <class Iterator>
Iterator Partition(Iterator begin, Iterator end, Iterator pivot) {
    auto pivot_value = *pivot;
    Iterator former_begin = begin;
    --end;

    while (begin <= end) {
        if (*begin < pivot_value) {
            ++begin;
        } else if (pivot_value < *end) {
            --end;
        } else {
            std::iter_swap(begin, end);
            ++begin;
            --end;
        }
    }

    if (begin == end + 2 and begin - 1 != former_begin) {
        return begin - 1;
    }
    return begin;
}

template <class Iterator>
unsigned FindKthStatistic(Iterator begin, Iterator end, Iterator statistic) {
    if (end - begin == 1) {
        return *begin;
    }
    Iterator pivot = PickPivot(begin, end);
    Iterator partition = Partition(begin, end, pivot);
    if (partition <= statistic) {
        return FindKthStatistic(partition, end, statistic);
    }
    return FindKthStatistic(begin, partition, statistic);
}

OutputType Solve(InputType input) {
    auto numbers = input.GenerateNumbers();

    auto median_position = numbers.begin() + numbers.size() / 2;
    auto median =
        static_cast<int64_t>(FindKthStatistic(numbers.begin(), numbers.end(), median_position));

    int64_t sum = 0;
    for (auto element : numbers) {
        sum += std::abs(median - element);
    }

    return OutputType{sum};
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
        ss << "\nExpected:\n" << expected << "\nReceived:\n" << output << "\n";
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
    Check("6 239 13", "8510257371");

    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::uniform_int_distribution<> distribution(1, 1 << 20);

    for (int test_case_id = 1; test_case_id < 1 << 5; ++test_case_id) {
        std::stringstream ss;
        ss << distribution(generator) << ' ' << distribution(generator) << ' '
           << distribution(generator);
        InputType input{ss};

        auto numbers = input.GenerateNumbers();
        auto median_position = numbers.begin() + numbers.size() / 2;
        std::nth_element(numbers.begin(), median_position, numbers.end());
        auto median = static_cast<int64_t>(*median_position);
        int64_t expected = 0;
        for (auto element : numbers) {
            expected += std::abs(median - element);
        }
        Check(ss.str(), std::to_string(expected));
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
