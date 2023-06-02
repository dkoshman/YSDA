// https://contest.yandex.ru/contest/29059/problems/

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

constexpr int32_t kAnswerModDivisor = 123'456'789;

namespace rng {

uint32_t GetSeed() {
    auto random_device = std::random_device{};
    static auto seed = random_device();
    return seed;
}

void PrintSeed(std::ostream &ostream = std::cerr) {
    ostream << "Seed = " << GetSeed() << std::endl;
}

std::mt19937 *GetEngine() {
    static std::mt19937 engine(GetSeed());
    return &engine;
}

}  // namespace rng

int32_t PositiveMod(int64_t value, int32_t divisor) {
    if (divisor == 0) {
        throw std::invalid_argument("Zero divisor.");
    }
    int64_t mod = value % divisor;
    if (mod < 0) {
        mod += divisor > 0 ? divisor : -divisor;
    }
    return static_cast<int32_t>(mod);
}

namespace io {

class InputType {
public:
    std::vector<int32_t> numbers_in_preorder;

    InputType() = default;

    explicit InputType(std::istream &in) {
        int32_t count_elements = 0;
        in >> count_elements;
        numbers_in_preorder.resize(count_elements);
        for (auto &element : numbers_in_preorder) {
            in >> element;
        }
    }
};

class OutputType {
public:
    OutputType() = default;

    explicit OutputType(const std::string &string) {
        std::stringstream ss{string};
        ss >> n_binary_search_trees_mod_kAnswerModDivisor;
    }

    explicit OutputType(int32_t n_binary_search_trees_mod_k)
        : n_binary_search_trees_mod_kAnswerModDivisor{n_binary_search_trees_mod_k} {
    }

    std::ostream &Write(std::ostream &out) const {
        out << n_binary_search_trees_mod_kAnswerModDivisor << '\n';
        return out;
    }

    bool operator!=(const OutputType &other) const {
        return n_binary_search_trees_mod_kAnswerModDivisor !=
               other.n_binary_search_trees_mod_kAnswerModDivisor;
    }

    int32_t n_binary_search_trees_mod_kAnswerModDivisor = 0;
};

std::ostream &operator<<(std::ostream &os, OutputType const &output) {
    return output.Write(os);
}

}  // namespace io

using io::InputType, io::OutputType;

class BinarySearchTreeCounter {
public:
    const int32_t kAnswerModDivisor = 0;

    BinarySearchTreeCounter(int32_t k_answer_mod_divisor,
                            const std::vector<int32_t> &unsorted_numbers)
        : kAnswerModDivisor{k_answer_mod_divisor},
          sorted_numbers_{unsorted_numbers},
          n_binary_search_trees_on_sorted_intervals_(unsorted_numbers.size() + 1) {

        std::sort(sorted_numbers_.begin(), sorted_numbers_.end());
        for (auto &vector : n_binary_search_trees_on_sorted_intervals_) {
            vector.resize(sorted_numbers_.size() + 1);
        }
    }

    int32_t CountAllPossibleFullSizedBinarySearchTreesModDivisor() {
        return static_cast<int32_t>(CountInterval(0, static_cast<int32_t>(sorted_numbers_.size())));
    }

private:
    std::vector<int32_t> sorted_numbers_;
    std::vector<std::vector<std::optional<int32_t>>> n_binary_search_trees_on_sorted_intervals_;

    [[nodiscard]] int32_t Mod(int64_t value) const {
        return PositiveMod(value, kAnswerModDivisor);
    }

    int64_t CountInterval(int32_t begin, int32_t end) {
        if (begin > end) {
            throw std::invalid_argument("Begin is less than end.");
        }

        auto &optional_answer = n_binary_search_trees_on_sorted_intervals_[begin][end];

        if (not optional_answer) {
            if (end - begin <= 1) {
                optional_answer = 1;
            } else {
                optional_answer = CountValidIntervalNoShorterThanTwo(begin, end);
            }
        }

        return optional_answer.value();
    }

    inline int64_t CountValidIntervalNoShorterThanTwo(int32_t begin, int32_t end) {
        int64_t count = 0;
        for (auto root = begin; root < end; ++root) {
            if (CanBeRoot(begin, root)) {
                count = Mod(CountInterval(begin, root) * CountInterval(root + 1, end) + count);
            }
        }
        return count;
    }

    [[nodiscard]] inline bool CanBeRoot(int32_t begin, int32_t root) const {
        return begin == root or
               (begin < root and sorted_numbers_[root - 1] < sorted_numbers_[root]);
    }
};

OutputType Solve(const InputType &input) {
    BinarySearchTreeCounter counter{kAnswerModDivisor, input.numbers_in_preorder};
    return OutputType{counter.CountAllPossibleFullSizedBinarySearchTreesModDivisor()};
}

namespace test {

class WrongAnswerException : public std::exception {
public:
    WrongAnswerException() = default;

    explicit WrongAnswerException(std::string const &message) : message{message.data()} {
    }

    [[nodiscard]] const char *what() const noexcept override {
        return message;
    }

    const char *message{};
};

class NotImplementedError : public std::logic_error {
public:
    NotImplementedError() : std::logic_error("Function not yet implemented."){};
};

OutputType BruteForceSolve(const InputType &input) {
    auto elements = input.numbers_in_preorder;
    std::sort(elements.begin(), elements.end());
    std::vector<std::vector<int32_t>> intervals_by_size(elements.size() + 1);
    for (auto &intervals : intervals_by_size) {
        intervals.resize(elements.size() + 1);
    }
    for (int32_t interval_size = 0; interval_size < 2; ++interval_size) {
        for (auto &count : intervals_by_size[interval_size]) {
            count = 1;
        }
    }
    size_t intervals_by_size_size = intervals_by_size.size();
    size_t elements_size = elements.size();
    for (size_t interval_size = 2; interval_size < intervals_by_size_size; ++interval_size) {
        for (size_t interval_start = 0; interval_start <= elements_size - interval_size;
             ++interval_start) {
            int32_t count = 0;
            for (size_t index = interval_start; index < interval_start + interval_size; ++index) {
                if (index == interval_start or elements[index] != elements[index - 1]) {
                    size_t left = index - interval_start;
                    size_t right = interval_start + interval_size - index - 1;
                    count =
                        PositiveMod(static_cast<int64_t>(intervals_by_size[left][interval_start]) *
                                            intervals_by_size[right][index + 1] +
                                        count,
                                    kAnswerModDivisor);
                }
            }
            intervals_by_size[interval_size][interval_start] = count;
        }
    }
    return OutputType{intervals_by_size.back().front()};
}

std::pair<InputType, std::optional<OutputType>> GenerateRandomTestIo(
    int32_t test_case_id) {
    int32_t n_numbers = 1 + test_case_id * 4;
    InputType input;
    input.numbers_in_preorder.resize(n_numbers);
    std::uniform_int_distribution<int32_t> distribution{INT32_MIN, INT32_MAX};
    auto &engine = *rng::GetEngine();
    for (auto &number : input.numbers_in_preorder) {
        number = distribution(engine);
    }

    try {
        auto expected_output = BruteForceSolve(input);
        return {input, expected_output};
    } catch (const NotImplementedError &e) {
        return {input, std::nullopt};
    }
}

class TimeItInMilliseconds {
public:
    std::chrono::time_point<std::chrono::steady_clock> begin;
    std::chrono::time_point<std::chrono::steady_clock> end;

    TimeItInMilliseconds() {
        Begin();
    }

    void Begin() {
        begin = std::chrono::steady_clock::now();
    }

    int64_t End() {
        end = std::chrono::steady_clock::now();
        return Duration();
    }

    [[nodiscard]] int64_t Duration() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    }
};

int64_t Check(const InputType &input,
              const std::optional<OutputType> &expected_output_optional = std::nullopt) {

    TimeItInMilliseconds time;
    auto output = Solve(input);
    time.End();

    if (expected_output_optional) {
        auto expected_output = expected_output_optional.value();

        if (output != expected_output) {
            std::stringstream ss;
            ss << "\n==============================Expected==============================\n"
               << expected_output
               << "\n==============================Received==============================\n"
               << output << "\n";
            throw WrongAnswerException{ss.str()};
        }
    }

    return time.Duration();
}

int64_t Check(const std::string &test_case, int32_t expected) {
    std::stringstream input_stream{test_case};
    return Check(InputType{input_stream}, OutputType{expected});
}

struct Stats {
    double mean = 0;
    double std = 0;
    double max = 0;
};

template <class Iterator>
Stats ComputeStats(Iterator begin, Iterator end) {
    auto size = end - begin;
    if (size == 0) {
        throw std::invalid_argument{"Empty container."};
    }

    auto mean = std::accumulate(begin, end, 0.0) / size;

    double std = 0;
    for (auto i = begin; i != end; ++i) {
        std += (*i - mean) * (*i - mean);
    }
    std = std::sqrt(std / size);

    auto max = static_cast<double>(*std::max_element(begin, end));

    return Stats{mean, std, max};
}

void Test() {
    rng::PrintSeed();

    Check(
        "2\n"
        "2 1\n",
        2);

    Check(
        "3\n"
        "10 10 10\n",
        1);

    Check(
        "3\n"
        "1 2 3\n",
        5);

    Check(
        "3\n"
        "1 2 1\n",
        3);

    Check(
        "3\n"
        "2 2 1\n",
        2);

    Check(
        "4\n"
        "2 3 1 2\n",
        7);

    Check(
        "1\n"
        "-100\n",
        1);

    int32_t n_test_cases = 100;
    std::vector<int64_t> durations;
    durations.reserve(n_test_cases);
    TimeItInMilliseconds time_it;

    for (int32_t test_case_id = 0; test_case_id < n_test_cases; ++test_case_id) {
        auto [input, expected_output_optional] = GenerateRandomTestIo(test_case_id);
        durations.emplace_back(Check(input, expected_output_optional));
    }

    auto duration_stats = ComputeStats(durations.begin(), durations.end());
    std::cerr << "Solve duration stats in milliseconds:\n"
              << "\tMean:\t" + std::to_string(duration_stats.mean) << '\n'
              << "\tStd:\t" + std::to_string(duration_stats.std) << '\n'
              << "\tMax:\t" + std::to_string(duration_stats.max) << '\n';

    std::cout << "OK\n";
}

}  // namespace test

void SetUp() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
}

int main(int argc, char *argv[]) {
    SetUp();
    if (argc > 1 && std::strcmp(argv[1], "test") == 0) {
        test::Test();
    } else {
        std::cout << Solve(InputType{std::cin});
    }
    return 0;
}
