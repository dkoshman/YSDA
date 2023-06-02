#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace io {

class Input {
public:
    std::vector<int32_t> prefix_function;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t string_size = 0;
        in >> string_size;
        prefix_function.resize(string_size);
        for (auto &value : prefix_function) {
            in >> value;
        }
    }
};

class Output {
public:
    std::vector<int32_t> z_function;

    Output() = default;

    explicit Output(std::vector<int32_t> prefix_function)
        : z_function{std::move(prefix_function)} {
    }

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        int32_t item = 0;
        while (ss >> item) {
            z_function.emplace_back(item);
        }
    }

    std::ostream &Write(std::ostream &out) const {
        for (auto item : z_function) {
            out << item << ' ';
        }
        return out;
    }

    bool operator!=(const Output &other) const {
        return z_function != other.z_function;
    }
};

std::ostream &operator<<(std::ostream &os, Output const &output) {
    return output.Write(os);
}

void SetUpFastIo() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
}

}  // namespace io

std::vector<int32_t> ZToPrefixFunction(const std::vector<int32_t> &z_function) {
    auto size = static_cast<int32_t>(z_function.size());
    std::vector<int32_t> prefix_function(size);

    for (int32_t i = 1; i < size; ++i) {
        int32_t pref_peak_index = i + z_function[i] - 1;
        prefix_function[pref_peak_index] =
            std::max(z_function[i], prefix_function[pref_peak_index]);
    }

    for (int32_t i = size - 2; i > 0; --i) {
        prefix_function[i] = std::max(prefix_function[i], prefix_function[i + 1] - 1);
    }

    return prefix_function;
}

io::Output Solve(const io::Input &input) {
    return io::Output{ZToPrefixFunction(input.prefix_function)};
}

namespace test {

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

namespace detail {

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

int64_t Timeit(const std::function<void()> &function) {
    detail::TimeItInMilliseconds time;
    function();
    time.End();
    return time.Duration();
}

struct Stats {
    double mean = 0;
    double std = 0;
    double max = 0;
};

std::ostream &operator<<(std::ostream &os, const Stats &stats) {
    os << "\tMean:\t" + std::to_string(stats.mean) << '\n'
       << "\tStd:\t" + std::to_string(stats.std) << '\n'
       << "\tMax:\t" + std::to_string(stats.max) << '\n';
    return os;
}

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

}  // namespace detail

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

struct TestIo {
    io::Input input;
    std::optional<io::Output> optional_expected_output;

    explicit TestIo(io::Input input) : input{std::move(input)} {
    }

    TestIo(io::Input input, io::Output output)
        : input{std::move(input)}, optional_expected_output{std::move(output)} {
    }
};

int32_t FindLargestBorderSizeFollowedByLetterInPrefix(const std::string &string,
                                                      const std::vector<int32_t> &prefix_function,
                                                      int32_t prefix_size) {
    auto letter = string[prefix_size];

    while (prefix_size > 0) {
        prefix_size = prefix_function[prefix_size - 1];
        if (string[prefix_size] == letter) {
            return prefix_size;
        }
    }

    return -1;
}

std::vector<int32_t> CalculatePrefixFunctionForString(const std::string &input) {
    std::vector<int32_t> prefix_function(input.size());
    for (int32_t index = 0; index < static_cast<int32_t>(input.size()); ++index) {
        prefix_function[index] =
            1 + FindLargestBorderSizeFollowedByLetterInPrefix(input, prefix_function, index);
    }
    return prefix_function;
}

void CalculateZFunction(const std::string &input, std::vector<int32_t> &z_function, int32_t index,
                        int32_t &max_block_index) {
    if (index == 0) {
        z_function[index] = input.size();
        return;
    }

    int32_t delta = index - max_block_index;
    int32_t z_value = -1;

    if (max_block_index == 0 or z_function[max_block_index] <= delta) {
        z_value = 0;
    } else if (z_function[delta] + delta >= z_function[max_block_index]) {
        z_value = z_function[max_block_index] - delta;
    }

    if (z_value == -1) {
        z_function[index] = z_function[delta];
    } else {
        while (index + z_value < static_cast<int32_t>(input.size()) and
               input[z_value] == input[index + z_value]) {
            ++z_value;
        }
        z_function[index] = z_value;
        max_block_index = index;
    }
}

std::vector<int32_t> CalculateZFunctionForString(const std::string &input) {
    std::vector<int32_t> z_function(input.size());

    int32_t max_block_index = 0;
    for (size_t index = 0; index < z_function.size(); ++index) {
        CalculateZFunction(input, z_function, index, max_block_index);
    }

    return z_function;
}

io::Output BruteForceSolve(const io::Input &input) {
    throw NotImplementedError{};
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t size = 1 + test_case_id;

    std::uniform_int_distribution<char> letters_dist{'a', 'b'};
    std::stringstream ss;
    for (int32_t i = 0; i < size; ++i) {
        ss << letters_dist(*rng::GetEngine());
    }
    auto string = ss.str();

    io::Input input;
    input.prefix_function = CalculateZFunctionForString(string);
    auto prefix_function = CalculatePrefixFunctionForString(string);

    return TestIo{input, io::Output{prefix_function}};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    return GenerateRandomTestIo(200'000);
}

class TimedChecker {
public:
    std::vector<int64_t> durations;

    void Check(const std::string &test_case, const std::string &expected) {
        std::stringstream input_stream{test_case};
        io::Input input{input_stream};
        io::Output expected_output{expected};
        TestIo test_io{input, expected_output};
        Check(test_io);
    }

    io::Output TimedSolve(const io::Input &input) {
        io::Output output;
        auto solve = [&output, &input]() { output = Solve(input); };

        durations.emplace_back(detail::Timeit(solve));
        return output;
    }

    void Check(TestIo test_io) {
        auto output = TimedSolve(test_io.input);

        if (not test_io.optional_expected_output) {
            try {
                test_io.optional_expected_output = BruteForceSolve(test_io.input);
            } catch (const NotImplementedError &e) {
            }
        }

        if (test_io.optional_expected_output) {
            auto &expected_output = test_io.optional_expected_output.value();

            if (output != expected_output) {
                Solve(test_io.input);

                std::stringstream ss;
                ss << "\n================================Expected================================\n"
                   << expected_output
                   << "\n================================Received================================\n"
                   << output << "\n";

                throw WrongAnswerException{ss.str()};
            }
        }
    }
};

std::ostream &operator<<(std::ostream &os, TimedChecker &timed_checker) {
    if (not timed_checker.durations.empty()) {
        auto duration_stats =
            detail::ComputeStats(timed_checker.durations.begin(), timed_checker.durations.end());
        std::cerr << duration_stats;
        timed_checker.durations.clear();
    }
    return os;
}

void Test() {
    rng::PrintSeed();

    TimedChecker timed_check;

    timed_check.Check(
        "8\n"
        "8 0 1 0 3 0 1 1\n",
        "0 0 1 0 1 2 3 1");

    std::cerr << "Basic tests OK:\n" << timed_check;

    int32_t n_random_test_cases = 100;

    try {

        for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {
            timed_check.Check(GenerateRandomTestIo(test_case_id));
        }

        std::cerr << "Random tests OK:\n" << timed_check;
    } catch (const NotImplementedError &e) {
    }

    int32_t n_stress_test_cases = 1;

    try {
        for (int32_t test_case_id = 0; test_case_id < n_stress_test_cases; ++test_case_id) {
            timed_check.Check(GenerateStressTestIo(test_case_id));
        }

        std::cerr << "Stress tests tests OK:\n" << timed_check;
    } catch (const NotImplementedError &e) {
    }

    std::cerr << "OK\n";
}

}  // namespace test

int main(int argc, char *argv[]) {

    io::SetUpFastIo();

    if (argc > 1 && std::strcmp(argv[1], "test") == 0) {
        test::Test();
    } else {
        std::cout << Solve(io::Input{std::cin});
    }

    return 0;
}
