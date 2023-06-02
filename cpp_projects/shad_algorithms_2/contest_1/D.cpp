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

namespace io {

class Input {
public:
    std::string string;

    Input() = default;

    explicit Input(std::istream &in) {
        in >> string;
    }
};

class Output {
public:
    int32_t max_substring_frequency;

    Output() = default;

    explicit Output(int32_t max_substring_frequency)
        : max_substring_frequency{max_substring_frequency} {
    }

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        ss >> max_substring_frequency;
    }

    std::ostream &Write(std::ostream &out) const {
        out << max_substring_frequency;
        return out;
    }

    bool operator!=(const Output &other) const {
        return max_substring_frequency != other.max_substring_frequency;
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

template <class Iterator = std::string::const_iterator>
class ZFunctionComputerWithMaxPrefixFrequency {
public:
    std::vector<int32_t> z_function;
    int32_t max_prefix_frequency = 1;

    void Compute(Iterator begin, Iterator end) {
        Reset(begin, end);

        for (int32_t index = 1; index < Size(); ++index) {
            z_function[index] = CalculateZFunctionAt(index);
        }
    }

    [[nodiscard]] int32_t Size() const {
        return end_ - begin_;
    }

private:
    Iterator begin_;
    Iterator end_;
    int32_t argmax_i_plus_z_i_ = 0;

    void Reset(Iterator begin, Iterator end) {
        begin_ = begin;
        end_ = end;
        argmax_i_plus_z_i_ = 1;
        z_function.clear();
        z_function.resize(Size());
        z_function[0] = Size();
    }

    [[nodiscard]] int32_t CalculateZFunctionAt(int32_t index) {
        int32_t index_minus_argmax = index - argmax_i_plus_z_i_;
        auto new_max_z_value = std::max(0, z_function[argmax_i_plus_z_i_] - index_minus_argmax);

        if (z_function[index_minus_argmax] < new_max_z_value) {
            return z_function[index_minus_argmax];
        }

        while (index + new_max_z_value < Size() and
               *(begin_ + new_max_z_value) == *(begin_ + index + new_max_z_value)) {
            ++new_max_z_value;
        }
        argmax_i_plus_z_i_ = index;

        max_prefix_frequency = std::max(max_prefix_frequency, 1 + new_max_z_value / index);

        return new_max_z_value;
    }
};

io::Output Solve(const io::Input &input) {
    ZFunctionComputerWithMaxPrefixFrequency computer;

    for (auto iter = input.string.begin(); iter < input.string.end(); ++iter) {
        computer.Compute(iter, input.string.end());
    }

    return io::Output{computer.max_prefix_frequency};
}

namespace test {

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

io::Output BruteForceSolve(const io::Input &input) {

    int32_t answer = 1;

    ZFunctionComputerWithMaxPrefixFrequency computer;

    for (auto string_iter = input.string.cbegin(); string_iter != input.string.cend();
         ++string_iter) {
        computer.Compute(string_iter, input.string.cend());

        for (int32_t index = 1; index < input.string.cend() - string_iter; ++index) {
            answer = std::max(answer, computer.z_function[index] / index + 1);
        }
    }
    std::cout << answer << std::endl;

    return io::Output{answer};
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t period_size = 1 + test_case_id / 5;
    int32_t n_periods = 1 + test_case_id / 10;

    std::uniform_int_distribution<char> letters_dist{'a', 'z'};
    std::stringstream ss;
    for (int32_t i = 0; i < period_size; ++i) {
        ss << letters_dist(*rng::GetEngine());
    }

    std::stringstream period_ss;
    for (int32_t i = 0; i < period_size; ++i) {
        period_ss << letters_dist(*rng::GetEngine());
    }
    auto period = period_ss.str();

    for (int32_t i = 0; i < n_periods; ++i) {
        ss << period;
    }

    for (int32_t i = 0; i < period_size; ++i) {
        ss << letters_dist(*rng::GetEngine());
    }

    io::Input input;
    input.string = ss.str();
    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    return GenerateRandomTestIo(500);
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
                ss << "\n================================Expected=============================="
                      "==\n"
                   << expected_output
                   << "\n================================Received=============================="
                      "==\n"
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

    timed_check.Check("xabaabaabaab", "3");

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
