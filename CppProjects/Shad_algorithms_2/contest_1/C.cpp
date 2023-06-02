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
    std::vector<int32_t> z_function;

    Output() = default;

    explicit Output(std::vector<int32_t> prefix_function) : z_function{std::move(prefix_function)} {
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

class ZFunctionComputer {
public:
    const std::string &string;
    std::vector<int32_t> z_function;

    explicit ZFunctionComputer(const std::string &string)
        : string{string}, z_function(string.size()), size_{static_cast<int32_t>(string.size())} {
    }

    std::vector<int32_t> Compute() {
        argmax_i_plus_z_i_ = 1;
        z_function[0] = size_;

        for (int32_t index = 1; index < size_; ++index) {
            z_function[index] = CalculateZFunctionAt(index);
        }

        return z_function;
    }

private:
    int32_t argmax_i_plus_z_i_ = 0;
    int32_t size_ = 0;

    [[nodiscard]] int32_t CalculateZFunctionAt(int32_t index) {
        int32_t index_minus_argmax = index - argmax_i_plus_z_i_;
        auto new_max_z_value = std::max(0, z_function[argmax_i_plus_z_i_] - index_minus_argmax);

        if (z_function[index_minus_argmax] < new_max_z_value) {
            return z_function[index_minus_argmax];
        }

        while (index + new_max_z_value < size_ and
               string[new_max_z_value] == string[index + new_max_z_value]) {
            ++new_max_z_value;
        }
        argmax_i_plus_z_i_ = index;
        return new_max_z_value;
    }
};

io::Output Solve(const io::Input &input) {
    return io::Output{ZFunctionComputer{input.string}.Compute()};
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
    auto size = static_cast<int32_t>(input.string.size());
    io::Output output;
    output.z_function.resize(size);

    for (int32_t index = 0; index < size; ++index) {
        int32_t z_value = 0;
        while (z_value + index < size) {
            if (input.string[z_value] != input.string[index + z_value]) {
                break;
            }
            ++z_value;
        }
        output.z_function[index] = z_value;
    }
    return output;
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t size = 1 + test_case_id;

    std::uniform_int_distribution<char> letters_dist{'a', 'c'};
    std::stringstream ss;
    for (int32_t i = 0; i < size; ++i) {
        ss << letters_dist(*rng::GetEngine());
    }

    io::Input input;
    input.string = ss.str();
    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    return GenerateRandomTestIo(1'000'000);
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

    timed_check.Check("abracadabra", "11 0 0 1 0 1 0 4 0 0 1");

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
