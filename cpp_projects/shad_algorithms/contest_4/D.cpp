#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <iomanip>
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

class InputType {
public:
    int32_t treap_size = 0;
    int32_t treap_height = 0;

    InputType() = default;

    explicit InputType(std::istream &in) {
        in >> treap_size >> treap_height;
    }
};

class OutputType {
public:
    double probability_treap_of_given_size_has_given_height = 0;

    OutputType() = default;

    explicit OutputType(double probability)
        : probability_treap_of_given_size_has_given_height{probability} {
    }

    std::ostream &Write(std::ostream &out) const {
        out << std::setprecision(10) << probability_treap_of_given_size_has_given_height;
        return out;
    }

    bool operator!=(const OutputType &other) const {
        return std::fabs(probability_treap_of_given_size_has_given_height -
                         other.probability_treap_of_given_size_has_given_height) > 1e-6;
    }
};

std::ostream &operator<<(std::ostream &os, OutputType const &output) {
    return output.Write(os);
}

}  // namespace io

using io::InputType, io::OutputType;

class TreapHeightProbabilityComputer {
public:
    double ComputeProbabilityThatTreapOfGivenSizeHasGivenHeight(int32_t treap_size,
                                                                int32_t treap_height) {
        if (treap_size <= treap_height) {
            return 0.0;
        }

        treap_size_ = treap_size;
        treap_height_ = treap_height;

        InitProbabilitiesTables();

        DynamicComputeProbabilities();

        return probability_size_height_equal_[treap_size_][treap_height_];
    }

private:
    int32_t treap_size_ = 0;
    int32_t treap_height_ = 0;

    std::vector<std::vector<double>> probability_size_height_less_;
    std::vector<std::vector<double>> probability_size_height_equal_;

    void InitProbabilitiesTables() {
        for (auto *probability_table :
             {&probability_size_height_less_, &probability_size_height_equal_}) {
            probability_table->clear();
            probability_table->resize(treap_size_ + 1);
            for (auto &probability_row : *probability_table) {
                probability_row.resize(treap_height_ + 1);
            }
        }

        for (int32_t treap_height = 0; treap_height <= treap_height_; ++treap_height) {
            probability_size_height_less_[0][treap_height] = 1.0;
        }

        probability_size_height_equal_[1][0] = 1.0;
    }

    [[nodiscard]] double GetProbabilityForSizeHeightLessFromSmallerTreaps(
        int32_t treap_size, int32_t treap_height) const {
        auto treap_height_one_less = treap_height - 1;
        return probability_size_height_less_[treap_size][treap_height_one_less] +
               probability_size_height_equal_[treap_size][treap_height_one_less];
    }

    [[nodiscard]] double GetProbabilityForSizeHeightEqualFromSmallerTreaps(
        int32_t treap_size, int32_t treap_height) const {
        auto treap_height_one_less = treap_height - 1;
        double probability = 0.;

        for (int32_t left_subtreap_size = 0; left_subtreap_size < treap_size;
             ++left_subtreap_size) {
            auto probability_of_left_subtreap_size = 1.0 / treap_size;

            auto right_subtreap_size = treap_size - left_subtreap_size - 1;

            auto probability_left_equal_right_equal =
                probability_size_height_equal_[left_subtreap_size][treap_height_one_less] *
                probability_size_height_equal_[right_subtreap_size][treap_height_one_less];

            auto probability_left_less_right_equal =
                probability_size_height_less_[left_subtreap_size][treap_height_one_less] *
                probability_size_height_equal_[right_subtreap_size][treap_height_one_less];

            auto probability_left_equal_right_less =
                probability_size_height_equal_[left_subtreap_size][treap_height_one_less] *
                probability_size_height_less_[right_subtreap_size][treap_height_one_less];

            probability += probability_of_left_subtreap_size *
                           (probability_left_equal_right_equal + probability_left_less_right_equal +
                            probability_left_equal_right_less);
        }

        return probability;
    }

    void DynamicComputeProbabilities() {
        for (int32_t treap_height = 1; treap_height <= treap_height_; ++treap_height) {
            for (int32_t treap_size = 1; treap_size <= treap_size_; ++treap_size) {

                probability_size_height_less_[treap_size][treap_height] =
                    GetProbabilityForSizeHeightLessFromSmallerTreaps(treap_size, treap_height);

                probability_size_height_equal_[treap_size][treap_height] =
                    GetProbabilityForSizeHeightEqualFromSmallerTreaps(treap_size, treap_height);
            }
        }
    }
};

OutputType Solve(const InputType &input) {
    auto computer = TreapHeightProbabilityComputer{};
    auto probability = computer.ComputeProbabilityThatTreapOfGivenSizeHasGivenHeight(
        input.treap_size, input.treap_height);
    return OutputType{probability};
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
    throw NotImplementedError{};
}

struct TestIo {
    InputType input;
    std::optional<OutputType> optional_expected_output = std::nullopt;

    explicit TestIo(InputType input) {
        try {
            optional_expected_output = BruteForceSolve(input);
        } catch (const NotImplementedError &e) {
        }
        this->input = std::move(input);
    }

    TestIo(InputType input, OutputType output)
        : input{std::move(input)}, optional_expected_output{std::move(output)} {
    }
};

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    InputType input;
    input.treap_size = test_case_id + 1;
    auto distribution = std::uniform_int_distribution<>{0, input.treap_size};
    input.treap_height = distribution(*rng::GetEngine());
    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    InputType input;
    input.treap_size = 1e2;
    input.treap_height = 10;
    return TestIo{input};
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

int64_t Check(const TestIo &test_io) {

    TimeItInMilliseconds time;
    auto output = Solve(test_io.input);
    time.End();

    if (test_io.optional_expected_output) {
        auto &expected_output = test_io.optional_expected_output.value();

        if (output != expected_output) {
            Solve(test_io.input);
            std::stringstream ss;
            ss << "\n==================================Expected==================================\n"
               << expected_output
               << "\n==================================Received==================================\n"
               << output << "\n";
            throw WrongAnswerException{ss.str()};
        }
    }

    return time.Duration();
}

int64_t Check(const std::string &test_case, double expected) {
    std::stringstream input_stream{test_case};
    return Check(TestIo{InputType{input_stream}, OutputType{expected}});
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

    Check("1 0\n", 1);
    Check("1 1\n", 0);
    Check("3 2\n", 0.666667);
    Check("3 0\n", 0);
    Check("3 100\n", 0);
    Check("1 1\n", 0);

    std::cerr << "Basic tests OK\n";

    std::vector<int64_t> durations;
    TimeItInMilliseconds time_it;

    int32_t n_random_test_cases = 100;

    for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {
        durations.emplace_back(Check(GenerateRandomTestIo(test_case_id)));
    }

    std::cerr << "Random tests OK\n";

    int32_t n_stress_test_cases = 1;
    for (int32_t test_case_id = 0; test_case_id < n_stress_test_cases; ++test_case_id) {
        durations.emplace_back(Check(GenerateStressTestIo(test_case_id)));
    }

    std::cerr << "Stress tests tests OK\n";

    auto duration_stats = ComputeStats(durations.begin(), durations.end());
    std::cerr << "Solve duration stats in milliseconds:\n"
              << "\tMean:\t" + std::to_string(duration_stats.mean) << '\n'
              << "\tStd:\t" + std::to_string(duration_stats.std) << '\n'
              << "\tMax:\t" + std::to_string(duration_stats.max) << '\n';

    std::cerr << "OK\n";
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
