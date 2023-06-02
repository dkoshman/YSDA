#include <algorithm>
#include <array>
#include <cassert>
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

class OnlineAgent {
public:
    struct SymbolPosition {
        enum class Where { Pattern, Text };

        Where where;
        int32_t index;
    };

    int32_t pattern_size = 0;
    int32_t text_size = 0;
    int32_t questions_per_round = 0;

    explicit OnlineAgent(std::istream &in = std::cin, std::ostream &out = std::cout,
                    int32_t questions_per_round = 5)
        : questions_per_round{questions_per_round}, in_{in}, out_{out} {
        in >> pattern_size >> text_size;
    }

    bool AskIfSymbolsEqual(SymbolPosition first, SymbolPosition second) {
        SaySymbolPosition(first);
        SaySymbolPosition(second);
        out_ << std::endl;

        std::string answer;
        in_ >> answer;
        return answer == "Yes";
    }

    void TellTheCurrentNumberOfPatternOccurrencesInText(int32_t n_occurrences) {
        out_ << "$ " << n_occurrences << std::endl;
    }

private:
    std::istream &in_;
    std::ostream &out_;

    void SaySymbolPosition(SymbolPosition position) {
        out_ << (position.where == SymbolPosition::Where::Pattern ? 's' : 't') << ' '
             << position.index + 1 << ' ';
    }
};

class Input {
public:
    OnlineAgent player;

    Input() = default;

    explicit Input(std::istream &in, std::ostream &out) : player{in, out} {
    }
};

void SetUpFastIo() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
}

}  // namespace io

template <class Player>
class AgentWrapperSymbolEqualityComparator {
public:
    AgentWrapperSymbolEqualityComparator(Player &player, const std::vector<int32_t> &z_function)
        : player_{player},
          z_function_{z_function},
          pattern_size_{player.pattern_size},
          size_of_text_prefix_seen_{1},
          questions_left_{player.questions_per_round},
          occurrences_{0} {
    }

    bool operator()(int32_t first, int32_t second) {
        if (first == pattern_size_ or second == pattern_size_) {
            return false;
        }

        if (questions_left_ == 0 or
            std::max(first, second) >= pattern_size_ + 1 + size_of_text_prefix_seen_) {
            SendAnswerToAgent();
        }

        --questions_left_;
        return player_.AskIfSymbolsEqual(PatternTextConcatenatedIndexToSymbolPosition(first),
                                         PatternTextConcatenatedIndexToSymbolPosition(second));
    }

    void SendAnswerToAgent() {
        occurrences_ += z_function_[size_of_text_prefix_seen_ + 1] == pattern_size_;
        player_.TellTheCurrentNumberOfPatternOccurrencesInText(occurrences_);
        questions_left_ = player_.questions_per_round;
        ++size_of_text_prefix_seen_;
    }

private:
    Player &player_;
    const std::vector<int32_t> &z_function_;
    int32_t pattern_size_;
    int32_t size_of_text_prefix_seen_;
    int32_t questions_left_;
    int32_t occurrences_;

    [[nodiscard]] io::OnlineAgent::SymbolPosition PatternTextConcatenatedIndexToSymbolPosition(
        int32_t index) const {
        if (index < pattern_size_) {
            return {io::OnlineAgent::SymbolPosition::Where::Pattern, index};
        } else {
            return {io::OnlineAgent::SymbolPosition::Where::Text, index - pattern_size_ - 1};
        }
    };
};

class ZFunctionComputer {
public:
    std::vector<int32_t> z_function;

    explicit ZFunctionComputer(int32_t string_size) : z_function(string_size), size_{string_size} {
    }

    template <class SymbolEqualityComparator>
    std::vector<int32_t> Compute(SymbolEqualityComparator *comparator) {
        argmax_i_plus_z_i_ = 1;
        z_function[0] = size_;

        for (int32_t index = 1; index < size_; ++index) {
            z_function[index] = CalculateZFunctionAt(index, *comparator);
        }

        return z_function;
    }

private:
    int32_t argmax_i_plus_z_i_ = 0;
    int32_t size_ = 0;

    template <class SymbolEqualityComparator>
    [[nodiscard]] int32_t CalculateZFunctionAt(int32_t index,
                                               SymbolEqualityComparator &comparator) {

        int32_t index_minus_argmax = index - argmax_i_plus_z_i_;
        auto new_max_z_value = std::max(0, z_function[argmax_i_plus_z_i_] - index_minus_argmax);

        if (z_function[index_minus_argmax] < new_max_z_value) {
            return z_function[index_minus_argmax];
        }

        while (index + new_max_z_value < size_ and
               comparator(new_max_z_value, index + new_max_z_value)) {
            ++new_max_z_value;
        }
        argmax_i_plus_z_i_ = index;
        return new_max_z_value;
    }
};

template <class Player>
void InteractWithAgentToFindAllPatternOccurrences(Player &player) {

    ZFunctionComputer z_function_computer{player.pattern_size + 1 + player.text_size};

    AgentWrapperSymbolEqualityComparator<Player> comparator(player, z_function_computer.z_function);

    z_function_computer.Compute(&comparator);

    comparator.SendAnswerToAgent();
}

void Solve(io::Input input) {
    InteractWithAgentToFindAllPatternOccurrences(input.player);
}

namespace test {

namespace rng {

uint32_t GetSeed() {
    auto random_device = std::random_device{};
    static auto seed = random_device();
    return 3433391386;
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

    explicit TestIo(io::Input input) : input{std::move(input)} {
    }
};

class TestPlayer {
public:
    int32_t pattern_size = 0;
    int32_t text_size = 0;
    int32_t questions_per_round = 5;

    explicit TestPlayer(int32_t pattern_size = 2, int32_t text_size = 10,
                        int32_t questions_per_round = 5, char from = 'a', char to = 'b')
        : pattern_size{pattern_size},
          text_size{text_size},
          questions_per_round{questions_per_round},
          questions_left_{questions_per_round},
          current_round_{0} {

        std::uniform_int_distribution<char> letter_distribution{from, to};
        for (int32_t i = 0; i < pattern_size; ++i) {
            pattern_ += letter_distribution(twister_);
        }
        for (int32_t i = 0; i < text_size; ++i) {
            text_ += letter_distribution(twister_);
        }
        for (int32_t round = 0; round < text_size; ++round) {
            int32_t occurrences = 0;
            for (int32_t j = 0; j < round - pattern_size + 2; ++j) {
                ++occurrences;
                for (int32_t k = 0; k < pattern_size; ++k) {
                    if (text_[j + k] != pattern_[k]) {
                        --occurrences;
                        break;
                    }
                }
            }
            answers_.push_back(occurrences);
        }
    }

    bool AskIfSymbolsEqual(io::OnlineAgent::SymbolPosition first, io::OnlineAgent::SymbolPosition second) {
        assert(questions_left_ > 0);
        --questions_left_;
        return GetLetter(first) == GetLetter(second);
    }

    void TellTheCurrentNumberOfPatternOccurrencesInText(int32_t answer) {
        assert(answers_[current_round_] == answer);
        ++current_round_;
        questions_left_ = questions_per_round;
    }

private:
    char GetLetter(io::OnlineAgent::SymbolPosition position) {
        return position.where == io::OnlineAgent::SymbolPosition::Where::Pattern
                   ? pattern_[position.index]
                   : text_[position.index];
    }

    std::mt19937 &twister_ = *rng::GetEngine();
    std::string pattern_;
    std::string text_;
    std::vector<int32_t> answers_;
    int32_t questions_left_;
    int32_t current_round_;
};

TestIo GenerateRandomTestIo(int32_t test_case_id) {

    throw NotImplementedError{};
    io::Input input;
    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    throw NotImplementedError{};
    io::Input input;
    return TestIo{input};
}

class TimedChecker {
public:
    std::vector<int64_t> durations;

    void TimedSolve(const io::Input &input) {
        auto solve = [&input]() { Solve(input); };

        durations.emplace_back(detail::Timeit(solve));
    }

    void Check(TestIo test_io) {
        TimedSolve(test_io.input);
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

    std::cerr << "Basic tests OK:\n" << timed_check;

    int32_t n_random_test_cases = 50;

    try {

        for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {

            TestPlayer player{1 + test_case_id, 1 + test_case_id * test_case_id};
            InteractWithAgentToFindAllPatternOccurrences(player);
        }

        std::cerr << "Random tests OK:\n" << timed_check;
    } catch (const NotImplementedError &e) {
    }

    int32_t n_stress_test_cases = 0;

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
        Solve(io::Input{std::cin, std::cout});
    }

    return 0;
}
