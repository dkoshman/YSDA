#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace io {

class Input {
public:
    static const char kWildcardChar = '?';
    std::string pattern_with_wildcards;
    std::string text;

    Input() = default;

    explicit Input(std::istream &in) {
        in >> pattern_with_wildcards >> text;
    }
};

class Output {
public:
    std::vector<int32_t> pattern_with_wildcards_occurrences_begin_in_text;

    Output() = default;

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        int32_t item = 0;
        while (ss >> item) {
            pattern_with_wildcards_occurrences_begin_in_text.emplace_back(item);
        }
    }

    std::ostream &Write(std::ostream &os) const {
        std::copy(pattern_with_wildcards_occurrences_begin_in_text.begin(),
                  pattern_with_wildcards_occurrences_begin_in_text.end(),
                  std::ostream_iterator<int32_t>(os, " "));
        return os;
    }

    bool operator!=(const Output &other) const {
        return pattern_with_wildcards_occurrences_begin_in_text !=
               other.pattern_with_wildcards_occurrences_begin_in_text;
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

bool IsEqual(char first, char second) {
    return first == second or first == io::Input::kWildcardChar or
           second == io::Input::kWildcardChar;
}

io::Output Solve(const io::Input &input);

struct Node;
using NodePointer = std::unique_ptr<Node>;
using NodeRefererence = Node &;

struct Edge {
    NodePointer next;
    char letter;
};

struct Node {
    Edge edge;
    NodePointer prefix_link;
    NodePointer terminal_link;
    bool is_terminal;
};

class Automaton {
public:
    NodeRefererence Root() const {
        return *root_;
    }

private:
    NodePointer root_;
};

struct IndexedWord {
    IndexedWord(std::string &&word, size_t id) : word{std::move(word)}, id{id} {
    }

    std::string word;
    size_t id;
};

class Builder {
public:
    void Add(std::string word, size_t id) {
        words_.emplace_back(std::move(word), id);
    }

private:
    std::vector<IndexedWord> words_;
};

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

io::Output BruteForceSolve(const io::Input &input) {
    auto pattern_size = static_cast<int32_t>(input.pattern_with_wildcards.size());
    auto text_size = static_cast<int32_t>(input.text.size());
    io::Output output;

    for (int32_t text_index = 0; text_index < text_size - pattern_size + 1; ++text_index) {
        int32_t match = 0;
        while (match < pattern_size and
               IsEqual(input.pattern_with_wildcards[match], input.text[text_index + match])) {
            ++match;
        }
        if (match == pattern_size) {
            output.pattern_with_wildcards_occurrences_begin_in_text.emplace_back(text_index);
        }
    }
    return output;
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    auto pattern_size = 1 + static_cast<int32_t>(std::log2(1 + test_case_id));
    auto text_size = 1 + test_case_id;
    auto wildcard_count = std::max(10, pattern_size / 3);

    std::uniform_int_distribution<char> text_dist{'a', 'z'};
    std::stringstream ss;
    std::unordered_set<int32_t> wildcards;
    std::uniform_int_distribution<int32_t> wildcard_dist{0, pattern_size - 1};
    io::Input input;

    for (int32_t i = 0; i < wildcard_count; ++i) {
        wildcards.insert(wildcard_dist(*rng::GetEngine()));
    }
    for (int32_t i = 0; i < pattern_size; ++i) {
        auto c = text_dist(*rng::GetEngine());
        if (wildcards.count(i)) {
            c = io::Input::kWildcardChar;
        }
        ss << c;
    }
    input.pattern_with_wildcards = ss.str();
    ss.str("");
    for (int32_t i = 0; i < text_size; ++i) {
        ss << text_dist(*rng::GetEngine());
    }
    input.text = ss.str();

    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    return GenerateRandomTestIo(100'000);
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

    TimedChecker timed_checker;

    timed_checker.Check(
        "ab?\n"
        "ababcabc\n",
        "0 2 5");
    timed_checker.Check(
        "???\n"
        "ababcabc\n",
        "0 1 2 3 4 5");
    timed_checker.Check(
        "z\n"
        "ababcabc\n",
        "");
    timed_checker.Check(
        "z\n"
        "zzzzz\n",
        "0 1 2 3 4");
    timed_checker.Check(
        "?z?\n"
        "zzzzz\n",
        "0 1 2");
    timed_checker.Check(
        "z?z\n"
        "zzzzz\n",
        "0 1 2");
    timed_checker.Check(
        "z?z\n"
        "zazaz\n",
        "0 2");
    timed_checker.Check(
        "z?a\n"
        "azzazza\n",
        "1 4");

    std::cerr << "Basic tests OK:\n" << timed_checker;

    int32_t n_random_test_cases = 100;

    try {

        for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {
            timed_checker.Check(GenerateRandomTestIo(test_case_id));
        }

        std::cerr << "Random tests OK:\n" << timed_checker;
    } catch (const NotImplementedError &e) {
    }

    int32_t n_stress_test_cases = 1;

    try {
        for (int32_t test_case_id = 0; test_case_id < n_stress_test_cases; ++test_case_id) {
            timed_checker.Check(GenerateStressTestIo(test_case_id));
        }

        std::cerr << "Stress tests tests OK:\n" << timed_checker;
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
