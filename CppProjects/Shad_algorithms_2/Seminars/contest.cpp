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
#include <utility>
#include <vector>

namespace meta {

namespace enable_if_implementation {

template <bool B, class T = void>
struct enable_if {};

template <class T>
struct enable_if<true, T> {
    typedef T type;
};

}  // namespace enable_if_implementation

template <bool B, class T = void>
using enable_if_t = typename enable_if_implementation::enable_if<B, T>::type;

namespace is_base_of_implementation {

template <typename Base, typename Derived, typename Check = void>
struct meta_best_matching_template_specialization;

template <typename Base, typename Derived>
struct meta_best_matching_template_specialization<
    Base, Derived, meta::enable_if_t<not std::is_class_v<Base> or not std::is_class_v<Derived>>> {

    using is_base = std::false_type;
};

template <typename Base, typename Derived>
struct meta_best_matching_template_specialization<
    Base, Derived, meta::enable_if_t<std::is_class_v<Base> and std::is_class_v<Derived>>> {

    static constexpr std::false_type best_matching_function_overload(...);

    static constexpr std::true_type best_matching_function_overload(Base *base_ptr);

    using is_base = decltype(best_matching_function_overload(static_cast<Base *>(nullptr)));
};

}  // namespace is_base_of_implementation

template <typename Base, typename Derived>
inline constexpr bool is_base_of_v =
    is_base_of_implementation::meta_best_matching_template_specialization<Base,
                                                                          Derived>::is_base::value;

namespace is_template_instance_implementation {

template <template <typename...> typename Template, typename TemplateInstance>
struct meta_best_matching_template_specialization : public std::false_type {};

template <template <typename...> typename Template, typename... TemplateArgs>
struct meta_best_matching_template_specialization<Template, Template<TemplateArgs...>>
    : public std::true_type {};

}  // namespace is_template_instance_implementation

template <template <typename...> typename Template, typename TemplateInstance>
inline constexpr bool is_template_instance_of =
    is_template_instance_implementation::meta_best_matching_template_specialization<
        Template, TemplateInstance>{};

namespace is_template_base_implementation {

template <template <typename...> typename Template, typename DerivedFromTemplateInstance,
          typename TCheck = void>
struct meta_best_matching_template_specialization;

template <template <typename...> typename Template, typename DerivedFromTemplateInstance>
struct meta_best_matching_template_specialization<
    Template, DerivedFromTemplateInstance,
    meta::enable_if_t<not std::is_class_v<DerivedFromTemplateInstance>>> {

    using is_base = std::false_type;
};

template <template <typename...> typename Template, typename DerivedFromTemplateInstance>
struct meta_best_matching_template_specialization<
    Template, DerivedFromTemplateInstance,
    meta::enable_if_t<std::is_class_v<DerivedFromTemplateInstance>>> : DerivedFromTemplateInstance {

    template <typename... TemplateArgs>
    static constexpr std::true_type is_base_test(Template<TemplateArgs...> *);

    static constexpr std::false_type is_base_test(...);

    using is_base = decltype(is_base_test(static_cast<DerivedFromTemplateInstance *>(nullptr)));
};

}  // namespace is_template_base_implementation

template <template <typename...> typename Template, typename DerivedFromTemplateInstance>
inline constexpr bool is_template_base_of =
    is_template_base_implementation::meta_best_matching_template_specialization<
        Template, DerivedFromTemplateInstance>::is_base::value;

}  // namespace meta

namespace utils {

template <class Dividend, class Divisor>
Divisor NonNegativeMod(Dividend value, Divisor divisor) {
    if (divisor == 0) {
        throw std::invalid_argument("Zero divisor.");
    }
    if (divisor < 0) {
        throw std::invalid_argument("Negative divisor.");
    }
    auto mod = value % divisor;
    if (mod < 0) {
        mod += divisor;
    }
    return static_cast<Divisor>(mod);
}

template <typename Integral>
static Integral LargestPowerOfTwoNotGreaterThan(Integral value) {
    if (value <= 0) {
        throw std::invalid_argument{"Non positive logarithm argument."};
    }
    Integral log_two = 0;
    while (value >>= 1) {
        ++log_two;
    }
    return log_two;
}

template <typename Number>
Number Squared(Number value) {
    return value * value;
}

template <typename Container, typename Comparator>
class RangeMinQueryResponder {
public:
    using Value = typename Container::value_type;

    explicit RangeMinQueryResponder(const Container &container, const Comparator &comparator)
        : container_{container}, comparator_{comparator} {
    }

    void Preprocess() {
        if (not rmq_preprocessed_.empty()) {
            return;
        }

        auto log_two = LargestPowerOfTwoNotGreaterThan(container_.size());

        rmq_preprocessed_.resize(log_two + 1);
        rmq_preprocessed_.front() = container_;

        for (size_t interval_size_log_two = 1; interval_size_log_two <= log_two;
             ++interval_size_log_two) {

            auto &rmq_preprocessed_by_interval_size = rmq_preprocessed_[interval_size_log_two];
            auto interval_size = 1 << interval_size_log_two;
            auto n_valid_begin_positions = container_.size() - interval_size + 1;
            rmq_preprocessed_by_interval_size.resize(n_valid_begin_positions);

            for (size_t begin = 0; begin < n_valid_begin_positions; ++begin) {

                auto &left_half_min = MinOnPreprocessedIntervalByEnd(begin + interval_size / 2,
                                                                     interval_size_log_two - 1);
                auto &right_half_min = MinOnPreprocessedIntervalByBegin(begin + interval_size / 2,
                                                                        interval_size_log_two - 1);

                MinOnPreprocessedIntervalByBegin(begin, interval_size_log_two) =
                    std::min(left_half_min, right_half_min, comparator_);
            }
        }
    }

    Value GetRangeMin(size_t begin, size_t end) {
        Preprocess();

        auto log_two = LargestPowerOfTwoNotGreaterThan(end - begin);

        auto min_on_left_overlapping_half = MinOnPreprocessedIntervalByBegin(begin, log_two);
        auto min_on_right_overlapping_half = MinOnPreprocessedIntervalByEnd(end, log_two);

        return std::min(min_on_left_overlapping_half, min_on_right_overlapping_half, comparator_);
    }

private:
    const Container &container_;
    const Comparator &comparator_;
    std::vector<std::vector<Value>> rmq_preprocessed_;

    template <typename Integral>
    static Integral LargestPowerOfTwoNotGreaterThan(Integral value) {
        Integral log_two = 0;
        while (value >>= 1) {
            ++log_two;
        }
        return log_two;
    }

    Value &MinOnPreprocessedIntervalByBegin(size_t begin, size_t interval_size_log_two) {
        return rmq_preprocessed_[interval_size_log_two][begin];
    }

    Value &MinOnPreprocessedIntervalByEnd(size_t end, size_t interval_size_log_two) {
        return rmq_preprocessed_[interval_size_log_two][end - (1 << interval_size_log_two)];
    }
};

template <class Key, class Value>
std::vector<Value> GetMapValues(const std::map<Key, Value> &map) {
    std::vector<Value> values;
    values.reserve(map.size());
    for (auto &pair : map) {
        values.emplace_back(pair.second);
    }
    return values;
}

template <typename Iterator, typename Comparator>
std::vector<size_t> ArgSort(Iterator begin, Iterator end) {
    Comparator comparator;
    std::vector<size_t> indices(end - begin);
    std::iota(indices.begin(), indices.end(), 0);
    auto arg_comparator = [&begin, &comparator](size_t left, size_t right) -> bool {
        return comparator(*(begin + left), *(begin + right));
    };
    std::sort(indices.begin(), indices.end(), arg_comparator);
    return indices;
}

struct ComparatorLess {
    template <typename T>
    bool operator()(const T &left, const T &right) const {
        return left < right;
    }
};

template <typename Iterator>
std::vector<size_t> ArgSort(Iterator begin, Iterator end) {
    return ArgSort<Iterator, ComparatorLess>(begin, end);
}

int32_t GetLeastSignificantBitPosition(int32_t x) {
    return (x & ~(x - 1));
}

}  // namespace utils

namespace io {

class Input {
public:
    Input() = default;

    explicit Input(std::istream &in);
};

class Output {
public:
    std::vector<int32_t> number_of_target_matrix_occurrences_up_to_one_error_;

    Output() = default;

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        int32_t item = 0;
        while (ss >> item) {
            smallest_super_string.emplace_back(item);
        }
    }

    std::ostream &Write(std::ostream &out) const {
        for (auto item : smallest_super_string) {
            out << item << '\n';
        }
        return out;
    }

    bool operator!=(const Output &other) const {
        return smallest_super_string != other.smallest_super_string;
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

io::Output Solve(const io::Input &input);

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
    throw NotImplementedError{};
}

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
