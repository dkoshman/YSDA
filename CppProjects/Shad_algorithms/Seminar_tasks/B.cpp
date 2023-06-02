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
    std::vector<int32_t> numbers;
    int64_t min_interval_sum = 0;
    int64_t max_interval_sum = 0;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t n_numbers = 0;
        in >> n_numbers;
        numbers.reserve(n_numbers);

        for (int32_t i = 0; i < n_numbers; ++i) {
            int32_t number = 0;
            in >> number;
            numbers.push_back(number);
        }

        in >> min_interval_sum >> max_interval_sum;
    }
};

class Output {
public:
    int64_t number_of_sub_intervals_with_sum_in_given_interval = 0;

    Output() = default;

    explicit Output(int64_t n_intervals)
        : number_of_sub_intervals_with_sum_in_given_interval{n_intervals} {
    }

    std::ostream &Write(std::ostream &out) const {
        out << number_of_sub_intervals_with_sum_in_given_interval;
        return out;
    }

    bool operator!=(const Output &other) const {
        return number_of_sub_intervals_with_sum_in_given_interval !=
               other.number_of_sub_intervals_with_sum_in_given_interval;
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

class MeetInTheMiddlePairsCounter {
public:
    explicit MeetInTheMiddlePairsCounter(std::vector<int64_t> numbers)
        : numbers_{std::move(numbers)} {
    }

    int64_t CountOrderedPairsWithDifferenceGreaterOrEqual(int64_t difference) {
        difference_ = difference;
        numbers_copy_ = numbers_copy_two_ = numbers_;
        return CountPairsOfElementsWithDifferenceGreaterOrEqualAndSort(
            numbers_copy_.begin(), numbers_copy_.end(), numbers_copy_two_.begin());
    }

private:
    std::vector<int64_t> numbers_;
    std::vector<int64_t> numbers_copy_;
    std::vector<int64_t> numbers_copy_two_;
    int64_t difference_ = 0;

    template <class Iterator>
    int64_t CountPairsWithDifferenceGreaterOrEqualDividedByTheMiddleOfIntervalWithSortedHalves(
        Iterator begin, Iterator mid, Iterator end) {

        auto left = begin;
        auto right = mid;
        int64_t count = 0;

        while (left < mid and right < end) {
            if (*right - *left >= difference_) {
                count += end - right;
                ++left;
            } else {
                ++right;
            }
        }

        return count;
    }

    template <class Iterator>
    void MergeSortedHalves(Iterator begin, Iterator mid, Iterator end,
                           Iterator sorted_buffer_begin) {

        auto left = begin;
        auto right = mid;

        while (left != mid or right != end) {
            if (right == end or (left != mid and *left < *right)) {
                *sorted_buffer_begin = *left;
                ++left;
            } else {
                *sorted_buffer_begin = *right;
                ++right;
            }
            ++sorted_buffer_begin;
        }
    }

    template <class Iterator>
    int64_t CountPairsOfElementsWithDifferenceGreaterOrEqualAndSort(Iterator begin, Iterator end,
                                                                    Iterator sorted_buffer_begin) {

        if (end - begin <= 1) {
            return 0;
        }

        auto mid = begin + (end - begin) / 2;
        auto sorted_buffer_mid = sorted_buffer_begin + (mid - begin);
        auto sorted_buffer_end = sorted_buffer_begin + (end - begin);

        auto pairs_to_the_left = CountPairsOfElementsWithDifferenceGreaterOrEqualAndSort(
            sorted_buffer_begin, sorted_buffer_mid, begin);

        auto pairs_to_the_right = CountPairsOfElementsWithDifferenceGreaterOrEqualAndSort(
            sorted_buffer_mid, sorted_buffer_end, mid);

        auto pairs_divided_by_mid =
            CountPairsWithDifferenceGreaterOrEqualDividedByTheMiddleOfIntervalWithSortedHalves(
                begin, mid, end);

        MergeSortedHalves(begin, mid, end, sorted_buffer_begin);

        return pairs_to_the_left + pairs_divided_by_mid + pairs_to_the_right;
    }
};

std::vector<int64_t> ComputePartialSums(const std::vector<int32_t> &numbers) {
    std::vector<int64_t> partial_sums;
    partial_sums.reserve(numbers.size() + 1);
    partial_sums.push_back(0);

    for (auto number : numbers) {
        partial_sums.push_back(partial_sums.back() + number);
    }

    return partial_sums;
}

io::Output Solve(const io::Input &input) {
    auto partial_sums = ComputePartialSums(input.numbers);

    MeetInTheMiddlePairsCounter counter{std::move(partial_sums)};

    return io::Output{
        counter.CountOrderedPairsWithDifferenceGreaterOrEqual(input.min_interval_sum) -
        counter.CountOrderedPairsWithDifferenceGreaterOrEqual(input.max_interval_sum + 1)};
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
        throw std::invalid_argument{"IsEmpty container."};
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

struct Interval {
    int64_t from = 0;
    int64_t to = 0;
};

int64_t CountPairsThroughTheMiddle(std::vector<int64_t>::iterator begin, size_t size,
                                   int64_t difference) {

    auto mid = begin + size / 2;
    auto end = begin + size;
    auto left = begin;
    auto right = mid;
    int64_t count = 0;

    while (left < mid and right < end) {
        if (*right - *left >= difference) {
            count += end - right;
            ++left;
        } else {
            ++right;
        }
    }

    return count;
}

void MergeSortedHalves(std::vector<int64_t>::iterator begin, size_t size,
                       std::vector<int64_t>::iterator output) {

    auto mid = begin + size / 2;
    auto end = begin + size;
    auto left = begin;
    auto right = mid;

    while (left != mid or right != end) {
        if (right == end or (left != mid and *left < *right)) {
            *output = *left;
            ++left;
        } else {
            *output = *right;
            ++right;
        }
        ++output;
    }
}

int64_t CountPairsOfElementsWithDifferenceGreaterThanGivenAndSort(
    std::vector<int64_t>::iterator begin, size_t size, Interval &interval,
    std::vector<int64_t>::iterator output) {

    if (size <= 1) {
        return 0;
    }

    auto half = size / 2;
    auto pairs_to_the_left =
        CountPairsOfElementsWithDifferenceGreaterThanGivenAndSort(output, half, interval, begin);
    auto pairs_to_the_right = CountPairsOfElementsWithDifferenceGreaterThanGivenAndSort(
        output + half, size - half, interval, begin + half);
    auto pairs_divided_by_mid = CountPairsThroughTheMiddle(begin, size, interval.from) -
                                CountPairsThroughTheMiddle(begin, size, interval.to + 1);

    MergeSortedHalves(begin, size, output);

    return pairs_to_the_left + pairs_divided_by_mid + pairs_to_the_right;
}

io::Output BruteForceSolve(const io::Input &input) {
    std::vector<int64_t> partial_sums;
    partial_sums.reserve(input.numbers.size() + 1);
    partial_sums.push_back(0);
    for (auto element : input.numbers) {
        partial_sums.push_back(partial_sums.back() + element);
    }
    auto partial_sums_sorted = partial_sums;

    Interval interval{input.min_interval_sum, input.max_interval_sum};
    auto answer = CountPairsOfElementsWithDifferenceGreaterThanGivenAndSort(
        partial_sums.begin(), partial_sums.size(), interval, partial_sums_sorted.begin());

    return io::Output{answer};
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {

    int32_t n_numbers = test_case_id + 1;
    int32_t max_abs_number_value = test_case_id + 1;
    std::uniform_int_distribution<int32_t> number_dist{-max_abs_number_value, max_abs_number_value};

    io::Input input;
    input.min_interval_sum = number_dist(*rng::GetEngine());
    input.max_interval_sum = number_dist(*rng::GetEngine());

    if (input.min_interval_sum > input.max_interval_sum) {
        std::swap(input.min_interval_sum, input.max_interval_sum);
    }

    input.numbers.reserve(n_numbers);
    for (int32_t i = 0; i < n_numbers; ++i) {
        input.numbers.push_back(number_dist(*rng::GetEngine()));
    }

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

    void Check(const std::string &test_case, int64_t expected) {
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
        "6\n"
        "-5 2 2 7 3 2\n"
        "4 6\n",
        3);

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
