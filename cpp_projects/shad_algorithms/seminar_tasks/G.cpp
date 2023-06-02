#include <algorithm>
#include <array>
#include <bitset>
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

namespace io {

struct Treasure {
    int64_t weight = 0;
    int64_t value = 0;

    Treasure() = default;

    Treasure(int64_t weight, int64_t value) : weight{weight}, value{value} {
    }
};

class Input {
public:
    static const int32_t kMaxNTreasures = 32;
    int64_t min_total_treasure_weight = 0;
    int64_t max_total_treasure_weight = 0;
    std::vector<Treasure> treasures;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t n_treasures = 0;
        in >> n_treasures >> min_total_treasure_weight >> max_total_treasure_weight;
        treasures.resize(n_treasures);
        for (auto &treasure : treasures) {
            in >> treasure.weight >> treasure.value;
        }
    }
};

class Output {
public:
    std::optional<int64_t> max_total_treasure_value;
    std::vector<int32_t> treasures_with_max_total_value_in_allowed_total_weight_range;

    Output() = default;

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        int32_t n_treasures = 0;
        ss >> n_treasures;
        int32_t item = 0;
        while (ss >> item) {
            treasures_with_max_total_value_in_allowed_total_weight_range.emplace_back(item - 1);
        }
    }

    std::ostream &Write(std::ostream &out) const {
        out << treasures_with_max_total_value_in_allowed_total_weight_range.size() << '\n';
        for (auto item : treasures_with_max_total_value_in_allowed_total_weight_range) {
            out << item + 1 << '\n';
        }
        return out;
    }

    bool operator!=(const Output &other) const {
        if (max_total_treasure_value and other.max_total_treasure_value) {
            return max_total_treasure_value.value() != other.max_total_treasure_value.value();
        } else {
            return treasures_with_max_total_value_in_allowed_total_weight_range !=
                   other.treasures_with_max_total_value_in_allowed_total_weight_range;
        }
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

struct TreasureSet {
    int64_t total_weight = 0;
    int64_t total_value = 0;
    int32_t treasure_set_id = 0;
};

bool ComparatorTreasureSetWeight(const TreasureSet &left, const TreasureSet &right) {
    return left.total_weight < right.total_weight;
}

bool ComparatorTreasureSetWeightGreater(const TreasureSet &left, const TreasureSet &right) {
    return left.total_weight > right.total_weight;
}

bool ComparatorTreasureSetValueGreater(const TreasureSet &left, const TreasureSet &right) {
    return left.total_value > right.total_value;
}

std::vector<TreasureSet> GenerateAllTreasureSets(const std::vector<io::Treasure> &treasures) {
    std::vector<TreasureSet> treasure_sets(1 << treasures.size());

    for (int32_t treasure_set_id = 0; treasure_set_id < static_cast<int32_t>(treasure_sets.size());
         ++treasure_set_id) {

        auto &treasure_set = treasure_sets[treasure_set_id];
        treasure_set.treasure_set_id = treasure_set_id;
        std::bitset<io::Input::kMaxNTreasures> bitset(treasure_set_id);

        for (size_t i = 0; i < treasures.size(); ++i) {
            if (bitset[i]) {
                treasure_set.total_weight += treasures[i].weight;
                treasure_set.total_value += treasures[i].value;
            }
        }
    }

    return treasure_sets;
}

template <typename Value>
class QueueWithExtremum {
public:
    using Comparator = std::function<bool(const Value &, const Value &)>;

    explicit QueueWithExtremum(Comparator comparator) : comparator_{std::move(comparator)} {
    }

    virtual ~QueueWithExtremum() = default;

    [[nodiscard]] bool IsEmpty() const {
        return sorted_deque_.empty();
    }

    virtual void Clear() {
        sorted_deque_.clear();
    }

    [[nodiscard]] virtual Value Extremum() const {
        return sorted_deque_.front();
    }

    virtual void Enqueue(Value value) {
        while (not sorted_deque_.empty() and comparator_(value, sorted_deque_.back())) {
            sorted_deque_.pop_back();
        }
        sorted_deque_.emplace_back(value);
    }

    virtual void Dequeue(Value value) {
        if (not comparator_(sorted_deque_.front(), value)) {
            sorted_deque_.pop_front();
        }
    }

protected:
    std::deque<Value> sorted_deque_;
    Comparator comparator_;
};

class IntervalWithValidCombinedTreasureWeightsAndMaxValue {
public:
    const int64_t min_total_treasure_weight = 0;
    const int64_t max_total_treasure_weight = 0;

    IntervalWithValidCombinedTreasureWeightsAndMaxValue(
        const std::vector<TreasureSet> &sorted_treasure_sets, int64_t min_total_treasure_weight,
        int64_t max_total_treasure_weight)
        : min_total_treasure_weight{min_total_treasure_weight},
          max_total_treasure_weight{max_total_treasure_weight},
          interval_begin_{sorted_treasure_sets.begin()},
          interval_end_{sorted_treasure_sets.begin()},
          treasure_sets_end_{sorted_treasure_sets.end()} {

        if (not std::is_sorted(sorted_treasure_sets.begin(), sorted_treasure_sets.end(),
                               ComparatorTreasureSetWeight)) {
            throw std::invalid_argument{"Treasure sets not sorted."};
        }
    }

    void Update(TreasureSet treasure_set) {
        if (previous_treasure_set_ and
            previous_treasure_set_->total_weight < treasure_set.total_weight) {
            throw std::invalid_argument{
                "Treasure sets must be iterated in descending weight order."};
        }

        UpdateIntervalEnd(treasure_set);
        UpdateIntervalBegin(treasure_set);

        previous_treasure_set_ = treasure_set;
    }

    [[nodiscard]] std::optional<TreasureSet> GetTreasureSetWithValidWeightAndMaxValue() const {
        if (interval_max_value_queue_.IsEmpty()) {
            return std::nullopt;
        } else {
            return interval_max_value_queue_.Extremum();
        }
    }

private:
    std::vector<TreasureSet>::const_iterator interval_begin_;
    std::vector<TreasureSet>::const_iterator interval_end_;
    std::vector<TreasureSet>::const_iterator treasure_sets_end_;
    QueueWithExtremum<TreasureSet> interval_max_value_queue_{ComparatorTreasureSetValueGreater};
    std::optional<TreasureSet> previous_treasure_set_;

    void UpdateIntervalEnd(TreasureSet treasure_set) {
        while (interval_end_ < treasure_sets_end_ and
               treasure_set.total_weight + interval_end_->total_weight <=
                   max_total_treasure_weight) {

            interval_max_value_queue_.Enqueue(*interval_end_);
            ++interval_end_;
        }
    }

    void UpdateIntervalBegin(TreasureSet treasure_set) {
        while (interval_begin_ < interval_end_ and
               treasure_set.total_weight + interval_begin_->total_weight <
                   min_total_treasure_weight) {

            interval_max_value_queue_.Dequeue(*interval_begin_);
            ++interval_begin_;
        }
    }
};

template <typename T>
std::pair<std::vector<T>, std::vector<T>> SplitInHalf(const std::vector<T> &vector) {
    auto half = static_cast<int32_t>(vector.size() / 2);
    return {{vector.begin(), vector.begin() + half}, {vector.begin() + half, vector.end()}};
}

io::Output Solve(const io::Input &input) {

    auto [left_treasures, right_treasures] = SplitInHalf(input.treasures);

    auto left_sets_by_decreasing_weight = GenerateAllTreasureSets(left_treasures);
    std::sort(left_sets_by_decreasing_weight.begin(), left_sets_by_decreasing_weight.end(),
              ComparatorTreasureSetWeightGreater);

    auto right_sets_by_increasing_weight = GenerateAllTreasureSets(right_treasures);
    std::sort(right_sets_by_increasing_weight.begin(), right_sets_by_increasing_weight.end(),
              ComparatorTreasureSetWeight);

    IntervalWithValidCombinedTreasureWeightsAndMaxValue interval{right_sets_by_increasing_weight,
                                                                 input.min_total_treasure_weight,
                                                                 input.max_total_treasure_weight};

    int64_t max_total_treasure_value = 0;
    int64_t max_left_set_id = 0;
    int64_t max_right_set_id = 0;

    for (auto left_treasure_set : left_sets_by_decreasing_weight) {

        interval.Update(left_treasure_set);
        auto right_max_treasure_set = interval.GetTreasureSetWithValidWeightAndMaxValue();

        if (right_max_treasure_set) {
            auto combined_sets_value =
                left_treasure_set.total_value + right_max_treasure_set->total_value;

            if (combined_sets_value > max_total_treasure_value) {
                max_total_treasure_value = combined_sets_value;
                max_left_set_id = left_treasure_set.treasure_set_id;
                max_right_set_id = right_max_treasure_set->treasure_set_id;
            }
        }
    }

    io::Output output;
    output.max_total_treasure_value = max_total_treasure_value;

    std::bitset<io::Input::kMaxNTreasures> bitset(max_left_set_id);
    for (size_t i = 0; i < left_treasures.size(); ++i) {
        if (bitset[i]) {
            output.treasures_with_max_total_value_in_allowed_total_weight_range.emplace_back(i);
        }
    }

    bitset = max_right_set_id;
    for (size_t i = 0; i < right_treasures.size(); ++i) {
        if (bitset[i]) {
            output.treasures_with_max_total_value_in_allowed_total_weight_range.emplace_back(
                i + left_treasures.size());
        }
    }

    return output;
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

io::Output BruteForceSolve(const io::Input &input) {

    if (input.treasures.size() > 10) {
        throw NotImplementedError{};
    }

    int64_t max_value = 0;
    int64_t max_treasure_set_id = 0;

    for (int64_t treasure_set_id = 0; treasure_set_id < 1l << input.treasures.size();
         ++treasure_set_id) {
        io::Treasure max_total_treasure;
        std::bitset<io::Input::kMaxNTreasures> bitset(treasure_set_id);

        for (size_t i = 0; i < input.treasures.size(); ++i) {
            if (bitset[i]) {
                max_total_treasure.value += input.treasures[i].value;
                max_total_treasure.weight += input.treasures[i].weight;
            }
        }

        if (input.min_total_treasure_weight <= max_total_treasure.weight and
            max_total_treasure.weight <= input.max_total_treasure_weight and
            max_total_treasure.value > max_value) {
            max_value = max_total_treasure.value;
            max_treasure_set_id = treasure_set_id;
        }
    }

    io::Output output;
    output.max_total_treasure_value = max_value;
    std::bitset<io::Input::kMaxNTreasures> bitset(max_treasure_set_id);
    for (size_t i = 0; i < input.treasures.size(); ++i) {
        if (bitset[i]) {
            output.treasures_with_max_total_value_in_allowed_total_weight_range.emplace_back(i);
        }
    }
    return output;
}

TestIo GenerateRandomTestIo(int64_t test_case_id) {
    auto n_treasures = std::min(32, 1 + static_cast<int32_t>(test_case_id) / 10);
    auto max_weight_value = std::min<int64_t>(1 + test_case_id, 1e15);
    int64_t min_total_weight_value = max_weight_value;
    int64_t max_total_weight_value =
        std::min<int64_t>(1 + max_weight_value * n_treasures / 2, 1e18);

    std::uniform_int_distribution<int64_t> weight_value_distribution{1, max_weight_value};
    auto &engine = *rng::GetEngine();

    io::Input input;

    input.min_total_treasure_weight = min_total_weight_value;
    input.max_total_treasure_weight = max_total_weight_value;

    while (static_cast<int32_t>(input.treasures.size()) < n_treasures) {
        input.treasures.emplace_back(weight_value_distribution(engine),
                                     weight_value_distribution(engine));
    }

    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    return GenerateRandomTestIo(1'000);
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

    TimedChecker timed_checker;

    timed_checker.Check(
        "3 6 8\n"
        "3 10\n"
        "7 3\n"
        "8 2",
        "1\n"
        "2");

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
