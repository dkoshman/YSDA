// https://contest.yandex.ru/contest/29062/problems/

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

struct Treasure {
    int32_t weight = 0;
    int32_t cost = 0;
    int32_t amount = 0;
};

std::istream &operator>>(std::istream &in, Treasure &treasure) {
    in >> treasure.weight >> treasure.cost >> treasure.amount;
    return in;
}

class Input {
public:
    int32_t backpack_capacity = 0;
    std::vector<Treasure> treasures;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t treasure_types = 0;
        in >> treasure_types >> backpack_capacity;

        treasures.resize(treasure_types);

        for (auto &treasure : treasures) {
            in >> treasure;
        }
    }
};

class Output {
public:
    int64_t maximum_backpack_value = 0;

    Output() = default;

    explicit Output(int64_t maximum_backpack_value)
        : maximum_backpack_value{maximum_backpack_value} {
    }

    std::ostream &Write(std::ostream &out) const {
        out << maximum_backpack_value;
        return out;
    }

    bool operator!=(const Output &other) const {
        return maximum_backpack_value != other.maximum_backpack_value;
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

template <typename Value>
class QueueWithExtremum {
public:
    using Comparator = std::function<bool(const Value &, const Value &)>;

    explicit QueueWithExtremum(Comparator comparator) : comparator_{std::move(comparator)} {
    }

    virtual ~QueueWithExtremum() = default;

    virtual void Clear() {
        sorted_deque_.clear();
    }

    [[nodiscard]] virtual Value Extremum() {
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

template <typename Value>
class QueueWithExtremumAndMonotonouslyGrowingValues {
    struct ValueWithTicks {
        Value value;
        int32_t ticks_when_enqueued = 0;

        ValueWithTicks() = default;
    };

public:
    explicit QueueWithExtremumAndMonotonouslyGrowingValues(
        std::function<bool(const Value &, const Value &)> comparator)
        : base_queue_{
              [comparator](const ValueWithTicks &left, const ValueWithTicks &right) -> bool {
                  return comparator(left.value, right.value);
              }} {
    }

    void SetNewGrowingRateAndTicksValueLivesFor(Value new_growing_rate,
                                                int32_t ticks_value_live_for) {
        growing_rate_ = new_growing_rate;
        ticks_value_live_for_ = ticks_value_live_for;
        Clear();
    }

    void Clear() {
        base_queue_.Clear();
        current_ticks_ = 0;
    }

    void IncrementTick() {
        ++current_ticks_;
    }

    [[nodiscard]] Value Extremum() {
        while (current_ticks_ - base_queue_.Extremum().ticks_when_enqueued >
               ticks_value_live_for_) {
            base_queue_.Dequeue(base_queue_.Extremum());
        }
        return AdjustedValueToGrown(base_queue_.Extremum().value);
    }

    void Enqueue(Value value) {
        base_queue_.Enqueue({GrownValueToAdjusted(value), current_ticks_});
    }

private:
    QueueWithExtremum<ValueWithTicks> base_queue_;
    Value growing_rate_ = 0;
    Value ticks_value_live_for_ = 0;
    int32_t current_ticks_ = 0;

    [[nodiscard]] Value GrownValueToAdjusted(Value value) const {
        return value - growing_rate_ * current_ticks_;
    }

    [[nodiscard]] Value AdjustedValueToGrown(Value value) const {
        return value + growing_rate_ * current_ticks_;
    }
};

class DynamicBackpack {
public:
    explicit DynamicBackpack(int32_t backpack_capacity)
        : backpack_capacity_{backpack_capacity},
          best_backpack_value_by_capacity_(backpack_capacity + 1),
          new_best_backpack_value_by_capacity_(backpack_capacity + 1) {
    }

    [[nodiscard]] int64_t GetBestBackpackValueForCurrentChoiceOfTreasuresByBackpackSize(
        int32_t backpack_size) const {
        return best_backpack_value_by_capacity_[backpack_size];
    }

    void AddTreasureChoice(io::Treasure treasure) {

        growing_queue_.SetNewGrowingRateAndTicksValueLivesFor(treasure.cost, treasure.amount);

        for (int64_t backpack_size_not_filled_with_this_treasure = 0;
             backpack_size_not_filled_with_this_treasure <=
             std::min(treasure.weight - 1, backpack_capacity_);
             ++backpack_size_not_filled_with_this_treasure) {

            growing_queue_.Clear();

            for (int64_t backpack_size = backpack_size_not_filled_with_this_treasure;
                 backpack_size <= backpack_capacity_; backpack_size += treasure.weight) {

                growing_queue_.Enqueue(best_backpack_value_by_capacity_[backpack_size]);

                new_best_backpack_value_by_capacity_[backpack_size] = growing_queue_.Extremum();

                growing_queue_.IncrementTick();
            }
        }

        std::swap(best_backpack_value_by_capacity_, new_best_backpack_value_by_capacity_);
    }

private:
    int32_t backpack_capacity_ = 0;
    std::vector<int64_t> best_backpack_value_by_capacity_;
    std::vector<int64_t> new_best_backpack_value_by_capacity_;
    QueueWithExtremumAndMonotonouslyGrowingValues<int64_t> growing_queue_{std::greater<>{}};
};

io::Output Solve(const io::Input &input) {

    DynamicBackpack backpack{input.backpack_capacity};

    for (auto &treasure : input.treasures) {
        backpack.AddTreasureChoice(treasure);
    }

    return io::Output{backpack.GetBestBackpackValueForCurrentChoiceOfTreasuresByBackpackSize(
        input.backpack_capacity)};
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

io::Output BruteForceSolve(const io::Input &input) {

    auto padded_size = input.backpack_capacity + 1;
    std::vector<int64_t> best_rucksack_with_current_number_of_treasure_types_of_given_size(
        padded_size);
    auto &dynamic_table = best_rucksack_with_current_number_of_treasure_types_of_given_size;
    std::vector<int64_t> new_table(padded_size);
    QueueWithExtremum<int64_t> queue{std::greater<>{}};

    for (auto &treasure : input.treasures) {
        for (int64_t remainder = 0; remainder < std::min(treasure.weight, padded_size);
             ++remainder) {

            queue.Clear();
            auto max_weight = static_cast<int64_t>(treasure.amount) * treasure.weight;
            auto max_cost = static_cast<int64_t>(treasure.amount) * treasure.cost;

            for (int64_t size = remainder, cost = 0; size < padded_size;
                 size += treasure.weight, cost += treasure.cost) {

                queue.Enqueue(dynamic_table[size] - cost);

                if (cost > max_cost) {
                    queue.Dequeue(dynamic_table[size - max_weight - treasure.weight] -
                                  (cost - max_cost - treasure.cost));
                }

                new_table[size] = queue.Extremum() + cost;
            }
        }
        std::swap(dynamic_table, new_table);
    }

    return io::Output{dynamic_table.back()};
}

int64_t FullSearch(const std::vector<io::Treasure> &trove, int64_t size, int32_t type = 0) {
    if (type >= static_cast<int64_t>(trove.size())) {
        return 0;
    }

    int64_t answer = 0;
    for (int64_t amount = 0; amount <= trove[type].amount; ++amount) {
        if (amount * trove[type].weight > size) {
            break;
        }
        answer =
            std::max(answer, amount * trove[type].cost +
                                 FullSearch(trove, size - amount * trove[type].weight, type + 1));
    }

    return answer;
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t n_treasure_types = 1 + test_case_id;
    int32_t backpack_size = test_case_id * 100;

    auto &engine = *rng::GetEngine();
    std::uniform_int_distribution<int32_t> distribution{1, n_treasure_types};

    io::Input input;
    input.backpack_capacity = backpack_size;
    input.treasures.resize(n_treasure_types);

    for (auto &treasure : input.treasures) {
        treasure.weight = distribution(engine);
        treasure.cost = distribution(engine);
        treasure.amount = distribution(engine);
    }

    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    return GenerateRandomTestIo(300);
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

    TimedChecker timed_check;

    timed_check.Check(
        "2 100\n"
        "2 1 100\n"
        "7 100 3\n",
        339);

    timed_check.Check(
        "2 100\n"
        "7 100 3\n"
        "2 1 100\n",
        339);

    timed_check.Check(
        "1 1\n"
        "2 1 100\n",
        0);

    timed_check.Check(
        "1 0\n"
        "2 1 100\n",
        0);

    timed_check.Check(
        "1 2\n"
        "2 1 100\n",
        1);

    timed_check.Check(
        "2 99999\n"
        "10 1000000000 1000000000\n"
        "2 1 2\n",
        9999L * 1000000000 + 2 * 1);

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
