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

int32_t GetValueWithOnlyLeastSignificantBit(int32_t x) {
    if (x == 0) {
        throw std::invalid_argument{"Zero does not have a least significant bit."};
    }
    return (x & ~(x - 1));
}

}  // namespace utils

namespace io {

class Input {
public:
    int32_t n_players = 0;
    int32_t josephs_counting_rhyme = 0;

    Input() = default;

    explicit Input(std::istream &in) {
        in >> n_players >> josephs_counting_rhyme;
    }
};

class Output {
public:
    std::vector<int32_t> are_nodes_connected;

    Output() = default;

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        int32_t item = 0;
        while (ss >> item) {
            players_drop_out_order.emplace_back(item - 1);
        }
    }

    std::ostream &Write(std::ostream &out) const {
        for (auto item : players_drop_out_order) {
            out << item + 1 << '\n';
        }
        return out;
    }

    bool operator!=(const Output &other) const {
        return players_drop_out_order != other.players_drop_out_order;
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

class RhymePlayersDropOutOrderByNPlayersOnTheLeft {
public:
    RhymePlayersDropOutOrderByNPlayersOnTheLeft(int32_t n_players, int32_t counting_rhyme)
        : n_players_{n_players}, counting_rhyme_{counting_rhyme} {
    }

    struct Iterator {

        Iterator(int32_t n_players, int32_t counting_rhyme)
            : counting_rhyme_{counting_rhyme}, n_players_in_game_{n_players} {
            this->operator++();
        }

        int32_t operator*() const {
            return n_players_on_the_left_including_self_;
        }

        Iterator &operator++() {
            if (n_players_in_game_ > 0) {
                auto n_players_on_the_right =
                    n_players_in_game_ - n_players_on_the_left_including_self_;

                auto right_shift = (counting_rhyme_ - 1) % n_players_in_game_;

                n_players_on_the_left_including_self_ =
                    right_shift > n_players_on_the_right
                        ? right_shift - n_players_on_the_right
                        : n_players_on_the_left_including_self_ + right_shift;
            }
            --n_players_in_game_;
            return *this;
        }

        friend bool operator==(const Iterator &left, const Iterator &right) {
            return left.IsExhausted() and right.IsExhausted();
        }

        friend bool operator!=(const Iterator &left, const Iterator &right) {
            return not operator==(left, right);
        }

    private:
        int32_t counting_rhyme_ = 0;
        int32_t n_players_on_the_left_including_self_ = 1;
        int32_t n_players_in_game_ = 0;

        [[nodiscard]] bool IsExhausted() const {
            return n_players_in_game_ == -1;
        }
    };

    Iterator begin() {
        return Iterator{n_players_, counting_rhyme_};
    }

    Iterator end() {
        return Iterator{0, 0};
    }

private:
    int32_t n_players_ = 0;
    int32_t counting_rhyme_ = 0;
};

namespace sqrt_decomposition {

class SqrtDecomposition {
public:
    explicit SqrtDecomposition(int32_t n_players)
        : is_player_in_game_(n_players, true),
          sqrt_{static_cast<int32_t>(std::sqrt(n_players))},
          sqrt_block_sums_(sqrt_, sqrt_) {
        sqrt_block_sums_.emplace_back(n_players - sqrt_ * sqrt_);
    }

    [[nodiscard]] int32_t FindPositionAtWhichPrefixSumIsEqual(int32_t prefix_sum) const {

        int32_t sqrt_block = 0;
        while (prefix_sum > sqrt_block_sums_[sqrt_block]) {
            prefix_sum -= sqrt_block_sums_[sqrt_block];
            ++sqrt_block;
        }

        auto player_id = sqrt_ * sqrt_block;
        for (; prefix_sum > 0; ++player_id) {
            if (is_player_in_game_[player_id]) {
                --prefix_sum;
            }
        }

        return player_id - 1;
    }

    void DropPlayer(int32_t player_id) {
        if (not is_player_in_game_[player_id]) {
            throw std::invalid_argument{"Player already dropped."};
        }
        is_player_in_game_[player_id] = false;
        --sqrt_block_sums_[player_id / sqrt_];
    }

private:
    std::vector<bool> is_player_in_game_;
    int32_t sqrt_ = 0;
    std::vector<int32_t> sqrt_block_sums_;
};

io::Output Solve(const io::Input &input) {
    SqrtDecomposition sqrt_decomposition{input.n_players};

    io::Output output;
    output.players_drop_out_order.reserve(input.n_players);

    for (auto n_players_on_the_left : RhymePlayersDropOutOrderByNPlayersOnTheLeft{
             input.n_players, input.josephs_counting_rhyme}) {
        auto player_id =
            sqrt_decomposition.FindPositionAtWhichPrefixSumIsEqual(n_players_on_the_left);

        sqrt_decomposition.DropPlayer(player_id);

        output.players_drop_out_order.emplace_back(player_id);
    }

    return output;
}

}  // namespace sqrt_decomposition

namespace fenwick {

[[nodiscard]] static int32_t GetParent(int32_t one_base_index) {
    return one_base_index - utils::GetValueWithOnlyLeastSignificantBit(one_base_index);
}

[[nodiscard]] static int32_t GetNextNodeWithSupersetResponsibilityRange(int32_t one_based_index) {
    return one_based_index + utils::GetValueWithOnlyLeastSignificantBit(one_based_index);
}

class ParentNodesIterable {
public:
    int32_t one_based_index = 0;

    ParentNodesIterable() = default;

    explicit ParentNodesIterable(int32_t zero_based_index) : one_based_index{zero_based_index + 1} {
    }

    struct Iterator {
        explicit Iterator(int32_t one_based_index) : one_based_index_{one_based_index} {
        }

        int32_t operator*() const {
            return one_based_index_;
        }

        Iterator &operator++() {
            one_based_index_ = GetParent(one_based_index_);
            return *this;
        }

        friend bool operator==(const Iterator &left, const Iterator &right) {
            return left.IsIteratorExhausted() and right.IsIteratorExhausted();
        }

        friend bool operator!=(const Iterator &left, const Iterator &right) {
            return not operator==(left, right);
        }

    private:
        int32_t one_based_index_ = 0;

        [[nodiscard]] bool IsIteratorExhausted() const {
            return one_based_index_ < 1;
        }
    };

    Iterator begin() {
        return Iterator{one_based_index};
    }

    Iterator end() {
        return Iterator{0};
    }
};

class NodesWithSupersetResponsibilityRangeIterable {
public:
    int32_t zero_based_index = 0;
    int32_t size = 0;

    explicit NodesWithSupersetResponsibilityRangeIterable(int32_t zero_based_index, size_t size)
        : zero_based_index{zero_based_index}, size{static_cast<int32_t>(size)} {
    }

    struct Iterator {
        explicit Iterator(int32_t zero_based_index, int32_t size)
            : one_based_index_{zero_based_index + 1}, size_{size} {
        }

        int32_t operator*() const {
            return one_based_index_;
        }

        Iterator &operator++() {
            one_based_index_ = GetNextNodeWithSupersetResponsibilityRange(one_based_index_);
            return *this;
        }

        friend bool operator==(const Iterator &left, const Iterator &right) {
            return left.IsIteratorExhausted() and right.IsIteratorExhausted();
        }

        friend bool operator!=(const Iterator &left, const Iterator &right) {
            return not operator==(left, right);
        }

    private:
        int32_t one_based_index_ = 0;
        int32_t size_ = 0;

        [[nodiscard]] bool IsIteratorExhausted() const {
            return one_based_index_ >= size_;
        }
    };

    Iterator begin() {
        return Iterator{zero_based_index, size};
    }

    Iterator end() {
        return Iterator{size, size};
    }
};

template <typename T>
std::vector<T> InitializeFenwickTree(const std::vector<T> &values) {
    std::vector<T> fenwick_tree{0};
    fenwick_tree.insert(fenwick_tree.end(), values.begin(), values.end());
    auto size = static_cast<int32_t>(fenwick_tree.size());

    for (int32_t one_based_index = 1; one_based_index < size; ++one_based_index) {
        auto next_superset_node = GetNextNodeWithSupersetResponsibilityRange(one_based_index);
        if (next_superset_node < size) {
            fenwick_tree[next_superset_node] += fenwick_tree[one_based_index];
        }
    }

    return fenwick_tree;
}

void AddValue(std::vector<int32_t> *fenwick_tree, int32_t index, int32_t value) {
    for (auto one_based_index :
         NodesWithSupersetResponsibilityRangeIterable{index, fenwick_tree->size()}) {
        (*fenwick_tree)[one_based_index] += value;
    }
}

int64_t GetPrefixSum(const std::vector<int32_t> &fenwick_tree, int32_t last_included_index) {
    int64_t sum = 0;
    for (auto one_based_index : ParentNodesIterable{last_included_index}) {
        sum += fenwick_tree[one_based_index];
    }
    return sum;
}

class FenwickTree {
public:
    explicit FenwickTree(int32_t size) : fenwick_tree_node_values_(size + 1) {
    }

    explicit FenwickTree(const std::vector<int32_t> &values)
        : fenwick_tree_node_values_{InitializeFenwickTree(values)} {
    }

    [[nodiscard]] size_t Size() const {
        return fenwick_tree_node_values_.size();
    }

    [[nodiscard]] int64_t GetRangeSum(int32_t from_including, int32_t to_including) const {
        return GetPrefixSum(to_including) - GetPrefixSum(from_including - 1);
    }

    [[nodiscard]] int64_t GetPrefixSum(int32_t last_included_index) const {
        return ::fenwick::GetPrefixSum(fenwick_tree_node_values_, last_included_index);
    }

    void AddValue(int32_t index, int32_t value) {
        ::fenwick::AddValue(&fenwick_tree_node_values_, index, value);
    }

private:
    std::vector<int32_t> fenwick_tree_node_values_;
};

template <size_t n_dimensions>
class MultiDimensionalFenwickTree {
public:
    using Coordinate = std::array<int32_t, n_dimensions>;

    explicit MultiDimensionalFenwickTree(int32_t size)
        : size_{static_cast<size_t>(size + 1)},
          fenwick_tree_node_values_(std::pow(size + 1, n_dimensions)) {
    }

    [[nodiscard]] int64_t GetRangeSum(const Coordinate &from_including,
                                      const Coordinate &to_including) const {
        auto from_minus_one = from_including;
        for (auto &i : from_minus_one) {
            --i;
        }
        Coordinate fenwick_coordinate{};
        return GetRangeSumRecursive(from_minus_one, to_including, &fenwick_coordinate);
    }

    [[nodiscard]] int64_t GetPrefixSum(const Coordinate &last_included_coordinate) const {
        Coordinate fenwick_coordinate{};
        return GetPrefixSumRecursive(last_included_coordinate, &fenwick_coordinate);
    }

    void AddValue(const Coordinate &coordinate, int32_t value) {
        Coordinate fenwick_coordinate{};
        AddValueRecursive(coordinate, value, &fenwick_coordinate);
    }

private:
    size_t size_ = 0;
    std::vector<int32_t> fenwick_tree_node_values_;

    [[nodiscard]] int64_t GetPrefixSumRecursive(const Coordinate &last_included_coordinate,
                                                Coordinate *fenwick_coordinate,
                                                size_t coordinate_id = 0) const {
        if (coordinate_id == n_dimensions) {
            return fenwick_tree_node_values_[GetFenwickNodeIndex(*fenwick_coordinate)];
        } else {
            int64_t sum = 0;
            for (auto one_based_index :
                 ParentNodesIterable{last_included_coordinate[coordinate_id]}) {

                (*fenwick_coordinate)[coordinate_id] = one_based_index;
                sum += GetPrefixSumRecursive(last_included_coordinate, fenwick_coordinate,
                                             coordinate_id + 1);
            }
            return sum;
        }
    }

    [[nodiscard]] int64_t GetRangeSumRecursive(const Coordinate &from, const Coordinate &to,
                                               Coordinate *fenwick_coordinate,
                                               size_t coordinate_id = 0) const {
        if (coordinate_id == n_dimensions) {
            return fenwick_tree_node_values_[GetFenwickNodeIndex(*fenwick_coordinate)];
        } else {
            int64_t sum = 0;

            for (auto one_based_index : ParentNodesIterable{to[coordinate_id]}) {

                (*fenwick_coordinate)[coordinate_id] = one_based_index;
                sum += GetRangeSumRecursive(from, to, fenwick_coordinate, coordinate_id + 1);
            }

            for (auto one_based_index : ParentNodesIterable{from[coordinate_id]}) {

                (*fenwick_coordinate)[coordinate_id] = one_based_index;
                sum -= GetRangeSumRecursive(from, to, fenwick_coordinate, coordinate_id + 1);
            }

            return sum;
        }
    }

    void AddValueRecursive(const Coordinate &coordinate, int32_t value,
                           Coordinate *fenwick_coordinate, size_t coordinate_id = 0) {
        if (coordinate_id == n_dimensions) {
            fenwick_tree_node_values_[GetFenwickNodeIndex(*fenwick_coordinate)] += value;
        } else {
            for (auto one_based_index :
                 NodesWithSupersetResponsibilityRangeIterable{coordinate[coordinate_id], size_}) {

                (*fenwick_coordinate)[coordinate_id] = one_based_index;
                AddValueRecursive(coordinate, value, fenwick_coordinate, coordinate_id + 1);
            }
        }
    }

    [[nodiscard]] int32_t GetFenwickNodeIndex(const Coordinate &coordinate) const {
        int32_t index = 0;
        for (auto i : coordinate) {
            index = index * size_ + i;
        }
        return index;
    }
};

int32_t FindIndexForPowerTwoFenwickTreeAtWhichPrefixSumIsEqualTo(
    const std::vector<int32_t> &fenwick_tree, int32_t prefix_sum) {
    int32_t low = 0;
    auto high = static_cast<int32_t>(fenwick_tree.size());

    while (high - low != 1) {
        auto mid = (low + high) / 2;
        if (fenwick_tree[mid] < prefix_sum) {
            prefix_sum -= fenwick_tree[mid];
            low = mid;
        } else {
            high = mid;
        }
    }

    return low;
}

io::Output Solve(const io::Input &input) {
    std::vector<int32_t> is_player_in_game(input.n_players, 1);
    auto log_two = static_cast<int32_t>(std::ceil(std::log2(input.n_players)));
    is_player_in_game.resize(1 << log_two);

    auto fenwick_tree = InitializeFenwickTree(is_player_in_game);

    io::Output output;
    output.players_drop_out_order.reserve(input.n_players);

    for (auto n_players_on_the_left : RhymePlayersDropOutOrderByNPlayersOnTheLeft{
             input.n_players, input.josephs_counting_rhyme}) {

        auto player_id = FindIndexForPowerTwoFenwickTreeAtWhichPrefixSumIsEqualTo(
            fenwick_tree, n_players_on_the_left);

        output.players_drop_out_order.emplace_back(player_id);

        AddValue(&fenwick_tree, player_id, -1);
    }

    return output;
}

}  // namespace fenwick

namespace range_sum_query {}

namespace cartesian_tree {}

io::Output Solve(const io::Input &input) {
    return fenwick::Solve(input);
    return sqrt_decomposition::Solve(input);
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
    if (input.n_players > 10'000) {
        throw NotImplementedError{};
    }

    std::vector<bool> is_player_in_game(input.n_players, true);
    io::Output output;
    output.players_drop_out_order.reserve(input.n_players);

    auto increment = [&input](int32_t &player_id) {
        ++player_id;
        if (player_id == input.n_players) {
            player_id = 0;
        }
    };

    auto find_next_player_in_game = [&is_player_in_game, &increment](int32_t &player_id) {
        while (not is_player_in_game[player_id]) {
            increment(player_id);
        }
    };

    int32_t player_id = 0;
    for (int32_t n_players_in_game = input.n_players; n_players_in_game > 0; --n_players_in_game) {
        find_next_player_in_game(player_id);

        for (auto rhyme = (input.josephs_counting_rhyme - 1) % n_players_in_game; rhyme > 0;
             --rhyme) {
            increment(player_id);
            find_next_player_in_game(player_id);
        }

        is_player_in_game[player_id] = false;
        output.players_drop_out_order.emplace_back(player_id);
    }

    return output;
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t n_players = 1 + test_case_id;
    std::uniform_int_distribution<int32_t> rhyme_distribution{1, n_players};
    io::Input input;
    input.n_players = n_players;
    input.josephs_counting_rhyme = rhyme_distribution(*rng::GetEngine());
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

    timed_checker.Check("5 3", "3 1 5 2 4");

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
