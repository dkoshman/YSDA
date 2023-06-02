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
#include <variant>
#include <vector>

namespace io {

template <size_t n_dimensions>
std::istream &operator>>(std::istream &is, std::array<int32_t, n_dimensions> &coordinate) {
    for (int &i : coordinate) {
        is >> i;
    }
    return is;
}

template <size_t n_dimensions>
struct AddValueAtCoordinate {
    struct Request {
        std::array<int32_t, n_dimensions> coordinate{};
        int32_t value = 0;

        Request() = default;

        explicit Request(std::istream &in) {
            in >> coordinate >> value;
        }
    };
};

template <size_t n_dimensions>
struct GetRangeSum {
    struct Request {
        std::array<int32_t, n_dimensions> from{};
        std::array<int32_t, n_dimensions> to{};

        Request() = default;

        explicit Request(std::istream &in) {
            in >> from >> to;
        }
    };
    struct Response {
        int64_t sum = 0;

        explicit Response(int64_t sum) : sum{sum} {
        }
    };
};

struct Stop {
    struct Request {};
};

template <size_t n_dimensions>
using Request = std::variant<typename AddValueAtCoordinate<n_dimensions>::Request,
                             typename GetRangeSum<n_dimensions>::Request, Stop::Request>;

template <size_t n_dimensions>
using Response = std::optional<typename GetRangeSum<n_dimensions>::Response>;

class Input {
public:
    static const size_t kDimensions = 3;
    int32_t cube_side = 0;
    std::vector<Request<kDimensions>> requests;

    Input() = default;

    explicit Input(std::istream &in) {
        in >> cube_side;
        int32_t request_code = 0;
        while (in >> request_code) {
            switch (request_code) {
                case 1:
                    requests.push_back(AddValueAtCoordinate<kDimensions>::Request{in});
                    break;
                case 2:
                    requests.push_back(GetRangeSum<kDimensions>::Request{in});
                    break;
                case 3:
                    requests.emplace_back(Stop::Request{});
                    break;
                default:
                    throw std::invalid_argument{"Unknown request code."};
            }
        }
    }
};

class Output {
public:
    std::vector<int64_t> range_sums;

    Output() = default;

    template <typename Responses>
    explicit Output(const Responses &responses) {
        for (auto response : responses) {
            range_sums.emplace_back(response.sum);
        }
    }

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        int32_t item = 0;
        while (ss >> item) {
            range_sums.emplace_back(item);
        }
    }

    std::ostream &Write(std::ostream &out) const {
        for (auto item : range_sums) {
            out << item << '\n';
        }
        return out;
    }

    bool operator!=(const Output &other) const {
        return range_sums != other.range_sums;
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

}  // namespace fenwick

io::Output Solve(const io::Input &input) {
    fenwick::MultiDimensionalFenwickTree<io::Input::kDimensions> fenwick_tree{input.cube_side};

    std::vector<io::GetRangeSum<io::Input::kDimensions>::Response> responses;

    for (auto request : input.requests) {
        if (std::holds_alternative<io::Stop::Request>(request)) {
            break;
        }

        if (auto add_request =
                std::get_if<io::AddValueAtCoordinate<io::Input::kDimensions>::Request>(&request)) {

            fenwick_tree.AddValue(add_request->coordinate, add_request->value);
        } else if (auto sum_request =
                       std::get_if<io::GetRangeSum<io::Input::kDimensions>::Request>(&request)) {

            responses.emplace_back(fenwick_tree.GetRangeSum(sum_request->from, sum_request->to));
        } else {

            throw std::invalid_argument{"Unknown request."};
        }
    }

    return io::Output{responses};
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
    std::vector<std::vector<std::vector<int32_t>>> cube(input.cube_side);
    for (auto &first : cube) {
        first.resize(input.cube_side);
        for (auto &second : first) {
            second.resize(input.cube_side);
        }
    }
    std::vector<io::GetRangeSum<io::Input::kDimensions>::Response> responses;

    for (auto request : input.requests) {
        if (std::holds_alternative<io::Stop::Request>(request)) {
            break;
        }
        if (auto add_request =
                std::get_if<io::AddValueAtCoordinate<io::Input::kDimensions>::Request>(&request)) {
            cube[add_request->coordinate[0]][add_request->coordinate[1]]
                [add_request->coordinate[2]] += add_request->value;
        } else if (auto sum_request =
                       std::get_if<io::GetRangeSum<io::Input::kDimensions>::Request>(&request)) {
            int64_t sum = 0;
            for (auto first = sum_request->from[0]; first <= sum_request->to[0]; ++first) {
                for (auto second = sum_request->from[1]; second <= sum_request->to[1]; ++second) {
                    for (auto third = sum_request->from[2]; third <= sum_request->to[2]; ++third) {
                        sum += cube[first][second][third];
                    }
                }
            }
            responses.emplace_back(sum);
        } else {
            throw std::invalid_argument{"Unknown request."};
        }
    }

    return io::Output{responses};
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {

    int32_t cube_side = std::min(128, 1 + test_case_id);
    int32_t n_requests = 1 + test_case_id;
    int32_t max_abs_value = std::min(20'000, 1 + test_case_id);

    std::uniform_int_distribution<int32_t> coordinate_distribution{0, cube_side - 1};
    std::uniform_int_distribution<int32_t> value_distribution{-max_abs_value, max_abs_value};
    auto &engine = *rng::GetEngine();

    io::Input input;
    input.cube_side = cube_side;
    while (static_cast<int32_t>(input.requests.size()) < n_requests) {
        if (coordinate_distribution(engine) % 2) {
            io::AddValueAtCoordinate<io::Input::kDimensions>::Request request;
            for (auto &i : request.coordinate) {
                i = coordinate_distribution(engine);
            }
            request.value = value_distribution(engine);
            input.requests.emplace_back(request);
        } else {
            io::GetRangeSum<io::Input::kDimensions>::Request request;
            for (size_t i = 0; i < request.from.size(); ++i) {
                auto from = coordinate_distribution(engine);
                auto to = coordinate_distribution(engine);
                if (to < from) {
                    std::swap(from, to);
                }
                request.from[i] = from;
                request.to[i] = to;
            }
            input.requests.emplace_back(request);
        }
    }
    input.requests.emplace_back(io::Stop::Request{});
    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    return GenerateRandomTestIo(10'000);
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
        "2\n"
        "2 1 1 1 1 1 1\n"
        "1 0 0 0 1\n"
        "1 0 1 0 3\n"
        "2 0 0 0 0 0 0\n"
        "2 0 0 0 0 1 0\n"
        "1 0 1 0 -2\n"
        "2 0 0 0 1 1 1\n"
        "3",
        "0\n"
        "1\n"
        "4\n"
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
