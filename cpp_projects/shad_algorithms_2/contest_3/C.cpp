#include <algorithm>
#include <array>
#include <cassert>
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
#include <unordered_map>
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
Number PowerOfTwo(Number value) {
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

template <typename T, typename I>
std::vector<T> Take(const std::vector<T> &values, const std::vector<I> &indices) {
    std::vector<T> slice;
    slice.reserve(values.size());
    for (auto i : indices) {
        slice.emplace_back(values[i]);
    }
    return slice;
}

}  // namespace utils

namespace sort {

std::vector<int32_t> ArgSortByArgs(
    int32_t size, std::function<bool(int32_t, int32_t)> arg_compare = std::less{}) {
    std::vector<int32_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), std::move(arg_compare));
    return indices;
}

template <typename Iterator>
std::vector<int32_t> ArgSortByValue(
    Iterator begin, Iterator end,
    std::function<bool(typename Iterator::value_type, typename Iterator::value_type)> comparator =
        std::less{}) {
    auto arg_comparator = [&begin, &comparator](int32_t left, int32_t right) -> bool {
        return comparator(*(begin + left), *(begin + right));
    };
    return ArgSortByArgs(end - begin, arg_comparator);
}

template <typename T>
std::vector<int32_t> SortedArgsToRanks(const std::vector<T> &sorted_args) {
    std::vector<int32_t> ranks(sorted_args.size());
    for (int32_t rank = 0; rank < static_cast<int32_t>(sorted_args.size()); ++rank) {
        ranks[sorted_args[rank]] = rank;
    }
    return ranks;
}

template <typename T>
std::vector<int32_t> RanksToSortedArgs(const std::vector<T> &ranks) {
    std::vector<int32_t> sorted_args(ranks.size());
    for (int32_t arg = 0; arg < static_cast<int32_t>(ranks.size()); ++arg) {
        sorted_args[ranks[arg]] = arg;
    }
    return sorted_args;
}

template <typename T, typename Comparator>
std::vector<T> MergeSorted(const std::vector<T> &left, const std::vector<T> &right,
                           Comparator left_right_comparator = std::less{}) {

    std::vector<T> sorted;
    sorted.reserve(left.size() + right.size());
    auto left_iter = left.begin();
    auto right_iter = right.begin();
    while (left_iter < left.end() and right_iter < right.end()) {
        if (left_right_comparator(*left_iter, *right_iter)) {
            sorted.emplace_back(*left_iter);
            ++left_iter;
        } else {
            sorted.emplace_back(*right_iter);
            ++right_iter;
        }
    }
    sorted.insert(sorted.end(), left_iter, left.end());
    sorted.insert(sorted.end(), right_iter, right.end());
    return sorted;
}

template <typename T>
std::vector<int32_t> SoftRank(const std::vector<T> &values,
                              const std::vector<int32_t> &sorted_indices) {
    if (values.empty()) {
        return {};
    }
    std::vector<int32_t> soft_ranks(values.size());
    auto prev = values[sorted_indices.front()];
    int32_t soft_rank = 1;
    for (auto i : sorted_indices) {
        if (values[i] == prev) {
            soft_ranks[i] = soft_rank;
        } else {
            prev = values[i];
            soft_ranks[i] = ++soft_rank;
        }
    }
    return soft_ranks;
}

template <typename T>
void CheckThatValueIsInAlphabet(T value, int32_t alphabet_size) {
    if (value < 0 or alphabet_size <= value) {
        throw std::invalid_argument{"Value must be non negative and not more than alphabet size."};
    }
}

class StableArgCountSorter {
public:
    std::vector<int32_t> sorted_indices;

    void Sort(const std::vector<int32_t> &values, int32_t alphabet_size) {
        for (auto value : values) {
            CheckThatValueIsInAlphabet(value, alphabet_size);
        }
        alphabet_counts_cum_sum_.resize(alphabet_size);
        std::fill(alphabet_counts_cum_sum_.begin(), alphabet_counts_cum_sum_.end(), 0);
        ComputeCumulativeAlphabetCounts(values);

        sorted_indices.resize(values.size());
        for (int32_t index = 0; index < static_cast<int32_t>(values.size()); ++index) {
            auto &value_sorted_position = alphabet_counts_cum_sum_[values[index]];
            sorted_indices[value_sorted_position] = index;
            ++value_sorted_position;
        }
    }

private:
    std::vector<int32_t> alphabet_counts_cum_sum_;

    void ComputeCumulativeAlphabetCounts(const std::vector<int32_t> &values) {
        for (auto i : values) {
            ++alphabet_counts_cum_sum_[i];
        }
        int32_t sum = 0;
        for (auto &i : alphabet_counts_cum_sum_) {
            auto count = i;
            i = sum;
            sum += count;
        }
    }
};

std::vector<int32_t> StableArgCountSort(const std::vector<int32_t> &values, int32_t alphabet_size) {
    StableArgCountSorter sorter;
    sorter.Sort(values, alphabet_size);
    return sorter.sorted_indices;
}

class StableArgRadixSorter {
public:
    std::vector<int32_t> sorted_indices;

    template <typename String>
    void SortEqualLengthStrings(const std::vector<String> &strings, int32_t alphabet_size) {
        radixes_.reserve(strings.size());
        sorted_indices.resize(strings.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);

        if (not strings.empty()) {
            for (int32_t nth = strings.front().size() - 1; nth >= 0; --nth) {
                BuildNthRadixes(strings, nth, alphabet_size);
                arg_count_sorter_.Sort(radixes_, alphabet_size);
                sorted_indices = utils::Take(sorted_indices, arg_count_sorter_.sorted_indices);
            }
        }
    }

private:
    std::vector<int32_t> radixes_;
    StableArgCountSorter arg_count_sorter_;

    template <typename String>
    void BuildNthRadixes(const std::vector<String> &strings, int32_t nth, int32_t alphabet_size) {
        radixes_.clear();
        for (auto index : sorted_indices) {
            auto radix = strings[index][nth];
            CheckThatValueIsInAlphabet(radix, alphabet_size);
            radixes_.emplace_back(radix);
        }
    }
};

template <typename String>
std::vector<int32_t> StableArgRadixSortEqualLengthStrings(const std::vector<String> &strings,
                                                          int32_t alphabet_size) {
    StableArgRadixSorter sorter;
    sorter.SortEqualLengthStrings(strings, alphabet_size);
    return sorter.sorted_indices;
}

}  // namespace sort

namespace io {

class Input {
public:
    std::string source_string;
    std::string target_string;
    int32_t max_edits = 0;

    Input() = default;

    explicit Input(std::istream &in) {
        in >> source_string >> target_string >> max_edits;
    }
};

class Output {
public:
    std::optional<int32_t> min_hamming_distance_after_edits;

    Output() = default;

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        int32_t item = 0;
        ss >> item;
        if (item != -1) {
            min_hamming_distance_after_edits = item;
        }
    }

    explicit Output(std::optional<int32_t> min_hamming_distance_after_edits)
        : min_hamming_distance_after_edits{min_hamming_distance_after_edits} {
    }

    std::ostream &Write(std::ostream &out) const {
        out << (min_hamming_distance_after_edits ? min_hamming_distance_after_edits.value() : -1);
        return out;
    }

    bool operator!=(const Output &other) const {
        return min_hamming_distance_after_edits != other.min_hamming_distance_after_edits;
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

std::optional<int32_t> ComputeHammingDistance(const std::string &first, const std::string &second) {
    if (first.size() != second.size()) {
        return std::nullopt;
    }
    int32_t distance = 0;
    for (auto f = first.begin(), s = second.begin(); f != first.end(); ++f, ++s) {
        if (*f != *s) {
            ++distance;
        }
    }
    return distance;
}

class LevenshteinMatrixDiagonal {
public:
    int32_t id = 0;
    std::vector<int32_t> hamming_distances;
    std::vector<bool> is_source_and_target_chars_equal;

    LevenshteinMatrixDiagonal(const std::string &source_string, const std::string &target_string,
                              int32_t id)
        : id{id} {
        auto source = source_string.begin() + std::max(0, id);
        auto target = target_string.begin() + std::max(0, -id);
        hamming_distances.resize(
            1 + std::min(source_string.end() - source, target_string.end() - target),
            INT32_MAX / 2);
        is_source_and_target_chars_equal.reserve(hamming_distances.size());
        for (; source < source_string.end() and target < target_string.end(); ++source, ++target) {
            is_source_and_target_chars_equal.emplace_back(*source == *target);
        }
    }

    [[nodiscard]] int32_t Size() const {
        return static_cast<int32_t>(hamming_distances.size());
    }

    int32_t &operator[](int32_t index) {
        return hamming_distances[index];
    }

    const int32_t &operator[](int32_t index) const {
        return hamming_distances[index];
    }

    void Update() {
        for (size_t i = 0; i + 1 < hamming_distances.size(); ++i) {
            hamming_distances[i + 1] =
                std::min(hamming_distances[i + 1], GetHammingDistanceAfterTraversingOneNode(i));
        }
    }

private:
    [[nodiscard]] inline int32_t GetHammingDistanceAfterTraversingOneNode(size_t node) const {
        return hamming_distances[node] +
               static_cast<int32_t>(not is_source_and_target_chars_equal[node]);
    }
};

class MinHammingDistanceFinder {
public:
    std::string source_string;
    std::string target_string;
    int32_t max_edits = 0;

    explicit MinHammingDistanceFinder(const io::Input &input)
        : source_string{input.source_string + std::string(input.max_edits, 'a')},
          target_string{input.target_string + std::string(input.max_edits, 'a')},
          max_edits{input.max_edits},
          equal_strings_size_diagonal_id_ {
              static_cast<int32_t>(input.source_string.size() - input.target_string.size())} {
    }

    std::optional<int32_t> Find() {
        InitializeFind();

        for (edits_made_ = 0; edits_made_ < max_edits; ++edits_made_) {
            RemoveInvalidCandidates();
            next_diagonal_candidates_by_id_map_ = diagonal_candidates_by_id_map_;

            for (auto &[id, diagonal] : diagonal_candidates_by_id_map_) {
                TryInserts(diagonal);
                TryDeletes(diagonal);
                TryChanges(diagonal);
            }

            for (auto &[id, diagonal] : next_diagonal_candidates_by_id_map_) {
                diagonal.Update();
            }

            std::swap(diagonal_candidates_by_id_map_, next_diagonal_candidates_by_id_map_);
        }

        return GetAnswer();
    }

private:
    int32_t equal_strings_size_diagonal_id_;
    std::map<int32_t, LevenshteinMatrixDiagonal> diagonal_candidates_by_id_map_;
    std::map<int32_t, LevenshteinMatrixDiagonal> next_diagonal_candidates_by_id_map_;
    int32_t edits_made_ = 0;

    LevenshteinMatrixDiagonal &AddDiagonal(std::map<int32_t, LevenshteinMatrixDiagonal> &map,
                                           int32_t id) const {
        auto [pair, was_inserted] = map.try_emplace(
            id, LevenshteinMatrixDiagonal{source_string, target_string, id});
        return pair->second;
    }

    static inline bool HasDiagonal(const std::map<int32_t, LevenshteinMatrixDiagonal> &map,
                                   int32_t id) {
        return map.count(id) > 0;
    }

    [[nodiscard]] inline bool IsValidDiagonalCandidateId(int32_t id) const {
        return abs(equal_strings_size_diagonal_id_ - id) <= max_edits - edits_made_;
    }

    void InitializeFind() {
        auto &diagonal = AddDiagonal(diagonal_candidates_by_id_map_, 0);
        diagonal[0] = 0;
        diagonal.Update();
    }

    void RemoveInvalidCandidates() {
        std::vector<int32_t> invalid_ids;
        invalid_ids.reserve(diagonal_candidates_by_id_map_.size());
        for (auto &[id, diagonal] : diagonal_candidates_by_id_map_) {
            if (not IsValidDiagonalCandidateId(id)) {
                invalid_ids.emplace_back(id);
            }
        }
        for (auto id : invalid_ids) {
            diagonal_candidates_by_id_map_.erase(id);
        }
    }

    void TryInserts(const LevenshteinMatrixDiagonal &diagonal) {
        TryEdit(diagonal, AddDiagonal(next_diagonal_candidates_by_id_map_, diagonal.id - 1),
                diagonal.id > 0 ? 1 : 0);
    }

    void TryDeletes(const LevenshteinMatrixDiagonal &diagonal) {
        TryEdit(diagonal, AddDiagonal(next_diagonal_candidates_by_id_map_, diagonal.id + 1),
                diagonal.id < 0 ? 1 : 0);
    }

    static void TryEdit(const LevenshteinMatrixDiagonal &diagonal,
                        LevenshteinMatrixDiagonal &diagonal_after_edit,
                        int32_t diagonal_index_offset) {
        for (int32_t index = 0;
             index < diagonal.Size() and diagonal_index_offset + index < diagonal_after_edit.Size();
             ++index) {
            auto &hamming_distance = diagonal_after_edit[diagonal_index_offset + index];
            hamming_distance = std::min(hamming_distance, diagonal[index]);
        }
    }

    void TryChanges(const LevenshteinMatrixDiagonal &diagonal) {
        auto &next_diagonal = next_diagonal_candidates_by_id_map_.at(diagonal.id);
        for (int32_t index = 0; index < diagonal.Size(); ++index) {
            auto &hamming_distance = next_diagonal[index];
            hamming_distance = std::min(hamming_distance, diagonal[index] - 1);
        }
    }

    [[nodiscard]] std::optional<int32_t> GetAnswer() const {
        if (HasDiagonal(diagonal_candidates_by_id_map_, equal_strings_size_diagonal_id_)) {
            return std::max(0, diagonal_candidates_by_id_map_.at(equal_strings_size_diagonal_id_)
                                   .hamming_distances.back());
        } else {
            return std::nullopt;
        }
    }
};

io::Output Solve(const io::Input &input) {
    return io::Output{MinHammingDistanceFinder{input}.Find()};
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
    if (input.source_string.size() > 10 or input.max_edits > 3) {
        throw NotImplementedError{};
    }

    auto min_char =
        std::min(*std::min_element(input.source_string.begin(), input.source_string.end()),
                 *std::min_element(input.target_string.begin(), input.target_string.end()));
    auto max_char =
        std::max(*std::max_element(input.source_string.begin(), input.source_string.end()),
                 *std::max_element(input.target_string.begin(), input.target_string.end()));

    std::vector<std::string> candidates{input.source_string};
    std::vector<std::string> next_candidates;
    io::Output output;

    for (int32_t edit = 0; edit < input.max_edits; ++edit) {
        for (auto &c : candidates) {
            auto distance = ComputeHammingDistance(c, input.target_string);
            if (distance) {
                if (not output.min_hamming_distance_after_edits) {
                    output.min_hamming_distance_after_edits = distance;
                } else {
                    output.min_hamming_distance_after_edits =
                        std::min(distance, output.min_hamming_distance_after_edits);
                }
            }
        }

        next_candidates.clear();

        for (auto &candidate : candidates) {
            for (auto b = candidate.begin(); b != candidate.end(); ++b) {
                auto candidate_copy = candidate;
                candidate_copy.erase(candidate_copy.begin() + (b - candidate.begin()));
                next_candidates.push_back(candidate_copy);
            }
        }

        for (auto ch = min_char; ch <= max_char; ++ch) {
            for (auto &candidate : candidates) {
                for (auto b = candidate.begin(); b <= candidate.end(); ++b) {
                    auto candidate_copy = candidate;
                    candidate_copy.insert(b - candidate.begin(), 1, ch);
                    next_candidates.push_back(candidate_copy);

                    if (b != candidate.end()) {
                        candidate_copy = candidate;
                        candidate_copy[b - candidate.begin()] = ch;
                        next_candidates.push_back(candidate_copy);
                    }
                }
            }
        }

        std::swap(candidates, next_candidates);
    }

    for (auto &c : candidates) {
        auto distance = ComputeHammingDistance(c, input.target_string);
        if (distance) {
            if (not output.min_hamming_distance_after_edits) {
                output.min_hamming_distance_after_edits = distance;
            } else {
                output.min_hamming_distance_after_edits =
                    std::min(distance, output.min_hamming_distance_after_edits);
            }
        }
    }
    return output;
}

std::string GenerateRandomString(int32_t size, char letter_from = 'a', char letter_to = 'z') {
    std::uniform_int_distribution<char> letters_dist{letter_from, letter_to};
    std::string string;
    for (int32_t i = 0; i < size; ++i) {
        string += letters_dist(*rng::GetEngine());
    }
    return string;
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t string_size = std::min(10'000, 1 + test_case_id / 30);
    int32_t max_edits = std::min(20, test_case_id / 20);
    char max_char = test_case_id < 1000 ? 'c' : 'z';

    std::uniform_int_distribution<int32_t> target_string_size_distribution{
        std::max(1, string_size - max_edits - 1), string_size + max_edits + 1};

    io::Input input;
    input.source_string = GenerateRandomString(string_size, 'a', max_char);
    input.target_string =
        GenerateRandomString(target_string_size_distribution(*rng::GetEngine()), 'a', max_char);
    input.max_edits = max_edits;
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

    TimedChecker timed_checker;

    timed_checker.Check(
        "abcdef\n"
        "xyz\n"
        "2",
        "-1");

    timed_checker.Check(
        "abcd\n"
        "bbb\n"
        "2",
        "1");

    timed_checker.Check(
        "aabcdef\n"
        "abcdefg\n"
        "2",
        "0");

    timed_checker.Check(
        "abccdef\n"
        "abcdefg\n"
        "1",
        "3");

    timed_checker.Check(
        "abcdefg\n"
        "abcgdef\n"
        "2",
        "0");

    timed_checker.Check(
        "aaaaaaaaaabbbbbbcasdcasdcasdccazxcvasdfasdasdvasdcasdacdbbbbbbbbaaaaaaaaaa\n"
        "bbbbbbcasdcasdcasdccazxcvasdfasdasdvasdcasdacdbbbbbbbbaaaaaaaaaaaaaaaaaaaa\n"
        "20",
        "0");

    timed_checker.Check(
        "abcdasdasdfasdffasdfasdfef\n"
        "z\n"
        "20",
        "-1");

    timed_checker.Check(
        "a\n"
        "zasdfasdfasdfasdfasdfasdfasdf\n"
        "20",
        "-1");

    timed_checker.Check(
        "a\n"
        "zasdfasdfasdf\n"
        "20",
        "0");

    timed_checker.Check(
        "a\n"
        "a\n"
        "20",
        "0");

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

    if (argc > 1 and std::strcmp(argv[1], "test") == 0) {
        test::Test();
    } else {
        std::cout << Solve(io::Input{std::cin});
    }

    return 0;
}
