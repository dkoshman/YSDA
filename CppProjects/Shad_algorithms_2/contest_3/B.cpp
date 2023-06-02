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

template <typename Container = std::vector<int32_t>, typename Comparator = decltype(std::less{})>
class RangeMinQueryResponder {
public:
    using Value = typename Container::value_type;

    explicit RangeMinQueryResponder(const Container &container,
                                    const Comparator &comparator = std::less{})
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

    Value GetRangeMin(int32_t begin, int32_t end) {
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
    std::string string;

    Input() = default;

    explicit Input(std::istream &in) {
        in >> string;
    }
};

class Output {
public:
    std::vector<int32_t> length_of_previous_longest_substring_starting_earlier;

    Output() = default;

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        int32_t item = 0;
        while (ss >> item) {
            length_of_previous_longest_substring_starting_earlier.emplace_back(item);
        }
    }

    std::ostream &Write(std::ostream &out) const {
        for (auto item : length_of_previous_longest_substring_starting_earlier) {
            out << item << '\n';
        }
        return out;
    }

    bool operator!=(const Output &other) const {
        return length_of_previous_longest_substring_starting_earlier !=
               other.length_of_previous_longest_substring_starting_earlier;
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

namespace suffix_array {

class RecursiveState {
public:
    std::vector<int32_t> padded_string;
    std::vector<int32_t> one_two_triplet_ranks;

    explicit RecursiveState(const std::vector<int32_t> &string) : padded_string{string} {
        padded_string.resize(string.size() + 3);
    }

    enum class CompareResult { Less, Equal, Greater };

    [[nodiscard]] CompareResult SingleLetterCompare(int32_t left, int32_t right) const {
        if (padded_string[left] == padded_string[right]) {
            return CompareResult::Equal;
        } else {
            return padded_string[left] < padded_string[right] ? CompareResult::Less
                                                              : CompareResult::Greater;
        }
    }

    [[nodiscard]] bool AllSuffixBlocksStringCompare(int32_t left, int32_t right) const {
        CompareResult compare_result;
        while ((compare_result = SingleLetterCompare(left, right)) == CompareResult::Equal) {
            ++left;
            ++right;
        }
        return compare_result == CompareResult::Less;
    }

    std::function<bool(int32_t, int32_t)> GetStringComparator() {
        return [this](int32_t left, int32_t right) -> bool {
            return AllSuffixBlocksStringCompare(left, right);
        };
    }

    [[nodiscard]] int32_t ConvertArgToOneTwoRank(int32_t arg) const {
        if (arg % 3 == 0) {
            throw std::invalid_argument{"Not from one or two mod 3 group."};
        }
        auto twos_start = (one_two_triplet_ranks.size() + 1) / 2;
        return arg % 3 == 1 ? one_two_triplet_ranks[arg / 3]
                            : one_two_triplet_ranks[twos_start + arg / 3];
    }

    [[nodiscard]] bool TripletGroupOneTwoCompare(int32_t left, int32_t right) const {
        return ConvertArgToOneTwoRank(left) < ConvertArgToOneTwoRank(right);
    }

    [[nodiscard]] bool TripletGroupZeroCompare(int32_t left, int32_t right) const {
        auto compare_result = SingleLetterCompare(left, right);
        if (compare_result == CompareResult::Equal) {
            return TripletGroupOneTwoCompare(left + 1, right + 1);
        }
        return compare_result == CompareResult::Less;
    }

    [[nodiscard]] bool TripletCompare(int32_t left, int32_t right) const {
        if (left % 3 == 0 and right % 3 == 0) {
            return TripletGroupZeroCompare(left, right);
        } else if (left % 3 != 0 and right % 3 != 0) {
            return TripletGroupOneTwoCompare(left, right);
        } else {
            auto compare_result = SingleLetterCompare(left, right);
            if (compare_result == CompareResult::Equal) {
                return TripletCompare(left + 1, right + 1);
            } else {
                return compare_result == CompareResult::Less;
            }
        }
    }

    std::function<bool(int32_t, int32_t)> GetTripletComparator() {
        return [this](int32_t left, int32_t right) -> bool { return TripletCompare(left, right); };
    }
};

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

class SuffixArrayBuilder {
public:
    std::vector<int32_t> Build(const std::string &string) {
        std::vector<int32_t> vector_string;
        vector_string.reserve(string.size());
        for (auto &i : string) {
            vector_string.emplace_back(i);
        }
        return Build(vector_string);
    }

    std::vector<int32_t> BuildFromLowerLatinLetters(const std::string &string) {
        std::vector<int32_t> vector_string;
        vector_string.reserve(string.size());
        for (auto &i : string) {
            vector_string.emplace_back(i - 'a' + 1);
        }
        return BuildFromPositiveValues(vector_string);
    }

    std::vector<int32_t> Build(std::vector<int32_t> string) {
        auto min = *std::min_element(string.begin(), string.end());
        auto max = 0;
        for (auto &i : string) {
            i -= min;
            max = std::max(max, i);
        }

        auto indices = StableArgCountSort(string, max + 1);
        string = utils::SoftRank(string, indices);
        return BuildFromPositiveValues(string);
    }

    std::vector<int32_t> BuildFromPositiveValues(const std::vector<int32_t> &string) {
        return UnPadEofMarker(BuildRecursivePadded(PadStringWithEofMarker(string)));
    }

private:
    StableArgRadixSorter arg_radix_sorter_;

    std::vector<int32_t> PadStringWithEofMarker(std::vector<int32_t> soft_ranks) {
        soft_ranks.push_back(0);
        return soft_ranks;
    }

    std::vector<int32_t> UnPadEofMarker(std::vector<int32_t> suffix_array) {
        suffix_array.erase(suffix_array.begin());
        return suffix_array;
    }

    std::vector<int32_t> BuildRecursivePadded(const std::vector<int32_t> &string) {
        RecursiveState state{string};

        if (string.size() <= 5) {
            return utils::ArgSortByArgs(static_cast<int32_t>(string.size()),
                                        state.GetStringComparator());
        }

        auto [one_two_indices, zero_indices] = BuildOneTwoAndZeroModThreeIndices(string.size());

        auto one_two_triplets_indices = ArgSortOneTwoTriplets(string, state);
        state.one_two_triplet_ranks = utils::SortedArgsToRanks(one_two_triplets_indices);
        one_two_indices = utils::Take(one_two_indices, one_two_triplets_indices);

        std::sort(zero_indices.begin(), zero_indices.end(), state.GetTripletComparator());

        return utils::MergeSorted(one_two_indices, zero_indices, state.GetTripletComparator());
    }

    static std::vector<std::array<int32_t, 3>> BuildOneTwoTriples(
        const std::vector<int32_t> &padded_string) {
        std::vector<std::array<int32_t, 3>> triples;
        auto unpadded_size = static_cast<int32_t>(padded_string.size() - 3);
        triples.reserve((unpadded_size + 1) * 2 / 3);
        for (int32_t i = 0; i < unpadded_size; ++i) {
            if (i % 3 == 1) {
                triples.push_back({padded_string[i], padded_string[i + 1], padded_string[i + 2]});
            }
        }
        for (int32_t i = 0; i < unpadded_size; ++i) {
            if (i % 3 == 2) {
                triples.push_back({padded_string[i], padded_string[i + 1], padded_string[i + 2]});
            }
        }
        return triples;
    }

    std::vector<int32_t> ArgSortOneTwoTriplets(const std::vector<int32_t> &string,
                                               const RecursiveState &state) {
        auto triples = BuildOneTwoTriples(state.padded_string);
        auto sorted_triples_indices = StableArgRadixSortEqualLengthStrings(
            triples, *std::max_element(string.begin(), string.end()) + 1);
        auto one_two_soft_ranks = utils::SoftRank(triples, sorted_triples_indices);
        return BuildFromPositiveValues(one_two_soft_ranks);
    }

    static std::pair<std::vector<int32_t>, std::vector<int32_t>> BuildOneTwoAndZeroModThreeIndices(
        size_t size) {
        std::vector<int32_t> one_two(size * 2 / 3);
        std::vector<int32_t> zero((size + 2) / 3);
        auto two_start = (one_two.size() + 1) / 2;
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i) {
            if (i % 3 == 1) {
                one_two[i / 3] = i;
            } else if (i % 3 == 2) {
                one_two[two_start + i / 3] = i;
            } else {
                zero[i / 3] = i;
            }
        }
        return {one_two, zero};
    }
};

std::vector<int32_t> BuildSuffixArray(const std::string &string) {
    return SuffixArrayBuilder{}.Build(string);
}

}  // namespace suffix_array

class LongestCommonPrefixComputer {
    const std::string &string_;

public:
    const std::vector<int32_t> suffix_array;
    const std::vector<int32_t> inverse_suffix_array;

    explicit LongestCommonPrefixComputer(const std::string &string)
        : string_{string},
          suffix_array{suffix_array::BuildSuffixArray(string)},
          inverse_suffix_array{utils::SortedArgsToRanks(suffix_array)},
          sorted_suffixes_neighbors_lcp_(string.size()),
          rmq_{sorted_suffixes_neighbors_lcp_} {
        PrecomputeSortedNeighborsLcp();
    }

    int32_t GetLongestCommonPrefix(int32_t first_suffix, int32_t second_suffix) {
        auto first_rank = inverse_suffix_array[first_suffix];
        auto second_rank = inverse_suffix_array[second_suffix];
        if (second_rank < first_rank) {
            std::swap(first_rank, second_rank);
        }
        return rmq_.GetRangeMin(first_rank, second_rank);
    }

private:
    std::vector<int32_t> sorted_suffixes_neighbors_lcp_;
    utils::RangeMinQueryResponder<> rmq_;

    void PrecomputeSortedNeighborsLcp() {
        auto size = static_cast<int32_t>(string_.size());
        int32_t previous_lcp = 0;
        for (int32_t pos = 0; pos < size; ++pos) {
            auto rank = inverse_suffix_array[pos];
            if (rank == size - 1) {
                continue;
            }
            auto next_sorted_neighbor_pos = suffix_array[rank + 1];
            sorted_suffixes_neighbors_lcp_[rank] = FindLongestCommonPrefixNotLessThan(
                pos, next_sorted_neighbor_pos, std::max(0, previous_lcp - 1));
            previous_lcp = sorted_suffixes_neighbors_lcp_[rank];
        }
    };

    [[nodiscard]] int32_t FindLongestCommonPrefixNotLessThan(int32_t first_suffix,
                                                             int32_t second_suffix,
                                                             int32_t min_length) const {
        for (auto first = string_.begin() + first_suffix + min_length,
                  second = string_.begin() + second_suffix + min_length;
             first != string_.end() and second != string_.end(); ++first, ++second) {
            if (*first == *second) {
                ++min_length;
            } else {
                break;
            }
        }
        return min_length;
    }
};

std::vector<std::optional<int32_t>> FindClosestLessLeftNeighborForEach(
    const std::vector<int32_t> &vector) {
    std::vector<std::optional<int32_t>> neighbors(vector.size());
    for (int32_t i = 1; i < static_cast<int32_t>(vector.size()); ++i) {
        for (int32_t neighbor = i - 1;; neighbor = neighbors[neighbor].value()) {
            if (vector[neighbor] < vector[i]) {
                neighbors[i] = neighbor;
                break;
            } else if (not neighbors[neighbor]) {
                break;
            }
        }
    }
    return neighbors;
}

std::vector<std::optional<int32_t>> FindClosestLessRightNeighborForEach(
    std::vector<int32_t> vector) {
    std::reverse(vector.begin(), vector.end());
    auto neighbors = FindClosestLessLeftNeighborForEach(vector);
    std::reverse(neighbors.begin(), neighbors.end());
    for (auto &i : neighbors) {
        if (i) {
            i = static_cast<int32_t>(vector.size()) - 1 - i.value();
        }
    }
    return neighbors;
}

io::Output Solve(const io::Input &input) {
    LongestCommonPrefixComputer lcp(input.string);
    auto left_neighbors = FindClosestLessLeftNeighborForEach(lcp.suffix_array);
    auto right_neighbors = FindClosestLessRightNeighborForEach(lcp.suffix_array);
    io::Output output;
    output.length_of_previous_longest_substring_starting_earlier.resize(input.string.size());

    for (int32_t rank = 0; rank < static_cast<int32_t>(input.string.size()); ++rank) {
        auto pos = lcp.suffix_array[rank];
        auto left = left_neighbors[rank];
        auto right = right_neighbors[rank];
        int32_t max_length = 0;
        if (left) {
            auto left_pos = lcp.suffix_array[left.value()];
            max_length = std::max(max_length, lcp.GetLongestCommonPrefix(pos, left_pos));
        }
        if (right) {
            auto right_pos = lcp.suffix_array[right.value()];
            max_length = std::max(max_length, lcp.GetLongestCommonPrefix(pos, right_pos));
        }
        output.length_of_previous_longest_substring_starting_earlier[pos] = max_length;
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

template <typename Iterator>
int32_t FindLongestCommonPrefix(Iterator first_begin, Iterator first_end, Iterator second_begin,
                                Iterator second_end) {
    int32_t length = 0;
    for (; first_begin != first_end and second_begin != second_end; ++first_begin, ++second_begin) {
        if (*first_begin != *second_begin) {
            break;
        }
        ++length;
    }
    return length;
}

int32_t FindLongestCommonPrefix(const std::string &first, const std::string &second) {
    return FindLongestCommonPrefix(first.begin(), first.end(), second.begin(), second.end());
}

io::Output BruteForceSolve(const io::Input &input) {
    if (input.string.size() >= 1000) {
        throw NotImplementedError{};
    }
    io::Output output;
    for (auto it = input.string.begin(); it != input.string.end(); ++it) {
        int32_t max_length = 0;
        for (auto previous = input.string.begin(); previous != it; ++previous) {
            max_length = std::max(
                max_length,
                FindLongestCommonPrefix(it, input.string.end(), previous, input.string.end()));
        }
        output.length_of_previous_longest_substring_starting_earlier.emplace_back(max_length);
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
    auto string_size = std::min(100'000, test_case_id + 1);
    io::Input input;
    input.string = GenerateRandomString(string_size, 'a', 'c');
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

    timed_checker.Check("ababaab",
                        "0\n"
                        "0\n"
                        "3\n"
                        "2\n"
                        "1\n"
                        "2\n"
                        "1");

    timed_checker.Check("aaaaa",
                        "0\n"
                        "4\n"
                        "3\n"
                        "2\n"
                        "1");

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
