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

    inline Value &MinOnPreprocessedIntervalByBegin(size_t begin, size_t interval_size_log_two) {
        return rmq_preprocessed_[interval_size_log_two][begin];
    }

    inline Value &MinOnPreprocessedIntervalByEnd(size_t end, size_t interval_size_log_two) {
        return rmq_preprocessed_[interval_size_log_two][end - (1 << interval_size_log_two)];
    }
};

template <typename T, typename I>
inline void Take(const std::vector<T> &values, const std::vector<I> &indices, std::vector<T> *out) {
    out->resize(values.size());
    int32_t index = 0;
    for (auto i : indices) {
        (*out)[index++] = values[i];
    }
}

template <typename T, typename I>
inline std::vector<T> Take(const std::vector<T> &values, const std::vector<I> &indices) {
    std::vector<T> slice;
    Take(values, indices, &slice);
    return slice;
}

inline std::vector<int32_t> ArgSortByArgs(
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
    std::vector<T> sorted(left.size() + right.size());
    auto left_iter = left.begin();
    auto right_iter = right.begin();
    auto left_end = left.end();
    auto right_end = right.end();
    int32_t index = 0;
    while (left_iter < left_end and right_iter < right_end) {
        auto lvalue = *left_iter;
        auto rvalue = *right_iter;
        if (left_right_comparator(lvalue, rvalue)) {
            sorted[index] = lvalue;
            ++left_iter;
        } else {
            sorted[index] = rvalue;
            ++right_iter;
        }
        ++index;
    }
    std::copy(left_iter, left_end, sorted.begin() + index);
    std::copy(right_iter, right_end, sorted.begin() + index + (left_end - left_iter));
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

size_t TotalStringsSize(const std::vector<std::string> &strings) {
    size_t size = 0;
    for (auto &string : strings) {
        size += string.size();
    }
    return size;
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
        //        for (auto value : values) {
        //            CheckThatValueIsInAlphabet(value, alphabet_size);
        //        }
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

    template <typename F>
    void Sort(int32_t size, int32_t alphabet_size, const F &get_value_by_index) {
        alphabet_counts_cum_sum_.resize(alphabet_size);
        std::fill(alphabet_counts_cum_sum_.begin(), alphabet_counts_cum_sum_.end(), 0);
        for (int32_t i = 0; i < size; ++i) {
            ++alphabet_counts_cum_sum_[get_value_by_index(i)];
        }
        int32_t sum = 0;
        for (auto &i : alphabet_counts_cum_sum_) {
            auto count = i;
            i = sum;
            sum += count;
        }

        sorted_indices.resize(size);
        for (int32_t index = 0; index < size; ++index) {
            auto &value_sorted_position = alphabet_counts_cum_sum_[get_value_by_index(index)];
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

class UnStableCountSorter {
public:
    template <typename F>
    void Sort(std::vector<int32_t> *values, int32_t alphabet_size,
              const F &get_what_to_compare_by_value) {
        alphabet_counts_cum_sum_.resize(alphabet_size);
        std::fill(alphabet_counts_cum_sum_.begin(), alphabet_counts_cum_sum_.end(), 0);
        for (auto v : *values) {
            ++alphabet_counts_cum_sum_[get_what_to_compare_by_value(v)];
        }
        int32_t sum = 0;
        for (auto &i : alphabet_counts_cum_sum_) {
            auto count = i;
            i = sum;
            sum += count;
        }
        auto begin = values->begin();
        auto end = values->end();
        auto it = begin;
        is_in_place_.resize(values->size());
        std::fill(is_in_place_.begin(), is_in_place_.end(), false);
        while (it != end) {
            auto &value_sorted_position =
                alphabet_counts_cum_sum_[get_what_to_compare_by_value(*it)];
            is_in_place_[value_sorted_position] = true;
            if (it != begin + value_sorted_position) {
                std::iter_swap(it, begin + value_sorted_position);
            } else {
                while (it != end and is_in_place_[it - begin]) {
                    ++it;
                }
            }
            ++value_sorted_position;
        }
    }

private:
    std::vector<int32_t> alphabet_counts_cum_sum_;
    std::vector<bool> is_in_place_;
};

template <typename T = int32_t>
class StableArgRadixSorter {
public:
    std::vector<T> sorted_indices;

    template <typename String>
    void SortEqualLengthStrings(const std::vector<String> &strings, int32_t alphabet_size) {
        sorted_indices.resize(strings.size());
        sorted_indices_buffer_.resize(strings.size());
        radixes_counts_cum_sum_.resize(alphabet_size + 1);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);

        if (not strings.empty()) {
            for (int32_t nth = strings.front().size() - 1; nth >= 0; --nth) {
                CountRadixes(strings, nth);
                StableSortByRadix(strings, nth);
            }
        }
    }

private:
    std::vector<int32_t> radixes_counts_cum_sum_;
    std::vector<T> sorted_indices_buffer_;

    template <typename String>
    void CountRadixes(const std::vector<String> &strings, int32_t nth) {
        std::fill(radixes_counts_cum_sum_.begin(), radixes_counts_cum_sum_.end(), 0);
        for (auto &string : strings) {
            ++radixes_counts_cum_sum_[string[nth] + 1];
        }
        int32_t prev = 0;
        for (auto &c : radixes_counts_cum_sum_) {
            c += prev;
            prev = c;
        }
    }

    template <typename String>
    inline void StableSortByRadix(const std::vector<String> &strings, int32_t nth) {
        for (auto i : sorted_indices) {
            auto radix = strings[i][nth];
            sorted_indices_buffer_[radixes_counts_cum_sum_[radix]] = i;
            ++radixes_counts_cum_sum_[radix];
        }
        std::swap(sorted_indices, sorted_indices_buffer_);
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
    std::vector<std::string> strings;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t n_strings = 0;
        in >> n_strings;
        strings.resize(n_strings);
        for (auto &string : strings) {
            in >> string;
        }
    }
};

class Output {
public:
    std::string largest_common_substring;

    Output() = default;

    explicit Output(std::string string) : largest_common_substring{std::move(string)} {
    }

    std::ostream &Write(std::ostream &out) const {
        out << largest_common_substring;
        return out;
    }

    bool operator!=(const Output &other) const {
        return largest_common_substring.size() != other.largest_common_substring.size();
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

namespace suffix_tree {

class Edge {
public:
    int32_t text_interval_begin = 0;
    int32_t text_interval_end = INT32_MAX;
    std::optional<int32_t> child_node;

    Edge() = default;

    explicit Edge(int32_t begin) : text_interval_begin{begin} {
    }

    Edge(int32_t begin, int32_t end, int32_t target_node)
        : text_interval_begin{begin}, text_interval_end{end}, child_node{target_node} {
    }

    [[nodiscard]] int32_t Length() const {
        return text_interval_end - text_interval_begin;
    }

    [[nodiscard]] bool IsLeaf() const {
        return not child_node;
    }

    Edge SplitChildEdge(int32_t child_edge_length, int32_t new_child_node) {
        auto child_edge = *this;
        child_edge.text_interval_begin += child_edge_length;

        text_interval_end = child_edge.text_interval_begin;
        child_node = new_child_node;

        return child_edge;
    }
};

struct Node {
    std::unordered_map<char, Edge> edges;
    std::optional<int32_t> link;

    Node() : link{0} {
    }

    Node(char letter, int32_t letter_position) : edges{{letter, Edge{letter_position}}} {
    }
};

class Location {
public:
    const std::string *text_;
    const std::vector<Node> *nodes_;
    int32_t node = 0;
    int32_t delta = 0;
    std::optional<char> first_edge_letter;

    Location(const std::string &text, const std::vector<Node> &nodes)
        : text_{&text}, nodes_{&nodes} {
    }

    [[nodiscard]] bool IsExplicit() const {
        return delta == 0;
    }

    [[nodiscard]] bool IsRoot() const {
        return node == 0;
    }

    [[nodiscard]] Edge GetImplicitEdge() const {
        return (*nodes_)[node].edges.at(first_edge_letter.value());
    }

    [[nodiscard]] char GetNextImplicitLetter() const {
        return (*text_)[GetImplicitEdge().text_interval_begin + delta];
    }

    [[nodiscard]] bool IsFirstImplicitNodeOnTheEdgeALeaf(Edge edge) const {
        return edge.text_interval_begin + delta == static_cast<int32_t>(text_->size());
    }

    [[nodiscard]] bool CanDescendByLetter(char letter) const {
        if (IsAtTextEnd()) {
            return false;
        } else if (IsExplicit()) {
            return (*nodes_)[node].edges.count(letter);
        } else {
            return GetNextImplicitLetter() == letter;
        }
    }

    [[nodiscard]] bool IsAtTextEnd() const {
        if (IsExplicit()) {
            return false;
        } else {
            return GetImplicitEdge().text_interval_begin + delta ==
                   static_cast<int32_t>(text_->size());
        }
    }

    void DescendByLetter(char letter) {
        if (IsExplicit()) {

            first_edge_letter = letter;
            Edge edge_to_descend = GetImplicitEdge();
            if (not edge_to_descend.IsLeaf() and edge_to_descend.Length() == 1) {
                node = edge_to_descend.child_node.value();
            } else {
                delta = 1;
            }
        } else {

            Edge edge_to_descend = GetImplicitEdge();

            if (not edge_to_descend.IsLeaf() and delta + 1 == edge_to_descend.Length()) {
                node = edge_to_descend.child_node.value();
                delta = 0;
            } else {
                ++delta;
            }
        }
    }

    [[nodiscard]] std::string GetAllSortedNextLetters() const {
        if (IsExplicit()) {
            std::string next_letters;
            for (auto c : (*nodes_)[node].edges) {
                next_letters += c.first;
            }
            return next_letters;
        } else {
            return {GetNextImplicitLetter()};
        }
    }
};

class SuffixTree {
public:
    explicit SuffixTree(size_t capacity = 0) : location_{text_, nodes_} {
        text_.reserve(capacity);
        nodes_.reserve(capacity);
        nodes_.emplace_back();
    }

    explicit SuffixTree(const std::string &text) : SuffixTree{text.size()} {
        AppendText(text);
    }

    [[nodiscard]] Location GetSearchLocation() const {
        return {text_, nodes_};
    }

    [[nodiscard]] bool Search(const std::string &word) const {
        auto location = GetSearchLocation();
        for (auto c : word) {
            if (not location.CanDescendByLetter(c)) {
                return false;
            }
            location.DescendByLetter(c);
        }
        return true;
    }

    void AppendText(const std::string &text) {
        for (auto c : text) {
            AppendLetter(c);
        }
    }

    void AppendLetter(char letter) {
        text_ += letter;
        std::optional<int32_t> suffix_link_from;

        while (not location_.CanDescendByLetter(letter)) {
            auto previous_location = location_;

            AddLastLetterAtLocation();

            if (suffix_link_from) {
                nodes_[suffix_link_from.value()].link = location_.node;
            }
            suffix_link_from = location_.node;

            TraverseSuffixLink(previous_location);
        }

        if (suffix_link_from) {
            nodes_[suffix_link_from.value()].link = location_.node;
        }

        location_.DescendByLetter(letter);
        if (location_.IsAtTextEnd()) {
            --location_.delta;
        }
    }

private:
    std::string text_;
    std::vector<Node> nodes_;
    Location location_;

    void AddLastLetterAtLocation() {
        if (location_.IsExplicit()) {
            nodes_[location_.node].edges.emplace(text_.back(), text_.size() - 1);
            return;
        }

        auto new_node = static_cast<int32_t>(nodes_.size());
        auto implicit_edge = location_.GetImplicitEdge();

        nodes_.emplace_back(text_.back(), text_.size() - 1);
        Edge edge_lower_half = implicit_edge.SplitChildEdge(location_.delta, new_node);
        nodes_[location_.node].edges[location_.first_edge_letter.value()] = implicit_edge;
        nodes_[new_node].edges[location_.GetNextImplicitLetter()] = edge_lower_half;

        location_.node = new_node;
    }

    void TraverseSuffixLink(Location previous_location) {
        if (location_.node == previous_location.node) {
            location_.node = nodes_[location_.node].link.value();
            return;
        }

        Edge previous_edge =
            nodes_[previous_location.node].edges[previous_location.first_edge_letter.value()];
        location_.node = previous_location.node;
        location_.delta = previous_location.delta;
        location_.first_edge_letter = previous_location.first_edge_letter;

        if (location_.IsRoot()) {
            ++previous_edge.text_interval_begin;
            location_.first_edge_letter = text_[previous_edge.text_interval_begin];
            --location_.delta;
        } else {
            location_.node = nodes_[location_.node].link.value();
        }

        Edge implicit_edge = location_.GetImplicitEdge();

        while (not implicit_edge.IsLeaf() and implicit_edge.Length() <= location_.delta) {

            previous_edge.text_interval_begin += implicit_edge.Length();
            location_.delta -= implicit_edge.Length();
            location_.node = implicit_edge.child_node.value();
            location_.first_edge_letter = text_[previous_edge.text_interval_begin];
            implicit_edge = location_.GetImplicitEdge();
        }
    }
};

}  // namespace suffix_tree

namespace suffix_array {

class RecursiveState {
public:
    std::vector<int32_t> padded_string;
    std::vector<int32_t> one_two_triplet_ranks;

    explicit RecursiveState(const std::vector<int32_t> &string) : padded_string{string} {
        padded_string.resize(string.size() + 3);
    }

    enum class CompareResult { Less, Equal, Greater };

    [[nodiscard]] inline CompareResult SingleLetterCompare(int32_t left, int32_t right) const {
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

    [[nodiscard]] inline int32_t ConvertArgToOneTwoRank(int32_t arg) const {
        //        if (arg % 3 == 0) {
        //            throw std::invalid_argument{"Not from one or two mod 3 group."};
        //        }
        auto [quot, rem] = std::div(arg, 3);
        if (rem == 1) {
            return one_two_triplet_ranks[quot];
        } else {
            auto twos_start = (one_two_triplet_ranks.size() + 1) / 2;
            return one_two_triplet_ranks[twos_start + quot];
        }
    }

    [[nodiscard]] inline bool TripletGroupOneTwoCompare(int32_t left, int32_t right) const {
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

    [[nodiscard]] bool GroupOneTwoToZeroCompare(int32_t left, int32_t right) const {
        auto compare_result = SingleLetterCompare(left, right);
        if (compare_result != CompareResult::Equal) {
            return compare_result == CompareResult::Less;
        }
        ++left;
        ++right;
        if (left % 3 == 2) {
            return TripletGroupOneTwoCompare(left, right);
        }
        compare_result = SingleLetterCompare(left, right);
        if (compare_result != CompareResult::Equal) {
            return compare_result == CompareResult::Less;
        }
        return TripletGroupOneTwoCompare(left + 1, right + 1);
    }
};

template <typename String>
std::vector<int32_t> StableArgRadixSortEqualLengthStrings(const std::vector<String> &strings,
                                                          int32_t alphabet_size) {
    sort::StableArgRadixSorter sorter;
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

        auto indices = sort::StableArgCountSort(string, max + 1);
        string = utils::SoftRank(string, indices);
        return BuildFromPositiveValues(string);
    }

    std::vector<int32_t> BuildFromPositiveValues(const std::vector<int32_t> &string) {
        return UnPadEofMarker(BuildRecursivePadded(PadStringWithEofMarker(string)));
    }

private:
    sort::StableArgCountSorter count_sorter_;

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

        SortZeroIndices(&zero_indices, state);

        return utils::MergeSorted(one_two_indices, zero_indices,
                                  [&state](int32_t left, int32_t right) -> bool {
                                      return state.GroupOneTwoToZeroCompare(left, right);
                                  });
    }

    void SortZeroIndices(std::vector<int32_t> *zero_indices, const RecursiveState &state) {
        //        std::vector<std::array<int32_t, 2>> zero_char_and_next_rank;
        //        zero_char_and_next_rank.reserve(zero_indices->size());
        //        int32_t max_char = 0;
        //        for (auto i : *zero_indices) {
        //            max_char = std::max(max_char, state.padded_string[i]);
        //            zero_char_and_next_rank.push_back(
        //                {state.padded_string[i], state.ConvertArgToOneTwoRank(i + 1)});
        //        }
        //        arg_radix_sorter_.SortEqualLengthStrings(zero_char_and_next_rank,
        //                         static_cast<int32_t>(state.padded_string.size()));
        //
        //        auto zi = utils::Take(*zero_indices, arg_radix_sorter_.sorted_indices);
        //        std::vector<int32_t> zero_buffer;
        //        zero_buffer.reserve(zero_indices.size());
        //        for (auto i : zero_indices) {
        //            zero_buffer.emplace_back(state.ConvertArgToOneTwoRank(i + 1));
        //        }
        auto alphabet_size = static_cast<int32_t>(state.padded_string.size());
        auto size = static_cast<int32_t>(zero_indices->size());

        count_sorter_.Sort(size, alphabet_size, [&state, &zero_indices](int32_t i) {
            return state.ConvertArgToOneTwoRank((*zero_indices)[i] + 1);
        });
        *zero_indices = utils::Take(*zero_indices, count_sorter_.sorted_indices);
        count_sorter_.Sort(size, alphabet_size, [&state, &zero_indices](int32_t i) {
            return state.padded_string[(*zero_indices)[i]];
        });
        *zero_indices = utils::Take(*zero_indices, count_sorter_.sorted_indices);
    }

    static std::vector<std::array<int32_t, 3>> BuildOneTwoTriples(
        const std::vector<int32_t> &padded_string) {
        std::vector<std::array<int32_t, 3>> triples;
        auto unpadded_size = static_cast<int32_t>(padded_string.size() - 3);
        triples.reserve((unpadded_size + 1) * 2 / 3);
        for (int32_t i = 1; i < unpadded_size; i += 3) {
            triples.push_back(GetTriple(padded_string, i));
        }
        for (int32_t i = 2; i < unpadded_size; i += 3) {
            triples.emplace_back(GetTriple(padded_string, i));
        }
        return triples;
    }

    static inline std::array<int32_t, 3> GetTriple(const std::vector<int32_t> &string,
                                                   int32_t index) {
        return {string[index], string[index + 1], string[index + 2]};
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
    std::string::const_iterator begin_;
    std::string::const_iterator end_;

public:
    const std::vector<int32_t> suffix_array;
    const std::vector<int32_t> inverse_suffix_array;

    explicit LongestCommonPrefixComputer(const std::string &string)
        : string_{string},
          begin_{string.begin()},
          end_{string.end()},
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

    [[nodiscard]] inline int32_t FindLongestCommonPrefixNotLessThan(int32_t first_suffix,
                                                                    int32_t second_suffix,
                                                                    int32_t min_length) const {
        for (auto first = begin_ + first_suffix + min_length,
                  second = begin_ + second_suffix + min_length;
             first != end_ and second != end_ and *first == *second;
             ++first, ++second, ++min_length) {
        }
        return min_length;
    }
};

class StringConverter {
public:
    const std::vector<std::string> &strings;
    std::vector<int32_t> super_string_pos_to_string_id;

    explicit StringConverter(const std::vector<std::string> &strings) : strings{strings} {
        cumulative_string_sizes_.emplace_back(0);
        super_string_pos_to_string_id.reserve(utils::TotalStringsSize(strings));
        int32_t string_id = 0;
        for (auto &string : strings) {
            cumulative_string_sizes_.emplace_back(cumulative_string_sizes_.back() + string.size() +
                                                  1);
            super_string_pos_to_string_id.resize(cumulative_string_sizes_.back(), string_id);
            ++string_id;
        }
    }

    [[nodiscard]] std::string ConcatenateStrings() const {
        std::ostringstream ss;
        auto buffer = '\0';
        for (auto &string : strings) {
            ss << string << buffer;
            ++buffer;
        }
        return ss.str();
    }

    [[nodiscard]] int32_t ClipLcp(int32_t position, int32_t lcp) const {
        auto string_id = super_string_pos_to_string_id[position];
        return std::min(lcp, cumulative_string_sizes_[string_id + 1] - position);
    }

private:
    std::vector<int32_t> cumulative_string_sizes_;
};

class StringIdCounter {
public:
    std::vector<int32_t> counts;
    bool has_all_strings = false;

    explicit StringIdCounter(int32_t n_strings) : counts(n_strings) {
    }

    inline void Insert(int32_t string_id) {
        ++counts[string_id];
        if (counts[string_id] == 1) {
            has_all_strings = std::all_of(counts.begin(), counts.end(),
                                          [](int32_t count) -> bool { return count != 0; });
        }
    }

    inline void Remove(int32_t string_id) {
        --counts[string_id];
        if (counts[string_id] == 0) {
            has_all_strings = false;
        }
    }

    [[nodiscard]] inline bool HasAllStrings() const {
        return has_all_strings;
    }
};

class SortedSuffixesWindow {
public:
    StringConverter converter;
    std::string super_string;
    LongestCommonPrefixComputer lcp_computer;
    StringIdCounter counter;
    int32_t begin = 0;
    int32_t end = 0;

    explicit SortedSuffixesWindow(const std::vector<std::string> &strings)
        : converter{strings},
          super_string{converter.ConcatenateStrings()},
          lcp_computer{super_string},
          counter{static_cast<int32_t>(strings.size())},
          begin{static_cast<int32_t>(strings.size())},
          end{static_cast<int32_t>(strings.size())} {
    }

    inline void IncrementBegin() {
        counter.Remove(converter.super_string_pos_to_string_id[lcp_computer.suffix_array[begin]]);
        ++begin;
    }

    inline void DecrementBegin() {
        --begin;
        counter.Insert(converter.super_string_pos_to_string_id[lcp_computer.suffix_array[begin]]);
    }

    inline void IncrementEnd() {
        counter.Insert(converter.super_string_pos_to_string_id[lcp_computer.suffix_array[end]]);
        ++end;
    }

    [[nodiscard]] inline int32_t GetBeginStringPosition() const {
        return lcp_computer.suffix_array[begin];
    }

    [[nodiscard]] inline int32_t GetLastStringPosition() const {
        return lcp_computer.suffix_array[end - 1];
    }
};

io::Output Solve(const io::Input &input) {
    io::Output output;
    if (input.strings.size() == 1) {
        output.largest_common_substring = input.strings.front();
        return output;
    }

    SortedSuffixesWindow window{input.strings};
    int32_t max_lcp = 0;
    int32_t max_begin = 0;
    while (window.end <= static_cast<int32_t>(window.super_string.size())) {
        if (not window.counter.HasAllStrings()) {
            if (window.end == static_cast<int32_t>(window.super_string.size())) {
                break;
            }
            window.IncrementEnd();
            continue;
        }
        while (window.counter.HasAllStrings()) {
            window.IncrementBegin();
        }
        window.DecrementBegin();

        auto window_start_pos = window.GetBeginStringPosition();
        auto window_last_pos = window.GetLastStringPosition();
        auto lcp = window.lcp_computer.GetLongestCommonPrefix(window_start_pos, window_last_pos);
        //        lcp = std::min(window.converter.ClipLcp(window_start_pos, lcp),
        //                       window.converter.ClipLcp(window_last_pos, lcp));
        if (lcp > max_lcp) {
            max_lcp = lcp;
            max_begin = window_start_pos;
        }
        window.IncrementBegin();
    }

    output.largest_common_substring = {window.super_string.begin() + max_begin,
                                       window.super_string.begin() + max_begin + max_lcp};
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
    size_t total_size = 0;
    for (auto &s : input.strings) {
        total_size += s.size();
    }
    if (total_size > 10'000) {
        throw NotImplementedError{};
    }
    io::Output output;
    auto smallest_string = std::min_element(input.strings.begin(), input.strings.end());
    for (auto smallest_begin = smallest_string->begin(); smallest_begin != smallest_string->end();
         ++smallest_begin) {
        auto min_common = static_cast<int32_t>(smallest_string->size());
        for (auto &string : input.strings) {
            auto max_pairwise_common = 0;
            for (auto begin = string.begin(); begin != string.end(); ++begin) {
                max_pairwise_common =
                    std::max(max_pairwise_common,
                             FindLongestCommonPrefix(smallest_begin, smallest_string->end(), begin,
                                                     string.end()));
            }
            min_common = std::min(min_common, max_pairwise_common);
        }
        if (min_common > static_cast<int32_t>(output.largest_common_substring.size())) {
            output.largest_common_substring = {smallest_begin, smallest_begin + min_common};
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

void TestSuffixTree(int32_t test_case_id) {
    auto string_size = test_case_id * test_case_id;
    auto string = GenerateRandomString(string_size, 'a', 'c');
    suffix_tree::SuffixTree suffix_tree{string};
    std::uniform_int_distribution<int32_t> word_sizes{1, 1 + test_case_id};

    for (int32_t i = 0; i <= test_case_id; ++i) {
        auto word = GenerateRandomString(word_sizes(*rng::GetEngine()), 'a', 'c');
        auto tree_search = suffix_tree.Search(word);
        auto search = string.find(word) != std::string::npos;
        if (tree_search != search) {
            tree_search = suffix_tree.Search(word);
            throw WrongAnswerException{};
        }
    }
}

void TestSuffixArray(int32_t test_case_id) {
    auto string_size = test_case_id + 1;
    auto string = GenerateRandomString(string_size, 'a', 'c');
    std::vector<std::string> suffixes;
    for (auto it = string.begin(); it != string.end(); ++it) {
        suffixes.emplace_back(it, string.end());
    }

    auto sorted_args = utils::ArgSortByValue(suffixes.begin(), suffixes.end());
    auto suffix_array = suffix_array::BuildSuffixArray(string);
    if (sorted_args != suffix_array) {
        suffix_array::BuildSuffixArray(string);
        throw WrongAnswerException{};
    }
}

void TestCountSort(int32_t test_case_id) {
    size_t size = test_case_id * test_case_id;
    auto max = test_case_id * 2;
    std::uniform_int_distribution<int32_t> distribution{0, max};
    std::vector<int32_t> values;
    while (values.size() < size) {
        values.emplace_back(distribution(*rng::GetEngine()));
    }
    auto indices = sort::StableArgCountSort(values, max + 1);
    std::vector<int32_t> sorted_values;
    for (auto i : indices) {
        sorted_values.push_back(values[i]);
    }
    //    auto sorted_values = values;
    std::sort(values.begin(), values.end());
    //    sort::UnStableCountSorter{}.Sort(&sorted_values, max + 1, [](auto i) { return i; });
    if (sorted_values != values) {
        sort::StableArgCountSort(values, max + 1);
        throw WrongAnswerException{};
    }
}

void TestRadixSortStableArgSort(int32_t test_case_id) {
    auto string_size = test_case_id + 1;
    auto alphabet_size = 127;
    size_t n_substrings = test_case_id;
    int32_t substring_size = string_size / 2;
    auto string = GenerateRandomString(string_size + substring_size, 'a', 'c');
    std::uniform_int_distribution<int32_t> distribution{0, string_size};
    std::vector<std::string> strings;
    while (strings.size() < n_substrings) {
        auto start = distribution(*rng::GetEngine());
        strings.emplace_back(string.begin() + start, string.begin() + start + substring_size);
    }
    auto indices = sort::StableArgRadixSortEqualLengthStrings(strings, alphabet_size);
    auto sorted_strings = utils::Take(strings, indices);
    auto expected_sorted_strings = strings;
    std::sort(expected_sorted_strings.begin(), expected_sorted_strings.end());
    if (sorted_strings != expected_sorted_strings) {
        sort::StableArgRadixSortEqualLengthStrings(strings, alphabet_size);
        throw WrongAnswerException{};
    }
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t n_strings = std::min(10, 1 + test_case_id / 10);
    int32_t substring_size = std::min(200'000, 1 + test_case_id / 10);
    auto max_char = 'c';

    std::uniform_int_distribution<int32_t> string_size_dist{0, substring_size};
    auto common_string = GenerateRandomString(substring_size, 'a', max_char);

    io::Input input;
    input.strings.resize(n_strings);
    for (auto &string : input.strings) {
        string = GenerateRandomString(string_size_dist(*rng::GetEngine()), 'a', max_char) +
                 common_string +
                 GenerateRandomString(string_size_dist(*rng::GetEngine()), 'a', max_char);
    }
    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    return GenerateRandomTestIo(2'000'000);
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
        "3\n"
        "abacaba\n"
        "mycabarchive\n"
        "acabistrue",
        "cab");

    std::cerr << "Basic tests OK:\n" << timed_checker;

    int32_t n_random_test_cases = 100;

    try {

        for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {
            TestSuffixTree(test_case_id);
            TestRadixSortStableArgSort(test_case_id);
            TestCountSort(test_case_id);
            TestSuffixArray(test_case_id);
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
