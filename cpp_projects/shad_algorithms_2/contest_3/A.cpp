#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <climits>
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
    std::string burrows_wheeler_transformed_string;

    Output() = default;

    explicit Output(std::string string) : burrows_wheeler_transformed_string{std::move(string)} {
    }

    std::ostream &Write(std::ostream &out) const {
        out << burrows_wheeler_transformed_string;
        return out;
    }

    bool operator!=(const Output &other) const {
        return burrows_wheeler_transformed_string != other.burrows_wheeler_transformed_string;
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

template <typename T, typename I>
std::vector<T> Take(const std::vector<T> &values, const std::vector<I> &indices) {
    std::vector<T> slice;
    slice.reserve(values.size());
    for (auto i : indices) {
        slice.emplace_back(values[i]);
    }
    return slice;
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

class StringStats {
public:
    explicit StringStats(const std::string &string) : char_at_position_occurrence_(string.size()) {
        for (int32_t i = 0; i < static_cast<int32_t>(string.size()); ++i) {
            auto &occurrences = char_nth_occurrence_position_[string[i]];
            char_at_position_occurrence_[i] = static_cast<int32_t>(occurrences.size());
            occurrences.push_back(i);
        }
    }

    [[nodiscard]] std::optional<int32_t> Select(char character, int32_t occurrence) const {
        return GetNthCharacterOccurrencePosition(character, occurrence);
    }

    [[nodiscard]] std::optional<int32_t> Start(char character) const {
        return GetNthCharacterOccurrencePosition(character, 0);
    }

    [[nodiscard]] int32_t Rank(int32_t position) const {
        return char_at_position_occurrence_[position];
    }

    [[nodiscard]] std::optional<int32_t> GetNthCharacterOccurrencePosition(
        char character, int32_t occurrence) const {
        auto &positions = char_nth_occurrence_position_.at(character);
        if (occurrence < static_cast<int32_t>(positions.size())) {
            return positions[occurrence];
        } else {
            return std::nullopt;
        }
    }

private:
    std::unordered_map<char, std::vector<int32_t>> char_nth_occurrence_position_;
    std::vector<int32_t> char_at_position_occurrence_;
};

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
    std::map<char, Edge> edges;
    std::optional<int32_t> link;

    Node() : link{0} {
    }

    Node(char letter, int32_t letter_position) : edges{{letter, Edge{letter_position}}} {
    }
};

class Location {
public:
    const std::string &text_;
    const std::vector<Node> &nodes_;
    int32_t node = 0;
    int32_t delta = 0;
    std::optional<char> first_edge_letter;

    Location(const std::string &text, const std::vector<Node> &nodes) : text_{text}, nodes_{nodes} {
    }

    [[nodiscard]] bool IsExplicit() const {
        return delta == 0;
    }

    [[nodiscard]] bool IsRoot() const {
        return node == 0;
    }

    [[nodiscard]] Edge GetImplicitEdge() const {
        return nodes_[node].edges.at(first_edge_letter.value());
    }

    [[nodiscard]] char GetNextImplicitLetter() const {
        return text_[GetImplicitEdge().text_interval_begin + delta];
    }

    [[nodiscard]] bool IsFirstImplicitNodeOnTheEdgeALeaf(Edge edge) const {
        return edge.text_interval_begin + delta == static_cast<int32_t>(text_.size());
    }

    [[nodiscard]] bool CanDescendByLetter(char letter) const {
        if (IsAtTextEnd()) {
            return false;
        } else if (IsExplicit()) {
            return nodes_[node].edges.count(letter);
        } else {
            return GetNextImplicitLetter() == letter;
        }
    }

    [[nodiscard]] bool IsAtTextEnd() const {
        if (IsExplicit()) {
            return false;
        } else {
            return GetImplicitEdge().text_interval_begin + delta ==
                   static_cast<int32_t>(text_.size());
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
            for (auto c : nodes_[node].edges) {
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

template <typename T>
void CheckThatValueIsInAlphabet(T value, int32_t alphabet_size) {
    if (value < 0 or alphabet_size <= value) {
        throw std::invalid_argument{"Value must be non negative and not more than alphabet size."};
    }
}

class StableArgCountSorter {
public:
    std::vector<int32_t> sorted_indices;

    template <typename Container>
    void Sort(const Container &values, int32_t alphabet_size) {
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
            for (int32_t nth = strings[0].size() - 1; nth >= 0; --nth) {
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

class SuffixArrayBuilder {
public:
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

        std::vector<std::array<int32_t, 2>> zero_char_and_next_rank;
        zero_char_and_next_rank.reserve(zero_indices.size());
        for (auto i : zero_indices) {
            zero_char_and_next_rank.push_back(
                {state.padded_string[i], state.ConvertArgToOneTwoRank(i + 1)});
        }
        arg_radix_sorter_.SortEqualLengthStrings(zero_char_and_next_rank,
                                                 static_cast<int32_t>(string.size()));
        zero_indices = utils::Take(zero_indices, arg_radix_sorter_.sorted_indices);
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
    std::vector<int32_t> vector_string;
    vector_string.reserve(string.size());
    for (auto &i : string) {
        vector_string.emplace_back(i - 'a' + 1);
    }
    return SuffixArrayBuilder{}.BuildFromPositiveValues(vector_string);
}

}  // namespace suffix_array

std::vector<int32_t> ArgSortCyclicShifts(const std::string &string) {
    auto double_string = string + string;
    auto double_suffix_array = suffix_array::BuildSuffixArray(double_string);

    std::vector<int32_t> suffix_array;
    suffix_array.reserve(string.size());
    for (auto i : double_suffix_array) {
        if (i < static_cast<int32_t>(string.size())) {
            suffix_array.emplace_back(i);
        }
    }
    return suffix_array;
}

char GetPreviousCharInCyclicString(const std::string &string, int32_t index) {
    auto size = static_cast<int32_t>(string.size());
    return string[(index - 1 + size) % size];
}

std::string BurrowsWheelerTransform(const std::string &string) {
    auto cyclic_shifts_indices = ArgSortCyclicShifts(string);
    auto transformed_string = string;
    for (size_t i = 0; i < string.size(); ++i) {
        transformed_string[i] = GetPreviousCharInCyclicString(string, cyclic_shifts_indices[i]);
    }
    return transformed_string;
}

io::Output Solve(const io::Input &input) {
    return io::Output{BurrowsWheelerTransform(input.string)};
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

    if (input.string.size() > 10'000) {
        throw NotImplementedError{};
    }

    std::vector<std::string> string_cycle_shifts(input.string.size());
    auto double_string = input.string + input.string;

    for (int32_t i = 0; i < static_cast<int32_t>(input.string.size()); ++i) {
        string_cycle_shifts[i] = {
            double_string.begin() + i,
            double_string.begin() + i + static_cast<int32_t>(input.string.size())};
    }

    std::sort(string_cycle_shifts.begin(), string_cycle_shifts.end());

    io::Output output;
    for (auto &string : string_cycle_shifts) {
        output.burrows_wheeler_transformed_string += string.back();
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
    auto indices = StableArgCountSort(values, max + 1);
    std::vector<int32_t> sorted_values;
    for (auto i : indices) {
        sorted_values.push_back(values[i]);
    }
    std::sort(values.begin(), values.end());
    if (sorted_values != values) {
        StableArgCountSort(values, max + 1);
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
    auto indices = StableArgRadixSortEqualLengthStrings(strings, alphabet_size);
    auto sorted_strings = utils::Take(strings, indices);
    auto expected_sorted_strings = strings;
    std::sort(expected_sorted_strings.begin(), expected_sorted_strings.end());
    if (sorted_strings != expected_sorted_strings) {
        StableArgRadixSortEqualLengthStrings(strings, alphabet_size);
        throw WrongAnswerException{};
    }
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

    timed_checker.Check("ababc", "cbaab");
    timed_checker.Check("a", "a");
    timed_checker.Check("aaaaa", "aaaaa");
    timed_checker.Check("abcde", "eabcd");

    std::cerr << "Basic tests OK:\n" << timed_checker;

    int32_t n_random_test_cases = 100;

    try {

        for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {
            timed_checker.Check(GenerateRandomTestIo(test_case_id));
            TestSuffixTree(test_case_id);
            TestSuffixArray(test_case_id);
            TestCountSort(test_case_id);
            TestRadixSortStableArgSort(test_case_id);
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
