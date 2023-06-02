#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <list>
#include <queue>
#include <random>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

template <typename IRadixCont>
void RadixSort(IRadixCont& container, const std::vector<uint32_t>& digits_count_by_rank) {
    IRadixCont output(container.size());
    for (uint32_t rank = 0; rank < digits_count_by_rank.size(); ++rank) {
        auto digits_count = digits_count_by_rank[rank];
        std::vector<uint32_t> counts(digits_count);

        for (const auto& radix_object : container) {
            ++counts[radix_object.GetDigit(rank)];
        }

        for (uint32_t digit = 1; digit < digits_count; ++digit) {
            counts[digit] += counts[digit - 1];
        }

        for (auto it = container.rbegin(); it != container.rend(); ++it) {
            auto index = --counts[it->GetDigit(rank)];
            output[index] = std::move(*it);
        }

        std::swap(output, container);
    }
}

struct Triplex {
    std::array<uint32_t, 3> ranks;
    uint32_t pos;

    Triplex() = default;
    Triplex(uint32_t pos, const std::vector<uint32_t>* data) : pos(pos) {
        for (int ii = 0; ii < 3; ++ii) {
            ranks[2 - ii] = pos + ii < data->size() ? (*data)[pos + ii] : 0;
        }
    }

    uint32_t GetDigit(uint32_t rank) const {
        return ranks[rank];
    }

    bool operator==(const Triplex& other) const {
        return ranks == other.ranks;
    }
};

struct String {
    std::vector<uint32_t> data;
    uint32_t alphabet_size;
};

String EnumerateTriplexes(const String& string) {
    std::vector<Triplex> triplexes;
    for (uint32_t pos = 0; pos < string.data.size(); ++pos) {
        triplexes.emplace_back(pos, &string.data);
    }

    RadixSort(triplexes,
              {string.alphabet_size + 1, string.alphabet_size + 1, string.alphabet_size + 1});

    uint32_t current_index = 0;
    std::vector<uint32_t> indexes(string.data.size(), current_index);
    for (uint32_t ii = 1; ii < triplexes.size(); ++ii) {
        current_index += !(triplexes[ii - 1] == triplexes[ii]);
        indexes[triplexes[ii].pos] = current_index;
    }
    return {indexes, current_index + 1};
}

String BuildS0S1(const String& triplexes_labels) {
    std::vector<uint32_t> s0_s1_data;
    for (uint32_t pos = 0; pos < triplexes_labels.data.size(); pos += 3) {
        s0_s1_data.push_back(triplexes_labels.data[pos] + 1);
    }
    s0_s1_data.push_back(0);
    for (uint32_t pos = 1; pos < triplexes_labels.data.size(); pos += 3) {
        s0_s1_data.push_back(triplexes_labels.data[pos] + 1);
    }
    return {s0_s1_data, triplexes_labels.alphabet_size + 1};
}

std::vector<uint32_t> BuildSuffixArray(const String& string);

std::vector<uint32_t> BuildS0S1SuffixArray(const String& triplexes_labels) {
    auto suffix_array = BuildSuffixArray(BuildS0S1(triplexes_labels));

    std::vector<uint32_t> result;
    result.reserve(suffix_array.size() - 1);
    for (uint32_t pos = 1; pos < suffix_array.size(); ++pos) {
        if (suffix_array[pos] < suffix_array[0]) {
            result.push_back(suffix_array[pos] * 3);
        } else {
            result.push_back((suffix_array[pos] - suffix_array[0] - 1) * 3 + 1);
        }
    }

    return result;
}

struct SuffixComparer {
    SuffixComparer(const String* string, const std::vector<uint32_t>& s0s1_suffix_array)
        : string(string), s0s1_rank_array(string->data.size(), -1) {
        int current_index = 0;
        for (auto suffix_index : s0s1_suffix_array) {
            s0s1_rank_array[suffix_index] = current_index++;
        }
        s0s1_alphabet_size = current_index;
    }

    bool Less(uint32_t lhs_suffix_index, uint32_t rhs_suffix_index) const {
        if (s0s1_rank_array[lhs_suffix_index] >= 0 && s0s1_rank_array[rhs_suffix_index] >= 0) {
            return s0s1_rank_array[lhs_suffix_index] < s0s1_rank_array[rhs_suffix_index];
        }
        if (string->data[lhs_suffix_index] == string->data[rhs_suffix_index]) {
            if (++lhs_suffix_index == string->data.size()) {
                return true;
            }
            if (++rhs_suffix_index == string->data.size()) {
                return false;
            }
            return Less(lhs_suffix_index, rhs_suffix_index);
        }
        return string->data[lhs_suffix_index] < string->data[rhs_suffix_index];
    }

    const String* string;
    std::vector<int> s0s1_rank_array;
    uint32_t s0s1_alphabet_size;
};

struct S2Suffix {
    S2Suffix() = default;
    S2Suffix(uint32_t pos, const SuffixComparer* suffix_comparer) : pos(pos) {
        ranks[0] = pos + 1 < suffix_comparer->s0s1_rank_array.size()
                       ? suffix_comparer->s0s1_rank_array[pos + 1] + 1
                       : 0;
        ranks[1] = suffix_comparer->string->data[pos];
    }

    uint32_t GetDigit(uint32_t rank) const {
        return ranks[rank];
    }

    uint32_t pos;
    std::array<uint32_t, 2> ranks;
};

std::vector<uint32_t> BuildS2SuffixArray(const SuffixComparer& suffix_comparer) {
    auto& string = *suffix_comparer.string;
    std::vector<S2Suffix> s2_suffixes;
    for (uint32_t pos = 2; pos < string.data.size(); pos += 3) {
        s2_suffixes.emplace_back(pos, &suffix_comparer);
    }
    RadixSort(s2_suffixes,
              {suffix_comparer.s0s1_alphabet_size + 1, suffix_comparer.string->alphabet_size});

    std::vector<uint32_t> s2_suffix_array;
    for (const auto& suffix : s2_suffixes) {
        s2_suffix_array.push_back(suffix.pos);
    }
    return s2_suffix_array;
}

std::vector<uint32_t> BuildSuffixArrayNaive(const std::vector<uint32_t>& string) {
    std::vector<std::vector<uint32_t>> suffixes;
    for (auto start = string.begin(); start != string.end(); ++start) {
        suffixes.emplace_back(start, string.end());
    }
    std::sort(suffixes.begin(), suffixes.end());

    std::vector<uint32_t> suffix_array;
    for (const auto& suffix : suffixes) {
        suffix_array.push_back(string.size() - suffix.size());
    }
    return suffix_array;
}

std::vector<uint32_t> BuildSuffixArray(const String& string) {
    if (string.data.size() < 10) {
        return BuildSuffixArrayNaive(string.data);
    }

    auto triplexes_labels = EnumerateTriplexes(string);
    auto s0s1_suffix_array = BuildS0S1SuffixArray(triplexes_labels);
    auto suffix_comparer = SuffixComparer(&string, s0s1_suffix_array);
    auto s2_suffix_array = BuildS2SuffixArray(suffix_comparer);

    std::vector<uint32_t> suffix_array(string.data.size());
    std::merge(
        s0s1_suffix_array.begin(), s0s1_suffix_array.end(), s2_suffix_array.begin(),
        s2_suffix_array.end(), suffix_array.begin(),
        [&suffix_comparer](uint32_t lhs, uint32_t rhs) { return suffix_comparer.Less(lhs, rhs); });

    return suffix_array;
}

std::vector<uint32_t> BuildRankArray(const std::vector<uint32_t>& suffix_array) {
    int current_index = 0;
    std::vector<uint32_t> rank_array(suffix_array.size());
    for (auto suffix_index : suffix_array) {
        rank_array[suffix_index] = current_index++;
    }
    return rank_array;
}

std::vector<uint32_t> BuildLCPArray(const String& string, const std::vector<uint32_t>& suffix_array,
                                    const std::vector<uint32_t>& rank_array) {
    std::vector<uint32_t> lcp(string.data.size() - 1);
    uint32_t current_lcp = 0;
    for (uint32_t suffix_a = 0; suffix_a < string.data.size(); ++suffix_a) {
        if (current_lcp > 0) {
            --current_lcp;
        }
        auto rank = rank_array[suffix_a];
        if (rank < lcp.size()) {
            auto suffix_b = suffix_array[rank + 1];
            while (suffix_a + current_lcp < string.data.size() &&
                   suffix_b + current_lcp < string.data.size() &&
                   string.data[suffix_a + current_lcp] == string.data[suffix_b + current_lcp]) {
                ++current_lcp;
            }
            lcp[rank] = current_lcp;
        }
    }
    return lcp;
}

String BuildString(const std::vector<std::string>& strings) {
    String result{{}, 27};
    uint32_t total_size = 0;
    for (auto& string : strings) {
        total_size += string.length() + 1;
    }
    result.data.reserve(total_size);
    for (auto& string : strings) {
        for (auto symbol : string) {
            result.data.push_back(symbol - 'a' + 1);
        }
        result.data.push_back(0);
    }

    result.data.pop_back();

    return result;
}

std::vector<std::string> ReadStrings(std::istream& input) {
    uint32_t count;
    input >> count;
    std::vector<std::string> strings(count);
    for (auto& string : strings) {
        input >> string;
    }
    return strings;
}

class MinStack {
public:
    void Push(uint32_t val) {
        data_.push(val);
        if (minimums_.empty() || val < minimums_.top()) {
            minimums_.push(val);
        } else {
            minimums_.push(minimums_.top());
        }
    }

    uint32_t Pop() {
        auto val = data_.top();
        data_.pop();
        minimums_.pop();
        return val;
    }

    uint32_t Size() const {
        return data_.size();
    }

    bool Empty() const {
        return data_.empty();
    }

    uint32_t Min() const {
        return minimums_.top();
    }

private:
    std::stack<uint32_t, std::vector<uint32_t>> data_;
    std::stack<uint32_t, std::vector<uint32_t>> minimums_;
};

class MinQueue {
public:
    void Push(uint32_t val) {
        tail_.Push(val);
    }

    uint32_t Pop() {
        if (head_.Empty()) {
            MoveTailToHead();
        }
        return head_.Pop();
    }

    uint32_t Size() const {
        return tail_.Size() + head_.Size();
    }

    bool Empty() const {
        return tail_.Empty() && head_.Empty();
    }

    uint32_t Min() const {
        if (tail_.Empty()) {
            return head_.Min();
        }
        if (head_.Empty()) {
            return tail_.Min();
        }
        return std::min(tail_.Min(), head_.Min());
    }

private:
    void MoveTailToHead() {
        while (!tail_.Empty()) {
            head_.Push(tail_.Pop());
        }
    }

private:
    MinStack tail_;
    MinStack head_;
};

class SuffixesCounter {
public:
    explicit SuffixesCounter(const std::vector<std::string>& strings) {
        for (auto& string : strings) {
            auto start = finishes_.empty() ? 0 : finishes_.back() + 1;
            finishes_.push_back(start + string.size());
        }
    }

    void Push(uint32_t index) {
        ++counts_[StringId(index)];
    }

    void Pop(uint32_t index) {
        auto string_id = StringId(index);
        if (--counts_[string_id] == 0) {
            counts_.erase(string_id);
        }
    }

    bool AllStringsCounted() const {
        return counts_.size() == finishes_.size();
    }

    std::pair<uint32_t, uint32_t> StringIdAndSymbolId(uint32_t index) const {
        auto string_id = StringId(index);
        auto start = string_id == 0 ? 0 : finishes_[string_id - 1] + 1;
        return {string_id, index - start};
    }

private:
    uint32_t StringId(uint32_t index) const {
        return std::upper_bound(finishes_.begin(), finishes_.end(), index) - finishes_.begin();
    }

    std::vector<uint32_t> finishes_;
    std::unordered_map<uint32_t, uint32_t> counts_;
};

std::string LCS(const std::vector<std::string>& strings) {
    if (strings.size() == 1) {
        return strings[0];
    }
    std::vector<uint32_t> suffix_array;
    std::vector<uint32_t> lcp_array;
    {
        auto string = BuildString(strings);
        suffix_array = BuildSuffixArray(string);
        auto rank_array = BuildRankArray(suffix_array);
        lcp_array = BuildLCPArray(string, suffix_array, rank_array);
    }

    MinQueue lcp_queue;
    SuffixesCounter counter(strings);
    auto left = strings.size() - 1;
    auto right = strings.size() - 1;
    counter.Push(suffix_array[left]);

    uint32_t best_length = 0;
    uint32_t start = suffix_array[left];

    while (right < lcp_array.size()) {
        if (counter.AllStringsCounted()) {
            if (auto current_length = lcp_queue.Min(); current_length > best_length) {
                best_length = current_length;
                start = suffix_array[left];
            }
            lcp_queue.Pop();
            counter.Pop(suffix_array[left]);
            ++left;
        } else {
            lcp_queue.Push(lcp_array[right]);
            counter.Push(suffix_array[++right]);
        }
    }

    if (counter.AllStringsCounted()) {
        if (auto current_length = lcp_queue.Min(); current_length > best_length) {
            best_length = current_length;
            start = suffix_array[left];
        }
    }

    auto [string_id, symbol_id] = counter.StringIdAndSymbolId(start);
    return strings[string_id].substr(symbol_id, best_length);
}

void TimeTest() {
    std::mt19937_64 gen(123);
    std::uniform_int_distribution<char> dist('a', 'z');
    std::vector<std::string> strings(10);
    for (auto& string : strings) {
        string.resize(200'000);
        for (auto& symbol : string) {
            symbol = dist(gen);
        }
    }
    std::cout << "TimeTest" << std::endl;
    auto time_start = std::chrono::system_clock::now();
    std::cout << LCS(strings) << std::endl;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now() - time_start)
                     .count()
              << "ms" << std::endl;
}

int main() {
    //    TimeTest();
    //        malloc_stats();

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    std::cout << LCS(ReadStrings(std::cin));
}
