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
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace io {

class Input {
public:
    std::vector<std::string> source_matrix;
    std::vector<std::string> target_matrix;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t n_rows = 0;
        int32_t n_cols = 0;
        in >> n_rows >> n_cols;
        source_matrix.resize(n_rows);
        for (auto &s : source_matrix) {
            in >> s;
        }

        in >> n_rows >> n_cols;
        target_matrix.resize(n_rows);
        for (auto &s : target_matrix) {
            in >> s;
        }
    }
};

class Output {
public:
    int32_t number_of_target_matrix_occurrences_up_to_one_error_ = 0;

    Output() = default;

    explicit Output(int32_t number) : number_of_target_matrix_occurrences_up_to_one_error_{number} {
    }

    std::ostream &Write(std::ostream &out) const {
        out << number_of_target_matrix_occurrences_up_to_one_error_;
        return out;
    }

    bool operator!=(const Output &other) const {
        return number_of_target_matrix_occurrences_up_to_one_error_ !=
               other.number_of_target_matrix_occurrences_up_to_one_error_;
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

bool AreStringsUnique(const std::vector<std::string> &strings) {
    std::unordered_set<std::string> set{strings.begin(), strings.end()};
    return set.size() == strings.size();
}

size_t TotalStringsSize(const std::vector<std::string> &strings) {
    size_t size = 0;
    for (auto &string : strings) {
        size += string.size();
    }
    return size;
}

std::vector<std::string> MakeUnique(const std::vector<std::string> &strings) {
    std::unordered_set<std::string> unique_strings{strings.begin(), strings.end()};
    return {unique_strings.begin(), unique_strings.end()};
}

}  // namespace utils

namespace prefix_tree {

template <typename GraphTraversal>
void BreadthFirstSearch(GraphTraversal *graph_traversal,
                        std::deque<typename GraphTraversal::Node> starting_nodes_queue) {

    auto &queue = starting_nodes_queue;

    for (size_t i = 0; i < queue.size(); ++i) {
        auto node = queue.front();
        queue.pop_front();

        queue.emplace_back(node);
    }

    while (not queue.empty()) {

        auto node = queue.front();
        queue.pop_front();

        for (const auto &adjacent_node : graph_traversal->NodesAdjacentTo(node)) {

            if (graph_traversal->ShouldTraverseEdge(node, adjacent_node)) {

                graph_traversal->OnEdgeTraverse(node, adjacent_node);
                queue.emplace_back(adjacent_node);
            }
        }
    }
}

struct PrefixTreeNode {
    std::optional<int32_t> string_id_that_terminates_here;
    std::unordered_map<char, int32_t> edges;
    int32_t prefix_link = 0;
    int32_t terminal_link = -1;
    std::optional<std::vector<int32_t>> cached_string_occurrences;
    std::unordered_map<char, int32_t> cached_prefix_link_transitions;
};

class PrefixTree {
public:
    explicit PrefixTree(const std::vector<std::string> &strings) {
        if (not utils::AreStringsUnique(strings)) {
            throw std::invalid_argument{"Provided strings are not unique."};
        }
        Initialize(strings);
    }

    void ResetTextIterator() {
        text_iter_node_ = 0;
    }

    [[nodiscard]] size_t GetCurrentSizeOfProcessedText() const {
        return size_of_processed_text_;
    }

    void SendNextTextLetter(char letter) {
        text_iter_node_ = FindNextLongestPrefixNodePrecededByLetterOrRoot(letter, text_iter_node_);
        ++size_of_processed_text_;
    }

    std::vector<int32_t> GetOccurrencesAtCurrentPosition() {
        return ComputeOccurrencesAtNode(text_iter_node_);
    }

    std::vector<int32_t> FindOccurrencesAtEndOfText(const std::string &text) {
        std::vector<int32_t> occurrences;
        ResetTextIterator();
        for (auto c : text) {
            SendNextTextLetter(c);
        }
        return GetOccurrencesAtCurrentPosition();
    }

    const std::vector<int32_t> &GetCachedOccurrencesAtCurrentPosition() {
        return GetCachedOccurrencesAtNode(text_iter_node_);
    }

protected:
    std::vector<PrefixTreeNode> nodes_{1};
    int32_t text_iter_node_ = 0;
    size_t size_of_processed_text_ = 0;

    void Initialize(const std::vector<std::string> &strings) {
        nodes_.reserve(utils::TotalStringsSize(strings) + 1);
        for (int32_t i = 0; i < static_cast<int32_t>(strings.size()); ++i) {
            AddString(strings[i], i);
        }
        ComputePrefixLinks();
        ComputeTerminalLinks();
    }

    [[nodiscard]] static bool IsRoot(int32_t node) {
        return node == 0;
    }

    [[nodiscard]] bool IsTerminal(int32_t node) const {
        return static_cast<bool>(nodes_[node].string_id_that_terminates_here);
    }

    [[nodiscard]] bool IsLeaf(int32_t node) const {
        return nodes_[node].edges.empty();
    }

    void AddString(const std::string &string, int32_t string_id) {
        int32_t node = 0;
        for (auto letter : string) {
            if (not GetNextDirectContinuationNode(letter, node)) {
                AddLetterAtNode(letter, node);
            }
            node = GetNextDirectContinuationNode(letter, node).value();
        }
        nodes_[node].string_id_that_terminates_here = string_id;
    }

    [[nodiscard]] std::optional<int32_t> GetNextDirectContinuationNode(char letter,
                                                                       int32_t node) const {
        auto &edges = nodes_[node].edges;
        auto search = edges.find(letter);
        return search == edges.end() ? std::nullopt : std::optional<int32_t>(search->second);
    }

    void AddLetterAtNode(char letter, int32_t node) {
        auto new_node = static_cast<int32_t>(nodes_.size());
        nodes_[node].edges[letter] = new_node;
        nodes_.emplace_back();
    }

    int32_t FindNextLongestPrefixNodePrecededByLetterOrRoot(char letter, int32_t node) {
        while (not GetNextDirectContinuationNode(letter, node)) {
            if (IsRoot(node)) {
                return 0;
            }
            node = nodes_[node].prefix_link;
        }
        return GetNextDirectContinuationNode(letter, node).value();
    };

    class Traversal {
    public:
        using Node = int32_t;
        using NodeIterable = std::vector<int32_t>;

        PrefixTree *tree = nullptr;

        explicit Traversal(PrefixTree *tree) : tree{tree} {
        }

        bool ShouldTraverseEdge(Node from, Node to) {
            return true;
        }

        NodeIterable NodesAdjacentTo(Node node) {
            std::vector<int32_t> adjacent_nodes;
            for (auto [letter, adjacent_node] : tree->nodes_[node].edges) {
                adjacent_nodes.emplace_back(adjacent_node);
            }
            return adjacent_nodes;
        }

        void OnEdgeTraverse(Node from, Node to) {
            if (PrefixTree::IsRoot(from)) {
                return;
            }
            auto letter = GetEdgeLetter(from, to);
            auto from_prefix = tree->nodes_[from].prefix_link;
            tree->nodes_[to].prefix_link =
                tree->FindNextLongestPrefixNodePrecededByLetterOrRoot(letter, from_prefix);
        }

        [[nodiscard]] char GetEdgeLetter(Node from, Node to) const {
            for (auto [letter, node] : tree->nodes_[from].edges) {
                if (node == to) {
                    return letter;
                }
            }
            throw std::invalid_argument{"No such edge"};
        }
    };

    void ComputePrefixLinks() {
        Traversal traversal{this};
        BreadthFirstSearch(&traversal, {0});
    }

    void ComputeTerminalLinks() {
        for (int32_t node = 0; node < static_cast<int32_t>(nodes_.size()); ++node) {
            nodes_[node].terminal_link = GetTerminalLink(node);
        }
    }

    int32_t GetTerminalLink(int32_t node) {
        if (nodes_[node].terminal_link != -1) {
            return nodes_[node].terminal_link;
        }
        while (not IsRoot(node)) {
            node = nodes_[node].prefix_link;
            if (IsTerminal(node)) {
                return node;
            }
        }
        return 0;
    }

    const std::vector<int32_t> &GetCachedOccurrencesAtNode(int32_t node) {
        auto &cached_occurrences = nodes_[node].cached_string_occurrences;
        if (not cached_occurrences) {
            cached_occurrences = ComputeOccurrencesAtNode(node);
        }
        return cached_occurrences.value();
    }

    std::vector<int32_t> ComputeOccurrencesAtNode(int32_t node) {
        std::vector<int32_t> occurrences;
        while (not IsRoot(node)) {
            if (nodes_[node].cached_string_occurrences) {
                occurrences.insert(occurrences.end(),
                                   nodes_[node].cached_string_occurrences->begin(),
                                   nodes_[node].cached_string_occurrences->end());
                return occurrences;
            }
            if (auto id = nodes_[node].string_id_that_terminates_here) {
                occurrences.emplace_back(id.value());
            }
            node = nodes_[node].terminal_link;
        }
        return occurrences;
    }
};

}  // namespace prefix_tree

template <typename Container>
std::vector<Container> Transpose(const std::vector<Container> &matrix) {
    std::vector<Container> transposed_matrix(matrix.front().size());
    for (auto &r : transposed_matrix) {
        r.resize(matrix.size());
    }
    for (size_t row = 0; row < matrix.size(); ++row) {
        for (size_t col = 0; col < matrix.front().size(); ++col) {
            transposed_matrix[col][row] = matrix[row][col];
        }
    }
    return transposed_matrix;
}

// std::vector<std::string> Transpose(const std::vector<std::string> &matrix) {
//     std::vector<std::string> transposed_matrix(matrix.front().size());
//     transposed_matrix
// }

class ZFunctionComputer {
public:
    const std::vector<int32_t> &string;
    std::vector<int32_t> z_function;

    explicit ZFunctionComputer(const std::vector<int32_t> &string)
        : string{string}, z_function(string.size()), size_{static_cast<int32_t>(string.size())} {
    }

    std::vector<int32_t> Compute() {
        argmax_i_plus_z_i_ = 1;
        z_function[0] = size_;

        for (int32_t index = 1; index < size_; ++index) {
            z_function[index] = CalculateZFunctionAt(index);
        }

        return z_function;
    }

private:
    int32_t argmax_i_plus_z_i_ = 0;
    int32_t size_ = 0;

    [[nodiscard]] int32_t CalculateZFunctionAt(int32_t index) {
        int32_t index_minus_argmax = index - argmax_i_plus_z_i_;
        auto new_max_z_value = std::max(0, z_function[argmax_i_plus_z_i_] - index_minus_argmax);

        if (z_function[index_minus_argmax] < new_max_z_value) {
            return z_function[index_minus_argmax];
        }

        while (index + new_max_z_value < size_ and
               string[new_max_z_value] == string[index + new_max_z_value]) {
            ++new_max_z_value;
        }
        argmax_i_plus_z_i_ = index;
        return new_max_z_value;
    }
};

std::vector<int32_t> PatternInTextZFunction(const std::vector<int32_t> &pattern,
                                            const std::vector<int32_t> &text) {
    auto concatenated = pattern;
    concatenated.push_back(-2);
    concatenated.insert(concatenated.end(), text.begin(), text.end());
    auto z_function = ZFunctionComputer{concatenated}.Compute();
    return {z_function.begin() + static_cast<int32_t>(pattern.size()) + 1, z_function.end()};
}

std::vector<bool> MatchUpToOneError(std::vector<int32_t> source, std::vector<int32_t> target) {
    auto forward_z_function = PatternInTextZFunction(target, source);
    std::reverse(target.begin(), target.end());
    std::reverse(source.begin(), source.end());
    auto reverse_z_function = PatternInTextZFunction(target, source);
    std::reverse(reverse_z_function.begin(), reverse_z_function.end());
    std::vector<bool> matches;
    matches.reserve(source.size() - target.size() + 1);
    auto target_size = static_cast<int32_t>(target.size());
    for (size_t i = target.size() - 1; i < forward_z_function.size(); ++i) {
        matches.emplace_back(forward_z_function[i - target.size() + 1] + reverse_z_function[i] >=
                             target_size - 1);
    }
    return matches;
}

std::vector<std::vector<bool>> FindRowMatches(const io::Input &input) {
    auto n_source_rows = static_cast<int32_t>(input.source_matrix.size());
    auto n_source_cols = static_cast<int32_t>(input.source_matrix.front().size());
    auto n_target_rows = static_cast<int32_t>(input.target_matrix.size());
    auto n_target_cols = static_cast<int32_t>(input.target_matrix.front().size());

    std::unordered_map<std::string, int32_t> unique_row_to_id;
    std::vector<std::string> unique_matrix;
    std::vector<int32_t> unique_row_ids;
    unique_matrix.reserve(n_target_rows);
    unique_row_ids.reserve(n_target_rows);
    for (int32_t row_index = 0; row_index < n_target_rows; ++row_index) {
        auto &row = input.target_matrix[row_index];
        auto find = unique_row_to_id.find(row);
        if (find == unique_row_to_id.end()) {
            auto new_id = static_cast<int32_t>(unique_matrix.size());
            unique_row_ids.emplace_back(new_id);
            unique_matrix.emplace_back(row);
            unique_row_to_id[row] = new_id;
        } else {
            unique_row_ids.emplace_back(find->second);
        }
    }

    prefix_tree::PrefixTree prefix_tree{unique_matrix};
    std::vector<std::vector<int32_t>> target_row_occurrences;
    target_row_occurrences.reserve(n_source_rows - n_target_rows + 1);
    for (auto &source_row : input.source_matrix) {
        target_row_occurrences.emplace_back();
        target_row_occurrences.back().reserve(n_source_cols - n_target_cols + 1);
        prefix_tree.ResetTextIterator();
        int32_t col = 0;
        for (auto &ch : source_row) {
            ++col;
            prefix_tree.SendNextTextLetter(ch);
            if (col >= n_target_cols) {
                auto occurrences = prefix_tree.GetOccurrencesAtCurrentPosition();
                assert(occurrences.size() <= 1);
                target_row_occurrences.back().emplace_back(
                    occurrences.empty() ? -1 : occurrences.front());
            }
        }
    }

    auto transposed_target_row_occurrences = Transpose(target_row_occurrences);
    std::vector<std::vector<bool>> matches;
    matches.reserve(transposed_target_row_occurrences.size());
    for (auto &row_occurrences : transposed_target_row_occurrences) {
        matches.emplace_back(MatchUpToOneError(row_occurrences, unique_row_ids));
    }
    return matches;
}

io::Output Solve(const io::Input &input) {
    auto row_matches = FindRowMatches(input);

    auto transposed_input = input;
    transposed_input.source_matrix = Transpose(transposed_input.source_matrix);
    transposed_input.target_matrix = Transpose(transposed_input.target_matrix);
    auto col_matches = FindRowMatches(transposed_input);

    io::Output output;
    for (size_t row = 0; row < row_matches.size(); ++row) {
        for (size_t col = 0; col < col_matches.size(); ++col) {
            output.number_of_target_matrix_occurrences_up_to_one_error_ +=
                row_matches[row][col] and col_matches[col][row];
        }
    }

    return output;
}

namespace test {

namespace rng {

uint32_t GetSeed() {
    auto random_device = std::random_device{};
    static auto seed = random_device();
    return 3236613196;
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

bool IsOccurrenceUpToOneError(const io::Input &input, int32_t occurrence_row,
                              int32_t occurrence_col) {
    auto n_target_rows = static_cast<int32_t>(input.target_matrix.size());
    auto n_target_cols = static_cast<int32_t>(input.target_matrix.front().size());
    int32_t n_errors = 0;
    for (int32_t row = 0; row < n_target_rows; ++row) {
        for (int32_t col = 0; col < n_target_cols; ++col) {
            n_errors += input.source_matrix[occurrence_row + row][occurrence_col + col] !=
                        input.target_matrix[row][col];
            if (n_errors > 1) {
                return false;
            }
        }
    }
    return true;
}

io::Output BruteForceSolve(const io::Input &input) {
    auto n_source_rows = static_cast<int32_t>(input.source_matrix.size());
    auto n_source_cols = static_cast<int32_t>(input.source_matrix.front().size());
    auto n_target_rows = static_cast<int32_t>(input.target_matrix.size());
    auto n_target_cols = static_cast<int32_t>(input.target_matrix.front().size());
    if (n_source_rows > 100 or n_source_cols > 100) {
        throw NotImplementedError{};
    }
    io::Output output;

    for (int32_t occurrence_row = 0; occurrence_row <= n_source_rows - n_target_rows;
         ++occurrence_row) {
        for (int32_t occurrence_col = 0; occurrence_col <= n_source_cols - n_target_cols;
             ++occurrence_col) {
            if (IsOccurrenceUpToOneError(input, occurrence_row, occurrence_col)) {
                ++output.number_of_target_matrix_occurrences_up_to_one_error_;
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

std::string MaybeMakeOneError(std::string string, double prob = 0.01) {
    std::uniform_int_distribution<int32_t> prob_dist{1, 100};
    std::uniform_int_distribution<int32_t> dist{0, static_cast<int32_t>(string.size() - 1)};
    std::uniform_int_distribution<char> letters_dist{'a', 'z'};
    if (prob_dist(*rng::GetEngine()) < prob * 100) {
        string[dist(*rng::GetEngine())] = letters_dist(*rng::GetEngine());
    }
    return string;
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t n_source_rows = std::min(1000, 1 + test_case_id);
    int32_t n_source_cols = std::min(1000, 1 + test_case_id);
    auto max_char = static_cast<char>('b' + ('z' - 'b') * std::min(1, test_case_id / 100));

    std::uniform_int_distribution<int32_t> row_dist{1, n_source_rows};
    std::uniform_int_distribution<int32_t> col_dist{1, n_source_cols};
    int32_t n_target_rows = row_dist(*rng::GetEngine());
    int32_t n_target_cols = col_dist(*rng::GetEngine());

    auto root_word = GenerateRandomString(n_target_cols, 'a', max_char);

    io::Input input;
    input.source_matrix.resize(n_source_rows);
    for (auto &s : input.source_matrix) {
        while (static_cast<int32_t>(s.size()) < n_source_cols) {
            s += MaybeMakeOneError(root_word);
        }
        s.resize(n_source_cols);
    }
    input.target_matrix.resize(n_target_rows);
    for (auto &s : input.target_matrix) {
        s = MaybeMakeOneError(root_word);
    }

    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    return GenerateRandomTestIo(1000);
}

class TimedChecker {
public:
    std::vector<int64_t> durations;

    void Check(const std::string &test_case, int32_t expected) {
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
        "5 5\n"
        "aazzz\n"
        "aazcb\n"
        "axzaa\n"
        "aapaa\n"
        "aapaa\n"
        "3 2\n"
        "aa\n"
        "aa\n"
        "aa",
        4);

    timed_checker.Check(
        "2 2\n"
        "ab\n"
        "ba\n"
        "2 2\n"
        "aa\n"
        "aa",
        0);

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
