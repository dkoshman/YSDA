#include <algorithm>
#include <array>
#include <bitset>
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
#include <utility>
#include <vector>

namespace io {

class Input {
public:
    int32_t target_string_size = 0;
    std::vector<std::string> strings;
    int32_t n_first_latin_alphabet_letters = 0;
    static const int32_t kMaxTotalStringsSize = 1'000;
    static const int32_t kModPrime = 1'000'000'007;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t n_strings = 0;
        in >> target_string_size >> n_strings >> n_first_latin_alphabet_letters;
        strings.resize(n_strings);
        for (auto &string : strings) {
            in >> string;
        }
    }
};

class Output {
public:
    int32_t n_strings_of_given_size_with_no_given_substrings_mod_prime = 0;

    Output() = default;

    explicit Output(int32_t answer)
        : n_strings_of_given_size_with_no_given_substrings_mod_prime{answer} {
    }

    std::ostream &Write(std::ostream &out) const {
        out << n_strings_of_given_size_with_no_given_substrings_mod_prime;
        return out;
    }

    bool operator!=(const Output &other) const {
        return n_strings_of_given_size_with_no_given_substrings_mod_prime !=
               other.n_strings_of_given_size_with_no_given_substrings_mod_prime;
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

size_t TotalStringsSize(const std::vector<std::string> &strings) {
    size_t size = 0;
    for (auto &string : strings) {
        size += string.size();
    }
    return size;
}

inline std::size_t CombineHashes(std::size_t first_hash, std::size_t second_hash) {
    return first_hash ^ (second_hash + 0x9e3779b9 + (first_hash << 6) + (first_hash >> 2));
}

template <typename First, typename Second, typename FirstHash = std::hash<First>,
          typename SecondHash = std::hash<Second>>
std::size_t HashPair(const First &first, const Second &second) {
    auto first_hash = FirstHash{};
    auto second_hash = SecondHash{};
    return CombineHashes(first_hash(first), second_hash(second));
}

template <typename First, typename Second, typename FirstHash = std::hash<First>,
          typename SecondHash = std::hash<Second>>
struct Pair {
    First first;
    Second second;

    Pair(First first, Second second) : first{std::move(first)}, second{std::move(second)} {
    }

    inline bool operator==(const Pair &other) const {
        return first == other.first and second == other.second;
    }

    struct HashFunction {
        inline std::size_t operator()(const Pair &pair) const {
            return HashPair<First, Second, FirstHash, SecondHash>(pair.first, pair.second);
        }
    };
};

template <typename Iterator, typename Hash = std::hash<typename Iterator::value_type>>
std::size_t HashOrderedContainer(Iterator begin, Iterator end) {
    auto hash = Hash{};
    size_t hash_value = 0;
    for (auto it = begin; it != end; ++it) {
        hash_value = CombineHashes(hash_value, hash(*it));
    }
    return hash_value;
}

template <class T = int32_t>
struct VectorHashFunction {
    inline std::size_t operator()(const std::vector<T> &vector) const {
        return HashOrderedContainer(vector.begin(), vector.end());
    }
};

template <class T = int32_t>
struct SetHashFunction {
    inline std::size_t operator()(const std::set<T> &set) const {
        return HashOrderedContainer(set.begin(), set.end());
    }
};

template <typename Input, typename Output, typename InputHashFunction = std::hash<Input>>
class DynamicComputer {
public:
    std::unordered_map<Input, Output, InputHashFunction> cache;

    virtual Output Compute(const Input &input) = 0;

    const Output &LookupCompute(const Input &input) {
        auto iterator = cache.find(input);
        if (iterator == cache.end()) {
            iterator = cache.emplace(input, Compute(input)).first;
        }
        return iterator->second;
    }
};

}  // namespace utils

namespace interface {

class VirtualBaseClass {
public:
    ~VirtualBaseClass() = default;
};

template <typename Node, typename NodeIterable = const std::vector<Node> &>
class GraphTraversal : public VirtualBaseClass {
public:
    virtual void OnTraverseStart() {
    }

    virtual bool ShouldNodeBeConsideredInThisTraversal(Node node) {
        return true;
    }

    virtual void OnNodeEnter(Node node) {
    }

    virtual void OnEdgeDiscovery(Node from, Node to) {
    }

    virtual void OnEdgeTraverse(Node from, Node to) {
    }

    virtual void OnEdgeBacktrack(Node to, Node from) {
    }

    virtual void OnNodeExit(Node node) {
    }

    virtual void SetNodeStateEntered(Node node) {
    }

    virtual void SetNodeStateExited(Node node) {
    }

    virtual bool ShouldTraverseEdge(Node from, Node to) {
        return true;
    }

    virtual NodeIterable NodesAdjacentTo(Node node) = 0;

    virtual void OnTraverseEnd() {
    }
};

template <typename State = int32_t>
class BaseAutomaton : public VirtualBaseClass {
public:
    [[nodiscard]] virtual State GetInitialState() const = 0;

    [[nodiscard]] virtual State Transition(const State &from, char letter) const = 0;

    [[nodiscard]] virtual bool IsTerminalState(const State &state) const = 0;
};

}  // namespace interface

namespace implementation {

class StopTraversalException : public std::exception {};

template <typename GraphTraversal>
void DepthFirstSearchRecursive(GraphTraversal *graph_traversal,
                               typename GraphTraversal::Node source_node) {

    graph_traversal->OnNodeEnter(source_node);
    graph_traversal->SetNodeStateEntered(source_node);

    for (auto adjacent_node : graph_traversal->NodesAdjacentTo(source_node)) {

        graph_traversal->OnEdgeDiscovery(source_node, adjacent_node);

        if (graph_traversal->ShouldTraverseEdge(source_node, adjacent_node)) {

            graph_traversal->OnEdgeTraverse(source_node, adjacent_node);
            DepthFirstSearchRecursive(graph_traversal, adjacent_node);
            graph_traversal->OnEdgeBacktrack(adjacent_node, source_node);
        }
    }

    graph_traversal->OnNodeExit(source_node);
    graph_traversal->SetNodeStateExited(source_node);
}

template <typename GraphTraversal>
void DepthFirstSearch(GraphTraversal *graph_traversal, typename GraphTraversal::Node source_node) {

    graph_traversal->OnTraverseStart();

    if (graph_traversal->ShouldNodeBeConsideredInThisTraversal(source_node)) {
        DepthFirstSearchRecursive(graph_traversal, source_node);
    }

    graph_traversal->OnTraverseEnd();
}

template <typename GraphTraversal>
void BreadthFirstSearch(GraphTraversal *graph_traversal,
                        std::deque<typename GraphTraversal::Node> starting_nodes_queue) {

    graph_traversal->OnTraverseStart();

    auto &queue = starting_nodes_queue;

    for (size_t i = 0; i < queue.size(); ++i) {
        auto node = queue.front();
        queue.pop_front();

        if (graph_traversal->ShouldNodeBeConsideredInThisTraversal(node)) {
            queue.emplace_back(node);
        }
    }

    while (not queue.empty()) {

        auto node = queue.front();
        queue.pop_front();

        graph_traversal->OnNodeEnter(node);
        graph_traversal->SetNodeStateEntered(node);

        for (const auto &adjacent_node : graph_traversal->NodesAdjacentTo(node)) {

            graph_traversal->OnEdgeDiscovery(node, adjacent_node);

            if (graph_traversal->ShouldTraverseEdge(node, adjacent_node)) {

                graph_traversal->OnEdgeTraverse(node, adjacent_node);
                queue.emplace_back(adjacent_node);
            }
        }

        graph_traversal->OnNodeExit(node);
        graph_traversal->SetNodeStateExited(node);
    }

    graph_traversal->OnTraverseEnd();
}

}  // namespace implementation

namespace io {

struct AutomatonTransition {
    int32_t from = 0;
    int32_t to = 0;
    char letter = '\0';

    AutomatonTransition(int32_t from, int32_t to, char letter)
        : from{from}, to{to}, letter{letter} {
    }
};

struct ImplicitEpsilonToSourceNfaAutomaton {
    int32_t initial_state = 0;
    int32_t n_states = 0;
    std::vector<AutomatonTransition> transitions;
    std::vector<int32_t> terminal_states;
    char alphabet_from = '\0';
    char alphabet_to = '\0';
};

}  // namespace io

io::ImplicitEpsilonToSourceNfaAutomaton BuildImplicitEpsilonToSourceNfaAutomaton(
    const io::Input &input) {
    io::ImplicitEpsilonToSourceNfaAutomaton automaton;
    automaton.alphabet_from = 'a';
    automaton.alphabet_to = static_cast<char>('a' + input.n_first_latin_alphabet_letters - 1);
    automaton.terminal_states.reserve(input.strings.size());
    automaton.transitions.reserve(utils::TotalStringsSize(input.strings));
    automaton.n_states = 1;

    for (auto &string : input.strings) {
        int32_t previous_state = automaton.initial_state;
        int32_t new_state = 0;
        for (auto letter : string) {
            new_state = automaton.n_states++;
            automaton.transitions.emplace_back(previous_state, new_state, letter);
            previous_state = new_state;
        }
        automaton.terminal_states.emplace_back(new_state);
    }

    return automaton;
}

using ImplicitEpsilonNfaStateSet = std::bitset<io::Input::kMaxTotalStringsSize + 10>;

class ImplicitEpsilonToSourceAutomaton
    : public interface::BaseAutomaton<ImplicitEpsilonNfaStateSet> {
public:
    using State = ImplicitEpsilonNfaStateSet;

    explicit ImplicitEpsilonToSourceAutomaton(
        const io::ImplicitEpsilonToSourceNfaAutomaton &automaton)
        : alphabet_from_{automaton.alphabet_from}, alphabet_to_{automaton.alphabet_to} {

        for (auto c = alphabet_from_; c <= alphabet_to_; ++c) {
            alphabet_.emplace_back(c);
        }

        for (auto i : automaton.terminal_states) {
            terminal_mask_[i] = true;
        }

        epsilon_nfa_transitions_.resize(LetterToIndex(alphabet_to_) + 1);
        for (auto &transition : epsilon_nfa_transitions_) {
            transition.resize(automaton.n_states);
        }

        for (auto &transition : automaton.transitions) {
            GetEpsilonNfaTransition(transition.from, transition.letter).emplace_back(transition.to);
        }

        initial_state_[automaton.initial_state] = true;

        source_transitions_.resize(LetterToIndex(alphabet_to_) + 1);
        for (auto letter = alphabet_from_; letter <= alphabet_to_; ++letter) {
            auto &source_transition = source_transitions_[LetterToIndex(letter)];
            source_transition = initial_state_;
            for (auto transition : GetEpsilonNfaTransition(automaton.initial_state, letter)) {
                source_transition[transition] = true;
            }
        }

        letter_masks_.resize(LetterToIndex(alphabet_to_) + 1);
        for (auto &transition : automaton.transitions) {
            letter_masks_[LetterToIndex(transition.letter)][transition.from] = true;
        }
        for (auto &mask : letter_masks_) {
            mask[automaton.initial_state] = false;
        }
    }

    [[nodiscard]] State GetInitialState() const override {
        return initial_state_;
    }

    [[nodiscard]] State Transition(const State &from, char letter) const override {
        auto to = from;
        to &= letter_masks_[LetterToIndex(letter)];
        to <<= 1;
        to |= source_transitions_[LetterToIndex(letter)];
        return to;
    }

    [[nodiscard]] bool IsTerminalState(const State &state) const override {
        return (state & terminal_mask_).any();
    }

    [[nodiscard]] const std::vector<char> &GetSortedAlphabet() const {
        return alphabet_;
    }

private:
    std::vector<char> alphabet_;
    State initial_state_;
    std::vector<std::vector<std::vector<int32_t>>> epsilon_nfa_transitions_;
    char alphabet_from_ = '\0';
    char alphabet_to_ = '\0';
    std::vector<State> source_transitions_;
    std::vector<State> letter_masks_;
    State terminal_mask_;

    [[nodiscard]] int32_t inline LetterToIndex(char letter) const {
        return letter - alphabet_from_;
    }

    [[nodiscard]] inline std::vector<int32_t> &GetEpsilonNfaTransition(int32_t epsilon_nfa_state,
                                                                       char letter) {
        return epsilon_nfa_transitions_[LetterToIndex(letter)][epsilon_nfa_state];
    }
};

inline int32_t ModPrime(int64_t value) {
    return utils::NonNegativeMod(value, io::Input::kModPrime);
}

class NStringsComputer {
public:
    const ImplicitEpsilonToSourceAutomaton &automaton;
    std::unordered_map<ImplicitEpsilonNfaStateSet, int32_t> state_to_id;
    std::vector<ImplicitEpsilonNfaStateSet> states;
    std::vector<std::vector<int32_t>> transitions;

    explicit NStringsComputer(const ImplicitEpsilonToSourceAutomaton &automaton)
        : automaton{automaton} {
    }

    int32_t Compute(const ImplicitEpsilonNfaStateSet &state, int32_t string_size) {
        state_counts_[GetStateId(state)] = 1;

        for (int32_t size = 0; size < string_size; ++size) {
            for (auto state_id = 0; state_id < state_counts_size_; ++state_id) {
                if (state_counts_[state_id] > 0) {
                    for (auto transition : GetTransition(state_id)) {
                        next_state_counts_[transition] += state_counts_[state_id];
                        next_state_counts_[transition] = ModPrime(next_state_counts_[transition]);
                    }
                }
            }

            std::swap(state_counts_, next_state_counts_);
            std::fill(next_state_counts_.begin(), next_state_counts_.end(), 0);
        }

        return SumUpCounts();
    }

private:
    std::vector<int32_t> state_counts_;
    std::vector<int32_t> next_state_counts_;
    int32_t state_counts_size_ = 0;

    int32_t GetStateId(const ImplicitEpsilonNfaStateSet &state) {
        auto [iter, has_emplaced] =
            state_to_id.emplace(state, static_cast<int32_t>(state_to_id.size()));
        if (has_emplaced) {
            states.emplace_back(state);
        }
        auto state_id = iter->second;
        if (state_counts_size_ <= state_id) {
            state_counts_size_ = 1 + 2 * state_id;
            state_counts_.resize(state_counts_size_);
            next_state_counts_.resize(state_counts_size_);
        }
        return state_id;
    }

    const std::vector<int32_t> &GetTransition(int32_t state_id) {
        for (auto id = static_cast<int32_t>(transitions.size()); id <= state_id; ++id) {
            transitions.emplace_back();
            auto &transition = transitions.back();
            auto state = states[id];
            for (auto letter : automaton.GetSortedAlphabet()) {
                auto new_state = automaton.Transition(state, letter);
                if (not automaton.IsTerminalState(new_state)) {
                    transition.emplace_back(GetStateId(new_state));
                }
            }
        }

        return transitions[state_id];
    }

    [[nodiscard]] int32_t SumUpCounts() const {
        int32_t sum = 0;
        for (auto count : state_counts_) {
            sum = ModPrime(sum + count);
        }
        return sum;
    }
};

int32_t ComputeNStringsOfGivenSizeWithNoGivenSubstrings(const io::Input &input) {
    auto io_automaton = BuildImplicitEpsilonToSourceNfaAutomaton(input);
    auto automaton = ImplicitEpsilonToSourceAutomaton{io_automaton};
    NStringsComputer computer{automaton};
    return computer.Compute(automaton.GetInitialState(), input.target_string_size);
}

io::Output Solve(const io::Input &input) {
    return io::Output{ComputeNStringsOfGivenSizeWithNoGivenSubstrings(input)};
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

io::ImplicitEpsilonToSourceNfaAutomaton BuildImplicitEpsilonToSourceNfaAutomaton(
    const io::Input &input) {
    io::ImplicitEpsilonToSourceNfaAutomaton automaton;
    automaton.n_states = 1;
    automaton.alphabet_from = 'a';
    automaton.alphabet_to = static_cast<char>('a' + input.n_first_latin_alphabet_letters - 1);
    automaton.terminal_states.reserve(input.strings.size());
    automaton.transitions.reserve(utils::TotalStringsSize(input.strings));

    for (auto &string : input.strings) {
        int32_t previous_state = automaton.initial_state;
        int32_t new_state = 0;
        for (auto letter : string) {
            new_state = automaton.n_states++;
            automaton.transitions.emplace_back(previous_state, new_state, letter);
            previous_state = new_state;
        }
        automaton.terminal_states.emplace_back(new_state);
    }

    return automaton;
}

using ImplicitEpsilonNfaStateSet = std::bitset<io::Input::kMaxTotalStringsSize>;

class ImplicitEpsilonToSourceAutomaton
    : public interface::BaseAutomaton<ImplicitEpsilonNfaStateSet> {
public:
    using State = ImplicitEpsilonNfaStateSet;

    explicit ImplicitEpsilonToSourceAutomaton(
        const io::ImplicitEpsilonToSourceNfaAutomaton &automaton)
        : initial_epsilon_nfa_state_{automaton.initial_state},
          n_states_{automaton.n_states},
          is_epsilon_nfa_state_terminal_(automaton.n_states),
          alphabet_from_{automaton.alphabet_from},
          alphabet_to_{automaton.alphabet_to} {

        for (auto c = alphabet_from_; c <= alphabet_to_; ++c) {
            alphabet_.emplace_back(c);
        }

        for (auto i : automaton.terminal_states) {
            is_epsilon_nfa_state_terminal_[i] = true;
        }

        epsilon_nfa_transitions_.resize(LetterToIndex(alphabet_to_) + 1);
        for (auto &transition : epsilon_nfa_transitions_) {
            transition.resize(automaton.n_states);
        }

        for (auto &transition : automaton.transitions) {
            GetEpsilonNfaTransition(transition.from, transition.letter).emplace_back(transition.to);
        }

        initial_state_[initial_epsilon_nfa_state_] = true;
    }

    [[nodiscard]] State GetInitialState() const override {
        return initial_state_;
    }

    [[nodiscard]] State Transition(const State &from, char letter) const override {
        auto to = GetInitialState();

        for (int32_t epsilon_nfa_state = 0; epsilon_nfa_state < n_states_; ++epsilon_nfa_state) {
            if (from[epsilon_nfa_state]) {
                for (auto transition : GetEpsilonNfaTransition(epsilon_nfa_state, letter)) {
                    to[transition] = true;
                }
            }
        }

        return to;
    }

    [[nodiscard]] bool IsTerminalState(const State &state) const override {
        for (int32_t epsilon_nfa_state = 0; epsilon_nfa_state < n_states_; ++epsilon_nfa_state) {
            if (state[epsilon_nfa_state] and IsTerminalEpsilonNfaState(epsilon_nfa_state)) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] const std::vector<char> &GetSortedAlphabet() const {
        return alphabet_;
    }

private:
    int32_t initial_epsilon_nfa_state_ = 0;
    int32_t n_states_ = 0;
    std::vector<char> alphabet_;
    State initial_state_;
    std::vector<bool> is_epsilon_nfa_state_terminal_;
    std::vector<std::vector<std::vector<int32_t>>> epsilon_nfa_transitions_;
    char alphabet_from_ = '\0';
    char alphabet_to_ = '\0';

    [[nodiscard]] int32_t inline LetterToIndex(char letter) const {
        return letter - alphabet_from_;
    }

    [[nodiscard]] inline bool IsTerminalEpsilonNfaState(int32_t epsilon_nfa_state) const {
        return is_epsilon_nfa_state_terminal_[epsilon_nfa_state];
    }

    [[nodiscard]] inline std::vector<int32_t> &GetEpsilonNfaTransition(int32_t epsilon_nfa_state,
                                                                       char letter) {
        return epsilon_nfa_transitions_[LetterToIndex(letter)][epsilon_nfa_state];
    }

    [[nodiscard]] inline const std::vector<int32_t> &GetEpsilonNfaTransition(
        int32_t epsilon_nfa_state, char letter) const {
        return epsilon_nfa_transitions_[LetterToIndex(letter)][epsilon_nfa_state];
    }
};

int32_t ModPrime(int64_t value) {
    return utils::NonNegativeMod(value, io::Input::kModPrime);
}

int32_t ComputeNStringsOfGivenSizeWithNoGivenSubstrings(const io::Input &input) {
    auto io_automaton = BuildImplicitEpsilonToSourceNfaAutomaton(input);
    auto automaton = ImplicitEpsilonToSourceAutomaton{io_automaton};

    std::unordered_map<ImplicitEpsilonNfaStateSet, int64_t> states_without_substrings_counts;
    states_without_substrings_counts.emplace(automaton.GetInitialState(), 1);

    for (int32_t string_size = 0; string_size < input.target_string_size; ++string_size) {
        std::unordered_map<ImplicitEpsilonNfaStateSet, int64_t> next_counts;

        for (auto &[state, count] : states_without_substrings_counts) {
            for (auto letter : automaton.GetSortedAlphabet()) {

                auto new_state = automaton.Transition(state, letter);
                if (not automaton.IsTerminalState(new_state)) {
                    auto [iter, has_emplaced] = next_counts.emplace(new_state, count);
                    if (not has_emplaced) {
                        iter->second = ModPrime(iter->second + count);
                    }
                }
            }
        }

        std::swap(states_without_substrings_counts, next_counts);
    }

    int64_t sum = 0;
    for (auto &[state, count] : states_without_substrings_counts) {
        sum += count;
    }
    return ModPrime(sum);
}

io::Output BruteForceSolve(const io::Input &input) {
    if (utils::TotalStringsSize(input.strings) > 300) {
        throw NotImplementedError{};
    }
    return io::Output{ComputeNStringsOfGivenSizeWithNoGivenSubstrings(input)};
}

std::string GenerateRandomString(int32_t size, char letter_from = 'a', char letter_to = 'z') {
    std::uniform_int_distribution<char> letters_dist{letter_from, letter_to};
    std::string string;
    for (int32_t i = 0; i < size; ++i) {
        string += letters_dist(*rng::GetEngine());
    }
    return string;
}

bool AreValidStringSplits(std::vector<int32_t> &string_splits) {
    std::sort(string_splits.begin(), string_splits.end());
    std::optional<int32_t> previous;
    for (auto split : string_splits) {
        if (previous and split - previous.value() <= 1) {
            return false;
        }
        previous = split;
    }
    return true;
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t target_string_size = 1 + (test_case_id > 1) + test_case_id / 2;
    int32_t total_strings_size = std::min(1000, 3 + test_case_id * 2);
    int32_t n_strings = std::min(100, 1 + target_string_size / 2);
    int32_t n_letters = std::min(26, 1 + (test_case_id > 1) + test_case_id / 30);

    io::Input input;
    input.target_string_size = target_string_size;
    input.n_first_latin_alphabet_letters = n_letters;

    std::uniform_int_distribution<int32_t> string_splits_dist{1, total_strings_size - 1};
    std::vector<int32_t> string_splits;
    do {
        string_splits.resize(n_strings + 1);
        for (auto &split : string_splits) {
            split = string_splits_dist(*rng::GetEngine());
        }
        string_splits.front() = 0;
        string_splits.back() = total_strings_size;
    } while (not AreValidStringSplits(string_splits));

    auto string =
        GenerateRandomString(total_strings_size, 'a', static_cast<char>('a' + n_letters - 1));

    std::optional<int32_t> previous;
    for (auto split : string_splits) {
        if (previous) {
            input.strings.emplace_back(string.begin() + previous.value(), string.begin() + split);
        }
        previous = split;
    }

    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    int32_t target_string_size = 1000;
    int32_t total_strings_size = 1000;
    int32_t n_strings = 40;
    int32_t n_letters = 26;

    io::Input input;
    input.target_string_size = target_string_size;
    input.n_first_latin_alphabet_letters = n_letters;

    std::uniform_int_distribution<int32_t> string_splits_dist{1, total_strings_size - 1};
    std::vector<int32_t> string_splits;
    do {
        string_splits.resize(n_strings + 1);
        for (auto &split : string_splits) {
            split = string_splits_dist(*rng::GetEngine());
        }
        string_splits.front() = 0;
        string_splits.back() = total_strings_size;
    } while (not AreValidStringSplits(string_splits));

    auto string = GenerateRandomString(total_strings_size, 'a', 'z');

    std::optional<int32_t> previous;
    for (auto split : string_splits) {
        if (previous) {
            input.strings.emplace_back(string.begin() + previous.value(), string.begin() + split);
        }
        previous = split;
    }

    return TestIo{input};
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
                ss << "\n================================Expected======================"
                      "==========\n"
                   << expected_output
                   << "\n================================Received======================"
                      "==========\n"
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
        "5 1 2\n"
        "a",
        1);
    timed_checker.Check(
        "5 2 1\n"
        "a\n"
        "aa",
        0);
    timed_checker.Check(
        "5 1 2\n"
        "ab",
        6);
    timed_checker.Check("5 0 2", 32);
    //    timed_checker.Check("", "");

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
