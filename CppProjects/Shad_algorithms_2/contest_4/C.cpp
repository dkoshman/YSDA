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
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace io {

struct AutomatonTransition {
    int32_t from = 0;
    int32_t to = 0;
    char letter = '\0';

    explicit AutomatonTransition(std::istream &istream) {
        istream >> from >> letter >> to;
    }
};

struct EpsilonNfaAutomaton {
    int32_t initial_state = 0;
    int32_t n_states = 0;
    std::vector<AutomatonTransition> transitions;
    std::vector<int32_t> terminal_states;
    char alphabet_from = '\0';
    char alphabet_to = '\0';
    char epsilon = '\0';

    EpsilonNfaAutomaton() = default;

    explicit EpsilonNfaAutomaton(std::istream &istream, char alphabet_from, char alphabet_to,
                                 char epsilon)
        : alphabet_from{alphabet_from}, alphabet_to{alphabet_to}, epsilon{epsilon} {
        int32_t n_transitions = 0;
        int32_t n_terminal_states = 0;
        istream >> n_states >> n_transitions >> n_terminal_states;

        terminal_states.resize(n_terminal_states);
        for (auto &terminal_state : terminal_states) {
            istream >> terminal_state;
        }

        transitions.reserve(n_transitions);
        while (transitions.size() < transitions.capacity()) {
            transitions.emplace_back(istream);
        }
    }
};

class Input {
public:
    EpsilonNfaAutomaton epsilon_nfa_automaton;
    std::string string;
    static const char kAlphabetFrom = 'a';
    static const char kAlphabetTo = 'z';
    static const char kEpsilon = '$';

    Input() = default;

    explicit Input(std::istream &in)
        : epsilon_nfa_automaton{in, kAlphabetFrom, kAlphabetTo, kEpsilon} {
        in >> string;
    }
};

class Output {
public:
    std::string largest_substring_recognized_by_automaton;
    std::string no_solution_string = "No solution";

    Output() = default;

    explicit Output(const std::string &string) {
        if (string != no_solution_string) {
            largest_substring_recognized_by_automaton = string;
        }
    }

    std::ostream &Write(std::ostream &out) const {
        out << (largest_substring_recognized_by_automaton.empty()
                    ? no_solution_string
                    : largest_substring_recognized_by_automaton);
        return out;
    }

    bool operator!=(const Output &other) const {
        return largest_substring_recognized_by_automaton !=
               other.largest_substring_recognized_by_automaton;
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

    [[nodiscard]] virtual State Transition(const State &from, char letter) = 0;

    [[nodiscard]] virtual bool IsTerminalState(const State &state) const = 0;

    [[nodiscard]] virtual bool IsLeafState(const State &state) const = 0;
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

using ImplicitEpsilonNfaStateSet = std::vector<std::optional<int32_t>>;

class AutomatonWithStringSize
    : public interface::BaseAutomaton<ImplicitEpsilonNfaStateSet> {
public:
    using State = ImplicitEpsilonNfaStateSet;

    explicit AutomatonWithStringSize(const io::EpsilonNfaAutomaton &automaton)
        : initial_epsilon_nfa_state_{automaton.initial_state},
          n_states_{automaton.n_states},
          is_epsilon_nfa_state_terminal_(automaton.n_states),
          alphabet_from_{automaton.alphabet_from},
          alphabet_to_{automaton.alphabet_to},
          epsilon_{automaton.epsilon} {

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

        initial_state_.resize(n_states_);
        initial_state_[initial_epsilon_nfa_state_] = 0;
        initial_state_ = BuildEpsilonClosure(initial_state_);
    }

    [[nodiscard]] State GetInitialState() const override {
        return initial_state_;
    }

    [[nodiscard]] State Transition(const State &from, char letter) override {
        State to = GetInitialState();

        for (int32_t epsilon_nfa_state = 0; epsilon_nfa_state < n_states_; ++epsilon_nfa_state) {
            auto string_size = from[epsilon_nfa_state];
            if (string_size) {
                for (auto transition : GetEpsilonNfaTransition(epsilon_nfa_state, letter)) {

                    auto &size = to[transition];
                    if (not size or (size.value() < string_size.value() + 1)) {
                        size = string_size.value() + 1;
                    }
                }
            }
        }

        return BuildEpsilonClosure(to);
    }

    [[nodiscard]] bool IsTerminalState(const State &state) const override {
        for (int32_t epsilon_nfa_state = 0; epsilon_nfa_state < n_states_; ++epsilon_nfa_state) {
            if (state[epsilon_nfa_state] and IsTerminalEpsilonNfaState(epsilon_nfa_state)) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] std::optional<int32_t> FindLongestTerminalEpsilonNfaState(
        const State &state) const {
        std::optional<int32_t> max_size;
        for (int32_t epsilon_nfa_state = 0; epsilon_nfa_state < n_states_; ++epsilon_nfa_state) {
            auto string_size = state[epsilon_nfa_state];
            if (string_size and IsTerminalEpsilonNfaState(epsilon_nfa_state) and
                max_size < string_size) {
                max_size = string_size;
            }
        }
        return max_size;
    }

    [[nodiscard]] bool IsLeafState(const State &state) const override {
        return state.empty();
    }

private:
    int32_t initial_epsilon_nfa_state_ = 0;
    int32_t n_states_;
    State initial_state_;
    std::vector<bool> is_epsilon_nfa_state_terminal_;
    std::vector<std::vector<std::vector<int32_t>>> epsilon_nfa_transitions_;
    char alphabet_from_ = '\0';
    char alphabet_to_ = '\0';
    char epsilon_ = '\0';

    [[nodiscard]] int32_t inline LetterToIndex(char letter) const {
        return letter == epsilon_ ? 0 : letter - alphabet_from_ + 1;
    }

    [[nodiscard]] inline bool IsTerminalEpsilonNfaState(int32_t epsilon_nfa_state) const {
        return is_epsilon_nfa_state_terminal_[epsilon_nfa_state];
    }

    [[nodiscard]] inline std::vector<int32_t> &GetEpsilonNfaTransition(int32_t epsilon_nfa_state,
                                                                       char letter) {
        return epsilon_nfa_transitions_[LetterToIndex(letter)][epsilon_nfa_state];
    }

    [[nodiscard]] State BuildEpsilonClosure(const State &state) {
        std::deque<int32_t> unprocessed_epsilon_nfa_states;
        auto closure = state;
        for (int32_t epsilon_nfa_state = 0; epsilon_nfa_state < n_states_; ++epsilon_nfa_state) {
            if (closure[epsilon_nfa_state]) {
                unprocessed_epsilon_nfa_states.emplace_back(epsilon_nfa_state);
            }
        }

        while (not unprocessed_epsilon_nfa_states.empty()) {
            auto epsilon_nfa_state = unprocessed_epsilon_nfa_states.front();
            unprocessed_epsilon_nfa_states.pop_front();

            for (auto transition : GetEpsilonNfaTransition(epsilon_nfa_state, epsilon_)) {
                if (closure[transition] < closure[epsilon_nfa_state]) {
                    closure[transition] = closure[epsilon_nfa_state];
                    unprocessed_epsilon_nfa_states.emplace_back(transition);
                }
            }
        }

        return closure;
    }
};

io::Output Solve(const io::Input &input) {
    AutomatonWithStringSize automaton{input.epsilon_nfa_automaton};
    auto initial_state = automaton.GetInitialState();
    auto state = initial_state;
    io::Output output;

    for (auto it = input.string.begin(); it != input.string.end(); ++it) {
        state = automaton.Transition(state, *it);
        auto max_size = automaton.FindLongestTerminalEpsilonNfaState(state);
        if (static_cast<int32_t>(output.largest_substring_recognized_by_automaton.size()) <
            max_size) {
            output.largest_substring_recognized_by_automaton = {it + 1 - max_size.value(), it + 1};
        }
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

io::Output BruteForceSolve(const io::Input &input) {
    throw NotImplementedError{};
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
    throw NotImplementedError{};
    io::Input input;
    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    throw NotImplementedError{};
    io::Input input;
    return TestIo{input};
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
        "7 6 2\n"
        "2 6\n"
        "0 a 1\n"
        "1 b 2\n"
        "0 $ 3\n"
        "3 a 4\n"
        "4 b 5\n"
        "5 c 6\n"
        "xabcd",
        "abc");
    timed_checker.Check(
        "2 1 1\n"
        "1\n"
        "0 x 1\n"
        "abc",
        "No solution");
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
