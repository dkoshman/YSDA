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
#include <unordered_set>
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
    int32_t n_states = 0;
    int32_t initial_state = 0;
    int32_t n_first_latin_letters_in_alphabet = 0;
    std::vector<AutomatonTransition> transitions;
    std::vector<int32_t> terminal_states;

    EpsilonNfaAutomaton() = default;

    explicit EpsilonNfaAutomaton(std::istream &istream) {
        int32_t n_terminal_states = 0;
        istream >> n_states >> n_terminal_states >> n_first_latin_letters_in_alphabet;
        terminal_states.resize(n_terminal_states);
        for (auto &terminal_state : terminal_states) {
            istream >> terminal_state;
        }
        transitions.reserve(n_states * n_first_latin_letters_in_alphabet);
        while (transitions.size() < transitions.capacity()) {
            transitions.emplace_back(istream);
        }
    }
};

class Input {
public:
    EpsilonNfaAutomaton first;
    EpsilonNfaAutomaton second;

    Input() = default;

    explicit Input(std::istream &in) : first{in}, second{in} {
    }
};

class Output {
public:
    bool are_automata_equivalent = false;
    std::string are_equivalent_string = "EQUIVALENT";
    std::string not_equivalent_string = "NOT EQUIVALENT";

    Output() = default;

    explicit Output(bool are_automata_equivalent)
        : are_automata_equivalent{are_automata_equivalent} {
    }

    explicit Output(const std::string &string) {
        are_automata_equivalent = string == are_equivalent_string;
    }

    std::ostream &Write(std::ostream &out) const {
        out << (are_automata_equivalent ? are_equivalent_string : not_equivalent_string);
        return out;
    }

    bool operator!=(const Output &other) const {
        return are_automata_equivalent != other.are_automata_equivalent;
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

template <class First, class Second>
std::size_t HashPair(const First &first, const Second &second) {
    auto first_hash = std::hash<First>{}(first);
    auto second_hash = std::hash<Second>{}(second);
    return CombineHashes(first_hash, second_hash);
}

template <typename T>
struct Pair {
    T first;
    T second;

    Pair(T first, T second) : first{std::move(first)}, second{std::move(second)} {
    }

    inline bool operator==(const Pair &other) const {
        return first == other.first and second == other.second;
    }

    struct HashFunction {
        inline std::size_t operator()(const Pair &pair) const {
            return utils::HashPair(pair.first, pair.second);
        }
    };
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

template <typename State>
class BaseAutomaton : public VirtualBaseClass {
public:
    [[nodiscard]] virtual State GetInitialState() const = 0;

    [[nodiscard]] virtual const std::vector<char> &GetSortedAlphabet() const = 0;

    [[nodiscard]] virtual State Transition(const State &from, char letter) const = 0;

    [[nodiscard]] virtual bool IsStateTerminal(const State &state) const = 0;
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

template <typename Automaton>
class AreAutomataEquivalentTraversal
    : public interface::GraphTraversal<utils::Pair<typename Automaton::State>> {
public:
    using State = typename Automaton::State;
    using Node = utils::Pair<State>;
    using NodeIterable = const std::vector<Node> &;

    AreAutomataEquivalentTraversal(const Automaton &first, const Automaton &second)
        : first{first}, second{second} {
    }

    [[nodiscard]] bool AreAutomataEquivalent() const {
        return are_automata_equivalent_;
    }

    void OnTraverseStart() override {
        are_automata_equivalent_ = true;
    }

    void OnNodeEnter(Node node) override {
        ConsiderNodeVisited(node);
        auto &first_state = node.first;
        auto &second_state = node.second;

        if (first.IsStateTerminal(first_state) != second.IsStateTerminal(second_state)) {
            are_automata_equivalent_ = false;
            throw implementation::StopTraversalException{};
        }
    }

    NodeIterable NodesAdjacentTo(Node node) override {
        adjacent_nodes_.clear();
        auto &first_state = node.first;
        auto &second_state = node.second;

        for (auto letter : first.GetSortedAlphabet()) {
            auto first_state_transition = first.Transition(first_state, letter);
            auto second_state_transition = second.Transition(second_state, letter);
            Node adjacent_node{first_state_transition, second_state_transition};

            if (not HasVisitedNode(adjacent_node)) {
                adjacent_nodes_.emplace_back(std::move(adjacent_node));
            }
        }

        return adjacent_nodes_;
    }

private:
    const Automaton &first;
    const Automaton &second;
    std::unordered_set<Node, typename Node::HashFunction> visited_nodes;
    std::vector<Node> adjacent_nodes_;
    bool are_automata_equivalent_ = true;

    void ConsiderNodeVisited(const Node &node) {
        visited_nodes.insert(node);
    }

    [[nodiscard]] bool HasVisitedNode(const Node &node) const {
        return visited_nodes.count(node);
    }
};

template <typename Automaton>
bool AreAutomataEquivalent(const Automaton &first, const Automaton &second) {
    if (first.GetSortedAlphabet() != second.GetSortedAlphabet()) {
        return false;
    }

    AreAutomataEquivalentTraversal<Automaton> traversal{first, second};
    auto first_initial_state = first.GetInitialState();
    auto second_initial_state = second.GetInitialState();
    try {
        implementation::BreadthFirstSearch(&traversal,
                                           {{first_initial_state, second_initial_state}});
    } catch (const implementation::StopTraversalException &) {
    }
    return traversal.AreAutomataEquivalent();
}

class ConnectedAutomaton : public interface::BaseAutomaton<int32_t> {
public:
    using State = int32_t;

    explicit ConnectedAutomaton(const io::EpsilonNfaAutomaton &automaton)
        : initial_state_{automaton.initial_state},
          is_state_terminal_(automaton.n_states),
          transition_(automaton.n_first_latin_letters_in_alphabet) {
        alphabet_.reserve(automaton.n_first_latin_letters_in_alphabet);
        for (auto c = 'a'; c < 'a' + automaton.n_first_latin_letters_in_alphabet; ++c) {
            alphabet_.emplace_back(c);
        }
        for (auto i : automaton.terminal_states) {
            is_state_terminal_[i] = true;
        }
        for (auto &letter_transition : transition_) {
            letter_transition.resize(automaton.n_states);
        }
        for (auto &transition : automaton.transitions) {
            transition_[transition.letter - 'a'][transition.from] = transition.to;
        }
    }

    [[nodiscard]] State GetInitialState() const override {
        return initial_state_;
    }

    [[nodiscard]] const std::vector<char> &GetSortedAlphabet() const override {
        return alphabet_;
    }

    [[nodiscard]] State Transition(const State &from, char letter) const override {
        return transition_[letter - 'a'][from];
    }

    [[nodiscard]] bool IsStateTerminal(const State &state) const override {
        return is_state_terminal_[state];
    }

private:
    int32_t initial_state_ = 0;
    std::vector<char> alphabet_;
    std::vector<bool> is_state_terminal_;
    std::vector<std::vector<int32_t>> transition_;
};

io::Output Solve(const io::Input &input) {
    ConnectedAutomaton first{input.first};
    ConnectedAutomaton second{input.second};
    return io::Output{AreAutomataEquivalent<ConnectedAutomaton>(first, second)};
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
        "4 1 2\n"
        "2\n"
        "0 a 1\n"
        "0 b 0\n"
        "1 a 1\n"
        "1 b 2\n"
        "2 a 3\n"
        "2 b 3\n"
        "3 a 3\n"
        "3 b 3\n"
        "2 1 2\n"
        "1\n"
        "0 a 1\n"
        "0 b 1\n"
        "1 a 1\n"
        "1 b 1",
        "NOT EQUIVALENT");

    timed_checker.Check(
        "4 3 1\n"
        "1 2 3\n"
        "0 a 1\n"
        "1 a 2\n"
        "2 a 3\n"
        "3 a 3\n"
        "2 1 1\n"
        "1\n"
        "0 a 1\n"
        "1 a 1",
        "EQUIVALENT");

    timed_checker.Check(
        "4 1 2\n"
        "2\n"
        "0 a 1\n"
        "0 b 0\n"
        "1 a 1\n"
        "1 b 2\n"
        "2 a 3\n"
        "2 b 3\n"
        "3 a 3\n"
        "3 b 3\n"
        "3 1 2\n"
        "2\n"
        "0 a 1\n"
        "0 b 0\n"
        "1 a 1\n"
        "1 b 2\n"
        "2 a 1\n"
        "2 b 0",
        "NOT EQUIVALENT");

    timed_checker.Check(
        "4 3 1\n"
        "1 2 3\n"
        "0 a 1\n"
        "1 a 2\n"
        "2 a 3\n"
        "3 a 3\n"
        "2 1 2\n"
        "1\n"
        "0 a 1\n"
        "1 a 1\n"
        "0 a 1\n"
        "1 a 1",
        "NOT EQUIVALENT");

    timed_checker.Check(
        "4 0 2\n"
        "\n"
        "0 a 1\n"
        "0 b 0\n"
        "1 a 1\n"
        "1 b 2\n"
        "2 a 3\n"
        "2 b 3\n"
        "3 a 3\n"
        "3 b 3\n"
        "3 0 2\n"
        "\n"
        "0 a 1\n"
        "0 b 0\n"
        "1 a 1\n"
        "1 b 2\n"
        "2 a 1\n"
        "2 b 0",
        "EQUIVALENT");

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
