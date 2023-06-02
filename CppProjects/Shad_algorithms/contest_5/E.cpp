#include <algorithm>
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

namespace io {

enum class Coin { heads, tails };

using Board = std::vector<std::vector<Coin>>;

class Input {
public:
    Board board;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t row_count = 0;
        int32_t col_count = 0;
        in >> row_count >> col_count;

        board.resize(row_count);
        for (auto &row : board) {
            row.reserve(col_count);
            for (int32_t col = 0; col < col_count; ++col) {
                char coin = 0;
                in >> coin;
                if (coin == '0') {
                    row.push_back(Coin::heads);
                } else if (coin == '1') {
                    row.push_back(Coin::tails);
                } else {
                    throw std::invalid_argument{"Unknown coin position " + std::to_string(coin)};
                }
            }
        }
    }
};

class Output {
public:
    std::optional<int32_t> smallest_number_of_coin_flips_to_create_checkerboard_pattern;

    Output() = default;

    explicit Output(std::optional<int32_t> answer)
        : smallest_number_of_coin_flips_to_create_checkerboard_pattern{answer} {
    }

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        int32_t item = 0;
        ss >> item;
        if (item != -1) {
            smallest_number_of_coin_flips_to_create_checkerboard_pattern = item;
        }
    }

    std::ostream &Write(std::ostream &out) const {
        out << (smallest_number_of_coin_flips_to_create_checkerboard_pattern
                    ? smallest_number_of_coin_flips_to_create_checkerboard_pattern.value()
                    : -1);
        return out;
    }

    bool operator!=(const Output &other) const {
        return smallest_number_of_coin_flips_to_create_checkerboard_pattern !=
               other.smallest_number_of_coin_flips_to_create_checkerboard_pattern;
    }
};

std::ostream &operator<<(std::ostream &os, Output const &output) {
    return output.Write(os);
}

}  // namespace io

using io::Input, io::Output;

namespace interface {

class NotImplementedError : public std::logic_error {
public:
    NotImplementedError() : std::logic_error("Function not yet implemented."){};
};

class VirtualBaseClass {
public:
    ~VirtualBaseClass() = default;
};

class ResizeInterface : public VirtualBaseClass {
public:
    virtual void Resize(size_t size) = 0;

    [[nodiscard]] virtual size_t Size() const = 0;
};

template <class NodeId = int32_t>
class ConnectByNodeIdsInterface : public VirtualBaseClass {
public:
    virtual void Connect(NodeId from, NodeId to) = 0;
};

template <class Edge>
class ConnectByEdgesInterface : public VirtualBaseClass {
public:
    virtual void AddEdge(Edge edge) = 0;
};

template <class Node, class NodeId = int32_t>
class NodeByIdInterface : public VirtualBaseClass {
public:
    virtual Node &operator[](NodeId node_id) = 0;

    virtual Node operator[](NodeId node_id) const = 0;
};

template <class NodeId = int32_t>
class AdjacentByIdInterface : public VirtualBaseClass {
public:
    //    [[nodiscard]] virtual NodeIdIterable NodeIdsAdjacentTo(NodeId node_id) const = 0;
};

template <class NodeId = int32_t>
class VisitAdjacentByIdInterface : public VirtualBaseClass {
public:
    virtual void VisitNodeIdsAdjacentTo(NodeId node_id,
                                        const std::function<void(NodeId node_id)> &function) = 0;
};

template <class NodeState>
class StateInterface : public VirtualBaseClass {
public:
    [[nodiscard]] virtual NodeState GetState() const = 0;

    virtual void SetState(NodeState state) = 0;
};

template <class NodeState, class NodeId = int32_t>
class StateByIdInterface : public VirtualBaseClass {
public:
    [[nodiscard]] virtual NodeState GetNodeState(NodeId node_id) const = 0;

    virtual void SetNodeState(NodeId node_id, NodeState state) = 0;

    virtual ~StateByIdInterface() = default;
};

class GraphTraversal {
public:
    template <class Node>
    void OnNodeEnter(Node &node) {
    }

    template <class Node>
    void OnEdgeDiscovery(Node &from, Node &to) {
    }

    template <class Edge>
    void OnEdgeDiscovery(Edge edge) {
    }

    template <class Node>
    void OnEdgeTraverse(Node &from, Node &to) {
    }

    template <class Edge>
    void OnEdgeTraverse(Edge &edge) {
    }

    template <class Node>
    void OnNodeExit(Node &node) {
    }

    template <class Node>
    void SetNodeStateEntered(Node &node) {
    }

    template <class Node>
    void SetNodeStateExited(Node &node) {
    }

    template <class Node>
    bool IsNodeUnvisited(Node &node) {
        throw NotImplementedError{};
    }

    template <class Node, class NodeIterable = std::vector<Node> &>
    NodeIterable NodesAdjacentTo(Node &node) {
        throw NotImplementedError{};
    }
};

template <class Value>
class GeneratorInterface : interface::VirtualBaseClass {
    virtual std::optional<Value> Next() = 0;
};

}  // namespace interface

namespace implementation {

struct DirectedEdge {
    int32_t from = 0;
    int32_t to = 0;

    DirectedEdge() = default;

    DirectedEdge(int32_t from, int32_t to) : from{from}, to{to} {
    }
};

struct UndirectedEdge : public DirectedEdge {
    using DirectedEdge::DirectedEdge;
};

class DirectedConnectByNodeIdsImplementation : public interface::ConnectByNodeIdsInterface<int32_t>,
                                               public interface::ResizeInterface {
public:
    std::vector<std::vector<int32_t>> adjacent_ids_table;

    void Resize(size_t size) override {
        adjacent_ids_table.resize(size);
    }

    [[nodiscard]] size_t Size() const override {
        return adjacent_ids_table.size();
    }

    void Connect(int32_t from, int32_t to) override {
        adjacent_ids_table[from].push_back(to);
    }
};

class UndirectedConnectByNodeIdsImplementation : public DirectedConnectByNodeIdsImplementation {
    void Connect(int32_t from, int32_t to) override {
        DirectedConnectByNodeIdsImplementation::Connect(from, to);
        if (from != to) {
            DirectedConnectByNodeIdsImplementation::Connect(to, from);
        }
    }
};

template <class Edge, class = std::enable_if_t<std::is_base_of_v<DirectedEdge, Edge>>>
class ConnectByEdgesImplementation : public interface::ConnectByEdgesInterface<Edge> {
public:
    std::vector<std::vector<Edge>> edges_table;

    void Resize(size_t size) override {
        edges_table.resize(size);
    }

    [[nodiscard]] size_t Size() const override {
        return edges_table.size();
    }

    void AddEdge(Edge edge) override {
        AddDirectedEdge(edge);

        if (std::is_base_of_v<UndirectedEdge, Edge> and edge.from != edge.to) {
            std::swap(edge.from, edge.to);
            AddDirectedEdge(edge);
        }
    }

    void AddDirectedEdge(Edge edge) {
        edges_table[edge.from].emplace_back(std::move(edge));
    }
};

template <class Node>
class NodeByIdImplementation : public interface::NodeByIdInterface<Node, int32_t> {
public:
    std::vector<Node> nodes;

    void Resize(size_t size) override {
        nodes.resize(size);
    }

    [[nodiscard]] size_t Size() const override {
        return nodes.size();
    }

    Node &operator[](int32_t node_id) override {
        return nodes[node_id];
    }

    Node operator[](int32_t node_id) const override {
        return nodes[node_id];
    }
};

class AdjacentByIdImplementation : public interface::AdjacentByIdInterface<int32_t>,
                                   public interface::ResizeInterface {
public:
    std::vector<std::vector<int32_t>> adjacent_ids_table;

    void Resize(size_t size) override {
        adjacent_ids_table.resize(size);
    }

    [[nodiscard]] size_t Size() const override {
        return adjacent_ids_table.size();
    }

    [[nodiscard]] const std::vector<int32_t> &NodeIdsAdjacentTo(int32_t node_id) const {
        return adjacent_ids_table[node_id];
    }
};

class DepthFirstSearch {
public:
    template <class Node>
    void Traverse(Node &starting_node) {
        TraverseRecursive(starting_node);
    }

private:
    template <class Node>
    void TraverseRecursive(Node &node) {

        OnNodeEnter(node);
        SetNodeStateEntered(node);

        for (auto &adjacent_node : NodeIdsAdjacentTo(node)) {

            OnEdgeDiscovery(node, adjacent_node);

            if (IsNodeUnvisited(adjacent_node)) {

                OnEdgeTraverse(node, adjacent_node);
                TraverseRecursive(adjacent_node);
            }
        }

        OnNodeExit(node);
        SetNodeStateExited(node);
    }
};

template <class GraphTraversal, class Node,
          class = std::enable_if_t<std::is_base_of_v<interface::GraphTraversal, GraphTraversal>>>
void BreadthFirstSearch(GraphTraversal *graph_traversal, std::deque<Node> starting_nodes_queue) {

    auto &queue = starting_nodes_queue;

    while (not queue.empty()) {
        auto node = queue.front();
        queue.pop_front();

        graph_traversal->OnNodeEnter(node);
        graph_traversal->SetNodeStateEntered(node);

        for (auto &adjacent_node : graph_traversal->NodesAdjacentTo(node)) {

            graph_traversal->OnEdgeDiscovery(node, adjacent_node);

            if (graph_traversal->IsNodeUnvisited(adjacent_node)) {

                graph_traversal->OnEdgeTraverse(node, adjacent_node);
                queue.emplace_back(adjacent_node);
            }
        }

        graph_traversal->OnNodeEnter(node);
        graph_traversal->SetNodeStateExited(node);
    }
}

template <class Generator,
          class = std::enable_if_t  // @formatter:off
          <std::is_base_of_v<       // @formatter:on
              interface::GeneratorInterface<typename Generator::Value>, Generator>>>
class GeneratorIterable : interface::VirtualBaseClass {
public:
    using Value = typename Generator::Value;

    struct Iterator {
        using IteratorCategory = std::forward_iterator_tag;
        using DifferenceType = std::ptrdiff_t;
        using ValueType = Value;
        using Pointer = Value *;
        using Reference = Value &;

        Iterator() = default;

        explicit Iterator(Generator generator) : optional_generator_{std::move(generator)} {
            PrecomputeValue();
        }

        ValueType operator*() const {
            return optional_value_.value();
        }

        Iterator &operator++() {
            if (IsGeneratorExhausted()) {
                throw std::runtime_error{"Generator is exhausted."};
            }
            PrecomputeValue();
            return *this;
        }

        friend bool operator==(const Iterator &left, const Iterator &right) {
            return left.IsGeneratorExhausted() and right.IsGeneratorExhausted();
        }

        friend bool operator!=(const Iterator &left, const Iterator &right) {
            return not operator==(left, right);
        }

    private:
        std::optional<Value> optional_value_;
        std::optional<Generator> optional_generator_;

        [[nodiscard]] bool IsGeneratorExhausted() const {
            return not optional_value_;
        }

        void PrecomputeValue() {
            optional_value_ = optional_generator_->Next();
        }
    };

    explicit GeneratorIterable(Generator generator) : generator_{std::move(generator)} {
    }

    Iterator begin() {
        return Iterator{generator_};
    }

    Iterator end() {
        return Iterator{};
    }

private:
    Generator generator_;
};

}  // namespace implementation

void FlipCoin(io::Coin *coin) {
    *coin = *coin == io::Coin::heads ? io::Coin::tails : io::Coin::heads;
}

void FlipAllCoinsOnTheBoard(io::Board *board) {
    for (auto &row : *board) {
        for (auto &coin : row) {
            FlipCoin(&coin);
        }
    }
}

void FlipCoinsOnTheBlackChessBoardSquares(io::Board *board) {
    for (size_t row = 0; row < board->size(); ++row) {
        for (size_t col = 0; col < board->front().size(); ++col) {
            if ((row + col) % 2 == 0) {
                FlipCoin(&(*board)[row][col]);
            }
        }
    }
}

class BoardNode {
public:
    io::Board board;

    explicit BoardNode(io::Board board) : board{std::move(board)} {
        UpdateUppermostLeftTailsCoordinates();
    }

    [[nodiscard]] int32_t NRows() const {
        return static_cast<int32_t>(board.size());
    }

    [[nodiscard]] int32_t NCols() const {
        return static_cast<int32_t>(board.front().size());
    }

    [[nodiscard]] bool HasAnyTails() const {
        return uppermost_left_tails_row_ < NRows();
    };

    [[nodiscard]] std::vector<BoardNode> ConstructChildren() const {
        if (not HasAnyTails()) {
            return {};
        }

        std::vector<BoardNode> children;
        for (auto &child : {DownwardChild(), RightChild()}) {
            if (child) {
                children.emplace_back(child.value());
            }
        }
        return children;
    }

    [[nodiscard]] int32_t GetNumberOfPairedCoinsFlips() const {
        return number_of_paired_coins_flips_;
    }

private:
    int32_t number_of_paired_coins_flips_ = 0;
    int32_t uppermost_left_tails_row_ = 0;
    int32_t uppermost_left_tails_col_ = 0;

    void UpdateUppermostLeftTailsCoordinates() {
        while (HasAnyTails() and
               board[uppermost_left_tails_row_][uppermost_left_tails_col_] != io::Coin::tails) {

            ++uppermost_left_tails_col_;

            if (uppermost_left_tails_col_ == NCols()) {

                ++uppermost_left_tails_row_;
                uppermost_left_tails_col_ = 0;
            }
        }
    }

    void FlipCoinAndTheOneDownwards(int32_t row, int32_t col) {
        FlipCoin(&board[row][col]);
        FlipCoin(&board[row + 1][col]);

        ++number_of_paired_coins_flips_;
    }

    void FlipCoinAndTheOneToTheRight(int32_t row, int32_t col) {
        FlipCoin(&board[row][col]);
        FlipCoin(&board[row][col + 1]);

        ++number_of_paired_coins_flips_;
    }

    [[nodiscard]] std::optional<BoardNode> DownwardChild() const {
        if (uppermost_left_tails_row_ >= NRows() - 1) {
            return std::nullopt;
        }

        auto child = *this;
        child.FlipCoinAndTheOneDownwards(uppermost_left_tails_row_, uppermost_left_tails_col_);
        child.UpdateUppermostLeftTailsCoordinates();
        return child;
    }

    [[nodiscard]] std::optional<BoardNode> RightChild() const {
        if (uppermost_left_tails_col_ >= NCols() - 1) {
            return std::nullopt;
        }

        auto child = *this;
        child.FlipCoinAndTheOneToTheRight(uppermost_left_tails_row_, uppermost_left_tails_col_);
        child.UpdateUppermostLeftTailsCoordinates();
        return child;
    }
};

class FoundShortestDistanceException : public std::exception {};

class BoardGraphTraversalInDirectionOfSmallerUpperLeftTailsCoordinates
    : public interface::GraphTraversal {
public:
    using Node = BoardNode;

    std::optional<int32_t> least_number_of_paired_coins_flips;

    void OnNodeEnter(Node &node) {
        if (not node.HasAnyTails() and not least_number_of_paired_coins_flips) {
            least_number_of_paired_coins_flips = node.GetNumberOfPairedCoinsFlips();
            throw FoundShortestDistanceException{};
        }
    }

    bool IsNodeUnvisited(Node &node) {
        return true;
    }

    std::vector<Node> NodesAdjacentTo(Node &node) {
        return node.ConstructChildren();
    }
};

std::optional<int32_t> FindLeastNumberOfPairedCoinsFlipsToAchieveBoardFullOfHeads(
    const io::Board &board) {

    BoardGraphTraversalInDirectionOfSmallerUpperLeftTailsCoordinates traversal;

    BoardNode board_node{board};

    try {
        implementation::BreadthFirstSearch(&traversal, std::deque<BoardNode>{board_node});
    } catch (const FoundShortestDistanceException &e) {
    }

    return traversal.least_number_of_paired_coins_flips;
}

Output Solve(const Input &input) {
    auto board = input.board;

    FlipCoinsOnTheBlackChessBoardSquares(&board);

    auto n_flips_for_tails_on_black_squares =
        FindLeastNumberOfPairedCoinsFlipsToAchieveBoardFullOfHeads(board);

    FlipAllCoinsOnTheBoard(&board);

    auto n_flips_for_heads_on_black_squares =
        FindLeastNumberOfPairedCoinsFlipsToAchieveBoardFullOfHeads(board);

    if (n_flips_for_tails_on_black_squares and n_flips_for_heads_on_black_squares) {
        return Output{std::min(n_flips_for_tails_on_black_squares.value(),
                               n_flips_for_heads_on_black_squares.value())};
    } else {
        return Output{
            std::max(n_flips_for_tails_on_black_squares, n_flips_for_heads_on_black_squares)};
    }
}

namespace test {

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
    Input input;
    std::optional<Output> optional_expected_output;

    explicit TestIo(Input input) : input{std::move(input)} {
    }

    TestIo(Input input, Output output)
        : input{std::move(input)}, optional_expected_output{std::move(output)} {
    }
};

Output BruteForceSolve(const Input &input) {
    throw NotImplementedError{};
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    throw NotImplementedError{};
    Input input;
    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    throw NotImplementedError{};
    Input input;
    return TestIo{input};
}

class TimedChecker {
public:
    std::vector<int64_t> durations;

    void Check(const std::string &test_case, const std::string &expected) {
        std::stringstream input_stream{test_case};
        Input input{input_stream};
        Output expected_output{expected};
        TestIo test_io{input, expected_output};
        Check(test_io);
    }

    void Check(TestIo test_io) {
        Output output;
        auto solve = [&output, &test_io]() { output = Solve(test_io.input); };

        durations.emplace_back(detail::Timeit(solve));

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

    TimedChecker timed_check;

    timed_check.Check(
        "2 3\n"
        "001\n"
        "011\n",
        "2");

    timed_check.Check(
        "2 2\n"
        "01\n"
        "00\n",
        "-1");

    timed_check.Check(
        "4 5\n"
        "01010\n"
        "10101\n"
        "01010\n"
        "10101\n"
        "01010\n",
        "0");

    timed_check.Check(
        "4 5\n"
        "01010\n"
        "11001\n"
        "01010\n"
        "10101\n"
        "01010\n",
        "1");

    timed_check.Check(
        "4 5\n"
        "01010\n"
        "11001\n"
        "01000\n"
        "10101\n"
        "01010\n",
        "-1");

    timed_check.Check(
        "1 1\n"
        "0\n",
        "0");

    timed_check.Check(
        "1 1\n"
        "1\n",
        "0");

    timed_check.Check(
        "3 1\n"
        "1\n"
        "1\n"
        "1\n",
        "2");

    timed_check.Check(
        "3 1\n"
        "0\n"
        "1\n"
        "1\n",
        "1");

    std::cerr << timed_check << "Basic tests OK\n";

    int32_t n_random_test_cases = 100;

    for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {
        timed_check.Check(GenerateRandomTestIo(test_case_id));
    }

    std::cerr << timed_check << "Random tests OK\n";

    int32_t n_stress_test_cases = 1;

    for (int32_t test_case_id = 0; test_case_id < n_stress_test_cases; ++test_case_id) {
        timed_check.Check(GenerateStressTestIo(test_case_id));
    }

    std::cerr << timed_check << "Stress tests tests OK\n";

    std::cerr << "OK\n";
}

}  // namespace test

void SetUp() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
}

int main(int argc, char *argv[]) {
    SetUp();
    if (argc > 1 && std::strcmp(argv[1], "test") == 0) {
        test::Test();
    } else {
        std::cout << Solve(Input{std::cin});
    }
    return 0;
}
