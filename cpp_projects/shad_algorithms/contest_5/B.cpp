#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <queue>
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

void PrintSeed(std::ostream& ostream = std::cerr) {
    ostream << "Seed = " << GetSeed() << std::endl;
}

std::mt19937* GetEngine() {
    static std::mt19937 engine(GetSeed());
    return &engine;
}

}  // namespace rng

template <class Dividend, class Divisor>
Divisor PositiveMod(Dividend value, Divisor divisor) {
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

namespace interface {

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
    virtual Node& operator[](NodeId node_id) = 0;

    virtual Node operator[](NodeId node_id) const = 0;
};

template <class NodeId = int32_t, class NodeIdIterable = const std::vector<NodeId>&>
class AdjacentByIdInterface : public VirtualBaseClass {
public:
    [[nodiscard]] virtual NodeIdIterable NodeIdsAdjacentTo(NodeId node_id) const = 0;
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

template <
    class Graph,
    class = std::enable_if_t < std::is_base_of_v <
        interface::AdjacentByIdInterface<typename Graph::NodeId, typename Graph::NodeIdIterable>,
        Graph>>,
    class = std::enable_if_t < std::is_base_of_v<
        interface::StateByIdInterface<typename Graph::NodeState, typename Graph::NodeId>, Graph>>>
struct GraphTraversal : public VirtualBaseClass {
    using NodeId = typename Graph::NodeId;

    Graph* graph;
    std::function<void(NodeId)> on_node_enter = [](NodeId) {};
    std::function<void(NodeId from, NodeId to)> on_edge_discovery = [](NodeId, NodeId) {};
    std::function<void(NodeId from, NodeId to)> on_edge_traverse = [](NodeId, NodeId) {};
    std::function<void(NodeId)> on_node_exit = [](NodeId) {};

    explicit GraphTraversal(Graph* graph) : graph{graph} {
    }

    virtual void Traverse() = 0;
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

    Node& operator[](int32_t node_id) override {
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

    [[nodiscard]] const std::vector<int32_t>& NodeIdsAdjacentTo(int32_t node_id) const override {
        return adjacent_ids_table[node_id];
    }
};

enum class NodeVisitedState { Unvisited, Entered, Exited };

template <class Graph>
class DepthFirstSearch : public interface::GraphTraversal<Graph> {
public:
    using interface::GraphTraversal<Graph>::on_node_enter;
    using interface::GraphTraversal<Graph>::on_edge_discovery;
    using interface::GraphTraversal<Graph>::on_edge_traverse;
    using interface::GraphTraversal<Graph>::on_node_exit;

    DepthFirstSearch(Graph* graph, typename Graph::NodeId starting_node_id)
        : interface::GraphTraversal<Graph>{graph},
          starting_node_id{starting_node_id} {}

          void Traverse() override {
        TraverseRecursive(starting_node_id);
    }

    typename Graph::NodeId starting_node_id;

private:
    using interface::GraphTraversal<Graph>::graph;

    void TraverseRecursive(typename Graph::NodeId node_id) {

        on_node_enter(node_id);
        graph->SetNodeState(node_id, NodeVisitedState::Entered);

        for (auto adjacent_node_id : graph->NodeIdsAdjacentTo(node_id)) {
            on_edge_discovery(node_id, adjacent_node_id);

            if (graph->GetNodeState(adjacent_node_id) == NodeVisitedState::Unvisited) {

                on_edge_traverse(node_id, adjacent_node_id);
                TraverseRecursive(adjacent_node_id);
            }
        }

        on_node_exit(node_id);
        graph->SetNodeState(node_id, NodeVisitedState::Exited);
    }
};

template <class Graph>
class BreadthFirstSearch : public interface::GraphTraversal<Graph> {
public:
    using interface::GraphTraversal<Graph>::on_node_enter;
    using interface::GraphTraversal<Graph>::on_edge_discovery;
    using interface::GraphTraversal<Graph>::on_edge_traverse;
    using interface::GraphTraversal<Graph>::on_node_exit;

    explicit BreadthFirstSearch(Graph* graph)
        : interface::GraphTraversal<Graph>{graph} {}

          void Traverse() override {
        auto& queue = starting_node_ids_queue;

        while (not queue.empty()) {
            auto node_id = queue.front();
            queue.pop();

            on_node_enter(node_id);
            graph->SetNodeState(node_id, NodeVisitedState::Entered);

            for (auto adjacent_node_id : graph->NodeIdsAdjacentTo(node_id)) {

                on_edge_discovery(node_id, adjacent_node_id);

                if (graph->GetNodeState(adjacent_node_id) == NodeVisitedState::Unvisited) {

                    on_edge_traverse(node_id, adjacent_node_id);
                    queue.emplace(adjacent_node_id);
                }
            }

            on_node_exit(node_id);
            graph->SetNodeState(node_id, NodeVisitedState::Exited);
        }
    }

    std::queue<typename Graph::NodeId>
        starting_node_ids_queue;

private:
    using interface::GraphTraversal<Graph>::graph;
};

}  // namespace implementation

struct Coordinate {
    int32_t line;
    int32_t column;

    Coordinate() = default;

    Coordinate(int32_t line, int32_t column) : line{line}, column{column} {
    }

    bool operator==(const Coordinate& other) const {
        return std::tie(line, column) == std::tie(other.line, other.column);
    }
};

using Path = std::vector<Coordinate>;

std::istream& operator>>(std::istream& is, Coordinate& coordinate) {
    is >> coordinate.line >> coordinate.column;
    return is;
}

struct Robot {
    static constexpr int32_t kFacingDirections = 4;
    static constexpr int32_t kLastTurns = 2;
    static constexpr int32_t kRobotStates = kFacingDirections * kLastTurns;

    enum class ClockwiseOrderedFacingDirection : int32_t { Down, Left, Up, Right };
    enum class LastTurn : int32_t { Left, Right };

    ClockwiseOrderedFacingDirection facing_direction;
    LastTurn last_turn;

    Robot(ClockwiseOrderedFacingDirection facing_direction, LastTurn last_turn)
        : facing_direction{facing_direction}, last_turn{last_turn} {
    }

    explicit Robot(int32_t state_id) {
        if (not IsValidStateId(state_id)) {
            throw std::invalid_argument{"Invalid state id " + std::to_string(state_id)};
        }
        facing_direction =
            static_cast<ClockwiseOrderedFacingDirection>(state_id % kFacingDirections);
        last_turn = static_cast<LastTurn>(state_id / kFacingDirections % kLastTurns);
    }

    [[nodiscard]] int32_t GetRobotStateId() const {
        auto state_id = kFacingDirections * static_cast<int32_t>(last_turn) +
                        static_cast<int32_t>(facing_direction);
        assert(IsValidStateId(state_id));
        return state_id;
    }

    static bool IsValidStateId(int32_t state_id) {
        return 0 <= state_id and state_id < kRobotStates;
    }

    void MakeTurn() {
        if (last_turn == LastTurn::Left) {
            ShiftFacingDirection(1);
            last_turn = LastTurn::Right;
        } else {
            ShiftFacingDirection(-1);
            last_turn = LastTurn::Left;
        }
    }

    void TurnAround() {
        ShiftFacingDirection(2);
    }

private:
    void ShiftFacingDirection(int32_t shift) {
        facing_direction = static_cast<ClockwiseOrderedFacingDirection>(
            PositiveMod(static_cast<int32_t>(facing_direction) + shift, kFacingDirections));
    }
};

struct Cell {
    enum class CellType { Blocked, Clear };

    CellType cell_type = CellType::Blocked;
    std::array<std::optional<int32_t>, Robot::kRobotStates> shorted_paths;

    std::optional<int32_t>& operator[](Robot robot) {
        return shorted_paths[robot.GetRobotStateId()];
    }
};

std::istream& operator>>(std::istream& is, Cell& cell) {
    char cell_type = '\0';
    is >> cell_type;

    if (cell_type == '.') {
        cell.cell_type = Cell::CellType::Clear;
    } else if (cell_type == '#') {
        cell.cell_type = Cell::CellType::Blocked;
    } else {
        throw std::invalid_argument{"Unknown cell type"};
    }
    return is;
}

struct Board {
    std::vector<std::vector<Cell>> cells;

    Board() = default;

    void Resize(int32_t n_rows, int32_t n_cols) {
        cells.resize(n_rows);
        for (auto& row : cells) {
            row.resize(n_cols);
        }
    }

    Cell& operator[](Coordinate coordinate) {
        return cells[coordinate.line][coordinate.column];
    }

    Cell operator[](Coordinate coordinate) const {
        return cells[coordinate.line][coordinate.column];
    }
};

namespace io {

class Input {
public:
    Board board;
    Coordinate start{};
    Coordinate finish{};

    Input() = default;

    explicit Input(std::istream& in) {
        int32_t line_count = 0;
        int32_t column_count = 0;
        in >> line_count >> column_count;

        board.Resize(line_count + 2, column_count + 2);

        for (int32_t line_index = 1; line_index < line_count + 1; ++line_index) {
            for (int32_t column_index = 1; column_index < column_count + 1; ++column_index) {
                auto& cell = board[{line_index, column_index}];
                in >> cell;
            }
        }

        in >> start >> finish;
    }
};

class Output {
public:
    std::optional<Path> path = std::nullopt;

    Output() = default;

    explicit Output(std::optional<Path> path) : path{std::move(path)} {
    }

    explicit Output(const std::string& string) {
        if (string == "-1") {
            return;
        }
        path = Path{};
        std::stringstream ss{string};
        Coordinate coordinate{};
        while (ss >> coordinate) {
            path.value().push_back(coordinate);
        }
    }

    std::ostream& Write(std::ostream& out) const {
        if (path) {
            out << path.value().size() - 1 << '\n';
            for (auto coordinate : path.value()) {
                out << coordinate.line << ' ' << coordinate.column << '\n';
            }
        } else {
            out << -1;
        }

        return out;
    }

    bool operator!=(const Output& other) const {
        return path != other.path;
    }
};

std::ostream& operator<<(std::ostream& os, Output const& output) {
    return output.Write(os);
}

}  // namespace io

using io::Input, io::Output;

using implementation::NodeVisitedState;

struct RobotOnACellNodeId {
    Coordinate coordinate;
    Robot robot;

    void GoForward() {
        switch (robot.facing_direction) {
            case Robot::ClockwiseOrderedFacingDirection::Down:
                --coordinate.line;
                break;
            case Robot::ClockwiseOrderedFacingDirection::Left:
                --coordinate.column;
                break;
            case Robot::ClockwiseOrderedFacingDirection::Up:
                ++coordinate.line;
                break;
            case Robot::ClockwiseOrderedFacingDirection::Right:
                ++coordinate.column;
                break;
        }
    }

    void GoBackward() {
        robot.TurnAround();
        GoForward();
        robot.TurnAround();
    }
};

class BoardGraph
    : public interface::AdjacentByIdInterface<RobotOnACellNodeId, std::vector<RobotOnACellNodeId>>,
      public interface::StateByIdInterface<NodeVisitedState, RobotOnACellNodeId> {
public:
    using NodeId = RobotOnACellNodeId;
    using NodeState = NodeVisitedState;
    using NodeIdIterable = std::vector<RobotOnACellNodeId>;

    Board& board;

    explicit BoardGraph(Board& board) : board{board} {
    }

    [[nodiscard]] NodeVisitedState GetNodeState(NodeId node_id) const override {
        return board[node_id.coordinate].shorted_paths[node_id.robot.GetRobotStateId()]
                   ? NodeVisitedState::Entered
                   : NodeVisitedState::Unvisited;
    }

    void SetNodeState(NodeId node_id, NodeVisitedState) override {
    }

    [[nodiscard]] NodeIdIterable NodeIdsAdjacentTo(NodeId node_id) const override {
        std::vector<NodeId> adjacent_candidates(3, node_id);

        adjacent_candidates[1].robot.TurnAround();
        adjacent_candidates[2].robot.MakeTurn();

        std::vector<RobotOnACellNodeId> adjacent_node_ids;
        for (auto& candidate_node_id : adjacent_candidates) {
            candidate_node_id.GoForward();
            if (IsInValidPosition(candidate_node_id)) {
                adjacent_node_ids.emplace_back(candidate_node_id);
            }
        }

        return adjacent_node_ids;
    }

    static std::vector<RobotOnACellNodeId> GetAllRobotStateVariations(Coordinate coordinate) {
        std::vector<RobotOnACellNodeId> nodes;
        for (int32_t state_id = 0; state_id < Robot::kRobotStates; ++state_id) {
            nodes.push_back({coordinate, Robot{state_id}});
        }
        return nodes;
    }

    std::optional<int32_t>& ShortestPathFromStartTo(RobotOnACellNodeId node_id) {
        auto& cell = board[node_id.coordinate];
        auto state_id = node_id.robot.GetRobotStateId();
        auto& shortest_path = cell.shorted_paths[state_id];
        return shortest_path;
    }

    [[nodiscard]] bool IsInValidPosition(RobotOnACellNodeId node_id) const {
        return board[node_id.coordinate].cell_type == Cell::CellType::Clear;
    }

    [[nodiscard]] std::vector<RobotOnACellNodeId> GetAdjacentParentNodeIds(
        RobotOnACellNodeId node_id) const {
        node_id.GoBackward();
        if (not IsInValidPosition(node_id)) {
            return {};
        }

        std::vector<RobotOnACellNodeId> adjacent_parent_node_ids(3, node_id);

        adjacent_parent_node_ids[1].robot.TurnAround();
        adjacent_parent_node_ids[2].robot.MakeTurn();

        return adjacent_parent_node_ids;
    }
};

struct ComparatorNodeIdsShortestPath {
    BoardGraph* graph;

    explicit ComparatorNodeIdsShortestPath(BoardGraph* graph) : graph{graph} {
    }

    bool operator()(const RobotOnACellNodeId& left, const RobotOnACellNodeId& right) const {
        auto left_path = graph->ShortestPathFromStartTo(left);
        auto right_path = graph->ShortestPathFromStartTo(right);
        return not right_path or (left_path and left_path.value() < right_path.value());
    }
};

void ComputeShortestPaths(BoardGraph* graph, Coordinate start) {

    implementation::BreadthFirstSearch bfs{graph};

    auto start_node_ids_variations = BoardGraph::GetAllRobotStateVariations(start);
    for (auto& node_id : start_node_ids_variations) {
        graph->ShortestPathFromStartTo(node_id) = 0;
        bfs.starting_node_ids_queue.push(node_id);
    }

    int32_t path_length = 0;
    bfs.on_node_enter = [&path_length, graph](RobotOnACellNodeId node_id) {
        if (path_length == graph->ShortestPathFromStartTo(node_id)) {
            ++path_length;
        }
    };

    bfs.on_edge_traverse = [&path_length, graph](RobotOnACellNodeId parent,
                                                 RobotOnACellNodeId child) {
        graph->ShortestPathFromStartTo(child) = path_length;
    };

    bfs.Traverse();
}

Path ReconstructShortestPathInReverse(RobotOnACellNodeId finish_node_id, Coordinate start,
                                      BoardGraph* graph) {
    Path path{finish_node_id.coordinate};

    while (not(finish_node_id.coordinate == start)) {
        for (const auto& previous_node_id : graph->GetAdjacentParentNodeIds(finish_node_id)) {
            if (graph->ShortestPathFromStartTo(previous_node_id) and
                graph->ShortestPathFromStartTo(previous_node_id).value() ==
                    graph->ShortestPathFromStartTo(finish_node_id).value() - 1) {

                finish_node_id = previous_node_id;
                if (not(finish_node_id.coordinate == path.back())) {
                    path.emplace_back(finish_node_id.coordinate);
                }
                break;
            }
        }
    }

    std::reverse(path.begin(), path.end());
    return path;
}

std::optional<Path> FindShortestPath(BoardGraph* graph, Coordinate start, Coordinate finish) {

    auto finish_variations = BoardGraph::GetAllRobotStateVariations(finish);

    auto finish_node_id = *std::min_element(finish_variations.begin(), finish_variations.end(),
                                            ComparatorNodeIdsShortestPath{graph});

    if (not(graph->ShortestPathFromStartTo(finish_node_id))) {
        return std::nullopt;
    }

    return ReconstructShortestPathInReverse(finish_node_id, start, graph);
}

Output Solve(Input input) {
    BoardGraph graph{input.board};

    ComputeShortestPaths(&graph, input.start);

    auto optional_shortest_path = FindShortestPath(&graph, input.start, input.finish);

    return Output{optional_shortest_path};
}

namespace test {

class WrongAnswerException : public std::exception {
public:
    WrongAnswerException() = default;

    explicit WrongAnswerException(std::string const& message) : message{message.data()} {
    }

    [[nodiscard]] const char* what() const noexcept override {
        return message;
    }

    const char* message{};
};

class NotImplementedError : public std::logic_error {
public:
    NotImplementedError() : std::logic_error("Function not yet implemented."){};
};

Output BruteForceSolve(Input input) {
    throw NotImplementedError{};
}

struct TestIo {
    Input input;
    std::optional<Output> optional_expected_output = std::nullopt;

    explicit TestIo(Input input) {
        try {
            optional_expected_output = BruteForceSolve(input);
        } catch (const NotImplementedError& e) {
        }
        this->input = std::move(input);
    }

    TestIo(Input input, Output output)
        : input{std::move(input)}, optional_expected_output{std::move(output)} {
    }
};

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

int64_t Check(const TestIo& test_io) {

    TimeItInMilliseconds time;
    auto output = Solve(test_io.input);
    time.End();

    if (test_io.optional_expected_output) {
        auto& expected_output = test_io.optional_expected_output.value();

        if (output != expected_output) {
            Solve(test_io.input);
            std::stringstream ss;
            ss << "\n==================================Expected================================"
                  "==\n"
               << expected_output
               << "\n==================================Received================================"
                  "==\n"
               << output << "\n";
            throw WrongAnswerException{ss.str()};
        }
    }

    return time.Duration();
}

int64_t Check(const std::string& test_case, const std::string& expected) {
    std::stringstream input_stream{test_case};
    return Check(TestIo{Input{input_stream}, Output{expected}});
}

struct Stats {
    double mean = 0;
    double std = 0;
    double max = 0;
};

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

void Test() {
    rng::PrintSeed();

    Check(
        "5 5\n"
        "#...#"
        "..#.#"
        "..#.#"
        "..#.#"
        "#...."
        "2 2\n"
        "2 4",

        "2 2\n"
        "3 2\n"
        "4 2\n"
        "5 2\n"
        "5 3\n"
        "5 4\n"
        "5 5\n"
        "5 4\n"
        "4 4\n"
        "3 4\n"
        "2 4");

    Check(
        "4 4"
        "...."
        "###."
        "...."
        "####"
        "1 1\n"
        "3 1",

        "-1");

    Check(
        "1 1"
        "."
        "1 1\n"
        "1 1",

        "1 1");

    Check(
        "1 3"
        "..."
        "1 3\n"
        "1 2",

        "1 3\n"
        "1 2");

    Check(
        "3 1"
        "#"
        "."
        "."
        "3 1\n"
        "2 1",

        "3 1\n"
        "2 1");

    std::cerr << "Basic tests OK\n";

    std::vector<int64_t> durations;
    TimeItInMilliseconds time_it;

    int32_t n_random_test_cases = 0;

    for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {
        durations.emplace_back(Check(GenerateRandomTestIo(test_case_id)));
    }

    if (n_random_test_cases > 0) {
        std::cerr << "Random tests OK\n";
    }

    int32_t n_stress_test_cases = 0;
    for (int32_t test_case_id = 0; test_case_id < n_stress_test_cases; ++test_case_id) {
        durations.emplace_back(Check(GenerateStressTestIo(test_case_id)));
    }

    if (n_stress_test_cases > 0) {
        std::cerr << "Stress tests tests OK\n";
    }

    if (not durations.empty()) {
        auto duration_stats = ComputeStats(durations.begin(), durations.end());
        std::cerr << "Solve duration stats in milliseconds:\n"
                  << "\tMean:\t" + std::to_string(duration_stats.mean) << '\n'
                  << "\tStd:\t" + std::to_string(duration_stats.std) << '\n'
                  << "\tMax:\t" + std::to_string(duration_stats.max) << '\n';
    }

    std::cerr << "OK\n";
}

}  // namespace test

void SetUp() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
}

int main(int argc, char* argv[]) {
    SetUp();
    if (argc > 1 && std::strcmp(argv[1], "test") == 0) {
        test::Test();
    } else {
        std::cout << Solve(Input{std::cin});
    }
    return 0;
}
