#include <algorithm>
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

void PrintSeed(std::ostream &ostream = std::cerr) {
    ostream << "Seed = " << GetSeed() << std::endl;
}

std::mt19937 *GetEngine() {
    static std::mt19937 engine(GetSeed());
    return &engine;
}

}  // namespace rng

namespace interface {

class ResizeInterface {
public:
    virtual void Resize(size_t size) = 0;

    [[nodiscard]] virtual size_t Size() const = 0;

    virtual ~ResizeInterface() = default;
};

template <class NodeId = int32_t>
class ConnectByNodeIdsInterface : public ResizeInterface {
public:
    virtual void Connect(NodeId from, NodeId to) = 0;
};

template <class Edge>
class ConnectByEdgesInterface : public ResizeInterface {
public:
    virtual void AddEdge(Edge edge) = 0;
};

template <class Node, class NodeId = int32_t>
class NodeByIdInterface : public ResizeInterface {
public:
    virtual Node &operator[](NodeId node_id) = 0;

    virtual Node operator[](NodeId node_id) const = 0;
};

template <class NodeId = int32_t, class NodeIdIterable = const std::vector<NodeId> &>
class AdjacentByIdInterface : public ResizeInterface {
public:
    [[nodiscard]] virtual NodeIdIterable NodeIdsAdjacentTo(NodeId node_id) const = 0;
};

template <class NodeState>
class StateInterface {
public:
    [[nodiscard]] virtual NodeState GetState() const = 0;

    virtual void SetState(NodeState state) = 0;

    virtual ~StateInterface() = default;
};

template <class NodeState, class NodeId = int32_t>
class StateByIdInterface {
public:
    [[nodiscard]] virtual NodeState GetNodeState(NodeId node_id) const = 0;

    virtual void SetNodeState(NodeId node_id, NodeState state) = 0;

    virtual ~StateByIdInterface() = default;
};

template <
    class Graph,
    class = std::enable_if_t<
        std::is_base_of_v<interface::AdjacentByIdInterface<typename Graph::NodeId>, Graph>>,
    class = std::enable_if_t < std::is_base_of_v <
        interface::StateByIdInterface<typename Graph::NodeState, typename Graph::NodeId>, Graph>>>
struct GraphTraversal {
    using NodeId = typename Graph::NodeId;

    Graph *graph;
    std::function<void(NodeId)> on_node_enter = [](NodeId) {};
    std::function<void(NodeId from, NodeId to)> on_edge_discovery = [](NodeId, NodeId) {};
    std::function<void(NodeId from, NodeId to)> on_edge_traverse = [](NodeId, NodeId) {};
    std::function<void(NodeId)> on_node_exit = [](NodeId) {};

    explicit GraphTraversal(Graph *graph) : graph{graph} {
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

class DirectedConnectByNodeIdsImplementation
    : public interface::ConnectByNodeIdsInterface<int32_t> {
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

class AdjacentByIdImplementation : public interface::AdjacentByIdInterface<int32_t> {
public:
    std::vector<std::vector<int32_t>> adjacent_ids_table;

    void Resize(size_t size) override {
        adjacent_ids_table.resize(size);
    }

    [[nodiscard]] size_t Size() const override {
        return adjacent_ids_table.size();
    }

    [[nodiscard]] const std::vector<int32_t> &NodeIdsAdjacentTo(int32_t node_id) const override {
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

    DepthFirstSearch(Graph *graph, typename Graph::NodeId starting_node_id)
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
    using interface::GraphTraversal<Graph>::graph;

    std::queue<typename Graph::NodeId> starting_node_ids_queue;

    void Traverse() override {
        auto &queue = starting_node_ids_queue;

        while (not queue.empty()) {
            auto node_id = queue.front();
            queue.pop();

            on_node_enter(node_id);
            graph.SetNodeState(node_id, NodeVisitedState::Entered);

            static auto function = [&node_id, &queue](auto &adjacent_node_id) {
                on_edge_discovery(node_id, adjacent_node_id);

                if (graph.GetNodeState(adjacent_node_id) == NodeVisitedState::Unvisited) {

                    on_edge_traverse(node_id, adjacent_node_id);
                    queue.emplace(adjacent_node_id);
                }
            };

            graph.VisitNodeIdsAdjacentTo(node_id, function);

            on_node_exit(node_id);
            graph.SetNodeState(node_id, NodeVisitedState::Exited);
        }
    }
};

}  // namespace implementation

using implementation::NodeVisitedState;

struct WeightedUndirectedEdge {
    int32_t from = 0;
    int32_t to = 0;
    int32_t weight = 0;

    WeightedUndirectedEdge() = default;

    WeightedUndirectedEdge(int32_t from, int32_t to, int32_t weight)
        : from{from}, to{to}, weight{weight} {
    }
};

bool ComparatorEdgeWeight(const WeightedUndirectedEdge &left, const WeightedUndirectedEdge &right) {
    return left.weight < right.weight;
}

class GraphWithWeightedEdgesAndNodeIds
    : public interface::ConnectByEdgesInterface<WeightedUndirectedEdge> {
public:
    using Edge = WeightedUndirectedEdge;

    void Resize(size_t size) override {
        edges_table.resize(size);
        adjacent_ids_table.resize(size);
    }

    [[nodiscard]] size_t Size() const override {
        return edges_table.size();
    }

    void AddEdge(WeightedUndirectedEdge edge) override {
        AddDirectedEdge(edge);
        if (edge.from != edge.to) {
            std::swap(edge.from, edge.to);
            AddDirectedEdge(edge);
        }
    }

    template <class DirectedEdge>
    void AddDirectedEdge(DirectedEdge edge) {
        edges_table[edge.from].emplace_back(edge);
        adjacent_ids_table[edge.from].emplace_back(edge.to);
    }

    std::vector<std::vector<Edge>> edges_table;
    std::vector<std::vector<int32_t>> adjacent_ids_table;
};

namespace io {

class Input {
public:
    GraphWithWeightedEdgesAndNodeIds graph;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t n_nodes = 0;
        int32_t n_edges = 0;
        in >> n_nodes >> n_edges;
        graph.Resize(n_nodes);

        for (int32_t index = 0; index < n_edges; ++index) {
            int32_t from = 0;
            int32_t to = 0;
            int32_t weight = 0;

            in >> from >> to >> weight;

            --from;
            --to;

            WeightedUndirectedEdge edge{from, to, weight};
            graph.AddEdge(edge);
        }
    }
};

class Output {
public:
    std::optional<int32_t> answer = std::nullopt;

    Output() = default;

    explicit Output(std::optional<int32_t> cheapest_escape_edge) : answer{cheapest_escape_edge} {
    }

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        int32_t item = 0;
        ss >> item;
        if (item != -1) {
            answer = item;
        }
    }

    std::ostream &Write(std::ostream &out) const {
        out << (answer ? answer.value() : -1) << '\n';
        return out;
    }

    bool operator!=(const Output &other) const {
        return answer != other.answer;
    }
};

std::ostream &operator<<(std::ostream &os, Output const &output) {
    return output.Write(os);
}

}  // namespace io

using io::Input, io::Output;

struct DfsNode {
    NodeVisitedState state = NodeVisitedState::Unvisited;
    int32_t node_id = 0;
    int32_t enter = 0;
    int32_t reach = 0;
    int32_t depth = 0;
    int32_t parent_node_id_in_dfs_spanning_tree = 0;

    [[nodiscard]] bool HasBackwardEdges() const {
        return reach < enter;
    }
};

bool ComparatorNodeDepthGreater(const DfsNode &left, const DfsNode &right) {
    return left.depth > right.depth;
}

class DfsGraph : public interface::AdjacentByIdInterface<>,
                 public interface::StateByIdInterface<NodeVisitedState>,
                 public interface::NodeByIdInterface<DfsNode> {
public:
    using NodeId = int32_t;
    using NodeState = NodeVisitedState;
    using Node = DfsNode;
    using Edge = WeightedUndirectedEdge;

    explicit DfsGraph(GraphWithWeightedEdgesAndNodeIds &&weighted_graph)
        : weighted_graph{std::move(weighted_graph)} {

        nodes.resize(this->weighted_graph.Size());
        for (int32_t node_id = 0; node_id < static_cast<int32_t>(nodes.size()); ++node_id) {
            nodes[node_id].node_id = node_id;
        }
    };

    void Resize(size_t size) override {
        nodes.resize(size);
        weighted_graph.Resize(size);
    }

    [[nodiscard]] size_t Size() const override {
        return nodes.size();
    }

    [[nodiscard]] const std::vector<int32_t> &NodeIdsAdjacentTo(int32_t node_id) const override {
        return weighted_graph.adjacent_ids_table[node_id];
    }

    Node &operator[](NodeId node_id) override {
        return nodes[node_id];
    }

    Node operator[](NodeId node_id) const override {
        return nodes[node_id];
    }

    [[nodiscard]] NodeState GetNodeState(NodeId node_id) const override {
        return operator[](node_id).state;
    }

    void SetNodeState(NodeId node_id, NodeState state) override {
        operator[](node_id).state = state;
    }

    void ResetNodeStates() {
        for (auto &node : nodes) {
            node.state = NodeVisitedState::Unvisited;
        }
    }

    template <class Edge>
    [[nodiscard]] bool IsDirectedEscapeEdge(Edge edge) const {
        return IsDirectedEdgeInDfsSpanningTree(edge) and not nodes[edge.to].HasBackwardEdges();
    }

    template <class Edge>
    [[nodiscard]] bool IsDirectedEdgeInDfsSpanningTree(Edge edge) const {
        return nodes[edge.to].parent_node_id_in_dfs_spanning_tree == edge.from;
    }

    [[nodiscard]] bool IsArticulationPoint(int32_t node_id) {
        auto doesnt_have_backward_edge = [this](int32_t node_id) {
            return not this->nodes[node_id].HasBackwardEdges();
        };
        auto adjacent_ids = NodeIdsAdjacentTo(node_id);
        auto a_neighbor_doesnt_have_backward_edge =
            std::any_of(adjacent_ids.begin(), adjacent_ids.end(), doesnt_have_backward_edge);
        return a_neighbor_doesnt_have_backward_edge;
    }

    void UpdateReach(int32_t node_id) {
        auto &node = nodes[node_id];
        auto reach = node.enter;

        for (auto adjacent_node_id : NodeIdsAdjacentTo(node_id)) {
            auto &adjacent_node = nodes[adjacent_node_id];

            if (node.depth < adjacent_node.depth) {
                reach = std::min(reach, adjacent_node.reach);

            } else if (node.parent_node_id_in_dfs_spanning_tree != adjacent_node_id) {
                reach = std::min(reach, adjacent_node.enter);
            }
        }

        node.reach = reach;
    }

    GraphWithWeightedEdgesAndNodeIds weighted_graph;
    std::vector<DfsNode> nodes;
};

void AssignEnterDepthAndParentToNodes(DfsGraph *graph_ptr, int32_t starting_node_id = 0) {
    auto &graph = *graph_ptr;

    graph.ResetNodeStates();

    implementation::DepthFirstSearch<DfsGraph> dfs{graph_ptr, starting_node_id};

    int32_t enter = 0;

    dfs.on_node_enter = [&enter, &graph](int32_t node_id) { graph[node_id].enter = enter++; };

    dfs.on_edge_traverse = [&graph](int32_t from, int32_t to) {
        graph[to].depth = graph[from].depth + 1;
        graph[to].parent_node_id_in_dfs_spanning_tree = from;
    };

    dfs.Traverse();
}

void CalculateReachForNodes(DfsGraph *graph) {
    auto nodes_by_decreasing_depth = graph->nodes;
    std::sort(nodes_by_decreasing_depth.begin(), nodes_by_decreasing_depth.end(),
              ComparatorNodeDepthGreater);

    for (const auto &vertex_sorted : nodes_by_decreasing_depth) {
        graph->UpdateReach(vertex_sorted.node_id);
    }
}

void VisitEscapeEdges(DfsGraph *graph, const std::function<void(DfsGraph::Edge &)> &edge_function) {
    for (auto &edges_from : graph->weighted_graph.edges_table) {
        for (auto &edge : edges_from) {
            if (graph->IsDirectedEscapeEdge(edge)) {
                edge_function(edge);
            }
        }
    }
}

std::optional<int32_t> FindLightestEscapeEdgeWeight(DfsGraph *graph) {
    std::optional<int32_t> lightest_weight;

    auto edge_function = [graph, &lightest_weight](DfsGraph::Edge &edge) {
        if (graph->IsDirectedEscapeEdge(edge)) {
            if (not lightest_weight) {
                lightest_weight = edge.weight;
            } else {
                lightest_weight = std::min(edge.weight, lightest_weight.value());
            }
        }
    };

    VisitEscapeEdges(graph, edge_function);

    return lightest_weight;
}

Output Solve(Input input) {
    DfsGraph graph{std::move(input.graph)};

    AssignEnterDepthAndParentToNodes(&graph);

    CalculateReachForNodes(&graph);

    auto lightest_escape_edge_weight = FindLightestEscapeEdgeWeight(&graph);

    return Output{lightest_escape_edge_weight};
}

namespace test {

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

Output BruteForceSolve(const Input &input) {
    throw NotImplementedError{};
}

struct TestIo {
    Input input;
    std::optional<Output> optional_expected_output = std::nullopt;

    explicit TestIo(Input input) {
        try {
            optional_expected_output = BruteForceSolve(input);
        } catch (const NotImplementedError &e) {
        }
        this->input = std::move(input);
    }

    TestIo(Input input, Output output)
        : input{std::move(input)}, optional_expected_output{std::move(output)} {
    }
};

template <class Graph, class = std::enable_if_t<
                           std::is_base_of_v<interface::ConnectByNodeIdsInterface<int32_t>, Graph>>>
Graph BuildTreeFromPruferCode(const std::vector<int32_t> &prufer_code) {
    Graph tree;
    tree.Resize(prufer_code.size() + 2);

    std::vector<int32_t> count(prufer_code.size() + 2);
    for (auto node_id : prufer_code) {
        ++count[node_id];
    }

    int32_t order = 0;
    int32_t next = 0;

    for (auto node_id : prufer_code) {
        if (count[next]) {
            while (count[order]) {
                ++order;
            }
            next = order;
        }

        tree.Connect(next, node_id);

        --count[node_id];
        --count[next];
        if (not count[node_id] and node_id < order) {
            next = node_id;
        }
    }

    while (count[next]) {
        ++next;
    }
    tree.Connect(next, static_cast<int32_t>(prufer_code.size() + 1));

    return tree;
}

class RandomWeightedGraph : public GraphWithWeightedEdgesAndNodeIds,
                            public interface::ConnectByNodeIdsInterface<int32_t> {
public:
    std::uniform_int_distribution<int32_t> weight_distribution{1, 1'000'000'000};

    [[nodiscard]] GraphWithWeightedEdgesAndNodeIds GetGraph() const {
        GraphWithWeightedEdgesAndNodeIds graph;
        graph.edges_table = edges_table;
        graph.adjacent_ids_table = adjacent_ids_table;
        return graph;
    }

    [[nodiscard]] size_t Size() const override {
        return GraphWithWeightedEdgesAndNodeIds::Size();
    }

    void Resize(size_t size) override {
        GraphWithWeightedEdgesAndNodeIds::Resize(size);
    }

    void Concatenate(const RandomWeightedGraph &other) {
        auto other_size = static_cast<int32_t>(other.edges_table.size());
        edges_table.insert(edges_table.end(), other.edges_table.begin(), other.edges_table.end());
        for (auto i = other_size; i < static_cast<int32_t>(edges_table.size()); ++i) {
            for (auto &edge : edges_table[i]) {
                edge.to += other_size;
                edge.from += other_size;
            }
        }

        adjacent_ids_table.insert(adjacent_ids_table.end(), other.adjacent_ids_table.begin(),
                                  other.adjacent_ids_table.end());
        for (auto i = other_size; i < static_cast<int32_t>(edges_table.size()); ++i) {
            for (auto &edge : adjacent_ids_table[i]) {
                edge += other_size;
            }
        }
    }

    void Connect(int32_t from, int32_t to) override {
        WeightedUndirectedEdge edge{from, to, weight_distribution(*rng::GetEngine())};
        GraphWithWeightedEdgesAndNodeIds::AddEdge(edge);
    }

    [[nodiscard]] RandomWeightedGraph GetComplementGraph() const {
        RandomWeightedGraph complement_edges;
        complement_edges.Resize(edges_table.size());

        std::vector<bool> flags;

        for (int32_t from = 0; from < static_cast<int32_t>(edges_table.size()); ++from) {
            flags.clear();
            flags.resize(edges_table.size());
            flags[from] = true;

            for (auto &edge : edges_table[from]) {
                flags[edge.to] = true;
            }

            for (int32_t to = 0; to < static_cast<int32_t>(flags.size()); ++to) {
                if (not flags[to]) {
                    complement_edges.Connect(from, to);
                }
            }
        }

        return complement_edges;
    }

    [[nodiscard]] std::vector<WeightedUndirectedEdge> GetAllEdges() const {
        std::vector<WeightedUndirectedEdge> result;
        for (auto &edges_from : edges_table) {
            result.insert(result.end(), edges_from.begin(), edges_from.end());
        }
        return result;
    }
};

std::vector<int32_t> GenerateRandomPruferCode(int32_t size) {
    if (size < 2) {
        throw std::invalid_argument{"Prufer code is defined for trees with at least 2 nodes."};
    }

    std::uniform_int_distribution<int32_t> node_id_distribution(0, size - 1);
    std::vector<int32_t> prufer_code;
    prufer_code.reserve(size - 2);

    for (int32_t i = 0; i < size - 2; ++i) {
        prufer_code.push_back(node_id_distribution(*rng::GetEngine()));
    }
    return prufer_code;
}

template <class Graph, class = std::enable_if_t<
                           std::is_base_of_v<interface::ConnectByNodeIdsInterface<int32_t>, Graph>>>
Graph GenerateRandomTree(int32_t size) {
    if (size < 2) {
        Graph tree;
        tree.Resize(size);
        return tree;
    }

    auto prufer_code = GenerateRandomPruferCode(size);

    auto tree = BuildTreeFromPruferCode<Graph>(prufer_code);

    return tree;
}

template <class Graph, class = std::enable_if_t<
                           std::is_base_of_v<interface::ConnectByNodeIdsInterface<int32_t>, Graph>>>
Graph GenerateRandomConnectedGraph(int32_t n_nodes, int32_t n_edges) {
    if (n_edges + 1 < n_nodes) {
        throw std::invalid_argument{"Not enough edges_table for a connected graph_."};
    }

    if (n_edges > n_nodes * (n_nodes - 1)) {
        throw std::invalid_argument{"Too many edges_table."};
    }

    auto graph = GenerateRandomTree<Graph>(n_nodes);

    auto complement_graph = graph.GetComplementGraph();
    auto complement_edges = complement_graph.GetAllEdges();
    std::shuffle(complement_edges.begin(), complement_edges.end(), *rng::GetEngine());

    int32_t n_edges_to_add = n_edges - (n_nodes - 1);
    for (int32_t i = 0; i < n_edges_to_add; ++i) {
        graph.Connect(complement_edges[i].from, complement_edges[i].to);
    }

    return graph;
}

TestIo
GenerateRandomTestIo(int32_t test_case_id) {

    auto n_nodes = test_case_id + 1;
    auto n_edges = (test_case_id + 1) * test_case_id / 2;

    Input input;

    if (n_nodes % 2) {
        auto random_graph = GenerateRandomConnectedGraph<RandomWeightedGraph>(n_nodes, n_edges);
        input.graph = random_graph.GetGraph();
        TestIo{input};
    }

    auto first = GenerateRandomConnectedGraph<RandomWeightedGraph>(n_nodes, n_edges);
    auto second = GenerateRandomConnectedGraph<RandomWeightedGraph>(n_nodes, n_edges);

    first.Concatenate(second);

    first.AddEdge({0, n_nodes, 0});

    input.graph = first.GetGraph();

    return TestIo{input, Output{"0"}};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    if (test_case_id == 0) {
        return GenerateRandomTestIo(1000);
    }
    int32_t n_nodes = 100;
    auto n_edges = n_nodes * (n_nodes - 1) / 2;
    Input input;
    auto random_graph = GenerateRandomConnectedGraph<RandomWeightedGraph>(n_nodes, n_edges);
    input.graph = random_graph.GetGraph();
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

int64_t Check(const TestIo &test_io) {

    TimeItInMilliseconds time;
    auto output = Solve(test_io.input);
    time.End();

    if (test_io.optional_expected_output) {
        auto &expected_output = test_io.optional_expected_output.value();

        if (output != expected_output) {
            Solve(test_io.input);
            std::stringstream ss;
            ss << "\n==================================Expected==================================\n"
               << expected_output
               << "\n==================================Received==================================\n"
               << output << "\n";
            throw WrongAnswerException{ss.str()};
        }
    }

    return time.Duration();
}

int64_t Check(const std::string &test_case, const std::string &expected) {
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

    Check("1 0\n", "-1");

    Check(
        "2 1\n"
        "1 2 10\n",
        "10");

    Check(
        "3 3\n"
        "1 2 10\n"
        "2 3 10\n"
        "3 1 10\n",
        "-1");

    Check(
        "7 6\n"
        "1 2 1\n"
        "1 3 2\n"
        "2 4 3\n"
        "2 5 4\n"
        "3 6 5\n"
        "3 7 6\n",
        "1");

    Check(
        "6 7\n"
        "1 2 1\n"
        "2 3 2\n"
        "3 1 3\n"
        "2 4 4\n"
        "4 5 5\n"
        "5 6 6\n"
        "6 2 7\n",
        "-1");

    Check(
        "7 8\n"
        "1 2 1\n"
        "2 3 2\n"
        "3 4 3\n"
        "4 1 4\n"
        "3 5 5\n"
        "5 6 6\n"
        "6 7 7\n"
        "7 5 8\n",
        "5");

    std::cerr << "Basic tests OK\n";

    std::vector<int64_t> durations;
    TimeItInMilliseconds time_it;

    int32_t n_random_test_cases = 100;

    for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {
        durations.emplace_back(Check(GenerateRandomTestIo(test_case_id)));
    }

    if (n_random_test_cases > 0) {
        std::cerr << "Random tests OK\n";
    }

    int32_t n_stress_test_cases = 2;
    for (int32_t test_case_id = 0; test_case_id < n_stress_test_cases; ++test_case_id) {
        durations.emplace_back(Check(GenerateStressTestIo(test_case_id)));
    }

    if (n_stress_test_cases > 0) {
        std::cerr << "Stress tests tests OK\n";
    }

    auto duration_stats = ComputeStats(durations.begin(), durations.end());
    std::cerr << "Solve duration stats in milliseconds:\n"
              << "\tMean:\t" + std::to_string(duration_stats.mean) << '\n'
              << "\tStd:\t" + std::to_string(duration_stats.std) << '\n'
              << "\tMax:\t" + std::to_string(duration_stats.max) << '\n';

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
