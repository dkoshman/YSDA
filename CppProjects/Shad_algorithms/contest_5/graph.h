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

namespace test {

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

std::vector<int32_t>
GenerateRandomPruferCode(int32_t size) {
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

template <
    class Graph, class = std::enable_if_t<std::is_base_of_v<interface::ResizeInterface, Graph>>,
    class =
        std::enable_if_t<std::is_base_of_v<interface::ConnectByNodeIdsInterface<int32_t>, Graph>>>
Graph GenerateRandomGraphWithLoopsAndParallelEdges(int32_t n_nodes, int32_t n_edges) {

    auto node_id_distribution = std::uniform_int_distribution<int32_t>{0, n_nodes - 1};

    Graph graph;
    graph.Resize(n_nodes);

    for (int32_t i = 0; i < n_edges; ++i) {
        auto from = node_id_distribution(*rng::GetEngine());
        auto to = node_id_distribution(*rng::GetEngine());
        graph.Connect(from, to);
    }

    return graph;
}

}  // namespace test

