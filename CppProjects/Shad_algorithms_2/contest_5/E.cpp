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
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace io {

struct Coordinate {
    int32_t first = 0;
    int32_t second = 0;

    Coordinate() = default;

    Coordinate(int32_t first, int32_t second) : first{first}, second{second} {
    }
};

std::istream &operator>>(std::istream &is, Coordinate &coordinate) {
    return is >> coordinate.first >> coordinate.second;
}

class Input {
public:
    int32_t n_rows = 0;
    int32_t n_cols = 0;
    std::vector<Coordinate> mountains;
    std::vector<Coordinate> gates;
    Coordinate first_city;
    Coordinate second_city;

    Input() = default;

    explicit Input(std::istream &in) {
        in >> n_rows >> n_cols;

        int32_t n_mountains = 0;
        int32_t n_gates = 0;
        in >> n_mountains >> n_gates;
        mountains.resize(n_mountains);
        for (auto &mountain : mountains) {
            in >> mountain;
            --mountain.first;
            --mountain.second;
        }
        gates.resize(n_gates);
        for (auto &gate : gates) {
            in >> gate;
            --gate.first;
            --gate.second;
        }

        in >> first_city >> second_city;
        --first_city.first;
        --first_city.second;
        --second_city.first;
        --second_city.second;
    }
};

class Output {
public:
    std::optional<int32_t> min_n_gates_to_separate_cities;
    std::vector<Coordinate> gates;

    Output() = default;

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        int32_t n_gates = 0;
        ss >> n_gates;
        if (n_gates != -1) {
            min_n_gates_to_separate_cities = n_gates;
            gates.resize(n_gates);
            for (auto &gate : gates) {
                ss >> gate;
            }
        }
    }

    std::ostream &Write(std::ostream &out) const {
        if (not min_n_gates_to_separate_cities) {
            out << -1;
        } else {
            out << min_n_gates_to_separate_cities.value() << '\n';
            for (auto gate : gates) {
                out << gate.first + 1 << ' ' << gate.second + 1 << '\n';
            }
        }
        return out;
    }

    bool operator!=(const Output &other) const {
        return min_n_gates_to_separate_cities != other.min_n_gates_to_separate_cities;
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

int32_t GetClosestNotSmallerPowerOfTwo(int32_t number) {
    return std::ceil(log2(number));
}

template <typename IteratorFirst, typename IteratorSecond>
class Zip {
public:
    using Value =
        std::pair<typename IteratorFirst::value_type, typename IteratorSecond::value_type>;

    struct Iterator {
        Iterator() = default;

        Iterator(IteratorFirst first_begin, IteratorFirst first_end, IteratorSecond second_begin,
                 IteratorSecond second_end)
            : first_iter_{first_begin},
              first_end_{first_end},
              second_iter_{second_begin},
              second_end_{second_end} {
        }

        Value operator*() const {
            return {*first_iter_, *second_iter_};
        }

        Iterator &operator++() {
            ++first_iter_;
            ++second_iter_;
            return *this;
        }

        friend bool operator==(const Iterator &left, const Iterator &right) {
            return left.IsExhausted() and right.IsExhausted();
        }

        friend bool operator!=(const Iterator &left, const Iterator &right) {
            return not operator==(left, right);
        }

    private:
        IteratorFirst first_iter_;
        IteratorFirst first_end_;
        IteratorSecond second_iter_;
        IteratorSecond second_end_;

        [[nodiscard]] bool IsExhausted() const {
            return first_iter_ == first_end_ or second_iter_ == second_end_;
        }
    };

    Zip(IteratorFirst first_begin, IteratorFirst first_end, IteratorSecond second_begin,
        IteratorSecond second_end)
        : first_begin_{first_begin},
          first_end_{first_end},
          second_begin_{second_begin},
          second_end_{second_end} {
    }

    Iterator begin() {  // NOLINT
        return Iterator{first_begin_, first_end_, second_begin_, second_end_};
    }

    Iterator end() {  // NOLINT
        return Iterator{};
    }

private:
    IteratorFirst first_begin_;
    IteratorFirst first_end_;
    IteratorSecond second_begin_;
    IteratorSecond second_end_;
};

int32_t BinarySearch(int32_t low, int32_t high,
                     const std::function<bool(int32_t)> &is_answer_no_more_than) {
    while (low < high) {
        auto mid = (low + high) / 2;
        if (is_answer_no_more_than(mid)) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return low;
}

template <typename Graph>
std::vector<std::pair<typename Graph::Node, typename Graph::Node>> GetAllEdges(const Graph &graph) {
    std::vector<std::pair<typename Graph::Node, typename Graph::Node>> edges;
    for (auto from = 0; from < graph.Size(); ++from) {
        for (auto to : graph.GetNodesAdjacentTo(from)) {
            edges.emplace_back(from, to);
        }
    }
    return edges;
}

template <typename Graph>
int32_t GetEdgeCount(const Graph &graph) {
    int32_t n_edges = 0;
    for (auto node = 0; node < graph.Size(); ++node) {
        n_edges += graph.GetNodesAdjacentTo(node).size();
    }
    return n_edges;
};

}  // namespace utils

namespace interface {

class NotImplementedError : public std::logic_error {
public:
    NotImplementedError() : std::logic_error("Function not yet implemented."){};
};

class VirtualBaseClass {
public:
    ~VirtualBaseClass() = default;
};

template <typename Node = int32_t, typename NodeIterable = const std::vector<Node> &>
class Graph : public VirtualBaseClass {
    [[nodiscard]] virtual NodeIterable GetNodesAdjacentTo(Node node) const = 0;
};

template <typename Node = int32_t, typename NodeIterable = const std::vector<Node> &>
class FilteredGraph : public Graph<Node, NodeIterable> {
    [[nodiscard]] virtual bool ShouldTraverseEdge(Node from, Node to) const = 0;
};

template <typename Node = int32_t, typename NodeIterable = const std::vector<Node> &>
class FlowNetwork : public FilteredGraph<Node, NodeIterable> {
public:
    [[nodiscard]] virtual Node GetInitialNode() const = 0;

    [[nodiscard]] virtual bool IsNodeTerminal(Node node) const = 0;

    [[nodiscard]] virtual int32_t GetEdgeConstraint(Node from, Node to) const = 0;

    [[nodiscard]] virtual int32_t GetEdgeFlowRate(Node from, Node to) const = 0;
};

template <typename Graph>
class GraphTraversal : public VirtualBaseClass {
public:
    using Node = typename Graph::Node;
    using NodeIterable = typename Graph::NodeIterable;

    virtual void OnTraverseStart() {
    }

    virtual void OnNodeEnter(Node node) {
    }

    virtual NodeIterable GetNodesAdjacentTo(Node node) = 0;

    virtual void OnEdgeDiscovery(Node from, Node to) {
    }

    virtual bool ShouldTraverseEdge(Node from, Node to) {
        return true;
    }

    virtual void OnNodeExit(Node node) {
    }

    virtual void OnEdgeTraverse(Node from, Node to) {
    }

    virtual void OnEdgeBacktrack(Node from, Node to) {
    }

    virtual void OnTraverseEnd() {
    }
};

}  // namespace interface

namespace implementation {

class StopTraverseException : public std::exception {};

template <typename GraphTraversal>
void BreadthFirstSearch(GraphTraversal *graph_traversal,
                        std::deque<typename GraphTraversal::Node> starting_nodes_queue) {

    graph_traversal->OnTraverseStart();

    auto &queue = starting_nodes_queue;

    while (not queue.empty()) {

        auto node = queue.front();
        queue.pop_front();

        graph_traversal->OnNodeEnter(node);

        for (const auto &adjacent_node : graph_traversal->GetNodesAdjacentTo(node)) {

            graph_traversal->OnEdgeDiscovery(node, adjacent_node);

            if (graph_traversal->ShouldTraverseEdge(node, adjacent_node)) {

                graph_traversal->OnEdgeTraverse(node, adjacent_node);
                queue.emplace_back(adjacent_node);
            }
        }

        graph_traversal->OnNodeExit(node);
    }

    graph_traversal->OnTraverseEnd();
}

template <typename GraphTraversal>
void DepthFirstSearchRecursive(GraphTraversal *graph_traversal,
                               typename GraphTraversal::Node source_node) {

    graph_traversal->OnNodeEnter(source_node);

    for (auto adjacent_node : graph_traversal->GetNodesAdjacentTo(source_node)) {

        graph_traversal->OnEdgeDiscovery(source_node, adjacent_node);

        if (graph_traversal->ShouldTraverseEdge(source_node, adjacent_node)) {

            graph_traversal->OnEdgeTraverse(source_node, adjacent_node);
            DepthFirstSearchRecursive(graph_traversal, adjacent_node);
            graph_traversal->OnEdgeBacktrack(source_node, adjacent_node);
        }
    }

    graph_traversal->OnNodeExit(source_node);
}

template <typename GraphTraversal>
void DepthFirstSearch(GraphTraversal *graph_traversal, typename GraphTraversal::Node source_node) {

    graph_traversal->OnTraverseStart();

    DepthFirstSearchRecursive(graph_traversal, source_node);

    graph_traversal->OnTraverseEnd();
}

template <typename GraphTraversal>
void GraphSearch(GraphTraversal *graph_traversal,
                 std::deque<typename GraphTraversal::Node> starting_nodes_queue) {
    BreadthFirstSearch(graph_traversal, starting_nodes_queue);
}

template <typename GraphTraversal>
void GraphSearch(GraphTraversal *graph_traversal, typename GraphTraversal::Node source_node) {
    DepthFirstSearch(graph_traversal, source_node);
}

template <typename Graph>
class BfsDistanceComputeTraversal : public interface::GraphTraversal<Graph> {
public:
    using Node = typename Graph::Node;
    using NodeIterable = typename Graph::NodeIterable;

    const Graph &graph;
    std::vector<std::optional<Node>> distance;

    explicit BfsDistanceComputeTraversal(const Graph &graph)
        : graph{graph}, distance(graph.Size()) {
    }

    void OnTraverseStart() override {
        std::fill(distance.begin(), distance.end(), std::nullopt);
    }

    NodeIterable GetNodesAdjacentTo(Node node) override {
        return graph.GetNodesAdjacentTo(node);
    }

    void OnNodeEnter(Node node) override {
        if (not distance[node]) {
            distance[node] = 0;
        }
    }

    bool ShouldTraverseEdge(Node from, Node to) override {
        return graph.ShouldTraverseEdge(from, to) and not distance[to];
    }

    void OnEdgeTraverse(Node from, Node to) override {
        distance[to] = distance[from].value() + 1;
    }
};

template <typename Graph>
std::vector<std::optional<typename Graph::Node>> ComputeDistancesFromNode(
    const Graph &graph, typename Graph::Node source_node) {

    BfsDistanceComputeTraversal traversal{graph};
    implementation::GraphSearch(&traversal, std::deque<typename Graph::Node>{source_node});
    return traversal.distance;
}

}  // namespace implementation

namespace path_traversal {

template <typename Node>
using OptionalPath = std::optional<std::vector<Node>>;

template <typename Graph>
using FindPath = std::function<OptionalPath<typename Graph::Node>()>;

template <typename Graph>
class PathTraversal : public interface::GraphTraversal<Graph> {
public:
    using Node = typename Graph::Node;

    [[nodiscard]] virtual bool IsTargetNode(Node node) const = 0;

    [[nodiscard]] virtual OptionalPath<Node> GetPath() const = 0;

protected:
    virtual void ThrowStopTraverseExceptionIfIsTargetNode(Node node) const {
        if (IsTargetNode(node)) {
            throw implementation::StopTraverseException{};
        }
    }

    [[nodiscard]] virtual OptionalPath<Node> CheckPath(const std::vector<Node> &path) const {
        if (not path.empty() and IsTargetNode(path.back())) {
            return path;
        } else {
            return std::nullopt;
        }
    }
};

template <typename PathTraversal, typename QueueOrNode>
OptionalPath<typename PathTraversal::Node> FindNonZeroPath(PathTraversal *traversal,
                                                           QueueOrNode from) {
    try {
        implementation::GraphSearch(traversal, from);
    } catch (implementation::StopTraverseException &) {
        return traversal->GetPath();
    }
    return std::nullopt;
}

template <typename Graph>
class DfsPathTraversal : public PathTraversal<Graph> {
public:
    using Node = typename Graph::Node;
    using NodeIterable = typename Graph::NodeIterable;

    const Graph &graph;

    explicit DfsPathTraversal(const Graph &graph) : graph{graph}, has_visited_node_(graph.Size()) {
    }

    void OnTraverseStart() override {
        path_.clear();
        std::fill(has_visited_node_.begin(), has_visited_node_.end(), false);
    }

    NodeIterable GetNodesAdjacentTo(Node node) override {
        return graph.GetNodesAdjacentTo(node);
    }

    bool ShouldTraverseEdge(Node from, Node to) override {
        return graph.ShouldTraverseEdge(from, to) and not has_visited_node_[to];
    }

    void OnNodeEnter(Node node) override {
        has_visited_node_[node] = true;
        path_.emplace_back(node);
        this->ThrowStopTraverseExceptionIfIsTargetNode(node);
    }

    void OnNodeExit(Node) override {
        path_.pop_back();
    }

    [[nodiscard]] OptionalPath<Node> GetPath() const override {
        return this->CheckPath(path_);
    }

private:
    std::vector<bool> has_visited_node_;
    std::vector<Node> path_;
};

template <typename Graph>
class BfsPathTraversal : public PathTraversal<Graph> {
public:
    using Node = typename Graph::Node;
    using NodeIterable = typename Graph::NodeIterable;

    const Graph &graph;

    explicit BfsPathTraversal(const Graph &graph)
        : graph{graph}, has_visited_node_(graph.Size()), parent_(graph.Size()) {
    }

    void OnTraverseStart() override {
        path_.clear();
        std::fill(has_visited_node_.begin(), has_visited_node_.end(), false);
        std::fill(parent_.begin(), parent_.end(), std::nullopt);
    }

    NodeIterable GetNodesAdjacentTo(Node node) override {
        return graph.GetNodesAdjacentTo(node);
    }

    bool ShouldTraverseEdge(Node from, Node to) override {
        return graph.ShouldTraverseEdge(from, to) and not has_visited_node_[to];
    }

    void OnEdgeTraverse(Node from, Node to) override {
        parent_[to] = from;
    }

    void OnNodeEnter(Node node) override {
        has_visited_node_[node] = true;
        if (this->IsTargetNode(node)) {
            BuildNonZeroPath(node);
            throw implementation::StopTraverseException{};
        }
    }

    [[nodiscard]] OptionalPath<Node> GetPath() const override {
        return this->CheckPath(path_);
    }

protected:
    std::vector<bool> has_visited_node_;
    std::vector<std::optional<Node>> parent_;
    std::vector<Node> path_;

    void BuildNonZeroPath(Node terminal_node) {
        auto node = terminal_node;
        for (; parent_[node]; node = parent_[node].value()) {
            path_.emplace_back(node);
        }
        path_.emplace_back(node);
        std::reverse(path_.begin(), path_.end());
    }
};

}  // namespace path_traversal

namespace decremental_dynamic_reachability {

template <typename Graph>
class DfsGeodesicSubtreePathTraversal : public path_traversal::PathTraversal<Graph> {
public:
    using Node = typename Graph::Node;
    using NodeIterable = typename Graph::NodeIterable;

    const Graph &graph;
    Node source_node;

    DfsGeodesicSubtreePathTraversal(const Graph &graph, Node source)
        : graph{graph},
          source_node{source},
          is_node_cut_(graph.Size()),
          bfs_distance_compute_traversal_{graph} {
        BuildGeodesicSubtree();
    }

    void BuildGeodesicSubtree() {
        std::fill(is_node_cut_.begin(), is_node_cut_.end(), false);
        implementation::BreadthFirstSearch(&bfs_distance_compute_traversal_, {source_node});
    }

    void CutNode(Node node) {
        is_node_cut_[node] = true;
    }

    void OnTraverseStart() override {
        path_.clear();
    }

    NodeIterable GetNodesAdjacentTo(Node node) override {
        return graph.GetNodesAdjacentTo(node);
    }

    bool ShouldTraverseEdge(Node from, Node to) override {
        return graph.ShouldTraverseEdge(from, to) and not is_node_cut_[to] and
               IsGeodesicEdge(from, to);
    }

    void OnNodeEnter(Node node) override {
        path_.emplace_back(node);
        this->ThrowStopTraverseExceptionIfIsTargetNode(node);
    }

    void OnNodeExit(Node) override {
        path_.pop_back();
    }

    void OnEdgeBacktrack(Node from, Node to) override {
        CutNode(to);
    }

    [[nodiscard]] path_traversal::OptionalPath<Node> GetPath() const override {
        return this->CheckPath(path_);
    }

private:
    std::vector<bool> is_node_cut_;
    std::vector<Node> path_;
    implementation::BfsDistanceComputeTraversal<Graph> bfs_distance_compute_traversal_;

    [[nodiscard]] bool IsGeodesicEdge(Node from, Node to) const {
        return bfs_distance_compute_traversal_.distance[to].value() ==
               bfs_distance_compute_traversal_.distance[from].value() + 1;
    }
};

template <typename Graph, typename Traversal>
class PathFinder {
public:
    Traversal traversal;

    PathFinder(const Graph &graph, typename Graph::Node source_node)
        : traversal{graph, source_node} {
    }

    path_traversal::OptionalPath<typename Graph::Node> FindPath() {
        if (auto path = TryToFindPathInCurrentGeodesicSubtree()) {
            return path;
        }
        traversal.BuildGeodesicSubtree();
        return TryToFindPathInCurrentGeodesicSubtree();
    }

private:
    path_traversal::OptionalPath<typename Graph::Node> TryToFindPathInCurrentGeodesicSubtree() {
        return path_traversal::FindNonZeroPath<>(&traversal, traversal.source_node);
    }
};

}  // namespace decremental_dynamic_reachability

namespace flow {

template <typename Node = int32_t, typename NodeIterable = const std::vector<Node> &>
class FlowNetwork : public interface::FlowNetwork<Node, NodeIterable> {
public:
    [[nodiscard]] virtual int32_t Size() const = 0;

    virtual void ResetFlows() = 0;

    [[nodiscard]] virtual int32_t GetResidualEdgeCapacity(Node from, Node to) const {
        return this->GetEdgeConstraint(from, to) - this->GetEdgeFlowRate(from, to);
    }

    [[nodiscard]] bool ShouldTraverseEdge(Node from, Node to) const override {
        return GetResidualEdgeCapacity(from, to) > 0;
    }

    virtual void SetEdgeConstraint(Node from, Node to, int32_t constraint) {
        throw interface::NotImplementedError {};
    }

    virtual void SetEdgeFlow(Node from, Node to, int32_t flow) {
        SafeSetEdgeDirectedFlow(from, to, flow);
        SafeSetEdgeDirectedFlow(to, from, -flow);
        CheckEdgeFlow(from, to);
    }

protected:
    virtual void SafeSetEdgeDirectedFlow(Node from, Node to, int32_t flow) = 0;

    virtual void CheckEdgeFlow(Node from, Node to) const {
        if (GetResidualEdgeCapacity(from, to) < 0 or GetResidualEdgeCapacity(to, from) < 0) {
            throw std::runtime_error{"Edge flow cannot be greater than constraint."};
        }
    }
};

template <typename Flow>
int32_t GetNodeExcess(const Flow &flow, typename Flow::Node node) {
    int32_t excess = 0;
    for (auto adjacent_node : flow.GetNodesAdjacentTo(node)) {
        excess += flow.GetEdgeFlowRate(adjacent_node, node);
    }
    return excess;
}

template <typename Flow>
int32_t GetNetworkFlowRate(const Flow &flow) {
    return -GetNodeExcess(flow, flow.GetInitialNode());
}

template <typename Flow>
int32_t GetMaxConstraint(const Flow &flow) {
    auto max_constraint = 0;
    for (auto [from, to] : utils::GetAllEdges(flow)) {
        auto constraint = flow.GetEdgeConstraint(from, to);
        max_constraint = std::max(max_constraint, constraint);
    }
    return max_constraint;
}

template <typename Flow>
void ScaleFlow(Flow *flow, double factor) {
    for (auto [from, to] : utils::GetAllEdges(*flow)) {
        auto constraint = flow->GetEdgeConstraint(from, to);
        auto flow_rate = flow->GetEdgeFlowRate(from, to);

        flow->SetEdgeConstraint(from, to, static_cast<int32_t>(constraint * factor));
        flow->SetEdgeFlow(from, to, static_cast<int32_t>(flow * factor));
    }
}

template <typename Flow>
void CheckPath(Flow *flow, const std::vector<typename Flow::Node> &path) {
    if (path.empty() or path.front() != flow->GetInitialNode() or
        not flow->IsNodeTerminal(path.back())) {
        throw std::invalid_argument{"Invalid path."};
    }
};

template <typename Flow>
void AugmentByPath(Flow *flow, const std::vector<typename Flow::Node> &path, int32_t flow_rate) {
    auto begin = path.begin();
    auto end = path.end();
    for (auto [from, to] : utils::Zip(begin, end - 1, begin + 1, end)) {
        auto edge_flow = flow->GetEdgeFlowRate(from, to);
        auto augmented_flow = edge_flow + flow_rate;
        flow->SetEdgeFlow(from, to, augmented_flow);
    }
}

template <typename Flow>
void AugmentByFlow(Flow *flow, const Flow &augmenting_flow) {
    for (auto [from, to] : utils::GetAllEdges(augmenting_flow)) {
        auto flow_rate = flow->GetEdgeFlowRate(from, to);
        flow_rate += augmenting_flow.GetEdgeFlowRate(from, to);
        flow->SetEdgeFlow(from, to, flow_rate);
    }
}

template <typename Flow>
int32_t ComputePathResidualFlow(const Flow &flow, const std::vector<typename Flow::Node> &path) {
    auto begin = path.begin();
    auto end = path.end();
    int32_t path_residual_flow = INT32_MAX;

    for (auto [from, to] : utils::Zip(begin, end - 1, begin + 1, end)) {
        auto edge_residual_flow = flow.GetResidualEdgeCapacity(from, to);
        path_residual_flow = std::min(path_residual_flow, edge_residual_flow);
    }
    return path_residual_flow;
}

template <typename Flow>
int32_t ComputePathMinFlow(const Flow &flow, const std::vector<typename Flow::Node> &path) {
    auto begin = path.begin();
    auto end = path.end();
    int32_t path_min_flow = INT32_MAX;

    for (auto [from, to] : utils::Zip(begin, end - 1, begin + 1, end)) {
        auto edge_residual_flow = flow.GetEdgeFlowRate(from, to);
        path_min_flow = std::min(path_min_flow, edge_residual_flow);
    }
    return path_min_flow;
}

template <typename Flow>
void SolveFlow(Flow *flow, const path_traversal::FindPath<Flow> &find_non_zero_flow_path) {
    while (auto path = find_non_zero_flow_path()) {
        auto path_residual_flow = ComputePathResidualFlow(*flow, path.value());
        AugmentByPath(flow, path.value(), path_residual_flow);
    }
}

template <typename Flow>
class FlowDfsPathTraversal : public path_traversal::DfsPathTraversal<Flow> {
public:
    using PathTraversal = path_traversal::DfsPathTraversal<Flow>;

    using PathTraversal::PathTraversal;

    [[nodiscard]] bool IsTargetNode(typename Flow::Node node) const override {
        return PathTraversal::graph.IsNodeTerminal(node);
    }
};

template <typename Flow>
void FordFulkersonSolveFlow(Flow *flow) {
    FlowDfsPathTraversal<Flow> traversal{*flow};
    auto initial_node = flow->GetInitialNode();

    path_traversal::FindPath<Flow> find_non_zero_path = [&traversal, initial_node]() {
        return path_traversal::FindNonZeroPath<>(&traversal, initial_node);
    };
    SolveFlow<Flow>(flow, find_non_zero_path);
}

template <typename Flow>
class FlowBfsPathTraversal : public path_traversal::BfsPathTraversal<Flow> {
public:
    using PathTraversal = path_traversal::BfsPathTraversal<Flow>;

    using PathTraversal::PathTraversal;

    [[nodiscard]] bool IsTargetNode(typename Flow::Node node) const override {
        return PathTraversal::graph.IsNodeTerminal(node);
    }
};

template <typename Flow>
void EdmondsKarpSolveFlow(Flow *flow) {
    FlowBfsPathTraversal<Flow> traversal{*flow};
    auto initial_node = std::deque<typename Flow::Node>{flow->GetInitialNode()};

    path_traversal::FindPath<Flow> find_non_zero_path = [&traversal, initial_node]() {
        return path_traversal::FindNonZeroPath<>(&traversal, initial_node);
    };
    SolveFlow<Flow>(flow, find_non_zero_path);
}

template <typename Flow>
class DinicTraversal
    : public decremental_dynamic_reachability::DfsGeodesicSubtreePathTraversal<Flow> {
public:
    using PathTraversal = decremental_dynamic_reachability::DfsGeodesicSubtreePathTraversal<Flow>;

    using PathTraversal::PathTraversal;

    [[nodiscard]] bool IsTargetNode(typename Flow::Node node) const override {
        return PathTraversal::graph.IsNodeTerminal(node);
    }
};

template <typename Flow>
void DinicSolveFlow(Flow *flow) {
    decremental_dynamic_reachability::PathFinder<Flow, DinicTraversal<Flow>> path_finder{
        *flow, flow->GetInitialNode()};

    path_traversal::FindPath<Flow> find_non_zero_path = [&path_finder]() {
        return path_finder.FindPath();
    };
    SolveFlow<Flow>(flow, find_non_zero_path);
}

template <typename Flow>
void ScalingFlowSolve(Flow *flow, std::function<void(Flow *)> solve_flow) {
    flow->ResetFlows();
    auto max_constraint = flow->GetMaxConstraint();
    auto log = utils::GetClosestNotSmallerPowerOfTwo(max_constraint + 1) - 1;
    auto iter_flow = *flow;

    for (int32_t scaling_constraint = 1 << log; scaling_constraint > 0; scaling_constraint >>= 1) {
        iter_flow.ResetFlows();
        for (auto [from, to] : flow->GetAllEdges()) {
            auto capacity = flow->GetResidualEdgeCapacity(from, to);
            auto binary_constrain = capacity >= scaling_constraint;
            iter_flow.SetEdgeConstraint(from, to, binary_constrain);
        }
        solve_flow(&iter_flow);
        iter_flow.Scale(scaling_constraint);
        flow->AugmentByFlow(iter_flow);
    }
}

}  // namespace flow

template <typename Flow>
class BfsMinCutTraversal : public interface::GraphTraversal<Flow> {
public:
    using Node = typename Flow::Node;
    using NodeIterable = typename Flow::NodeIterable;

    const Flow &flow;

    explicit BfsMinCutTraversal(const Flow &graph) : flow{graph}, has_visited_node_(graph.Size()) {
    }

    void OnTraverseStart() override {
        std::fill(has_visited_node_.begin(), has_visited_node_.end(), false);
        edges_to_cut_candidates_.clear();
    }

    NodeIterable GetNodesAdjacentTo(Node node) override {
        return flow.GetNodesAdjacentTo(node);
    }

    bool ShouldTraverseEdge(Node from, Node to) override {
        if (has_visited_node_[to]) {
            return false;
        }

        auto is_saturated = flow.GetResidualEdgeCapacity(from, to) == 0;
        auto is_flowing_from_source = flow.GetEdgeFlowRate(from, to) > 0;
        auto is_candidate_to_cut = is_saturated and is_flowing_from_source;
        if (is_candidate_to_cut) {
            edges_to_cut_candidates_.emplace_back(from, to);
        }
        return not is_saturated;
    }

    void OnNodeEnter(Node node) override {
        has_visited_node_[node] = true;
    }

    std::vector<std::pair<Node, Node>> BuildMinCut() {
        std::vector<std::pair<Node, Node>> min_cut;
        min_cut.reserve(edges_to_cut_candidates_.size());
        for (auto [from, to] : edges_to_cut_candidates_) {
            if (not has_visited_node_[to]) {
                auto from_coordinate = flow.coordinate_converter.FromNodeId(from);
                auto to_coordinate = flow.coordinate_converter.FromNodeId(to);
                assert(from_coordinate.row == to_coordinate.row and
                       from_coordinate.col == to_coordinate.col);
                min_cut.emplace_back(from, to);
            }
        }
        assert(static_cast<int32_t>(min_cut.size()) == flow::GetNetworkFlowRate(flow));
        return min_cut;
    }

protected:
    std::vector<bool> has_visited_node_;
    std::vector<std::pair<Node, Node>> edges_to_cut_candidates_;
};

template <typename Flow>
std::vector<std::pair<typename Flow::Node, typename Flow::Node>> FindMinCut(Flow *flow) {
    flow::DinicSolveFlow<Flow>(flow);
    BfsMinCutTraversal traversal{*flow};
    implementation::DepthFirstSearch(&traversal, flow->GetInitialNode());
    auto min_cut = traversal.BuildMinCut();
    //    CheckMinCut(*flow, min_cut);
    return min_cut;
}

struct NodeCoordinate {
    int32_t row = 0;
    int32_t col = 0;
    bool is_input = false;
};

class GridCoordinateConverter {
public:
    int32_t n_rows = 0;
    int32_t n_cols = 0;
    int32_t grid_size = 0;
    int32_t n_nodes = 0;
    int32_t n_edges = 0;

    GridCoordinateConverter() = default;

    GridCoordinateConverter(int32_t n_rows, int32_t n_cols)
        : n_rows{n_rows},
          n_cols{n_cols},
          grid_size{n_rows * n_cols},
          n_nodes{grid_size * 2},
          n_edges{grid_size * 10} {
    }

    [[nodiscard]] int32_t ToNodeId(NodeCoordinate coordinate) const {
        return (not coordinate.is_input) * grid_size + CodeCoordinate(coordinate);
    }

    [[nodiscard]] NodeCoordinate FromNodeId(int32_t cell_id) const {
        NodeCoordinate coordinate;
        coordinate.is_input = cell_id < grid_size;
        if (not coordinate.is_input) {
            cell_id -= grid_size;
        }
        coordinate.row = cell_id / n_cols;
        coordinate.col = cell_id % n_cols;
        return coordinate;
    }

    [[nodiscard]] int32_t ToEdgeId(int32_t from, int32_t to) const {
        auto from_coordinate = FromNodeId(from);
        auto to_coordinate = FromNodeId(to);
        auto row_delta = from_coordinate.row - to_coordinate.row;
        auto col_delta = from_coordinate.col - to_coordinate.col;

        if (from_coordinate.is_input == to_coordinate.is_input or
            std::abs(row_delta) + std::abs(col_delta) >= 2) {
            throw std::invalid_argument{"No such edge."};
        }

        return 10 * CodeCoordinate(from_coordinate) + 2 * CodeEdgeDirection(row_delta, col_delta) +
               from_coordinate.is_input;
    }

private:
    [[nodiscard]] int32_t CodeCoordinate(NodeCoordinate coordinate) const {
        return coordinate.row * n_cols + coordinate.col;
    }

    [[nodiscard]] int32_t CodeEdgeDirection(int32_t row_delta, int32_t col_delta) const {
        return 2 * (row_delta + 1) + col_delta;
    }
};

class FlowBuilder;

class FlowNetwork : public flow::FlowNetwork<> {
public:
    using Node = int32_t;
    using NodeIterable = const std::vector<Node> &;

    GridCoordinateConverter coordinate_converter;

    FlowNetwork() = default;

    [[nodiscard]] int32_t Size() const override {
        return coordinate_converter.n_nodes;
    }

    void ResetFlows() override {
        std::fill(flow_.begin(), flow_.end(), 0);
    }

    void UndirectedConnect(Node from, Node to) {
        assert(from != to);
        adjacent_nodes_[from].emplace_back(to);
        adjacent_nodes_[to].emplace_back(from);
    }

    void SetEdgeConstraint(Node from, Node to, int32_t constraint) override {
        auto edge = coordinate_converter.ToEdgeId(from, to);
        constraint_[edge] = constraint;
    }

    [[nodiscard]] NodeIterable GetNodesAdjacentTo(Node node) const override {
        return adjacent_nodes_[node];
    }

    [[nodiscard]] Node GetInitialNode() const override {
        return initial_node_;
    }

    [[nodiscard]] Node GetTerminalNode() const {
        return terminal_node_;
    }

    [[nodiscard]] bool IsNodeTerminal(Node node) const override {
        return node == terminal_node_;
    }

    [[nodiscard]] int32_t GetEdgeConstraint(Node from, Node to) const override {
        auto edge = coordinate_converter.ToEdgeId(from, to);
        return constraint_[edge];
    }

    [[nodiscard]] int32_t GetEdgeFlowRate(Node from, Node to) const override {
        auto edge = coordinate_converter.ToEdgeId(from, to);
        return flow_[edge];
    }

protected:
    void SafeSetEdgeDirectedFlow(Node from, Node to, int32_t flow) override {
        auto edge = coordinate_converter.ToEdgeId(from, to);
        flow_[edge] = flow;
    }

private:
    int32_t initial_node_ = 0;
    int32_t terminal_node_ = 0;
    std::vector<std::vector<Node>> adjacent_nodes_;
    std::vector<int32_t> flow_;
    std::vector<int32_t> constraint_;

    friend class FlowBuilder;
};

class FlowBuilder {
public:
    static FlowNetwork Build(const io::Input &input) {
        FlowNetwork flow;
        flow.coordinate_converter = GridCoordinateConverter{input.n_rows, input.n_cols};
        Resize(&flow);

        flow.initial_node_ = flow.coordinate_converter.ToNodeId(
            {input.first_city.first, input.first_city.second, false});
        flow.terminal_node_ = flow.coordinate_converter.ToNodeId(
            {input.second_city.first, input.second_city.second, true});

        BuildEdges(&flow, input);

        return flow;
    }

private:
    static constexpr int32_t kUnboundEdgeConstraint = INT32_MAX / 2;

    static void Resize(FlowNetwork *flow_ptr) {
        auto &flow = *flow_ptr;

        flow.adjacent_nodes_.resize(flow.coordinate_converter.n_nodes);
        flow.flow_.resize(flow.coordinate_converter.n_edges);
        flow.constraint_.resize(flow.coordinate_converter.n_edges);
    }

    static void BuildEdges(FlowNetwork *flow_ptr, const io::Input &input) {
        auto &flow = *flow_ptr;

        for (int32_t row = 0; row < input.n_rows; ++row) {
            for (int32_t col = 0; col < input.n_cols; ++col) {
                ConnectIfCan(flow_ptr, input, row, col, row - 1, col);
                ConnectIfCan(flow_ptr, input, row, col, row, col - 1);
                ConnectIfCan(flow_ptr, input, row, col, row, col);
                ConnectIfCan(flow_ptr, input, row, col, row, col + 1);
                ConnectIfCan(flow_ptr, input, row, col, row + 1, col);
            }
        }

        for (auto mountain : input.mountains) {
            auto [from, to] =
                ToEdge(flow_ptr, mountain.first, mountain.second, mountain.first, mountain.second);
            flow.SetEdgeConstraint(to, from, 0);
        }

        for (auto gate : input.gates) {
            auto [from, to] = ToEdge(flow_ptr, gate.first, gate.second, gate.first, gate.second);
            flow.SetEdgeConstraint(to, from, 1);
        }
    }

    static void ConnectIfCan(FlowNetwork *flow_ptr, const io::Input &input, int32_t from_row,
                             int32_t from_col, int32_t to_row, int32_t to_col) {
        if (to_row < 0 or to_row >= input.n_rows or to_col < 0 or to_col >= input.n_cols) {
            return;
        }
        auto [from, to] = ToEdge(flow_ptr, from_row, from_col, to_row, to_col);
        flow_ptr->UndirectedConnect(from, to);

        if (from_row == to_row and from_col == to_col) {
            flow_ptr->SetEdgeConstraint(to, from, kUnboundEdgeConstraint);
        } else {
            flow_ptr->SetEdgeConstraint(from, to, kUnboundEdgeConstraint);
        }
    }

    static std::pair<int32_t, int32_t> ToEdge(FlowNetwork *flow_ptr, int32_t from_row,
                                              int32_t from_col, int32_t to_row, int32_t to_col) {
        auto from = flow_ptr->coordinate_converter.ToNodeId({from_row, from_col, false});
        auto to = flow_ptr->coordinate_converter.ToNodeId({to_row, to_col, true});
        return {from, to};
    }
};

template <typename Flow>
class InterCityWithAllGatesClosedTraversal : public interface::GraphTraversal<Flow> {
public:
    using Node = typename Flow::Node;
    using NodeIterable = typename Flow::NodeIterable;

    const Flow &flow;

    explicit InterCityWithAllGatesClosedTraversal(const Flow &flow)
        : flow{flow}, has_visited_node_(flow.Size()) {
    }

    void OnTraverseStart() override {
        std::fill(has_visited_node_.begin(), has_visited_node_.end(), false);
    }

    NodeIterable GetNodesAdjacentTo(Node node) override {
        return flow.GetNodesAdjacentTo(node);
    }

    bool ShouldTraverseEdge(Node from, Node to) override {
        //        auto from_coordinate = flow.coordinate_converter.FromNodeId(from);
        //        auto to_coordinate = flow.coordinate_converter.FromNodeId(to);
        //        assert(from_coordinate.is_input != to_coordinate.is_input);
        return not has_visited_node_[to] and flow.GetEdgeConstraint(from, to) > 1;
    }

    void OnNodeEnter(Node node) override {
        has_visited_node_[node] = true;
    }

    [[nodiscard]] bool IsTerminalNodeVisited() const {
        return has_visited_node_[flow.GetTerminalNode()];
    }

protected:
    std::vector<bool> has_visited_node_;
};

bool CanCitiesBeSeparatedByGates(const FlowNetwork &flow) {
    InterCityWithAllGatesClosedTraversal traversal(flow);
    implementation::DepthFirstSearch(&traversal, flow.GetInitialNode());
    return not traversal.IsTerminalNodeVisited();
}

io::Output Solve(const io::Input &input) {
    io::Output output;
    auto flow = FlowBuilder::Build(input);

    if (not CanCitiesBeSeparatedByGates(flow)) {
        //                VisualizeSolution(input, output);
        return output;
    }

    auto min_cut = FindMinCut(&flow);
    output.min_n_gates_to_separate_cities = min_cut.size();
    output.gates.reserve(min_cut.size());
    for (auto [from, to] : min_cut) {
        auto coordinate = flow.coordinate_converter.FromNodeId(from);
        output.gates.emplace_back(coordinate.row, coordinate.col);
    }
    //    VisualizeSolution(input, output);
    return output;
}

int main(int argc, char *argv[]) {

    io::SetUpFastIo();

    std::cout << Solve(io::Input{std::cin});

    return 0;
}
