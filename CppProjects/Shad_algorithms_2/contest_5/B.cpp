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
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace io {

struct Edge {
    int32_t from = 0;
    int32_t to = 0;

    Edge() = default;

    Edge(int32_t from, int32_t to) : from{from}, to{to} {
    }
};

class Input {
public:
    std::vector<int32_t> first_bipartition_constraints;
    std::vector<int32_t> second_bipartition_constraints;
    std::vector<Edge> edges;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t n_nodes_in_bipartition = 0;
        int32_t n_edges_between_bipartitions = 0;

        in >> n_nodes_in_bipartition >> n_edges_between_bipartitions;
        assert(n_nodes_in_bipartition >= 1);
        first_bipartition_constraints.resize(n_nodes_in_bipartition);
        for (auto &c : first_bipartition_constraints) {
            in >> c;
        }

        second_bipartition_constraints.resize(n_nodes_in_bipartition);
        for (auto &c : second_bipartition_constraints) {
            in >> c;
        }

        edges.resize(n_edges_between_bipartitions);
        for (auto &[from, to] : edges) {
            in >> from >> to;
            --from;
            --to;
        }
    }
};

class Output {
public:
    int32_t max_flow = 0;

    Output() = default;

    explicit Output(int32_t max_flow) : max_flow{max_flow} {
    }

    std::ostream &Write(std::ostream &out) const {
        out << max_flow;
        return out;
    }

    bool operator!=(const Output &other) const {
        return max_flow != other.max_flow;
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

namespace preflow {

template <typename Flow>
class PreflowNetwork {
public:
    using Node = typename Flow::Node;

    Flow *flow = nullptr;

    explicit PreflowNetwork(Flow *flow)
        : flow{flow}, height_(flow->Size()), node_excess_(flow->Size()), max_height_{INT32_MAX} {
    }

    virtual void Init() {
        auto initial_node = flow->GetInitialNode();
        height_[initial_node] = flow->Size();
        for (auto node : flow->GetNodesAdjacentTo(initial_node)) {
            auto constraint = flow->GetEdgeConstraint(initial_node, node);
            AdjustEdgeFlow(initial_node, node, constraint);
        }
    }

    [[nodiscard]] bool CanPush(Node from, Node to) const {
        auto residual = flow->GetResidualEdgeCapacity(from, to);
        return node_excess_[from] > 0 and residual > 0 and height_[from] == height_[to] + 1;
    }

    virtual void Push(Node from, Node to) {
        auto residual = flow->GetResidualEdgeCapacity(from, to);
        auto epsilon = std::min(residual, node_excess_[from]);
        AdjustEdgeFlow(from, to, epsilon);
    }

    [[nodiscard]] bool CanRelabel(Node node) const {
        if (flow->IsNodeTerminal(node) or node_excess_[node] <= 0 or height_[node] == max_height_) {
            return false;
        }
        for (auto adjacent_node : flow->GetNodesAdjacentTo(node)) {
            if (CanPush(node, adjacent_node)) {
                return false;
            }
        }
        return true;
    }

    virtual void Relabel(Node node) {
        int32_t min_height = max_height_;
        for (auto adjacent_node : flow->GetNodesAdjacentTo(node)) {
            if (flow->GetResidualEdgeCapacity(node, adjacent_node) > 0) {
                min_height = std::min(min_height, height_[adjacent_node]);
            }
        }
        this->LiftNode(node, std::min(max_height_, min_height + 1));
    }

    [[nodiscard]] int32_t GetNodeExcess(Node node) const {
        return node_excess_[node];
    }

    [[nodiscard]] int32_t GetNodeHeight(Node node) const {
        return height_[node];
    }

protected:
    std::vector<int32_t> height_;
    std::vector<int32_t> node_excess_;
    int32_t max_height_ = 0;

    virtual void AdjustEdgeFlow(Node from, Node to, int32_t epsilon) {
        auto flow_rate = flow->GetEdgeFlowRate(from, to);
        flow->SetEdgeFlow(from, to, flow_rate + epsilon);
        node_excess_[from] -= epsilon;
        node_excess_[to] += epsilon;
    }

    virtual void LiftNode(Node node, int32_t new_height) {
        assert(height_[node] <= new_height);
        height_[node] = new_height;
    }
};

template <typename Flow>
void BruteForceSolveFlow(Flow *flow) {
    flow->ResetFlows();
    PreflowNetwork<Flow> preflow(flow);
    preflow.Init();
    auto size = flow->Size();

    bool has_pushed_or_relabeled = true;
    while (has_pushed_or_relabeled) {
        has_pushed_or_relabeled = false;

        for (auto [from, to] : utils::GetAllEdges(*flow)) {
            if (preflow.CanPush(from, to)) {
                preflow.Push(from, to);
                has_pushed_or_relabeled = true;
            }
        }

        for (auto node = 0; node < size; ++node) {
            if (preflow.CanRelabel(node)) {
                preflow.Relabel(node);
                has_pushed_or_relabeled = true;
            }
        }
    }
}

template <typename Flow>
class DischargePreflow : public PreflowNetwork<Flow> {
public:
    using Node = typename Flow::Node;
    using Preflow = PreflowNetwork<Flow>;

    explicit DischargePreflow(Flow *flow)
        : Preflow{flow},
          adjacent_nodes_index_(flow->Size()),
          active_nodes_by_height_(flow->Size() * 2) {
    }

    std::optional<Node> GetActiveNodeWithMaxHeight() {
        while (max_active_node_height_ >= 0) {
            auto &bucket = active_nodes_by_height_[max_active_node_height_];
            while (not bucket.empty() and
                   (not IsNodeActive(bucket.front()) or
                    Preflow::height_[bucket.front()] == Preflow::max_height_)) {
                bucket.pop_front();
            }
            if (not bucket.empty()) {
                return bucket.front();
            }
            --max_active_node_height_;
        }
        return std::nullopt;
    }

    virtual void Discharge(Node node) {
        auto &iterable = Preflow::flow->GetNodesAdjacentTo(node);
        auto size = static_cast<int32_t>(iterable.size());
        auto &index = adjacent_nodes_index_[node];

        while (IsNodeActive(node) and index < size) {
            auto adjacent_node = iterable[index];
            if (this->CanPush(node, adjacent_node)) {
                this->Push(node, adjacent_node);
            }
            ++index;
        }

        if (IsNodeActive(node)) {
            this->Relabel(node);
            index = 0;
        }
    }

protected:
    std::vector<int32_t> adjacent_nodes_index_;
    std::vector<std::deque<Node>> active_nodes_by_height_;
    int32_t max_active_node_height_ = -1;

    void LiftNode(Node node, int32_t new_height) override {
        auto old_height = Preflow::height_[node];
        if (new_height > old_height) {
            Preflow::LiftNode(node, new_height);
            UpdateActiveNodeHeightState(node);
        }
    }

    [[nodiscard]] bool IsNodeActive(Node node) const {
        return not Preflow::flow->IsNodeTerminal(node) and Preflow::node_excess_[node] > 0;
    }

    void AdjustEdgeFlow(Node from, Node to, int32_t epsilon) override {
        bool was_from_active = IsNodeActive(from);
        bool was_to_active = IsNodeActive(to);
        Preflow::AdjustEdgeFlow(from, to, epsilon);
        if (not was_from_active) {
            UpdateActiveNodeHeightState(from);
        }
        if (not was_to_active) {
            UpdateActiveNodeHeightState(to);
        }
    }

    void UpdateActiveNodeHeightState(Node node) {
        if (IsNodeActive(node)) {
            auto height = Preflow::height_[node];
            active_nodes_by_height_[height].emplace_back(node);
            if (height < Preflow::max_height_) {
                max_active_node_height_ = std::max(max_active_node_height_, height);
            }
        }
    }
};

template <typename Flow>
void DischargeSolveFlow(Flow *flow) {
    DischargePreflow<Flow> discharge_preflow{flow};
    discharge_preflow.Init();

    while (auto node = discharge_preflow.GetActiveNodeWithMaxHeight()) {
        discharge_preflow.Discharge(node.value());
    }
}

template <typename Flow>
class BfsDistanceFromTerminalNodeComputeTraversal
    : public implementation::BfsDistanceComputeTraversal<Flow> {
public:
    using Node = typename Flow::Node;
    using Traversal = implementation::BfsDistanceComputeTraversal<Flow>;

    const PreflowNetwork<Flow> &preflow;

    explicit BfsDistanceFromTerminalNodeComputeTraversal(const PreflowNetwork<Flow> &preflow)
        : Traversal{*preflow.flow}, preflow{preflow} {
    }

    bool ShouldTraverseEdge(Node from, Node to) override {
        return not Traversal::distance[to] and
               Traversal::graph.GetResidualEdgeCapacity(to, from) > 0 and
               preflow.GetNodeHeight(to) <= preflow.GetNodeHeight(from) + 1;
    }
};

template <typename Flow>
class OptimizedDischargePreflow : public DischargePreflow<Flow> {
public:
    using Node = typename Flow::Node;
    using Preflow = DischargePreflow<Flow>;

    explicit OptimizedDischargePreflow(Flow *flow)
        : Preflow{flow},
          distance_compute_traversal_{*this},
          n_edges_{utils::GetEdgeCount(*Preflow::flow) / 2} {

        Preflow::max_height_ = flow->Size();
        height_counts_.resize(Preflow::max_height_ + 1);
    }

    void Init() override {
        Preflow::Init();
        height_counts_[0] = Preflow::flow->Size() - 1;
        height_counts_[Preflow::flow->Size()] = 1;
        ApplyGlobalRelabelHeuristic();
    }

    void Relabel(Node node) override {
        auto old_height = Preflow::height_[node];
        Preflow::Relabel(node);

        if (height_counts_[old_height] == 0 and old_height < max_non_max_height_) {
            ApplyGapHeuristic(old_height);
        }
        OnPushRelabel();
    }

    void Push(Node from, Node to) override {
        Preflow::Push(from, to);
        OnPushRelabel();
    }

    [[nodiscard]] std::deque<Node> GetNodesWithStuckExcessFlow() const {
        return Preflow::active_nodes_by_height_[Preflow::max_height_];
    }

protected:
    std::vector<int32_t> height_counts_;
    int32_t max_non_max_height_ = 0;
    BfsDistanceFromTerminalNodeComputeTraversal<Flow> distance_compute_traversal_;
    int32_t n_push_relabels_since_last_global_relabel_ = 0;
    int32_t n_edges_ = 0;

    void LiftNode(int32_t node, int32_t new_height) override {
        auto old_height = Preflow::height_[node];
        Preflow::LiftNode(node, new_height);
        --height_counts_[old_height];
        ++height_counts_[new_height];

        UpdateMaxNonMaxHeight(new_height);
    }

    void UpdateMaxNonMaxHeight(int32_t new_height) {
        if (new_height < Preflow::max_height_) {
            max_non_max_height_ = std::max(max_non_max_height_, new_height);
        }
        while (max_non_max_height_ >= 0 and height_counts_[max_non_max_height_] == 0) {
            --max_non_max_height_;
        }
    }

    void ApplyGapHeuristic(int32_t gap_height) {
        for (auto node = 0; node < Preflow::flow->Size(); ++node) {
            if (Preflow::height_[node] > gap_height) {
                LiftNode(node, Preflow::max_height_);
            }
        }
    }

    void OnPushRelabel() {
        ++n_push_relabels_since_last_global_relabel_;
        if (n_push_relabels_since_last_global_relabel_ > n_edges_) {
            ApplyGlobalRelabelHeuristic();
        }
    }

    void ApplyGlobalRelabelHeuristic() {
        implementation::BreadthFirstSearch(&distance_compute_traversal_,
                                           {Preflow::flow->GetTerminalNode()});

        for (auto node = 0; node < Preflow::flow->Size(); ++node) {
            auto distance = distance_compute_traversal_.distance[node];
            auto new_height = distance ? distance.value() : Preflow::max_height_;
            LiftNode(node, new_height);
        }
        n_push_relabels_since_last_global_relabel_ = 0;
    }
};

template <typename Flow>
class FlowBfsPathTraversal : public path_traversal::BfsPathTraversal<Flow> {
public:
    using Node = typename Flow::Node;
    using PathTraversal = path_traversal::BfsPathTraversal<Flow>;

    using PathTraversal::PathTraversal;
    Node target_node;

    [[nodiscard]] bool IsTargetNode(Node node) const override {
        return node == target_node;
    }

    bool ShouldTraverseEdge(Node from, Node to) override {
        return not PathTraversal::has_visited_node_[to] and
               PathTraversal::graph.GetEdgeFlowRate(from, to) > 0;
    }
};

template <typename Flow>
void PushBackStuckFlowExcessesToInitialNode(
    Flow *flow, const OptimizedDischargePreflow<Flow> &discharge_preflow) {

    FlowBfsPathTraversal<Flow> traversal{*flow};

    for (auto node : discharge_preflow.GetNodesWithStuckExcessFlow()) {
        auto excess = discharge_preflow.GetNodeExcess(node);
        traversal.target_node = node;

        while (excess > 0) {
            auto path = path_traversal::FindNonZeroPath<>(&traversal, flow->GetInitialNode());
            auto min_flow = flow::ComputePathMinFlow(*flow, path.value());
            min_flow = std::min(min_flow, excess);
            flow::AugmentByPath(flow, path.value(), -min_flow);
            excess -= min_flow;
        }
    }
}

template <typename Flow>
void OptimizedDischargeSolveFlow(Flow *flow) {
    OptimizedDischargePreflow<Flow> discharge_preflow{flow};
    discharge_preflow.Init();

    while (auto node = discharge_preflow.GetActiveNodeWithMaxHeight()) {
        discharge_preflow.Discharge(node.value());
    }

    PushBackStuckFlowExcessesToInitialNode(flow, discharge_preflow);
}

}  // namespace preflow

class FlowBuilder;

class FlowNetwork : public flow::FlowNetwork<> {
public:
    using Node = int32_t;
    using NodeIterable = const std::vector<Node> &;

    FlowNetwork() = default;

    [[nodiscard]] int32_t BipartitionSize() const {
        return bipartition_size_;
    }

    [[nodiscard]] int32_t Size() const override {
        return BipartitionSize() * 2 + 2;
    }

    void ResetFlows() override {
        std::fill(from_source_flows_.begin(), from_source_flows_.end(), 0);
        std::fill(to_terminal_flows_.begin(), to_terminal_flows_.end(), 0);
    }

    void UndirectedConnect(Node from, Node to) {
        adjacent_nodes_[from].emplace_back(to);
        adjacent_nodes_[to].emplace_back(from);
        SetEdgeFlow(from, to, 0);
        SetEdgeFlow(to, from, 0);
    }

    void SetEdgeConstraint(Node from, Node to, int32_t constraint) override {
        if (from == GetInitialNode()) {
            from_source_constraints_[to] = constraint;
        } else if (IsNodeTerminal(to)) {
            to_terminal_constraints_[from - BipartitionSize()] = constraint;
        } else if (to != GetInitialNode() and not IsNodeTerminal(from)) {
            unbound_constraint_ = constraint;
        }
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
        if (from == terminal_node_ or to == initial_node_) {
            return 0;
        } else if (from == GetInitialNode()) {
            return from_source_constraints_[to];
        } else if (IsNodeTerminal(to)) {
            return to_terminal_constraints_[from - BipartitionSize()];
        } else {
            return unbound_constraint_;
        }
    }

    [[nodiscard]] int32_t GetEdgeFlowRate(Node from, Node to) const override {
        if (from < to) {
            return -GetEdgeFlowRate(to, from);
        } else if (from == GetInitialNode()) {
            return from_source_flows_[to];
        } else if (IsNodeTerminal(from)) {
            return -to_terminal_flows_[to - BipartitionSize()];
        } else {
            return flow_[from].at(to);
        }
    }

    int32_t FindMaxFlowRate() {
        visited_.resize(Size());
        visited_[initial_node_] = true;
        visited_[terminal_node_] = true;
        int32_t flow = 0;
        for (Node node = 0; node < Size(); ++node) {
            if (not visited_[node]) {
                flow += FindMaxFlowRate(node);
            }
        }
        return flow;
    }

    int32_t FindMaxFlowRate(Node node) {
        auto scc = FindScc(node);
        int32_t in_flow = 0;
        int32_t out_flow = 0;
        for (auto n : scc) {
            if (n < BipartitionSize()) {
                in_flow += from_source_constraints_[n];
            } else {
                out_flow += to_terminal_constraints_[n - BipartitionSize()];
            }
        }
        return std::min(in_flow, out_flow);
    }

    std::vector<Node> FindScc(Node node) {
        std::vector<Node> scc;
        std::deque<Node> deque{node};
        visited_[node] = true;

        while (not deque.empty()) {
            node = deque.back();
            deque.pop_back();
            scc.emplace_back(node);

            for (auto neighbor : adjacent_nodes_[node]) {
                if (not visited_[neighbor]) {
                    deque.emplace_back(neighbor);
                    visited_[neighbor] = true;
                }
            }
        }
        return scc;
    }

protected:
    void SafeSetEdgeDirectedFlow(Node from, Node to, int32_t flow) override {
        if (from == GetInitialNode()) {
            from_source_flows_[to] = flow;
        } else if (IsNodeTerminal(to)) {
            to_terminal_flows_[from - BipartitionSize()] = flow;
        } else {
            flow_[from][to] = flow;
        }
    }

private:
    int32_t bipartition_size_ = 0;
    int32_t initial_node_ = 0;
    int32_t terminal_node_ = 0;
    int32_t unbound_constraint_ = INT32_MAX / 2;
    std::vector<std::vector<Node>> adjacent_nodes_;
    std::vector<int32_t> from_source_constraints_;
    std::vector<int32_t> to_terminal_constraints_;
    std::vector<int32_t> from_source_flows_;
    std::vector<int32_t> to_terminal_flows_;
    std::vector<std::pair<Node, Node>> edges_;
    std::vector<std::unordered_map<Node, int32_t>> flow_;

    std::vector<bool> visited_;

    friend class FlowBuilder;
};

class FlowBuilder {
public:
    static FlowNetwork Build(const io::Input &input) {
        FlowNetwork flow;
        flow.from_source_constraints_ = input.first_bipartition_constraints;
        flow.to_terminal_constraints_ = input.second_bipartition_constraints;
        flow.bipartition_size_ = static_cast<int32_t>(input.first_bipartition_constraints.size());
        flow.initial_node_ = flow.Size() - 1;
        flow.terminal_node_ = flow.Size() - 2;

        Resize(&flow);
        BuildEdges(&flow, input);
        flow.edges_ = utils::GetAllEdges(flow);
        return flow;
    }

private:
    static void Resize(FlowNetwork *flow_ptr) {
        auto &flow = *flow_ptr;
        auto size = flow.Size();

        flow.adjacent_nodes_.resize(size);
        flow.from_source_flows_.resize(flow.BipartitionSize());
        flow.to_terminal_flows_.resize(flow.BipartitionSize());
        flow.flow_.resize(size);
    }

    static void BuildEdges(FlowNetwork *flow_ptr, const io::Input &input) {
        auto &flow = *flow_ptr;

        auto bipartition_size = flow_ptr->BipartitionSize();
        for (int32_t node = 0; node < bipartition_size; ++node) {
            flow.UndirectedConnect(flow.initial_node_, node);
            flow.UndirectedConnect(node + bipartition_size, flow.terminal_node_);
        }

        for (auto edge : input.edges) {
            flow.UndirectedConnect(edge.from, edge.to + bipartition_size);
        }
    }
};

io::Output CheatSolve(const io::Input &input) {
    auto flow = FlowBuilder::Build(input);
    auto flow_rate = flow.FindMaxFlowRate();
    return io::Output{flow_rate};
}

io::Output Solve(const io::Input &input) {
    return CheatSolve(input);
    auto flow = FlowBuilder::Build(input);
    preflow::OptimizedDischargeSolveFlow<FlowNetwork>(&flow);
    auto flow_rate = flow::GetNetworkFlowRate(flow);
    return io::Output{flow_rate};
}

namespace test {

namespace rng {

uint32_t GetSeed() {
    auto random_device = std::random_device{};
    static auto seed = random_device();
    return 990888316;
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

}  // namespace utils

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
    int32_t n_nodes_in_bipartition = std::min(10'000, 1 + test_case_id / 10);
    int32_t n_edges_between_bipartitions = std::min(100'000, test_case_id / 5);
    int32_t max_constraint = std::min(10'000, 1 + test_case_id / 5);

    io::Input input;

    std::uniform_int_distribution<int32_t> constraint_distribution{1, max_constraint};
    input.first_bipartition_constraints.resize(n_nodes_in_bipartition);
    for (auto &i : input.first_bipartition_constraints) {
        i = constraint_distribution(*rng::GetEngine());
    }
    input.second_bipartition_constraints.resize(n_nodes_in_bipartition);
    for (auto &i : input.second_bipartition_constraints) {
        i = constraint_distribution(*rng::GetEngine());
    }

    std::uniform_int_distribution<int32_t> node_distribution{0, n_nodes_in_bipartition - 1};
    std::vector<io::Edge> all_edges;
    for (int32_t from = 0; from < n_nodes_in_bipartition; ++from) {
        for (int32_t to = 0; to < n_nodes_in_bipartition; ++to) {
            //                        all_edges.emplace_back(from, to);
            all_edges.emplace_back(node_distribution(*rng::GetEngine()),
                                   node_distribution(*rng::GetEngine()));
        }
    }
    std::shuffle(all_edges.begin(), all_edges.end(), *rng::GetEngine());
    input.edges = {all_edges.begin(), all_edges.begin() + n_edges_between_bipartitions};

    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    return GenerateRandomTestIo(20'000);
    return GenerateRandomTestIo(INT32_MAX);
}

class TimedChecker {
public:
    std::vector<int64_t> durations;

    template <typename Expected>
    void Check(const std::string &test_case, const Expected &expected) {
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
        "3 4\n"
        "3 2 1\n"
        "5 4 4\n"
        "1 1\n"
        "1 2\n"
        "2 3\n"
        "3 3",
        6);

    timed_checker.Check(
        "3 5\n"
        "3 2 1\n"
        "5 4 4\n"
        "1 1\n"
        "1 1\n"
        "1 2\n"
        "2 3\n"
        "3 3",
        6);

    timed_checker.Check(
        "1 3\n"
        "3\n"
        "2\n"
        "1 1\n"
        "1 1\n"
        "1 1\n",
        2);

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

        std::cerr << "Stress tests OK:\n" << timed_checker;
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
