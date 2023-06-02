#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <functional>
#include <iostream>
#include <optional>
#include <utility>
#include <vector>

namespace io {

struct TrustRelation {
    int32_t from = 0;
    int32_t to = 0;

    TrustRelation() = default;

    TrustRelation(int32_t from, int32_t to) : from{from}, to{to} {
    }
};

struct Input {
    std::vector<int32_t> gold_bars;
    std::vector<TrustRelation> trust_relations;
};

Input ReadInputFromStream(std::istream &in) {
    Input input;
    int32_t n_people = 0;
    int32_t n_relations = 0;
    in >> n_people >> n_relations;

    input.gold_bars.resize(n_people);
    for (auto &gold_bar : input.gold_bars) {
        in >> gold_bar;
    }

    input.trust_relations.resize(n_relations);
    for (auto &trust_relation : input.trust_relations) {
        in >> trust_relation.from >> trust_relation.to;
        --trust_relation.from;
        --trust_relation.to;
    }

    return input;
}

}  // namespace io

namespace utils {

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

        friend bool operator==(Iterator left, Iterator right) {
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
            return first_iter_ == first_end_ || second_iter_ == second_end_;
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

template <typename Predicate>
int32_t BinarySearch(int32_t low, int32_t high, const Predicate &is_answer_no_more_than) {
    if (high < low) {
        throw std::invalid_argument{"Low must be not greater than high."};
    }
    while (low < high) {
        auto mid = low < 0 && high > 0 ? (low + high) / 2 : low + (high - low) / 2;
        if (is_answer_no_more_than(mid)) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return low;
}

}  // namespace utils

namespace interface {

class VirtualBaseClass {
public:
    virtual ~VirtualBaseClass() = default;
};

template <typename Node = int32_t, typename NodeIterable = const std::vector<Node> &>
class Graph : public VirtualBaseClass {
    [[nodiscard]] virtual NodeIterable GetNodesAdjacentTo(Node node) const = 0;
};

template <typename Node = int32_t, typename NodeIterable = const std::vector<Node> &>
class FlowNetwork : public Graph<Node, NodeIterable> {
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
        return graph.GetResidualEdgeCapacity(from, to) > 0 and not distance[to];
    }

    void OnEdgeTraverse(Node from, Node to) override {
        distance[to] = distance[from].value() + 1;
    }
};

}  // namespace implementation

namespace path_traversal {

template <typename Node>
using OptionalPath = std::optional<std::vector<Node>>;

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
        return graph.GetResidualEdgeCapacity(from, to) > 0 and not is_node_cut_[to] and
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

template <typename FlowNetworkBuilder, typename N = int32_t,
          typename NIterable = const std::vector<N> &>
class FlowNetwork : public interface::FlowNetwork<N, NIterable> {
public:
    using Node = N;
    using NodeIterable = NIterable;

    FlowNetwork() = default;

    [[nodiscard]] virtual int32_t Size() const {
        return static_cast<int32_t>(adjacent_nodes_.size());
    }

    [[nodiscard]] NIterable GetNodesAdjacentTo(N node) const override {
        return adjacent_nodes_[node];
    }

    [[nodiscard]] N GetInitialNode() const override {
        return initial_node_;
    }

    [[nodiscard]] N GetTerminalNode() const {
        return terminal_node_;
    }

    [[nodiscard]] bool IsNodeTerminal(N node) const override {
        return node == terminal_node_;
    }

    [[nodiscard]] int32_t GetEdgeConstraint(N from, N to) const override {
        return constraints_[from][to];
    }

    [[nodiscard]] int32_t GetEdgeFlowRate(N from, N to) const override {
        return flows_[from][to];
    }

    [[nodiscard]] virtual int32_t GetResidualEdgeCapacity(N from, N to) const {
        return this->GetEdgeConstraint(from, to) - this->GetEdgeFlowRate(from, to);
    }

    virtual void SetEdgeFlow(N from, N to, int32_t flow) {
        SafeSetEdgeDirectedFlow(from, to, flow);
        SafeSetEdgeDirectedFlow(to, from, -flow);
        CheckEdgeFlow(from, to);
    }

protected:
    std::vector<std::vector<N>> adjacent_nodes_;
    std::vector<std::vector<int32_t>> flows_;
    std::vector<std::vector<int32_t>> constraints_;
    int32_t initial_node_ = 0;
    int32_t terminal_node_ = 0;

    friend FlowNetworkBuilder;

    void SafeSetEdgeDirectedFlow(N from, N to, int32_t flow) {
        flows_[from][to] = flow;
    }

    virtual void CheckEdgeFlow(N from, N to) const {
        if (GetResidualEdgeCapacity(from, to) < 0 || GetResidualEdgeCapacity(to, from) < 0) {
            throw std::runtime_error{"Edge graph cannot be greater than constraint."};
        }
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
int32_t ComputePathResidualFlow(const Flow &flow, const std::vector<typename Flow::Node> &path) {
    if (path.size() <= 1) {
        throw std::invalid_argument{"Path must have at least two nodes."};
    }
    auto begin = path.begin();
    auto end = path.end();
    std::optional<int32_t> path_residual_flow;

    for (auto [from, to] : utils::Zip(begin, end - 1, begin + 1, end)) {
        auto edge_residual_flow = flow.GetResidualEdgeCapacity(from, to);
        if (path_residual_flow) {
            path_residual_flow = std::min(path_residual_flow.value(), edge_residual_flow);
        } else {
            path_residual_flow = edge_residual_flow;
        }
    }
    return path_residual_flow.value();
}

template <typename Flow, typename FindPath>
void ComputeMaxFlow(Flow *flow, const FindPath &find_non_zero_flow_path) {
    while (auto path = find_non_zero_flow_path()) {
        auto path_residual_flow = ComputePathResidualFlow(*flow, path.value());
        AugmentByPath(flow, path.value(), path_residual_flow);
    }
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
void DinicComputeMaxFlow(Flow *flow) {
    decremental_dynamic_reachability::PathFinder<Flow, DinicTraversal<Flow>> path_finder{
        *flow, flow->GetInitialNode()};

    auto find_non_zero_path = [&path_finder]() { return path_finder.FindPath(); };
    ComputeMaxFlow<Flow>(flow, find_non_zero_path);
}

}  // namespace flow

class GoldBarFlowBuilder {
public:
    const io::Input &input;
    int32_t target_max_gold_bars = 0;
    static constexpr int32_t kUnboundEdgeConstraint = INT32_MAX / 2;

    using Flow = flow::FlowNetwork<GoldBarFlowBuilder>;

    GoldBarFlowBuilder(const io::Input &input, int32_t target_max_gold_bars)
        : input{input}, target_max_gold_bars{target_max_gold_bars} {
    }

    Flow Build() {
        Flow flow;
        Resize(&flow, static_cast<int32_t>(input.gold_bars.size()) + 2);
        flow.initial_node_ = flow.Size() - 1;
        flow.terminal_node_ = flow.Size() - 2;
        BuildEdgeConnections(&flow);
        return flow;
    }

private:
    static void Resize(Flow *flow_ptr, int32_t n_nodes) {
        auto &flow = *flow_ptr;

        flow.adjacent_nodes_.resize(n_nodes);
        flow.flows_.resize(n_nodes);
        flow.constraints_.resize(n_nodes);

        for (auto &node_flow : flow.flows_) {
            node_flow.resize(n_nodes);
        }
        for (auto &constraint : flow.constraints_) {
            constraint.resize(n_nodes);
        }
    }

    void BuildEdgeConnections(Flow *flow_ptr) {
        auto &flow = *flow_ptr;

        for (int32_t node = 0; node < flow.Size(); ++node) {
            if (node != flow.GetInitialNode() and not flow.IsNodeTerminal(node)) {
                BuildEdge(flow_ptr, flow.initial_node_, node, false);
                BuildEdge(flow_ptr, node, flow.terminal_node_, false);
            }
        }

        for (auto trust_relation : input.trust_relations) {
            BuildEdge(flow_ptr, trust_relation.from, trust_relation.to, true);
        }
    }

    template <typename Node>
    void BuildEdge(Flow *flow_ptr, Node from, Node to, bool does_first_trust_second) {
        auto &flow = *flow_ptr;
        flow.adjacent_nodes_[from].emplace_back(to);
        flow.adjacent_nodes_[to].emplace_back(from);
        flow.constraints_[from][to] = GetEdgeConstraint(flow, from, to, does_first_trust_second);
    }

    int32_t GetEdgeConstraint(const Flow &flow, Flow::Node from, Flow::Node to,
                              bool does_first_trust_second) {
        if (from == flow.GetInitialNode()) {
            return std::max(0, input.gold_bars[to] - target_max_gold_bars);
        }
        if (flow.IsNodeTerminal(to)) {
            return std::max(0, target_max_gold_bars - input.gold_bars[from]);
        }
        if (does_first_trust_second) {
            return kUnboundEdgeConstraint;
        }
        return 0;
    }
};

template <typename Flow>
[[nodiscard]] int32_t ComputeMaxGoldBarsAfterRedistributionAccordingToFlow(
    const Flow &flow, const std::vector<int32_t> &gold_bars) {

    auto initial_node = flow.GetInitialNode();
    auto terminal_node = flow.GetTerminalNode();
    int32_t max = 0;

    for (int32_t node = 0; node < flow.Size(); ++node) {
        if (node != initial_node and node != terminal_node) {
            auto flow_adjusted_gold_bars = gold_bars[node] -
                                           flow.GetEdgeFlowRate(initial_node, node) +
                                           flow.GetEdgeFlowRate(node, terminal_node);
            max = std::max(max, flow_adjusted_gold_bars);
        }
    }
    return max;
}

template <typename ComputeMaxFlow>
bool CanMaxGoldBarsBeNoMoreThan(const io::Input &input, int32_t target_max_gold_bars,
                                const ComputeMaxFlow &compute_max_flow) {
    GoldBarFlowBuilder builder{input, target_max_gold_bars};
    auto flow = builder.Build();
    compute_max_flow(&flow);
    return ComputeMaxGoldBarsAfterRedistributionAccordingToFlow(flow, input.gold_bars) <=
           target_max_gold_bars;
}

template <typename ComputeMaxFlow>
int32_t FindMinimalMaxGoldBarsAmongPeopleAfterRedistributionToTrustees(
    const io::Input &input, const ComputeMaxFlow &compute_max_flow) {

    auto is_answer_no_more_than = [&input, &compute_max_flow](int32_t gold_bars) -> bool {
        return CanMaxGoldBarsBeNoMoreThan(input, gold_bars, compute_max_flow);
    };

    int32_t low = 0;
    int32_t high = 1 + *std::max_element(input.gold_bars.begin(), input.gold_bars.end());
    auto minimal_max_gold_bars_among_people =
        utils::BinarySearch(low, high, is_answer_no_more_than);
    return minimal_max_gold_bars_among_people;
}

int32_t Solve(const io::Input &input) {
    return FindMinimalMaxGoldBarsAmongPeopleAfterRedistributionToTrustees(
        input, flow::DinicComputeMaxFlow<flow::FlowNetwork<GoldBarFlowBuilder>>);
}

int main(int argc, char *argv[]) {

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cout << Solve(io::ReadInputFromStream(std::cin)) << '\n';

    return 0;
}
