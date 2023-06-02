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
#include <utility>
#include <vector>

namespace io {

class Input {
public:
    std::vector<int32_t> boat_capacity;
    std::vector<std::vector<int32_t>> passenger_boat_preferences;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t n_boats = 0;
        int32_t n_passengers = 0;
        in >> n_boats >> n_passengers;

        boat_capacity.resize(n_boats);
        for (auto &i : boat_capacity) {
            in >> i;
        }

        passenger_boat_preferences.resize(n_passengers);
        for (auto &i : passenger_boat_preferences) {
            int32_t n_preferences = 0;
            in >> n_preferences;
            i.resize(n_preferences);
            for (auto &j : i) {
                in >> j;
                --j;
            }
        }
    }
};

class Output {
public:
    std::optional<int32_t> min_n_of_boats_needed_to_seat_all_passengers;

    Output() = default;

    explicit Output(std::optional<int32_t> answer)
        : min_n_of_boats_needed_to_seat_all_passengers{answer} {
    }

    explicit Output(int32_t answer) {
        if (answer != -1) {
            min_n_of_boats_needed_to_seat_all_passengers = answer;
        }
    }

    std::ostream &Write(std::ostream &out) const {
        out << (min_n_of_boats_needed_to_seat_all_passengers
                    ? min_n_of_boats_needed_to_seat_all_passengers.value()
                    : -1);
        return out;
    }

    bool operator!=(const Output &other) const {
        return min_n_of_boats_needed_to_seat_all_passengers !=
               other.min_n_of_boats_needed_to_seat_all_passengers;
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
        for (auto to : graph.NodesAdjacentTo(from)) {
            edges.emplace_back(from, to);
        }
    }
    return edges;
}

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

    virtual bool ShouldStop() {
        return false;
    }
};

template <typename Graph>
class PathTraversal : public GraphTraversal<typename Graph::Node, typename Graph::NodeIterable> {
public:
    using Node = typename Graph::Node;

    [[nodiscard]] virtual bool IsTargetNode(Node node) const = 0;

    [[nodiscard]] virtual std::optional<std::vector<Node>> GetPath() const = 0;
};

template <typename Node = int32_t, typename NodeIterable = const std::vector<Node> &>
class FlowNetwork : public VirtualBaseClass {
public:
    virtual void ResetFlows() = 0;

    [[nodiscard]] virtual NodeIterable NodesAdjacentTo(Node node) const = 0;

    [[nodiscard]] virtual Node GetInitialNode() const = 0;

    [[nodiscard]] virtual bool IsNodeTerminal(Node node) const = 0;

    [[nodiscard]] virtual int32_t GetEdgeConstraint(Node from, Node to) const = 0;

    [[nodiscard]] virtual int32_t GetEdgeFlow(Node from, Node to) const = 0;

    [[nodiscard]] virtual int32_t GetResidualEdgeCapacity(Node from, Node to) const = 0;

    [[nodiscard]] virtual bool ShouldTraverseEdge(Node from, Node to) const = 0;

    virtual void SetEdgeConstraint(Node from, Node to, int32_t constraint) = 0;

    virtual void AugmentByPath(const std::vector<Node> &path, int32_t flow) = 0;
};

}  // namespace interface

namespace implementation {

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

    while (not queue.empty() and not graph_traversal->ShouldStop()) {

        auto node = queue.front();
        queue.pop_front();

        graph_traversal->OnNodeEnter(node);
        graph_traversal->SetNodeStateEntered(node);

        for (const auto &adjacent_node : graph_traversal->NodesAdjacentTo(node)) {

            if (graph_traversal->ShouldStop()) {
                return;
            }

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

template <typename GraphTraversal>
void DepthFirstSearchRecursive(GraphTraversal *graph_traversal,
                               typename GraphTraversal::Node source_node) {

    if (graph_traversal->ShouldStop()) {
        return;
    }

    graph_traversal->OnNodeEnter(source_node);
    graph_traversal->SetNodeStateEntered(source_node);

    for (auto adjacent_node : graph_traversal->NodesAdjacentTo(source_node)) {
        if (graph_traversal->ShouldStop()) {
            return;
        }

        graph_traversal->OnEdgeDiscovery(source_node, adjacent_node);

        if (graph_traversal->ShouldTraverseEdge(source_node, adjacent_node)) {

            graph_traversal->OnEdgeTraverse(source_node, adjacent_node);
            DepthFirstSearchRecursive(graph_traversal, adjacent_node);
            if (graph_traversal->ShouldStop()) {
                return;
            }
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
void GraphSearch(GraphTraversal *graph_traversal,
                 std::deque<typename GraphTraversal::Node> starting_nodes_queue) {
    BreadthFirstSearch(graph_traversal, starting_nodes_queue);
}

template <typename GraphTraversal>
void GraphSearch(GraphTraversal *graph_traversal, typename GraphTraversal::Node source_node) {
    DepthFirstSearch(graph_traversal, source_node);
}

template <typename Graph>
class PathTraversal : public interface::PathTraversal<Graph> {
public:
    using Node = typename Graph::Node;
    using NodeIterable = typename Graph::NodeIterable;

    const Graph &graph;

    explicit PathTraversal(const Graph &graph) : graph{graph}, has_visited_node_(graph.Size()) {
    }

    void OnTraverseStart() override {
        std::fill(has_visited_node_.begin(), has_visited_node_.end(), false);
    }

    NodeIterable NodesAdjacentTo(Node node) override {
        return graph.NodesAdjacentTo(node);
    }

    void OnNodeEnter(Node node) override {
        has_visited_node_[node] = true;
    }

    bool ShouldTraverseEdge(Node from, Node to) override {
        return graph.ShouldTraverseEdge(from, to) and not has_visited_node_[to];
    }

protected:
    std::vector<bool> has_visited_node_;
};

template <typename Node>
using OptionalPath = std::optional<std::vector<Node>>;

template <typename PathTraversal, typename QueueOrNode>
OptionalPath<typename PathTraversal::Node> FindNonZeroPath(PathTraversal *traversal,
                                                           QueueOrNode from) {
    GraphSearch(traversal, from);
    return traversal->GetPath();
}

template <typename Graph>
class BfsDepthTraversal
    : public interface::GraphTraversal<typename Graph::Node, typename Graph::NodeIterable> {
public:
    using Node = typename Graph::Node;
    using NodeIterable = typename Graph::NodeIterable;

    const Graph &graph;
    std::vector<std::optional<Node>> depths;

    explicit BfsDepthTraversal(const Graph &graph) : graph{graph}, depths(graph.Size()) {
    }

    NodeIterable NodesAdjacentTo(Node node) override {
        return graph.NodesAdjacentTo(node);
    }

    void OnNodeEnter(Node node) override {
        if (not depths[node]) {
            depths[node] = 0;
        }
    }

    bool ShouldTraverseEdge(Node from, Node to) override {
        return graph.ShouldTraverseEdge(from, to) and not depths[to];
    }

    void OnEdgeTraverse(Node from, Node to) override {
        depths[to] = depths[from].value() + 1;
    }
};

template <typename Graph>
std::vector<std::optional<typename Graph::Node>> FindNodeDepths(const Graph &graph,
                                                                typename Graph::Node source_node) {
    BfsDepthTraversal traversal{graph};
    implementation::GraphSearch(&traversal, std::deque<typename Graph::Node>{source_node});
    return traversal.depths;
}

template <typename Graph>
class StatefulDfsDecrementalDynamicReachabilityPathTraversal
    : public interface::PathTraversal<Graph> {
public:
    using Node = typename Graph::Node;
    using NodeIterable = typename Graph::NodeIterable;

    const Graph &graph;
    Node source;

    StatefulDfsDecrementalDynamicReachabilityPathTraversal(const Graph &graph, Node source)
        : graph{graph}, source{source}, is_node_cut_(graph.Size()) {
        Reset();
    }

    void Reset() {
        std::fill(is_node_cut_.begin(), is_node_cut_.end(), false);
        depths_ = implementation::FindNodeDepths(graph, source);
    }

    implementation::OptionalPath<Node> FindPath() {
        if (auto path = implementation::FindNonZeroPath<>(this, source)) {
            return path;
        }
        Reset();
        return implementation::FindNonZeroPath<>(this, source);
    };

    void CutNode(Node node) {
        is_node_cut_[node] = true;
    }

    void OnTraverseStart() override {
        path_.clear();
        should_stop_ = false;
    }

    bool ShouldStop() override {
        return should_stop_;
    }

    NodeIterable NodesAdjacentTo(Node node) override {
        return graph.NodesAdjacentTo(node);
    }

    bool ShouldTraverseEdge(Node from, Node to) override {
        return not ShouldStop() and graph.ShouldTraverseEdge(from, to) and not is_node_cut_[to] and
               depths_[to].value() == depths_[from].value() + 1;
    }

    void OnNodeEnter(Node node) override {
        if (not ShouldStop()) {
            path_.emplace_back(node);
            if (this->IsTargetNode(node)) {
                should_stop_ = true;
            }
        }
    }

    void OnNodeExit(Node) override {
        if (not ShouldStop()) {
            path_.pop_back();
        }
    }

    void OnEdgeBacktrack(Node to, Node from) override {
        CutNode(to);
    }

    [[nodiscard]] implementation::OptionalPath<Node> GetPath() const override {
        if (not path_.empty() and this->IsTargetNode(path_.back())) {
            return path_;
        } else {
            return std::nullopt;
        }
    }

private:
    std::vector<std::optional<int32_t>> depths_;
    std::vector<bool> is_node_cut_;
    std::vector<Node> path_;
    bool should_stop_ = false;
};

template <typename Node = int32_t, typename NodeIterable = const std::vector<Node> &>
class FlowNetwork : public interface::FlowNetwork<Node, NodeIterable> {
public:
    [[nodiscard]] int32_t GetResidualEdgeCapacity(Node from, Node to) const override {
        return this->GetEdgeConstraint(from, to) - this->GetEdgeFlow(from, to);
    }

    [[nodiscard]] bool ShouldTraverseEdge(Node from, Node to) const override {
        return GetResidualEdgeCapacity(from, to) > 0;
    }

    void AugmentByPath(const std::vector<Node> &path, int32_t flow) override {
        if (path.front() != this->GetInitialNode() or not this->IsNodeTerminal(path.back())) {
            throw std::invalid_argument{"Invalid path."};
        }

        auto begin = path.begin();
        auto end = path.end();
        for (auto [from, to] : utils::Zip(begin, end - 1, begin + 1, end)) {
            auto edge_flow = this->GetEdgeFlow(from, to);
            auto augmented_flow = edge_flow + flow;
            SetEdgeFlow(from, to, augmented_flow);
            SetEdgeFlow(to, from, -augmented_flow);
        }
    }

    void AugmentByFlow(const FlowNetwork &augmenting_flow_network) {
        for (auto [from, to] : GetAllEdges()) {
            auto flow = this->GetEdgeFlow(from, to);
            flow += augmenting_flow_network.GetEdgeFlow(from, to);
            SetEdgeFlow(from, to, flow);
        }
    }

    [[nodiscard]] int32_t GetNetworkFlowRate() const {
        int32_t flow_rate = 0;
        auto initial_node = this->GetInitialNode();
        for (auto node : this->NodesAdjacentTo(initial_node)) {
            flow_rate += this->GetEdgeFlow(initial_node, node);
        }
        return flow_rate;
    }

    [[nodiscard]] const std::vector<std::pair<Node, Node>> &GetAllEdges() const {
        throw interface::NotImplementedError {};
    }

    [[nodiscard]] virtual int32_t GetMaxConstraint() const {
        auto max_constraint = 0;
        for (auto [from, to] : GetAllEdges()) {
            auto constraint = this->GetEdgeConstraint(from, to);
            max_constraint = std::max(max_constraint, constraint);
        }
        return max_constraint;
    }

    void virtual Scale(double factor) {
        for (auto [from, to] : GetAllEdges()) {
            auto constraint = this->GetEdgeConstraint(from, to);
            auto flow = this->GetEdgeFlow(from, to);

            this->SetEdgeConstraint(from, to, static_cast<int32_t>(constraint * factor));
            this->SetEdgeFlow(from, to, static_cast<int32_t>(flow * factor));
        }
    }

protected:
    virtual void SetEdgeFlow(Node from, Node to, int32_t flow) {
        if (flow > this->GetEdgeConstraint(from, to)) {
            throw std::invalid_argument{"Edge flow_ cannot be greater than constraint."};
        }
        SafeSetEdgeFlow(from, to, flow);
    }

    virtual void SafeSetEdgeFlow(Node from, Node to, int32_t flow) = 0;
};

}  // namespace implementation

namespace flow {

template <typename Flow>
class NonZeroFlowPathDfsTraversal : public implementation::PathTraversal<Flow> {
public:
    using Node = typename Flow::Node;
    using PathTraversal = implementation::PathTraversal<Flow>;

    explicit NonZeroFlowPathDfsTraversal(const Flow &flow) : PathTraversal{flow} {
    }

    [[nodiscard]] bool IsTargetNode(Node node) const override {
        return PathTraversal::graph.IsNodeTerminal(node);
    }

    void OnTraverseStart() override {
        PathTraversal::OnTraverseStart();
        path_.clear();
        should_stop_ = false;
    }

    bool ShouldStop() override {
        return should_stop_;
    }

    void OnNodeEnter(Node node) override {
        PathTraversal::OnNodeEnter(node);
        if (not ShouldStop()) {
            path_.emplace_back(node);
            if (IsTargetNode(node)) {
                should_stop_ = true;
            }
        }
    }

    void OnNodeExit(Node) override {
        if (not ShouldStop()) {
            path_.pop_back();
        }
    }

    [[nodiscard]] implementation::OptionalPath<Node> GetPath() const override {
        if (IsTargetNode(path_.back())) {
            return path_;
        } else {
            return std::nullopt;
        }
    }

private:
    std::vector<Node> path_;
    bool should_stop_ = false;
};

template <typename Flow>
class NonZeroFlowPathBfsTraversal : public implementation::PathTraversal<Flow> {
public:
    using Node = typename Flow::Node;
    using PathTraversal = implementation::PathTraversal<Flow>;

    explicit NonZeroFlowPathBfsTraversal(const Flow &flow)
        : PathTraversal{flow}, parent_(flow.Size()) {
    }

    [[nodiscard]] bool IsTargetNode(Node node) const override {
        return PathTraversal::graph.IsNodeTerminal(node);
    }

    void OnTraverseStart() override {
        PathTraversal::OnTraverseStart();
        path_.clear();
        should_stop_ = false;
        std::fill(parent_.begin(), parent_.end(), std::nullopt);
    }

    bool ShouldStop() override {
        return should_stop_;
    }

    void OnEdgeTraverse(Node from, Node to) override {
        if (not ShouldStop()) {
            parent_[to] = from;
        }
    }

    void OnNodeEnter(Node node) override {
        PathTraversal::OnNodeEnter(node);
        if (IsTargetNode(node)) {
            BuildNonZeroPath(node);
            should_stop_ = true;
        }
    }

    [[nodiscard]] implementation::OptionalPath<Node> GetPath() const override {
        if (not path_.empty() and this->IsTargetNode(path_.back())) {
            return path_;
        } else {
            return std::nullopt;
        }
    }

private:
    std::vector<std::optional<Node>> parent_;
    std::vector<Node> path_;
    bool should_stop_ = false;

    void BuildNonZeroPath(Node terminal_node) {
        auto node = terminal_node;
        for (; parent_[node]; node = parent_[node].value()) {
            path_.emplace_back(node);
        }
        path_.emplace_back(node);
        std::reverse(path_.begin(), path_.end());
    }
};

template <typename Flow>
using FindPath = std::function<implementation::OptionalPath<typename Flow::Node>()>;

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
void SolveFlow(Flow *flow, const FindPath<Flow> &find_non_zero_flow_path) {
    while (auto path = find_non_zero_flow_path()) {
        auto path_residual_flow = ComputePathResidualFlow(*flow, path.value());
        flow->AugmentByPath(path.value(), path_residual_flow);
    }
}

template <typename Flow>
void FordFulkersonSolveFlow(Flow *flow) {
    NonZeroFlowPathDfsTraversal traversal{*flow};
    auto initial_node = flow->GetInitialNode();

    FindPath<Flow> find_non_zero_path = [&traversal, initial_node]() {
        return implementation::FindNonZeroPath<>(&traversal, initial_node);
    };
    SolveFlow<Flow>(flow, find_non_zero_path);
}

template <typename Flow>
void EdmondsKarpSolveFlow(Flow *flow) {
    NonZeroFlowPathBfsTraversal traversal{*flow};
    auto initial_node = std::deque<typename Flow::Node>{flow->GetInitialNode()};

    FindPath<Flow> find_non_zero_path = [&traversal, initial_node]() {
        return implementation::FindNonZeroPath<>(&traversal, initial_node);
    };
    SolveFlow<Flow>(flow, find_non_zero_path);
}

template <typename Flow>
class DinicTraversal
    : public implementation::StatefulDfsDecrementalDynamicReachabilityPathTraversal<Flow> {
public:
    using S = implementation::StatefulDfsDecrementalDynamicReachabilityPathTraversal<Flow>;

    explicit DinicTraversal(const Flow &flow) : S{flow, flow.GetInitialNode()} {
    }

    [[nodiscard]] bool IsTargetNode(typename Flow::Node node) const override {
        return S::graph.IsNodeTerminal(node);
    }
};

template <typename Flow>
void DinicSolveFlow(Flow *flow) {
    DinicTraversal<Flow> traversal{*flow};
    FindPath<Flow> find_non_zero_path = [&traversal]() { return traversal.FindPath(); };
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

class FlowBuilder;

class FlowNetwork : public implementation::FlowNetwork<> {
public:
    using Node = int32_t;
    using NodeIterable = const std::vector<Node> &;

    const io::Input *input = nullptr;
    int32_t n_boats = 0;

    FlowNetwork() = default;

    [[nodiscard]] int32_t Size() const {
        return static_cast<int32_t>(flows_.size());
    }

    void ResetFlows() override {
        for (auto &f : flows_) {
            std::fill(f.begin(), f.end(), 0);
        }
    }

    void UndirectedConnect(Node from, Node to) {
        adjacent_nodes_[from].emplace_back(to);
        adjacent_nodes_[to].emplace_back(from);
    }

    void SetEdgeConstraint(Node from, Node to, int32_t constraint) override {
    }

    [[nodiscard]] NodeIterable NodesAdjacentTo(Node node) const override {
        return adjacent_nodes_[node];
    }

    [[nodiscard]] Node GetInitialNode() const override {
        return initial_node_;
    }

    [[nodiscard]] bool IsNodeTerminal(Node node) const override {
        return node == terminal_node_;
    }

    void SetRemovedBoats(std::vector<bool> is_boat_removed) {
        is_boat_removed_ = std::move(is_boat_removed);
        n_removed_boats_ = std::accumulate(is_boat_removed_.begin(), is_boat_removed_.end(), 0);
    }

    void RemoveBoat(int32_t boat) {
        if (not is_boat_removed_[boat]) {
            ++n_removed_boats_;
            is_boat_removed_[boat] = true;
        }
    }

    void KeepBoat(int32_t boat) {
        if (is_boat_removed_[boat]) {
            --n_removed_boats_;
            is_boat_removed_[boat] = false;
        }
    }

    [[nodiscard]] int32_t GetNRemovedBoats() const {
        return n_removed_boats_;
    }

    [[nodiscard]] int32_t GetEdgeConstraint(Node from, Node to) const override {
        if (from == GetInitialNode() and to < n_passengers_) {
            return 1;
        } else if (from < n_passengers_ and to >= n_passengers_ and not IsNodeTerminal(to)) {
            return 1;
        } else if (from >= n_passengers_ and not IsNodeTerminal(from) and IsNodeTerminal(to)) {
            auto boat = from - n_passengers_;
            return is_boat_removed_[boat] ? 0 : input->boat_capacity[boat];
        } else {
            return 0;
        }
    }

    [[nodiscard]] int32_t GetEdgeFlowRate(Node from, Node to) const override {
        return flows_[from][to];
    }

    [[nodiscard]] bool AreSourceEdgesSaturated() const {
        auto initial_node = GetInitialNode();
        for (auto node : NodesAdjacentTo(initial_node)) {
            if (GetResidualEdgeCapacity(initial_node, node) != 0) {
                return false;
            }
        }
        return true;
    }

    std::vector<bool> is_boat_removed_;

protected:
    void SafeSetEdgeFlow(Node from, Node to, int32_t flow) override {
        flows_[from][to] = flow;
    }

private:
    int32_t initial_node_ = 0;
    int32_t terminal_node_ = 0;
    std::vector<std::vector<Node>> adjacent_nodes_;
    std::vector<std::vector<int32_t>> flows_;
    int32_t n_passengers_ = 0;
    int32_t n_removed_boats_ = 0;

    friend class FlowBuilder;
};

class FlowBuilder {
public:
    static FlowNetwork Build(const io::Input &input) {
        FlowNetwork flow;
        flow.input = &input;
        flow.n_boats = static_cast<int32_t>(input.boat_capacity.size());
        flow.n_passengers_ = static_cast<int32_t>(input.passenger_boat_preferences.size());
        auto n_nodes = flow.n_boats + flow.n_passengers_ + 2;

        Resize(&flow, n_nodes);

        flow.initial_node_ = flow.Size() - 1;
        flow.terminal_node_ = flow.Size() - 2;

        BuildEdges(&flow, input);

        return flow;
    }

private:
    static void Resize(FlowNetwork *flow, int32_t n_nodes) {
        flow->adjacent_nodes_.resize(n_nodes);
        flow->is_boat_removed_.resize(flow->n_boats);

        flow->flows_.resize(n_nodes);
        for (auto &i : flow->flows_) {
            i.resize(n_nodes);
        }
    }

    static void BuildEdges(FlowNetwork *flow, const io::Input &input) {
        auto initial_node = flow->GetInitialNode();
        auto terminal_node = flow->terminal_node_;

        for (int32_t i = 0; i < flow->n_passengers_; ++i) {
            flow->UndirectedConnect(initial_node, i);
            for (auto boat : input.passenger_boat_preferences[i]) {
                flow->UndirectedConnect(i, boat + flow->n_passengers_);
            }
        }

        for (int32_t i = 0; i < flow->n_boats; ++i) {
            flow->UndirectedConnect(flow->n_passengers_ + i, terminal_node);
        }
    }
};

class Solver {
public:
    const io::Input &input;

    explicit Solver(const io::Input &input)
        : input{input}, flow_{FlowBuilder::Build(input)}, min_boats_{flow_.n_boats + 1} {
    }

    std::optional<int32_t> Solve() {
        if (not CanFitAllPassengers()) {
            return std::nullopt;
        }
        RecursiveSolve(0);
        //        std::cerr << n_calls_ << '\n';
        return min_boats_;
    }

private:
    FlowNetwork flow_;
    int32_t min_boats_ = 0;
    int32_t n_calls_ = 0;

    void RecursiveSolve(int32_t current_boat) {
        min_boats_ = std::min(min_boats_, flow_.n_boats - flow_.GetNRemovedBoats());
        while (current_boat < flow_.n_boats and
               current_boat - flow_.GetNRemovedBoats() < min_boats_) {
            auto removed_boats = flow_.is_boat_removed_;
            flow_.RemoveBoat(current_boat);
            ++current_boat;
            if (CanFitAllPassengers()) {
                RecursiveSolve(current_boat);
            }
            flow_.SetRemovedBoats(removed_boats);
        }
    }

    bool CanFitAllPassengers() {
        ++n_calls_;
        flow_.ResetFlows();
        flow::DinicSolveFlow<FlowNetwork>(&flow_);
        return flow_.AreSourceEdgesSaturated();
    }
};

io::Output Solve(const io::Input &input) {
    Solver solver{input};
    return io::Output{solver.Solve()};
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

class RecursiveBfSolver {
public:
    const io::Input &input;

    explicit RecursiveBfSolver(const io::Input &input)
        : input{input},
          n_boats_{static_cast<int32_t>(input.boat_capacity.size())},
          n_passengers_{static_cast<int32_t>(input.passenger_boat_preferences.size())} {
    }

    std::optional<int32_t> Solve() {
        std::vector<int32_t> seats(n_boats_);
        auto min = RecursiveSolve(n_passengers_ - 1, seats);
        if (min == INT32_MAX) {
            return std::nullopt;
        } else {
            return min;
        }
    }

private:
    int32_t n_boats_ = 0;
    int32_t n_passengers_ = 0;

    int32_t RecursiveSolve(int32_t passenger, const std::vector<int32_t> &seats) {
        if (passenger < 0) {
            int32_t n_non_empty_boats = 0;
            for (int32_t i = 0; i < n_boats_; ++i) {
                if (seats[i] > 0) {
                    if (seats[i] > input.boat_capacity[i]) {
                        return INT32_MAX;
                    }
                    ++n_non_empty_boats;
                }
            }
            return n_non_empty_boats;
        }
        int32_t min = INT32_MAX;
        for (auto boat : input.passenger_boat_preferences[passenger]) {
            auto seats_copy = seats;
            ++seats_copy[boat];
            min = std::min(min, RecursiveSolve(passenger - 1, seats_copy));
        }
        return min;
    }
};

io::Output BruteForceSolve(const io::Input &input) {
    if (input.boat_capacity.size() > 10) {
        throw NotImplementedError{};
    }
    return io::Output{RecursiveBfSolver{input}.Solve()};
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t n_boats = std::min(16, 1 + test_case_id / 20);
    int32_t n_passengers = std::min(100, 1 + test_case_id / 10);

    std::uniform_int_distribution<int32_t> capacity_distribution{0, n_passengers};
    std::uniform_int_distribution<int32_t> n_preferences_distribution{n_boats / 2, n_boats};

    io::Input input;
    input.boat_capacity.resize(n_boats);
    for (auto &i : input.boat_capacity) {
        i = capacity_distribution(*rng::GetEngine());
    }

    std::vector<int32_t> preferences(n_boats);
    std::iota(preferences.begin(), preferences.end(), 0);
    input.passenger_boat_preferences.resize(n_passengers);
    for (auto &i : input.passenger_boat_preferences) {
        auto n_preferences = n_preferences_distribution(*rng::GetEngine());
        std::shuffle(preferences.begin(), preferences.end(), *rng::GetEngine());
        i = {preferences.begin(), preferences.begin() + n_preferences};
    }

    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
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

    TimedChecker timed_checker;

    timed_checker.Check(
        "3 5\n"
        "1\n"
        "2\n"
        "3\n"
        "3 1 2 3\n"
        "2 1 2\n"
        "1 1\n"
        "2 1 2\n"
        "3 1 2 3",
        3);

    timed_checker.Check(
        "3 5\n"
        "1\n"
        "2\n"
        "3\n"
        "3 1 2 3\n"
        "2 1 2\n"
        "0 \n"
        "2 1 2\n"
        "3 1 2 3",
        -1);

    timed_checker.Check(
        "3 3\n"
        "1\n"
        "2\n"
        "3\n"
        "3 1 2 3\n"
        "2 1 3\n"
        "3 1 2 3\n",
        1);

    timed_checker.Check(
        "3 3\n"
        "1\n"
        "2\n"
        "3\n"
        "2 1 3\n"
        "3 1 2 3\n"
        "2 2 3\n",
        1);

    timed_checker.Check(
        "3 3\n"
        "1\n"
        "2\n"
        "3\n"
        "2 1 3\n"
        "3 1 2 3\n"
        "2 2 3\n",
        1);

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
