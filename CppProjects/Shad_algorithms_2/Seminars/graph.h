#ifndef GRAPH_H
#define GRAPH_H

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

namespace meta {

namespace enable_if_implementation {

template <bool B, class T = void>
struct enable_if {};

template <class T>
struct enable_if<true, T> {
    typedef T type;
};

}  // namespace enable_if_implementation

template <bool B, class T = void>
using enable_if_t = typename enable_if_implementation::enable_if<B, T>::type;

namespace is_base_of_implementation {

template <typename Base, typename Derived, typename Check = void>
struct meta_best_matching_template_specialization;

template <typename Base, typename Derived>
struct meta_best_matching_template_specialization<
    Base, Derived, meta::enable_if_t<not std::is_class_v<Base> or not std::is_class_v<Derived>>> {

    using is_base = std::false_type;
};

template <typename Base, typename Derived>
struct meta_best_matching_template_specialization<
    Base, Derived, meta::enable_if_t<std::is_class_v<Base> and std::is_class_v<Derived>>> {

    static constexpr std::false_type best_matching_function_overload(...);

    static constexpr std::true_type best_matching_function_overload(Base *base_ptr);

    using is_base = decltype(best_matching_function_overload(static_cast<Base *>(nullptr)));
};

}  // namespace is_base_of_implementation

template <typename Base, typename Derived>
inline constexpr bool is_base_of_v =
    is_base_of_implementation::meta_best_matching_template_specialization<Base,
                                                                          Derived>::is_base::value;

namespace is_template_instance_implementation {

template <template <typename...> typename Template, typename TemplateInstance>
struct meta_best_matching_template_specialization : public std::false_type {};

template <template <typename...> typename Template, typename... TemplateArgs>
struct meta_best_matching_template_specialization<Template, Template<TemplateArgs...>>
    : public std::true_type {};

}  // namespace is_template_instance_implementation

template <template <typename...> typename Template, typename TemplateInstance>
inline constexpr bool is_template_instance_of =
    is_template_instance_implementation::meta_best_matching_template_specialization<
        Template, TemplateInstance>{};

namespace is_template_base_implementation {

template <template <typename...> typename Template, typename DerivedFromTemplateInstance,
          typename TCheck = void>
struct meta_best_matching_template_specialization;

template <template <typename...> typename Template, typename DerivedFromTemplateInstance>
struct meta_best_matching_template_specialization<
    Template, DerivedFromTemplateInstance,
    meta::enable_if_t<not std::is_class_v<DerivedFromTemplateInstance>>> {

    using is_base = std::false_type;
};

template <template <typename...> typename Template, typename DerivedFromTemplateInstance>
struct meta_best_matching_template_specialization<
    Template, DerivedFromTemplateInstance,
    meta::enable_if_t<std::is_class_v<DerivedFromTemplateInstance>>> : DerivedFromTemplateInstance {

    template <typename... TemplateArgs>
    static constexpr std::true_type is_base_test(Template<TemplateArgs...> *);

    static constexpr std::false_type is_base_test(...);

    using is_base = decltype(is_base_test(static_cast<DerivedFromTemplateInstance *>(nullptr)));
};

}  // namespace is_template_base_implementation

template <template <typename...> typename Template, typename DerivedFromTemplateInstance>
inline constexpr bool is_template_base_of =
    is_template_base_implementation::meta_best_matching_template_specialization<
        Template, DerivedFromTemplateInstance>::is_base::value;

}  // namespace meta

namespace interface {

class NotImplementedError : public std::logic_error {
public:
    NotImplementedError() : std::logic_error("Function not yet implemented."){};
};

class VirtualBaseClass {
public:
    ~VirtualBaseClass() = default;
};

class SizeInterface : public VirtualBaseClass {
public:
    [[nodiscard]] virtual size_t Size() const = 0;
};

template <typename Node>
class ConnectByNodesInterface : public VirtualBaseClass {
public:
    virtual void Connect(Node from, Node to) = 0;
};

template <typename Edge>
class ConnectByEdgesInterface : public VirtualBaseClass {
public:
    virtual void AddEdge(Edge edge) = 0;
};

template <typename Node, typename NodeId = int32_t>
class NodeByIdInterface : public VirtualBaseClass {
public:
    virtual Node &operator[](NodeId node_id) = 0;

    virtual Node operator[](NodeId node_id) const = 0;
};

template <typename Node, typename NodeIterable>
class AdjacentInterface : public VirtualBaseClass {
public:
    [[nodiscard]] virtual NodeIterable NodesAdjacentTo(Node node) const = 0;
};

template <typename Node>
class VisitAdjacentInterface : public VirtualBaseClass {
public:
    void VisitNodesAdjacentTo(Node node, const std::function<void(Node node)> &function) {
        throw NotImplementedError{};
    }
};

template <typename NodeState>
class StateInterface : public VirtualBaseClass {
public:
    [[nodiscard]] virtual NodeState GetState() const = 0;

    virtual void SetState(NodeState state) = 0;
};

template <typename NodeState, typename Node>
class NodeStateInterface : public VirtualBaseClass {
public:
    [[nodiscard]] virtual NodeState GetNodeState(Node node) const = 0;

    virtual void SetNodeState(Node node, NodeState state) = 0;
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

    virtual bool ShouldTraverseEdge(Node from, Node to) = 0;

    virtual NodeIterable NodesAdjacentTo(Node node) = 0;

    virtual void OnTraverseEnd() {
    }
};

template <typename N, typename NodeIterable = const std::vector<N> &>
struct GraphTraversalLambdas {
    using Node = N;

    std::function<void()> OnTraverseStart = []() {};
    std::function<void(N node)> OnNodeEnter = [](N) {};
    std::function<void(N from, N to)> OnEdgeDiscovery = [](N, N) {};
    std::function<void(N from, N to)> OnEdgeTraverse = [](N, N) {};
    std::function<void(N from, N to)> OnEdgeBacktrack = [](N, N) {};
    std::function<void(N node)> OnNodeExit = [](N) {};
    std::function<void(N node)> SetNodeStateEntered = [](N) {};
    std::function<void(N node)> SetNodeStateExited = [](N) {};
    std::function<bool(N node)> ShouldVisitNode = [](N) {
        throw NotImplementedError{};
        return false;
    };
    std::function<NodeIterable(N node)> NodesAdjacentTo = [](N) {
        throw NotImplementedError{};
        return NodeIterable{};
    };
    std::function<void()> OnTraverseEnd = []() {};
};

template <typename Value>
class GeneratorInterface : public interface::VirtualBaseClass {
    virtual std::optional<Value> Next() = 0;
};

class Treap {
public:
    template <typename T>
    void SetLeftChild(T new_left_child) {
        throw NotImplementedError{};
    }

    template <typename Treap>
    void SetRightChild(Treap new_right_child) {
        throw NotImplementedError{};
    }

    template <typename Treap>
    Treap DropLeftChild() {
        throw NotImplementedError{};
    }

    template <typename Treap>
    Treap DropRightChild() {
        throw NotImplementedError{};
    }

    template <typename Treap>
    [[nodiscard]] bool CompareKey(const Treap &other) const {
        throw NotImplementedError{};
    }

    template <typename Treap>
    [[nodiscard]] bool ComparePriority(const Treap &other) const {
        throw NotImplementedError{};
    }
};

}  // namespace interface

namespace implementation {

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

template <typename GraphTraversal,
          typename = meta::enable_if_t<
              meta::is_template_base_of<interface::GraphTraversal, GraphTraversal> or
              meta::is_template_base_of<interface::GraphTraversalLambdas, GraphTraversal>>>
void DepthFirstSearch(GraphTraversal *graph_traversal, typename GraphTraversal::Node source_node) {

    graph_traversal->OnTraverseStart();

    if (graph_traversal->ShouldNodeBeConsideredInThisTraversal(source_node)) {
        DepthFirstSearchRecursive(graph_traversal, source_node);
    }

    graph_traversal->OnTraverseEnd();
}

template <typename GraphTraversal,
          typename = meta::enable_if_t<
              meta::is_template_base_of<interface::GraphTraversal, GraphTraversal> or
              meta::is_template_base_of<interface::GraphTraversalLambdas, GraphTraversal>>>
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

template <typename Generator,
          typename = meta::enable_if_t  // @formatter:off
          <meta::is_base_of_v<          // @formatter:on
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

template <typename Treap>
bool ComparatorTreapKey(const Treap &left, const Treap &right) {
    return left.CompareKey(right);
}

template <typename TreapPtr>
bool ComparatorTreapPtrKey(const TreapPtr &left, const TreapPtr &right) {
    return left->CompareKey(*right);
}

template <typename Treap>
bool ComparatorTreapPriority(const Treap &left, const Treap &right) {
    return left.ComparePriority(right);
}

template <typename TreapPtr>
bool ComparatorTreapPtrPriority(const TreapPtr &left, const TreapPtr &right) {
    return left->ComparePriority(*right);
}

template <typename NodePtrIterator, typename NodeComparator>
typename NodePtrIterator::value_type ConstructCartesianTree(NodePtrIterator begin,
                                                            NodePtrIterator end,
                                                            const NodeComparator &comparator) {

    if (begin == end) {
        return {};
    }

    auto min_iter = std::min_element(begin, end, comparator);
    auto min_node = *min_iter;

    min_node->SetLeftChild(ConstructCartesianTree(begin, min_iter, comparator));

    min_node->SetRightChild(ConstructCartesianTree(min_iter + 1, end, comparator));

    return min_node;
}

template <typename TreapPtrIterator>
typename TreapPtrIterator::value_type ConstructTreap(TreapPtrIterator begin, TreapPtrIterator end) {

    std::sort(begin, end, ComparatorTreapPtrKey<typename TreapPtrIterator::value_type>);

    return ConstructCartesianTree(
        begin, end, ComparatorTreapPtrPriority<typename TreapPtrIterator::value_type>);
}

template <typename TreapPtr>
TreapPtr Merge(TreapPtr left, TreapPtr right) {
    if (not left) {
        return right;
    }

    if (not right) {
        return left;
    }

    if (left->ComparePriority(*right)) {

        auto right_child = left->DropRightChild();
        left->SetRightChild(Merge(right_child, right));
        return left;
    } else {

        auto left_child = right->DropLeftChild();
        right->SetLeftChild(Merge(left, left_child));
        return right;
    }
}

template <typename Value>
struct Pair {
    Value left;
    Value right;
};

template <typename TreapPtr>
Pair<TreapPtr> Split(TreapPtr treap, int32_t key_value) {
    if (not treap) {
        return {};
    }

    if (treap->CompareKey(key_value)) {

        auto right_child = treap->DropRightChild();
        auto right_child_split = Split(right_child, key_value);
        treap->SetRightChild(right_child_split.left);
        return {treap, right_child_split.right};
    } else {

        auto left_child = treap->DropLeftChild();
        auto left_child_split_by_index = Split(left_child, key_value);
        treap->SetLeftChild(left_child_split_by_index.right);
        return {left_child_split_by_index.left, treap};
    }
}

template <typename Graph>
class TarjanStronglyConnectedComponentDfsTraversal : public interface::GraphTraversal<int32_t> {
public:
    int32_t n_strongly_connected_components = 0;
    std::vector<int32_t> node_to_scc_map;

    using Node = int32_t;
    using NodeIterable = typename Graph::NodeIterable;

    explicit TarjanStronglyConnectedComponentDfsTraversal(const Graph &graph) : graph_{graph} {
        node_to_scc_map.resize(graph_.Size(), -1);
        stack_.reserve(graph_.Size());
        node_position_on_stack_.resize(graph_.Size());
        stack_low_links_.resize(graph_.Size());
    }

    [[nodiscard]] bool IsNodeOnStack(Node node) const {
        return static_cast<bool>(node_position_on_stack_[node]);
    }

    [[nodiscard]] bool IsNodeUnvisited(Node node) const {
        return not IsNodeOnStack(node) and node_to_scc_map[node] == -1;
    }

    bool ShouldNodeBeConsideredInThisTraversal(Node node) override {
        return IsNodeUnvisited(node);
    }

    void OnNodeEnter(Node node) override {
        PutOnStack(node);
    }

    void PutOnStack(Node node) {
        node_position_on_stack_[node] = stack_.size();
        stack_low_links_[node] = stack_.size();
        stack_.emplace_back(node);
    }

    Node PopFromStack() {
        auto node = stack_.back();
        stack_.pop_back();
        node_to_scc_map[node] = n_strongly_connected_components;
        node_position_on_stack_[node].reset();
        return node;
    }

    bool ShouldTraverseEdge(Node from, Node to) override {
        if (IsNodeOnStack(to)) {
            UpdateLowLink(from, to);
        }
        return IsNodeUnvisited(to);
    }

    void UpdateLowLink(Node from, Node to) {
        stack_low_links_[from] = std::min(stack_low_links_[from], stack_low_links_[to]);
    }

    NodeIterable NodesAdjacentTo(Node node) override {
        return graph_.NodesAdjacentTo(node);
    }

    void OnEdgeBacktrack(Node to, Node from) override {
        UpdateLowLink(from, to);
    }

    void OnNodeExit(Node node) override {
        if (IsSccRoot(node)) {
            while (PopFromStack() != node) {
            }
            ++n_strongly_connected_components;
        }
    }

    [[nodiscard]] bool IsSccRoot(Node node) const {
        return node_position_on_stack_[node] == stack_low_links_[node];
    }

private:
    const Graph &graph_;
    std::vector<int32_t> stack_;
    std::vector<std::optional<int32_t>> node_position_on_stack_;
    std::vector<int32_t> stack_low_links_;
};

template <typename Graph>
std::vector<int32_t> FindStronglyConnectedComponents(const Graph &graph) {
    TarjanStronglyConnectedComponentDfsTraversal tarjan_traversal{graph};

    for (int32_t node = 0; node < graph.Size(); ++node) {
        implementation::DepthFirstSearch(&tarjan_traversal, node);
    }

    return tarjan_traversal.node_to_scc_map;
}

}  // namespace implementation

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

struct UndirectedEdge {
    int32_t from = 0;
    int32_t to = 0;

    UndirectedEdge(int32_t from, int32_t to) : from{from}, to{to} {
        if (from > to) {
            std::swap(from, to);
        }
    }
};

std::vector<UndirectedEdge> BuildTreeEdgesFromPruferCode(const std::vector<int32_t> &prufer_code) {
    std::vector<UndirectedEdge> edges;
    edges.reserve(prufer_code.size() + 2);

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

        edges.emplace_back(next, node_id);

        --count[node_id];
        --count[next];
        if (not count[node_id] and node_id < order) {
            next = node_id;
        }
    }

    while (count[next]) {
        ++next;
    }

    edges.emplace_back(next, static_cast<int32_t>(prufer_code.size() + 1));

    return edges;
}

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

std::vector<UndirectedEdge> GenerateRandomTreeEdges(int32_t size) {
    if (size < 2) {
        return {};
    }

    auto prufer_code = GenerateRandomPruferCode(size);

    return BuildTreeEdgesFromPruferCode(prufer_code);
}

std::vector<UndirectedEdge> GenerateRandomUndirectedConnectedGraphEdges(int32_t n_nodes,
                                                                        int32_t n_edges) {
    if (n_edges + 1 < n_nodes) {
        throw std::invalid_argument{"Not enough edges_table for a connected graph_."};
    }

    if (n_edges > n_nodes * (n_nodes - 1)) {
        throw std::invalid_argument{"Too many edges_table."};
    }

    if (n_nodes == 0 or n_edges == 0) {
        return {};
    }

    auto edges = GenerateRandomTreeEdges(n_nodes);

    std::vector<std::vector<int32_t>> adjacency(n_nodes);
    for (auto edge : edges) {
        adjacency[edge.from].emplace_back(edge.to);
    }

    std::vector<bool> flags(n_nodes);
    std::vector<UndirectedEdge> complement_edges;
    for (int32_t from = 0; from < n_nodes; ++from) {
        flags.clear();
        flags.resize(n_nodes);

        for (auto to : adjacency[from]) {
            flags[to] = true;
        }

        for (int32_t to = from + 1; to < n_nodes; ++to) {
            if (not flags[to]) {
                complement_edges.emplace_back(from, to);
            }
        }
    }

    std::shuffle(complement_edges.begin(), complement_edges.end(), *rng::GetEngine());

    int32_t n_edges_to_add = n_edges - (n_nodes - 1);
    edges.insert(edges.end(), complement_edges.begin(), complement_edges.begin() + n_edges_to_add);

    return edges;
}

std::vector<UndirectedEdge> GenerateRandomGraphWithLoopsAndParallelEdges(int32_t n_nodes,
                                                                         int32_t n_edges) {
    std::vector<UndirectedEdge> edges;
    auto node_id_distribution = std::uniform_int_distribution<int32_t>{0, n_nodes - 1};

    edges.reserve(n_nodes);

    for (int32_t i = 0; i < n_edges; ++i) {
        auto from = node_id_distribution(*rng::GetEngine());
        auto to = node_id_distribution(*rng::GetEngine());
        edges.emplace_back(from, to);
    }

    return edges;
}

}  // namespace test

#endif  // GRAPH_H
