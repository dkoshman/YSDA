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

namespace rng {

uint32_t GetSeed() {
    auto random_device = std::random_device{};
    static auto seed = random_device();
    return 990288894;
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

struct UndirectedEdge {
    int32_t from = 0;
    int32_t to = 0;

    UndirectedEdge(int32_t from, int32_t to) : from{from}, to{to} {
    }
};

struct FindLeastCommonAncestorDepth {
    struct Request {
        int32_t first = 0;
        int32_t second = 0;

        Request(int32_t first, int32_t second) : first{first}, second{second} {
        }
    };

    struct Response {
        int32_t lca_depth = 0;

        explicit Response(int32_t lca_depth) : lca_depth{lca_depth} {
        }
    };
};

class Input {
public:
    std::vector<UndirectedEdge> edges;
    int32_t n_nodes = 0;
    int32_t root_id = 0;
    std::vector<FindLeastCommonAncestorDepth::Request> requests;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t n_edges = 0;
        in >> n_nodes >> n_edges >> root_id;
        --root_id;
        edges.reserve(n_edges);

        for (int32_t edge_id = 0; edge_id < n_edges; ++edge_id) {
            int32_t from = 0;
            int32_t to = 0;
            in >> from >> to;
            --from;
            --to;
            edges.emplace_back(from, to);
        }

        int32_t n_requests = 0;
        in >> n_requests;
        requests.reserve(n_requests);
        for (int32_t request_id = 0; request_id < n_requests; ++request_id) {
            int32_t first = 0;
            int32_t second = 0;
            in >> first >> second;
            --first;
            --second;
            requests.emplace_back(first, second);
        }
    }
};

class Output {
public:
    std::vector<int32_t> least_common_ancestor_depths;

    Output() = default;

    explicit Output(const std::vector<FindLeastCommonAncestorDepth::Response> &responses) {
        least_common_ancestor_depths.reserve(responses.size());
        for (auto &response : responses) {
            least_common_ancestor_depths.emplace_back(response.lca_depth);
        }
    }

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        int32_t item = 0;
        while (ss >> item) {
            least_common_ancestor_depths.emplace_back(item);
        }
    }

    std::ostream &Write(std::ostream &out) const {
        for (auto item : least_common_ancestor_depths) {
            out << item << '\n';
        }
        return out;
    }

    bool operator!=(const Output &other) const {
        return least_common_ancestor_depths != other.least_common_ancestor_depths;
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

    static constexpr std::true_type best_matching_function_overload(Base *);

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
    std::enable_if_t<std::is_class_v<DerivedFromTemplateInstance>>> : DerivedFromTemplateInstance {

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

class ResizeInterface : public VirtualBaseClass {
public:
    virtual void Resize(size_t size) = 0;

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

    virtual void OnNodeEnter(Node node) {
    }

    virtual void OnEdgeDiscovery(Node from, Node to) {
    }

    virtual void OnEdgeTraverse(Node from, Node to) {
    }

    virtual void OnEdgeBacktrack(Node from, Node to) {
    }

    virtual void OnNodeExit(Node node) {
    }

    virtual void SetNodeStateEntered(Node node) {
    }

    virtual void SetNodeStateExited(Node node) {
    }

    virtual bool ShouldVisitNode(Node node) = 0;

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

class DirectedConnectByNodeIdsImplementation : public interface::ConnectByNodesInterface<int32_t>,
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

template <typename Edge, typename = meta::enable_if_t<meta::is_base_of_v<DirectedEdge, Edge>>>
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

        if (meta::is_base_of_v<UndirectedEdge, Edge> and edge.from != edge.to) {
            std::swap(edge.from, edge.to);
            AddDirectedEdge(edge);
        }
    }

    void AddDirectedEdge(Edge edge) {
        edges_table[edge.from].emplace_back(std::move(edge));
    }
};

template <typename Node>
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

class AdjacentByIdImplementation
    : public interface::AdjacentInterface<int32_t, const std::vector<int32_t> &>,
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

template <typename GraphTraversal>
void DepthFirstSearchRecursive(GraphTraversal *graph_traversal,
                               typename GraphTraversal::Node source_node) {

    graph_traversal->OnNodeEnter(source_node);
    graph_traversal->SetNodeStateEntered(source_node);

    for (auto adjacent_node : graph_traversal->NodesAdjacentTo(source_node)) {

        graph_traversal->OnEdgeDiscovery(source_node, adjacent_node);

        if (graph_traversal->ShouldVisitNode(adjacent_node)) {

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

    if (graph_traversal->ShouldVisitNode(source_node)) {
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

        if (graph_traversal->ShouldVisitNode(node)) {
            queue.emplace_back(node);
        }
    }

    while (not queue.empty()) {
        auto node = queue.front();
        queue.pop_front();

        graph_traversal->OnNodeEnter(node);
        graph_traversal->SetNodeStateEntered(node);

        for (auto adjacent_node : graph_traversal->NodesAdjacentTo(node)) {

            graph_traversal->OnEdgeDiscovery(node, adjacent_node);

            if (graph_traversal->ShouldVisitNode(adjacent_node)) {

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

template <typename Container, typename Comparator>
class RangeMinQueryResponder {
public:
    using Value = typename Container::value_type;

    explicit RangeMinQueryResponder(const Container &container, const Comparator &comparator)
        : container_{container}, comparator_{comparator} {
    }

    void Preprocess() {
        if (not rmq_preprocessed_.empty()) {
            return;
        }

        auto log_two = LargestPowerOfTwoNotGreaterThan(container_.size());

        rmq_preprocessed_.resize(log_two + 1);
        rmq_preprocessed_.front() = container_;

        for (size_t interval_size_log_two = 1; interval_size_log_two <= log_two;
             ++interval_size_log_two) {

            auto &rmq_preprocessed_by_interval_size = rmq_preprocessed_[interval_size_log_two];
            auto interval_size = 1 << interval_size_log_two;
            auto n_valid_begin_positions = container_.size() - interval_size + 1;
            rmq_preprocessed_by_interval_size.resize(n_valid_begin_positions);

            for (size_t begin = 0; begin < n_valid_begin_positions; ++begin) {

                auto &left_half_min = MinOnPreprocessedIntervalByEnd(begin + interval_size / 2,
                                                                     interval_size_log_two - 1);
                auto &right_half_min = MinOnPreprocessedIntervalByBegin(begin + interval_size / 2,
                                                                        interval_size_log_two - 1);

                MinOnPreprocessedIntervalByBegin(begin, interval_size_log_two) =
                    std::min(left_half_min, right_half_min, comparator_);
            }
        }
    }

    Value GetRangeMin(size_t begin, size_t end) {
        Preprocess();

        auto log_two = LargestPowerOfTwoNotGreaterThan(end - begin);

        auto min_on_left_overlapping_half = MinOnPreprocessedIntervalByBegin(begin, log_two);
        auto min_on_right_overlapping_half = MinOnPreprocessedIntervalByEnd(end, log_two);

        return std::min(min_on_left_overlapping_half, min_on_right_overlapping_half, comparator_);
    }

private:
    const Container &container_;
    const Comparator &comparator_;
    std::vector<std::vector<Value>> rmq_preprocessed_;

    template <typename Integral>
    static Integral LargestPowerOfTwoNotGreaterThan(Integral value) {
        Integral log_two = 0;
        while (value >>= 1) {
            ++log_two;
        }
        return log_two;
    }

    Value &MinOnPreprocessedIntervalByBegin(size_t begin, size_t interval_size_log_two) {
        return rmq_preprocessed_[interval_size_log_two][begin];
    }

    Value &MinOnPreprocessedIntervalByEnd(size_t end, size_t interval_size_log_two) {
        return rmq_preprocessed_[interval_size_log_two][end - (1 << interval_size_log_two)];
    }
};

}  // namespace implementation

class UndirectedGraph : public interface::ResizeInterface,
                        public interface::ConnectByEdgesInterface<io::UndirectedEdge>,
                        public interface::ConnectByNodesInterface<int32_t>,
                        public interface::AdjacentInterface<int32_t, const std::vector<int32_t> &> {
public:
    using Node = int32_t;
    using Edge = io::UndirectedEdge;
    using NodeIterable = const std::vector<int32_t> &;

    std::vector<std::vector<Node>> adjacent_nodes;

    void Resize(size_t size) override {
        if (size > adjacent_nodes.size()) {
            adjacent_nodes.resize(size);
        }
    }

    [[nodiscard]] size_t Size() const override {
        return adjacent_nodes.size();
    }

    void Connect(Node from, Node to) override {
        Resize(from + 1);
        Resize(to + 1);
        adjacent_nodes[from].emplace_back(to);
        adjacent_nodes[to].emplace_back(from);
    }

    void AddEdge(Edge edge) override {
        adjacent_nodes[edge.from].emplace_back(edge.to);
        adjacent_nodes[edge.to].emplace_back(edge.from);
    }

    [[nodiscard]] NodeIterable NodesAdjacentTo(Node node) const override {
        return adjacent_nodes[node];
    }
};

template <typename Graph,
          typename =
              meta::enable_if_t<std::is_same_v<typename Graph::Node, int32_t> and
                                meta::is_template_base_of<interface::AdjacentInterface, Graph>>>
class SpanningTreeGraphTraversal : public interface::GraphTraversal<int32_t> {
public:
    std::vector<int32_t> nodes_in_enter_order;
    std::vector<int32_t> node_to_spanning_tree_parent;

    using Node = int32_t;
    using NodeIterable = typename Graph::NodeIterable;

    explicit SpanningTreeGraphTraversal(const Graph &graph) : graph_{graph} {
        nodes_in_enter_order.reserve(graph_.Size());
    }

    void OnTraverseStart() override {
        nodes_in_enter_order.clear();
        node_to_spanning_tree_parent.clear();
        node_to_spanning_tree_parent.resize(graph_.Size(), -1);
    }

    bool ShouldVisitNode(Node node) override {
        return node_to_spanning_tree_parent[node] == -1;
    }

    NodeIterable NodesAdjacentTo(Node node) override {
        return graph_.NodesAdjacentTo(node);
    }

    void OnNodeEnter(Node node) override {
        if (ShouldVisitNode(node)) {
            node_to_spanning_tree_parent[node] = node;
        }
        nodes_in_enter_order.emplace_back(node);
    }

    void OnEdgeTraverse(Node from, Node to) override {
        node_to_spanning_tree_parent[to] = from;
    }

private:
    const Graph &graph_;
};

template <typename GraphIn, typename GraphOut = interface::ConnectByNodesInterface<int32_t>,
          typename = meta::enable_if_t<
              std::is_same_v<typename GraphIn::Node, int32_t> and
              meta::is_template_base_of<interface::AdjacentInterface, GraphIn> and
              meta::is_base_of_v<interface::ConnectByNodesInterface<int32_t>, GraphOut>>>
class SingleCycleInEnterOrderInGraphExcludingDirectedSpanningTreeBfsTraversal
    : public interface::GraphTraversal<int32_t> {
public:
    std::vector<int32_t> node_to_cycle_id_map;

    using Node = int32_t;
    using NodeIterable = typename GraphIn::NodeIterable;

    SingleCycleInEnterOrderInGraphExcludingDirectedSpanningTreeBfsTraversal(
        const GraphIn &graph, const std::vector<int32_t> &node_to_spanning_tree_parent,
        GraphOut *tree_of_cycles = nullptr)
        : graph_{graph},
          node_to_spanning_tree_parent_{node_to_spanning_tree_parent},
          tree_of_cycles_{tree_of_cycles} {
        node_to_cycle_id_map.clear();
        node_to_cycle_id_map.resize(graph_.Size(), -1);
    }

    void OnTraverseStart() override {
        is_new_cycle_ = true;
    }

    bool ShouldVisitNode(Node node) override {
        return IsNodeUnvisited(node) and not IsGoingToTraverseDirectedSpanningTreeEdge(node);
    }

    NodeIterable NodesAdjacentTo(Node node) override {
        return graph_.NodesAdjacentTo(node);
    }

    void OnNodeEnter(Node node) override {
        if (is_new_cycle_) {
            current_cycle_id_ = current_cycle_id_ ? current_cycle_id_.value() + 1 : 0;
            is_new_cycle_ = false;
        }
        node_to_cycle_id_map[node] = current_cycle_id_.value();
        current_entered_node_ = node;
    }

    void OnEdgeDiscovery(Node from, Node to) override {
        if (tree_of_cycles_ and not IsNodeUnvisited(to) and
            node_to_cycle_id_map[to] < current_cycle_id_) {

            tree_of_cycles_->Connect(node_to_cycle_id_map[to], current_cycle_id_.value());
        }
    }

    void OnNodeExit(Node node) override {
        current_entered_node_.reset();
    }

private:
    const GraphIn &graph_;
    const std::vector<int32_t> &node_to_spanning_tree_parent_;
    GraphOut *tree_of_cycles_;
    std::optional<int32_t> current_cycle_id_;
    std::optional<int32_t> current_entered_node_;
    bool is_new_cycle_ = true;

    [[nodiscard]] bool IsNodeUnvisited(Node node) const {
        return node_to_cycle_id_map[node] == -1;
    }

    [[nodiscard]] bool IsGoingToTraverseDirectedSpanningTreeEdge(Node node) const {
        return current_entered_node_ and
               node_to_spanning_tree_parent_[node] == current_entered_node_.value();
    }
};

template <typename GraphIn, typename GraphOut = interface::ConnectByNodesInterface<int32_t>>
std::vector<int32_t> CompressUndirectedGraphWithCyclesToTree(const GraphIn &graph, int32_t root = 0,
                                                             GraphOut *tree = nullptr) {

    SpanningTreeGraphTraversal spanning_tree_traversal{graph};
    implementation::DepthFirstSearch(&spanning_tree_traversal, root);

    SingleCycleInEnterOrderInGraphExcludingDirectedSpanningTreeBfsTraversal single_cycle_traversal{
        graph, spanning_tree_traversal.node_to_spanning_tree_parent, tree};

    for (auto node : spanning_tree_traversal.nodes_in_enter_order) {
        implementation::BreadthFirstSearch(&single_cycle_traversal, {node});
    }

    return single_cycle_traversal.node_to_cycle_id_map;
}

template <typename Tree,
          typename =
              meta::enable_if_t<meta::is_template_base_of<interface::AdjacentInterface, Tree>>>
class EulerBfsGraphTraverse : public interface::GraphTraversal<int32_t> {
public:
    using Node = typename Tree::Node;
    using NodeIterable = typename Tree::NodeIterable;

    std::vector<Node> euler_tour;
    std::vector<int32_t> first_node_occurrence_position_in_euler_tour;

    explicit EulerBfsGraphTraverse(const Tree &tree) : tree_{tree} {
    }

    void OnTraverseStart() override {
        euler_tour.clear();
        euler_tour.reserve(tree_.Size() * 2);
        first_node_occurrence_position_in_euler_tour.clear();
        first_node_occurrence_position_in_euler_tour.resize(tree_.Size(), -1);
    }

    bool ShouldVisitNode(Node node) override {
        return first_node_occurrence_position_in_euler_tour[node] == -1;
    }

    NodeIterable NodesAdjacentTo(Node node) override {
        return tree_.NodesAdjacentTo(node);
    }

    void OnNodeEnter(Node node) override {
        first_node_occurrence_position_in_euler_tour[node] = euler_tour.size();
        euler_tour.emplace_back(node);
    }

    void OnEdgeBacktrack(Node from, Node to) override {
        euler_tour.emplace_back(to);
    }

protected:
    const Tree &tree_;
};

template <typename Graph,
          typename =
              meta::enable_if_t<meta::is_template_base_of<interface::AdjacentInterface, Graph>>>
class ComputeDistancesBfsTraverse : public interface::GraphTraversal<typename Graph::Node> {
public:
    using Node = typename Graph::Node;
    using NodeIterable = typename Graph::NodeIterable;

    std::vector<std::optional<int32_t>> distances_from_root;

    explicit ComputeDistancesBfsTraverse(const Graph &graph) : graph_{graph} {
    }

    void OnTraverseStart() override {
        distances_from_root.clear();
        distances_from_root.resize(graph_.Size());
    }

    bool ShouldVisitNode(Node node) override {
        return not distances_from_root[node];
    }

    NodeIterable NodesAdjacentTo(Node node) override {
        return graph_.NodesAdjacentTo(node);
    }

    void OnNodeEnter(Node node) override {
        if (ShouldVisitNode(node)) {
            distances_from_root[node] = 0;
        }
    }

    void OnEdgeTraverse(Node from, Node to) override {
        distances_from_root[to] = distances_from_root[from].value() + 1;
    }

protected:
    const Graph &graph_;
};

struct IsomorphicTask {
    io::Input input;
    UndirectedGraph tree;
};

IsomorphicTask CompressGraphCyclesAndTransformInput(io::Input input) {
    UndirectedGraph graph;
    graph.Resize(input.n_nodes);

    for (auto &edge : input.edges) {
        graph.AddEdge(edge);
    }

    UndirectedGraph tree;
    tree.adjacent_nodes.reserve(graph.Size());
    tree.Resize(1);
    auto node_to_cycle_id = CompressUndirectedGraphWithCyclesToTree(graph, input.root_id, &tree);

    input.root_id = node_to_cycle_id[input.root_id];

    for (auto &request : input.requests) {
        request.first = node_to_cycle_id[request.first];
        request.second = node_to_cycle_id[request.second];
    }

    return {input, tree};
}

io::Output SolveIsomorphicTask(const io::Input &input, const UndirectedGraph &tree) {

    EulerBfsGraphTraverse euler_traverse{tree};
    implementation::DepthFirstSearch(&euler_traverse, input.root_id);

    ComputeDistancesBfsTraverse distances_traverse(tree);
    implementation::BreadthFirstSearch(&distances_traverse, {input.root_id});

    auto node_depths = distances_traverse.distances_from_root;

    auto tree_nodes_comparator = [&node_depths](int32_t left, int32_t right) {
        return node_depths[left].value() < node_depths[right].value();
    };

    implementation::RangeMinQueryResponder rmq{euler_traverse.euler_tour, tree_nodes_comparator};

    std::vector<io::FindLeastCommonAncestorDepth::Response> responses;
    responses.reserve(input.requests.size());

    for (auto &request : input.requests) {

        auto euler_tour_begin =
            euler_traverse.first_node_occurrence_position_in_euler_tour[request.first];
        auto euler_tour_end_minus_one =
            euler_traverse.first_node_occurrence_position_in_euler_tour[request.second];

        if (euler_tour_end_minus_one < euler_tour_begin) {
            std::swap(euler_tour_end_minus_one, euler_tour_begin);
        }

        auto lca = rmq.GetRangeMin(euler_tour_begin, euler_tour_end_minus_one + 1);

        responses.emplace_back(node_depths[lca].value());
    }

    return io::Output{responses};
}

io::Output Solve(const io::Input &input) {
    auto [isomorphic_input, tree] = CompressGraphCyclesAndTransformInput(input);
    return SolveIsomorphicTask(isomorphic_input, tree);
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

template <typename Iterator>
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

struct UndirectedEdge {
    int32_t from = 0;
    int32_t to = 0;

    UndirectedEdge(int32_t from, int32_t to) : from{from}, to{to} {
        if (from > to) {
            std::swap(from, to);
        }
    }
};

std::vector<UndirectedEdge> BuildTreeFromPruferCode(const std::vector<int32_t> &prufer_code) {
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

    return BuildTreeFromPruferCode(prufer_code);
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

TestIo GenerateRandomTestIo(int32_t test_case_id) {

    int32_t n_nodes = 2 + test_case_id;
    int32_t n_edges = 1 + test_case_id + (test_case_id * 2 - test_case_id) / 2;
    int32_t n_requests = 2 + test_case_id;

    io::Input input;

    input.n_nodes = n_nodes;

    std::uniform_int_distribution nodes_distribution{0, n_nodes - 1};
    input.root_id = nodes_distribution(*rng::GetEngine());

    auto edges = GenerateRandomUndirectedConnectedGraphEdges(n_nodes, n_edges);
    for (auto edge : edges) {
        input.edges.emplace_back(edge.from, edge.to);
    }

    for (int32_t i = 0; i < n_requests; ++i) {
        input.requests.emplace_back(nodes_distribution(*rng::GetEngine()),
                                    nodes_distribution(*rng::GetEngine()));
    }

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
        "6 6\n"
        "1\n"
        "1 2\n"
        "2 3\n"
        "3 4\n"
        "4 2\n"
        "4 5\n"
        "3 6\n"
        "2\n"
        "5 6\n"
        "6 6\n",
        "1\n"
        "2");

    timed_check.Check(
        "6 6\n"
        "2\n"
        "1 2\n"
        "2 3\n"
        "3 4\n"
        "4 2\n"
        "4 5\n"
        "3 6\n"
        "2\n"
        "5 6\n"
        "6 6\n",
        "0\n"
        "1");

    std::cerr << timed_check << "Basic tests OK\n";

    int32_t n_random_test_cases = 100;

    try {

        for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {
            timed_check.Check(GenerateRandomTestIo(test_case_id));
        }

        std::cerr << timed_check << "Random tests OK\n";
    } catch (const NotImplementedError &e) {
    }

    int32_t n_stress_test_cases = 1;

    try {
        for (int32_t test_case_id = 0; test_case_id < n_stress_test_cases; ++test_case_id) {
            timed_check.Check(GenerateStressTestIo(test_case_id));
        }

        std::cerr << timed_check << "Stress tests tests OK\n";
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
