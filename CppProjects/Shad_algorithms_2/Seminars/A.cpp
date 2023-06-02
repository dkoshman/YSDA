#include <algorithm>
#include <array>
#include <bitset>
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

struct BooleanVariable {
    int32_t id = 0;
    bool is_negated = false;

    BooleanVariable() = default;

    explicit BooleanVariable(int32_t id, bool is_negated = false) : id{id}, is_negated{is_negated} {
    }

    [[nodiscard]] bool Bool(bool value) const {
        return value == not is_negated;
    }

    BooleanVariable operator not() const {
        return BooleanVariable{id, not is_negated};
    }
};

struct Disjunction {
    BooleanVariable first;
    BooleanVariable second;

    Disjunction(int32_t first_encoded_value, int32_t second_encoded_value)
        : first{std::abs(first_encoded_value) - 1, first_encoded_value < 0},
          second{std::abs(second_encoded_value) - 1, second_encoded_value < 0} {
    }

    [[nodiscard]] bool Bool(bool first_value, bool second_value) const {
        return first.Bool(first_value) or second.Bool(second_value);
    }
};

class Input {
public:
    int32_t n_boolean_variables = 0;
    std::vector<Disjunction> disjunctions;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t n_disjunctions = 0;
        in >> n_boolean_variables >> n_disjunctions;
        disjunctions.reserve(n_disjunctions);

        for (int32_t i = 0; i < n_disjunctions; i++) {
            int32_t first = 0;
            int32_t second = 0;
            in >> first >> second;
            disjunctions.emplace_back(first, second);
        }
    }
};

class Output {
public:
    bool is_two_satisfiable_with_first_bool_var_set_to_true = false;

    Output() = default;

    explicit Output(bool is_satisfiable)
        : is_two_satisfiable_with_first_bool_var_set_to_true{is_satisfiable} {
    }

    explicit Output(const std::string &string) {
        is_two_satisfiable_with_first_bool_var_set_to_true = string == "yes";
    }

    std::ostream &Write(std::ostream &out) const {
        out << (is_two_satisfiable_with_first_bool_var_set_to_true ? "yes" : "no");
        return out;
    }

    bool operator!=(const Output &other) const {
        return is_two_satisfiable_with_first_bool_var_set_to_true !=
               other.is_two_satisfiable_with_first_bool_var_set_to_true;
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
                                               public interface::SizeInterface {
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
      public interface::SizeInterface {
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

        for (auto adjacent_node : graph_traversal->NodesAdjacentTo(node)) {

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

class ImplicationGraph
    : public interface::ConnectByEdgesInterface<io::Disjunction>,
      public interface::AdjacentInterface<int32_t, const std::vector<int32_t> &> {
public:
    using Node = int32_t;
    using NodeIterable = const std::vector<int32_t> &;
    using Edge = io::Disjunction;

    int32_t n_bool_vars;
    std::vector<std::vector<Node>> adjacency_table;

    explicit ImplicationGraph(int32_t n_bool_vars)
        : n_bool_vars{n_bool_vars}, adjacency_table(Size()) {
    }

    [[nodiscard]] int32_t Size() const {
        return n_bool_vars * 2;
    }

    [[nodiscard]] static int32_t GetNodeId(io::BooleanVariable var) {
        return var.id * 2 + var.is_negated;
    }

    void AddImplication(io::BooleanVariable from, io::BooleanVariable to) {
        adjacency_table[GetNodeId(from)].emplace_back(GetNodeId(to));
    }

    void AddDisjunction(io::BooleanVariable first, io::BooleanVariable second) {
        AddImplication(not first, second);
        AddImplication(not second, first);
    }

    void AddEdge(Edge edge) override {
        AddDisjunction(edge.first, edge.second);
    }

    [[nodiscard]] NodeIterable NodesAdjacentTo(Node node) const override {
        return adjacency_table[node];
    }

    [[nodiscard]] bool IsSatisfiable() const {
        auto sccs = FindStronglyConnectedComponents(*this);
        for (int32_t i = 0; i < n_bool_vars; ++i) {
            io::BooleanVariable var{i};
            if (sccs[GetNodeId(var)] == sccs[GetNodeId(not var)]) {
                return false;
            }
        }
        return true;
    }
};

io::Output Solve(const io::Input &input) {

    ImplicationGraph graph{input.n_boolean_variables};

    io::BooleanVariable first{0};
    graph.AddImplication(not first, first);

    for (auto disjunction : input.disjunctions) {
        graph.AddEdge(disjunction);
    }

    return io::Output{graph.IsSatisfiable()};
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

template <size_t N>
bool IsSatisfying(const std::vector<io::Disjunction> &disjunctions,
                  const std::bitset<N> &bool_vars) {
    auto does_satisfy_one_disjunction = [&bool_vars](const io::Disjunction &disjunction) -> bool {
        return disjunction.Bool(bool_vars[disjunction.first.id], bool_vars[disjunction.second.id]);
    };
    return std::all_of(disjunctions.begin(), disjunctions.end(), does_satisfy_one_disjunction);
}

io::Output BruteForceSolve(const io::Input &input) {
    if (input.n_boolean_variables > 20) {
        throw NotImplementedError{};
    }

    std::vector<bool> bool_vars(input.n_boolean_variables);
    bool_vars[0] = true;

    for (size_t i = 0; i < static_cast<size_t>(2) << input.n_boolean_variables; ++i) {
        std::bitset<20> bits{i};
        if (bits[0] and IsSatisfying(input.disjunctions, bits)) {
            return io::Output{true};
        }
    }

    return io::Output{false};
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t n_boolean_variables = 2 + test_case_id;
    int32_t n_disjunctions = 1 + test_case_id;

    std::uniform_int_distribution<int32_t> var_distribution{-n_boolean_variables,
                                                            n_boolean_variables};

    io::Input input;
    input.n_boolean_variables = n_boolean_variables;
    while (static_cast<int32_t>(input.disjunctions.size()) < n_disjunctions) {
        auto first = var_distribution(*rng::GetEngine());
        auto second = var_distribution(*rng::GetEngine());
        if (std::abs(first) == std::abs(second) or first == 0 or second == 0) {
            continue;
        }
        input.disjunctions.emplace_back(first, second);
    }

    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    return GenerateRandomTestIo(1000);
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
        "4 3\n"
        "1 2\n"
        "-2 -3\n"
        "2 4",
        "yes");

    timed_check.Check(
        "2 4\n"
        "1 2\n"
        "1 -2\n"
        "-1 2\n"
        "-1 -2",
        "no");

    std::cerr << "Basic tests OK:\n" << timed_check;

    int32_t n_random_test_cases = 10;

    try {

        for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {
            timed_check.Check(GenerateRandomTestIo(test_case_id));
        }

        std::cerr << "Random tests OK:\n" << timed_check;
    } catch (const NotImplementedError &e) {
    }

    int32_t n_stress_test_cases = 1;

    try {
        for (int32_t test_case_id = 0; test_case_id < n_stress_test_cases; ++test_case_id) {
            timed_check.Check(GenerateStressTestIo(test_case_id));
        }

        std::cerr << "Stress tests tests OK:\n" << timed_check;
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
