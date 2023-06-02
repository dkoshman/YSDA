
#ifndef STRING_H
#define STRING_H

#include <array>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <unordered_set>
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

namespace utils {

template <class Key, class Value>
std::vector<Value> GetMapValues(const std::map<Key, Value> &map) {
    std::vector<Value> values;
    values.reserve(map.size());
    for (auto &pair : map) {
        values.emplace_back(pair.target_string_);
    }
    return values;
}

bool AreStringsUnique(const std::vector<std::string> &strings) {
    std::unordered_set<std::string> set{strings.begin(), strings.end()};
    return set.size() == strings.size();
}

size_t TotalStringsSize(const std::vector<std::string> &strings) {
    size_t size = 0;
    for (auto &string : strings) {
        size += string.size();
    }
    return size;
}

std::vector<int32_t> ArgSortByArgs(
    int32_t size, std::function<bool(int32_t, int32_t)> arg_compare = std::less{}) {
    std::vector<int32_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), std::move(arg_compare));
    return indices;
}

template <typename Iterator>
std::vector<int32_t> ArgSortByValue(
    Iterator begin, Iterator end,
    std::function<bool(typename Iterator::value_type, typename Iterator::value_type)> comparator =
        std::less{}) {
    auto arg_comparator = [&begin, &comparator](int32_t left, int32_t right) -> bool {
        return comparator(*(begin + left), *(begin + right));
    };
    return ArgSortByArgs(end - begin, arg_comparator);
}

template <typename T>
std::vector<int32_t> SortedArgsToRanks(const std::vector<T> &sorted_args) {
    std::vector<int32_t> ranks(sorted_args.size());
    for (int32_t rank = 0; rank < static_cast<int32_t>(sorted_args.size()); ++rank) {
        ranks[sorted_args[rank]] = rank;
    }
    return ranks;
}

template <typename T>
std::vector<int32_t> RanksToSortedArgs(const std::vector<T> &ranks) {
    std::vector<int32_t> sorted_args(ranks.size());
    for (int32_t arg = 0; arg < static_cast<int32_t>(ranks.size()); ++arg) {
        sorted_args[ranks[arg]] = arg;
    }
    return sorted_args;
}

template <typename T, typename I>
std::vector<T> Take(const std::vector<T> &values, const std::vector<I> &indices) {
    std::vector<T> slice;
    slice.reserve(values.size());
    for (auto i : indices) {
        slice.emplace_back(values[i]);
    }
    return slice;
}

template <typename T, typename Comparator>
std::vector<T> MergeSorted(const std::vector<T> &left, const std::vector<T> &right,
                           Comparator left_right_comparator = std::less{}) {

    std::vector<T> sorted;
    sorted.reserve(left.size() + right.size());
    auto left_iter = left.begin();
    auto right_iter = right.begin();
    while (left_iter < left.end() and right_iter < right.end()) {
        if (left_right_comparator(*left_iter, *right_iter)) {
            sorted.emplace_back(*left_iter);
            ++left_iter;
        } else {
            sorted.emplace_back(*right_iter);
            ++right_iter;
        }
    }
    sorted.insert(sorted.end(), left_iter, left.end());
    sorted.insert(sorted.end(), right_iter, right.end());
    return sorted;
}

template <typename T>
std::vector<int32_t> SoftRank(const std::vector<T> &values,
                              const std::vector<int32_t> &sorted_indices) {
    if (values.empty()) {
        return {};
    }
    std::vector<int32_t> soft_ranks(values.size());
    auto prev = values[sorted_indices.front()];
    int32_t soft_rank = 1;
    for (auto i : sorted_indices) {
        if (values[i] == prev) {
            soft_ranks[i] = soft_rank;
        } else {
            prev = values[i];
            soft_ranks[i] = ++soft_rank;
        }
    }
    return soft_ranks;
}

}  // namespace utils

class PrefixFunctionComputer {
public:
    const std::string &string;
    std::vector<int32_t> prefix_function;

    explicit PrefixFunctionComputer(const std::string &string)
        : string{string}, prefix_function(string.size()) {
    }

    std::vector<int32_t> Compute() {

        for (int32_t prefix = 0; prefix < static_cast<int32_t>(string.size()); ++prefix) {

            auto border = FindLargestBorderFollowedBySameLetterAsPrefix(prefix);

            prefix_function[prefix] = border ? 1 + border.value() : 0;
        }

        return prefix_function;
    }

private:
    [[nodiscard]] char GetLetterAfterPrefix(int32_t prefix_size) const {
        return string[prefix_size];
    }

    std::optional<int32_t> FindLargestBorderFollowedBySameLetterAsPrefix(int32_t prefix_size) {
        auto letter_after_prefix = GetLetterAfterPrefix(prefix_size);
        auto border = GetLargestNonDegenerateBorderSizeForPrefix(prefix_size);

        while (border and GetLetterAfterPrefix(border.value()) != letter_after_prefix) {
            border = GetLargestNonDegenerateBorderSizeForPrefix(border.value());
        }

        return border;
    }

    [[nodiscard]] std::optional<int32_t> GetLargestNonDegenerateBorderSizeForPrefix(
        int32_t prefix_size) const {

        return prefix_size >= 1 ? std::optional<int32_t>(prefix_function[prefix_size - 1])
                                : std::nullopt;
    }
};

class ZFunctionComputer {
public:
    const std::string &string;
    std::vector<int32_t> z_function;

    explicit ZFunctionComputer(const std::string &string)
        : string{string}, z_function(string.size()), size_{static_cast<int32_t>(string.size())} {
    }

    std::vector<int32_t> Compute() {
        argmax_i_plus_z_i_ = 1;
        z_function[0] = size_;

        for (int32_t index = 1; index < size_; ++index) {
            z_function[index] = CalculateZFunctionAt(index);
        }

        return z_function;
    }

private:
    int32_t argmax_i_plus_z_i_ = 0;
    int32_t size_ = 0;

    [[nodiscard]] int32_t CalculateZFunctionAt(int32_t index) {
        int32_t index_minus_argmax = index - argmax_i_plus_z_i_;
        auto new_max_z_value = std::max(0, z_function[argmax_i_plus_z_i_] - index_minus_argmax);

        if (z_function[index_minus_argmax] < new_max_z_value) {
            return z_function[index_minus_argmax];
        }

        while (index + new_max_z_value < size_ and
               string[new_max_z_value] == string[index + new_max_z_value]) {
            ++new_max_z_value;
        }
        argmax_i_plus_z_i_ = index;
        return new_max_z_value;
    }
};

std::vector<int32_t> PatternInTextZFunction(const std::string &pattern, const std::string &text) {
    auto z_function = ZFunctionComputer{pattern + '\0' + text}.Compute();
    return {z_function.begin() + static_cast<int32_t>(pattern.size()) + 1, z_function.end()};
}

std::optional<size_t> FindPatternBeginInText(const std::string &pattern, const std::string &text) {
    auto z_function = PatternInTextZFunction(pattern, text);

    for (size_t i = 0; i < z_function.size(); ++i) {
        if (z_function[i] == static_cast<int32_t>(pattern.size())) {
            return i;
        }
    }
    return std::nullopt;
}

int32_t BiggestSuffixOfFirstThatIsPrefixOfSecond(const std::string &first,
                                                 const std::string &second) {
    auto z_function = PatternInTextZFunction(second, first);

    for (size_t z = 0; z < z_function.size(); ++z) {
        if (z + z_function[z] == z_function.size()) {
            return z_function[z];
        }
    }
    return 0;
}

namespace prefix_tree {

struct PrefixTreeNode {
    std::optional<int32_t> string_id_that_terminates_here;
    std::map<char, int32_t> edges;
    int32_t prefix_link = 0;
    int32_t terminal_link = -1;
    std::optional<std::vector<int32_t>> cached_string_occurrences;
    std::map<char, int32_t> cached_prefix_link_transitions;
};

class PrefixTree {
public:
    explicit PrefixTree(const std::vector<std::string> &strings) {
        if (not utils::AreStringsUnique(strings)) {
            throw std::invalid_argument{"Provided strings are not unique."};
        }
        Initialize(strings);
    }

    void ResetTextIterator() {
        text_iter_node_ = 0;
    }

    [[nodiscard]] size_t GetCurrentSizeOfProcessedText() const {
        return size_of_processed_text_;
    }

    void SendNextTextLetter(char letter) {
        text_iter_node_ = FindNextLongestPrefixNodePrecededByLetterOrRoot(letter, text_iter_node_);
        ++size_of_processed_text_;
    }

    std::vector<int32_t> GetOccurrencesAtCurrentPosition() {
        return ComputeOccurrencesAtNode(text_iter_node_);
    }

    const std::vector<int32_t> &GetCachedOccurrencesAtCurrentPosition() {
        return GetCachedOccurrencesAtNode(text_iter_node_);
    }

protected:
    std::vector<PrefixTreeNode> nodes_{1};
    int32_t text_iter_node_ = 0;
    size_t size_of_processed_text_ = 0;

    void Initialize(const std::vector<std::string> &strings) {
        nodes_.reserve(utils::TotalStringsSize(strings) + 1);
        for (int32_t i = 0; i < static_cast<int32_t>(strings.size()); ++i) {
            AddString(strings[i], i);
        }
        ComputePrefixLinks();
        ComputeTerminalLinks();
    }

    [[nodiscard]] static bool IsRoot(int32_t node) {
        return node == 0;
    }

    [[nodiscard]] bool IsTerminal(int32_t node) const {
        return static_cast<bool>(nodes_[node].string_id_that_terminates_here);
    }

    [[nodiscard]] bool IsLeaf(int32_t node) const {
        return nodes_[node].edges.empty();
    }

    void AddString(const std::string &string, int32_t string_id) {
        int32_t node = 0;
        for (auto letter : string) {
            if (not GetNextDirectContinuationNode(letter, node)) {
                AddLetterAtNode(letter, node);
            }
            node = GetNextDirectContinuationNode(letter, node).value();
        }
        nodes_[node].string_id_that_terminates_here = string_id;
    }

    [[nodiscard]] std::optional<int32_t> GetNextDirectContinuationNode(char letter,
                                                                       int32_t node) const {
        auto &edges = nodes_[node].edges;
        auto search = edges.find(letter);
        return search == edges.end() ? std::nullopt : std::optional<int32_t>(search->second);
    }

    void AddLetterAtNode(char letter, int32_t node) {
        auto new_node = static_cast<int32_t>(nodes_.size());
        nodes_[node].edges[letter] = new_node;
        nodes_.emplace_back();
    }

    int32_t FindNextLongestPrefixNodePrecededByLetterOrRoot(char letter, int32_t node) {
        while (not GetNextDirectContinuationNode(letter, node)) {
            if (IsRoot(node)) {
                return 0;
            }
            node = nodes_[node].prefix_link;
        }
        return GetNextDirectContinuationNode(letter, node).value();
    };

    class Traversal : public interface::GraphTraversal<int32_t, std::vector<int32_t>> {
    public:
        using Node = int32_t;
        using NodeIterable = std::vector<int32_t>;

        PrefixTree *tree = nullptr;

        explicit Traversal(PrefixTree *tree) : tree{tree} {
        }

        bool ShouldTraverseEdge(Node from, Node to) override {
            return true;
        }

        NodeIterable NodesAdjacentTo(Node node) override {
            std::vector<int32_t> adjacent_nodes;
            for (auto [letter, adjacent_node] : tree->nodes_[node].edges) {
                adjacent_nodes.emplace_back(adjacent_node);
            }
            return adjacent_nodes;
        }

        void OnEdgeTraverse(Node from, Node to) override {
            if (PrefixTree::IsRoot(from)) {
                return;
            }
            auto letter = GetEdgeLetter(from, to);
            auto from_prefix = tree->nodes_[from].prefix_link;
            tree->nodes_[to].prefix_link =
                tree->FindNextLongestPrefixNodePrecededByLetterOrRoot(letter, from_prefix);
        }

        [[nodiscard]] char GetEdgeLetter(Node from, Node to) const {
            for (auto [letter, node] : tree->nodes_[from].edges) {
                if (node == to) {
                    return letter;
                }
            }
            throw std::invalid_argument{"No such edge"};
        }
    };

    void ComputePrefixLinks() {
        Traversal traversal{this};
        implementation::BreadthFirstSearch(&traversal, {0});
    }

    void ComputeTerminalLinks() {
        for (int32_t node = 0; node < static_cast<int32_t>(nodes_.size()); ++node) {
            nodes_[node].terminal_link = GetTerminalLink(node);
        }
    }

    int32_t GetTerminalLink(int32_t node) {
        if (nodes_[node].terminal_link != -1) {
            return nodes_[node].terminal_link;
        }
        while (not IsRoot(node)) {
            node = nodes_[node].prefix_link;
            if (IsTerminal(node)) {
                return node;
            }
        }
        return 0;
    }

    const std::vector<int32_t> &GetCachedOccurrencesAtNode(int32_t node) {
        auto &cached_occurrences = nodes_[node].cached_string_occurrences;
        if (not cached_occurrences) {
            cached_occurrences = ComputeOccurrencesAtNode(node);
        }
        return cached_occurrences.value();
    }

    std::vector<int32_t> ComputeOccurrencesAtNode(int32_t node) {
        std::vector<int32_t> occurrences;
        while (not IsRoot(node)) {
            if (nodes_[node].cached_string_occurrences) {
                occurrences.insert(occurrences.end(),
                                   nodes_[node].cached_string_occurrences->begin(),
                                   nodes_[node].cached_string_occurrences->end());
                return occurrences;
            }
            if (auto id = nodes_[node].string_id_that_terminates_here) {
                occurrences.emplace_back(id.value());
            }
            node = nodes_[node].terminal_link;
        }
        return occurrences;
    }
};

}  // namespace prefix_tree

namespace suffix_array {

class RecursiveState {
public:
    std::vector<int32_t> padded_string;
    std::vector<int32_t> one_two_triplet_ranks;

    explicit RecursiveState(const std::vector<int32_t> &string) : padded_string{string} {
        padded_string.resize(string.size() + 3);
    }

    enum class CompareResult { Less, Equal, Greater };

    [[nodiscard]] CompareResult SingleLetterCompare(int32_t left, int32_t right) const {
        if (padded_string[left] == padded_string[right]) {
            return CompareResult::Equal;
        } else {
            return padded_string[left] < padded_string[right] ? CompareResult::Less
                                                              : CompareResult::Greater;
        }
    }

    [[nodiscard]] bool AllSuffixBlocksStringCompare(int32_t left, int32_t right) const {
        CompareResult compare_result;
        while ((compare_result = SingleLetterCompare(left, right)) == CompareResult::Equal) {
            ++left;
            ++right;
        }
        return compare_result == CompareResult::Less;
    }

    std::function<bool(int32_t, int32_t)> GetStringComparator() {
        return [this](int32_t left, int32_t right) -> bool {
            return AllSuffixBlocksStringCompare(left, right);
        };
    }

    [[nodiscard]] int32_t ConvertArgToOneTwoRank(int32_t arg) const {
        if (arg % 3 == 0) {
            throw std::invalid_argument{"Not from one or two mod 3 group."};
        }
        auto twos_start = (one_two_triplet_ranks.size() + 1) / 2;
        return arg % 3 == 1 ? one_two_triplet_ranks[arg / 3]
                            : one_two_triplet_ranks[twos_start + arg / 3];
    }

    [[nodiscard]] bool TripletGroupOneTwoCompare(int32_t left, int32_t right) const {
        return ConvertArgToOneTwoRank(left) < ConvertArgToOneTwoRank(right);
    }

    [[nodiscard]] bool TripletGroupZeroCompare(int32_t left, int32_t right) const {
        auto compare_result = SingleLetterCompare(left, right);
        if (compare_result == CompareResult::Equal) {
            return TripletGroupOneTwoCompare(left + 1, right + 1);
        }
        return compare_result == CompareResult::Less;
    }

    [[nodiscard]] bool TripletCompare(int32_t left, int32_t right) const {
        if (left % 3 == 0 and right % 3 == 0) {
            return TripletGroupZeroCompare(left, right);
        } else if (left % 3 != 0 and right % 3 != 0) {
            return TripletGroupOneTwoCompare(left, right);
        } else {
            auto compare_result = SingleLetterCompare(left, right);
            if (compare_result == CompareResult::Equal) {
                return TripletCompare(left + 1, right + 1);
            } else {
                return compare_result == CompareResult::Less;
            }
        }
    }

    std::function<bool(int32_t, int32_t)> GetTripletComparator() {
        return [this](int32_t left, int32_t right) -> bool { return TripletCompare(left, right); };
    }
};

template <typename T>
void CheckThatValueIsInAlphabet(T value, int32_t alphabet_size) {
    if (value < 0 or alphabet_size <= value) {
        throw std::invalid_argument{"Value must be non negative and not more than alphabet size."};
    }
}

class StableArgCountSorter {
public:
    std::vector<int32_t> sorted_indices;

    void Sort(const std::vector<int32_t> &values, int32_t alphabet_size) {
        for (auto value : values) {
            CheckThatValueIsInAlphabet(value, alphabet_size);
        }
        alphabet_counts_cum_sum_.resize(alphabet_size);
        std::fill(alphabet_counts_cum_sum_.begin(), alphabet_counts_cum_sum_.end(), 0);
        ComputeCumulativeAlphabetCounts(values);

        sorted_indices.resize(values.size());
        for (int32_t index = 0; index < static_cast<int32_t>(values.size()); ++index) {
            auto &value_sorted_position = alphabet_counts_cum_sum_[values[index]];
            sorted_indices[value_sorted_position] = index;
            ++value_sorted_position;
        }
    }

private:
    std::vector<int32_t> alphabet_counts_cum_sum_;

    void ComputeCumulativeAlphabetCounts(const std::vector<int32_t> &values) {
        for (auto i : values) {
            ++alphabet_counts_cum_sum_[i];
        }
        int32_t sum = 0;
        for (auto &i : alphabet_counts_cum_sum_) {
            auto count = i;
            i = sum;
            sum += count;
        }
    }
};

std::vector<int32_t> StableArgCountSort(const std::vector<int32_t> &values, int32_t alphabet_size) {
    StableArgCountSorter sorter;
    sorter.Sort(values, alphabet_size);
    return sorter.sorted_indices;
}

class StableArgRadixSorter {
public:
    std::vector<int32_t> sorted_indices;

    template <typename String>
    void SortEqualLengthStrings(const std::vector<String> &strings, int32_t alphabet_size) {
        radixes_.reserve(strings.size());
        sorted_indices.resize(strings.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);

        if (not strings.empty()) {
            for (int32_t nth = strings.front().size() - 1; nth >= 0; --nth) {
                BuildNthRadixes(strings, nth, alphabet_size);
                arg_count_sorter_.Sort(radixes_, alphabet_size);
                sorted_indices = utils::Take(sorted_indices, arg_count_sorter_.sorted_indices);
            }
        }
    }

private:
    std::vector<int32_t> radixes_;
    StableArgCountSorter arg_count_sorter_;

    template <typename String>
    void BuildNthRadixes(const std::vector<String> &strings, int32_t nth, int32_t alphabet_size) {
        radixes_.clear();
        for (auto index : sorted_indices) {
            auto radix = strings[index][nth];
            CheckThatValueIsInAlphabet(radix, alphabet_size);
            radixes_.emplace_back(radix);
        }
    }
};

template <typename String>
std::vector<int32_t> StableArgRadixSortEqualLengthStrings(const std::vector<String> &strings,
                                                          int32_t alphabet_size) {
    StableArgRadixSorter sorter;
    sorter.SortEqualLengthStrings(strings, alphabet_size);
    return sorter.sorted_indices;
}

class SuffixArrayBuilder {
public:
    std::vector<int32_t> Build(const std::string &string) {
        std::vector<int32_t> vector_string;
        vector_string.reserve(string.size());
        for (auto &i : string) {
            vector_string.emplace_back();
        }
        return Build(vector_string);
    }

    std::vector<int32_t> BuildFromLowerLatinLetters(const std::string &string) {
        std::vector<int32_t> vector_string;
        vector_string.reserve(string.size());
        for (auto &i : string) {
            vector_string.emplace_back(i - 'a' + 1);
        }
        return BuildFromPositiveValues(vector_string);
    }

    std::vector<int32_t> Build(std::vector<int32_t> string) {
        auto min = *std::min_element(string.begin(), string.end());
        auto max = 0;
        for (auto &i : string) {
            i -= min;
            max = std::max(max, i);
        }

        auto indices = StableArgCountSort(string, max + 1);
        string = utils::SoftRank(string, indices);
        return BuildFromPositiveValues(string);
    }

    std::vector<int32_t> BuildFromPositiveValues(const std::vector<int32_t> &string) {
        return UnPadEofMarker(BuildRecursivePadded(PadStringWithEofMarker(string)));
    }

private:
    StableArgRadixSorter arg_radix_sorter_;

    std::vector<int32_t> PadStringWithEofMarker(std::vector<int32_t> soft_ranks) {
        soft_ranks.push_back(0);
        return soft_ranks;
    }

    std::vector<int32_t> UnPadEofMarker(std::vector<int32_t> suffix_array) {
        suffix_array.erase(suffix_array.begin());
        return suffix_array;
    }

    std::vector<int32_t> BuildRecursivePadded(const std::vector<int32_t> &string) {
        RecursiveState state{string};

        if (string.size() <= 5) {
            return utils::ArgSortByArgs(static_cast<int32_t>(string.size()),
                                        state.GetStringComparator());
        }

        auto [one_two_indices, zero_indices] = BuildOneTwoAndZeroModThreeIndices(string.size());

        auto one_two_triplets_indices = ArgSortOneTwoTriplets(string, state);
        state.one_two_triplet_ranks = utils::SortedArgsToRanks(one_two_triplets_indices);
        one_two_indices = utils::Take(one_two_indices, one_two_triplets_indices);

        std::sort(zero_indices.begin(), zero_indices.end(), state.GetTripletComparator());

        return utils::MergeSorted(one_two_indices, zero_indices, state.GetTripletComparator());
    }

    static std::vector<std::array<int32_t, 3>> BuildOneTwoTriples(
        const std::vector<int32_t> &padded_string) {
        std::vector<std::array<int32_t, 3>> triples;
        auto unpadded_size = static_cast<int32_t>(padded_string.size() - 3);
        triples.reserve((unpadded_size + 1) * 2 / 3);
        for (int32_t i = 0; i < unpadded_size; ++i) {
            if (i % 3 == 1) {
                triples.push_back({padded_string[i], padded_string[i + 1], padded_string[i + 2]});
            }
        }
        for (int32_t i = 0; i < unpadded_size; ++i) {
            if (i % 3 == 2) {
                triples.push_back({padded_string[i], padded_string[i + 1], padded_string[i + 2]});
            }
        }
        return triples;
    }

    std::vector<int32_t> ArgSortOneTwoTriplets(const std::vector<int32_t> &string,
                                               const RecursiveState &state) {
        auto triples = BuildOneTwoTriples(state.padded_string);
        auto sorted_triples_indices = StableArgRadixSortEqualLengthStrings(
            triples, *std::max_element(string.begin(), string.end()) + 1);
        auto one_two_soft_ranks = utils::SoftRank(triples, sorted_triples_indices);
        return BuildFromPositiveValues(one_two_soft_ranks);
    }

    static std::pair<std::vector<int32_t>, std::vector<int32_t>> BuildOneTwoAndZeroModThreeIndices(
        size_t size) {
        std::vector<int32_t> one_two(size * 2 / 3);
        std::vector<int32_t> zero((size + 2) / 3);
        auto two_start = (one_two.size() + 1) / 2;
        for (int32_t i = 0; i < static_cast<int32_t>(size); ++i) {
            if (i % 3 == 1) {
                one_two[i / 3] = i;
            } else if (i % 3 == 2) {
                one_two[two_start + i / 3] = i;
            } else {
                zero[i / 3] = i;
            }
        }
        return {one_two, zero};
    }
};

std::vector<int32_t> BuildSuffixArray(const std::string &string) {
    return SuffixArrayBuilder{}.Build(string);
}

}  // namespace suffix_array

namespace burrows_wheeler {

std::vector<int32_t> ArgSortCyclicShifts(const std::string &string) {
    auto double_string = string + string;
    auto double_suffix_array = suffix_array::BuildSuffixArray(double_string);

    std::vector<int32_t> suffix_array;
    suffix_array.reserve(string.size());
    for (auto i : double_suffix_array) {
        if (i < static_cast<int32_t>(string.size())) {
            suffix_array.emplace_back(i);
        }
    }
    return suffix_array;
}

char GetPreviousCharInCyclicString(const std::string &string, int32_t index) {
    auto size = static_cast<int32_t>(string.size());
    return string[(index - 1 + size) % size];
}

std::string BurrowsWheelerTransform(const std::string &string) {
    auto cyclic_shifts_indices = ArgSortCyclicShifts(string);
    auto transformed_string = string;
    for (size_t i = 0; i < string.size(); ++i) {
        transformed_string[i] = GetPreviousCharInCyclicString(string, cyclic_shifts_indices[i]);
    }
    return transformed_string;
}

}  // namespace burrows_wheeler

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

std::string GenerateRandomString(int32_t size, char letter_from = 'a', char letter_to = 'z') {
    std::uniform_int_distribution<char> letters_dist{letter_from, letter_to};
    std::string string;
    for (int32_t i = 0; i < size; ++i) {
        string += letters_dist(*rng::GetEngine());
    }
    return string;
}

}  // namespace test

#endif  // STRING_H
