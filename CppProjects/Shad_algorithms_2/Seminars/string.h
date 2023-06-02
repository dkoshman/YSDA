#ifndef STRING_H
#define STRING_H

#include <deque>
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
        values.emplace_back(pair.second);
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

struct Node {
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
    std::vector<Node> nodes_{1};
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

namespace suffix_tree {

class Edge {
public:
    int32_t text_interval_begin = 0;
    int32_t text_interval_end = INT32_MAX;
    std::optional<int32_t> child_node;

    Edge() = default;

    explicit Edge(int32_t begin) : text_interval_begin{begin} {
    }

    Edge(int32_t begin, int32_t end, int32_t target_node)
        : text_interval_begin{begin}, text_interval_end{end}, child_node{target_node} {
    }

    [[nodiscard]] int32_t Length() const {
        return text_interval_end - text_interval_begin;
    }

    [[nodiscard]] bool IsLeaf() const {
        return not child_node;
    }

    Edge SplitChildEdge(int32_t child_edge_length, int32_t new_child_node) {
        auto child_edge = *this;
        child_edge.text_interval_begin += child_edge_length;

        text_interval_end = child_edge.text_interval_begin;
        child_node = new_child_node;

        return child_edge;
    }
};

struct Node {
    std::map<char, Edge> edges;
    std::optional<int32_t> suffix_link;

    Node() : suffix_link{0} {
    }

    Node(char letter, int32_t letter_position) : edges{{letter, Edge{letter_position}}} {
    }
};

struct Location {
    int32_t node;
    int32_t delta = 0;
    std::optional<char> first_edge_letter;

    explicit Location(int32_t node) : node{node} {
    }

    [[nodiscard]] bool IsExplicit() const {
        return delta == 0;
    }

    [[nodiscard]] bool IsRoot() const {
        return node == 0;
    }
};

class SuffixTree {
public:
    explicit SuffixTree(size_t capacity) : location_{0} {
        text_.reserve(capacity);
        nodes_.reserve(capacity);
        nodes_.emplace_back();
    }

    [[nodiscard]] bool Search(std::string word) const {
        std::optional<int32_t> node = 0;

        for (auto word_iter = word.begin(); word_iter != word.end();) {

            auto search = nodes_[node.value()].edges.find(*word_iter);
            if (search == nodes_[node.value()].edges.end()) {
                return false;
            }

            Edge edge = search->second;
            for (auto slice = edge.text_interval_begin;
                 slice < edge.text_interval_end and word_iter < word.end(); ++slice, ++word_iter) {
                if (text_[slice] != *word_iter) {
                    return false;
                }
            }
            node = edge.child_node;
        }
        return true;
    }

    void AppendLetter(char letter) {
        text_ += letter;
        std::optional<int32_t> suffix_link_from;

        while (not LocationHasLastLetter()) {
            Location previous_location = location_;

            AddLastLetterAtLocation();

            if (suffix_link_from) {
                nodes_[suffix_link_from.value()].suffix_link = location_.node;
            }
            suffix_link_from = location_.node;

            TraverseSuffixLink(previous_location);
        }

        if (suffix_link_from) {
            nodes_[suffix_link_from.value()].suffix_link = location_.node;
        }

        DescendByLastLetter();
    }

private:
    std::string text_;
    std::vector<Node> nodes_;
    Location location_;

    Edge GetImplicitEdge() {
        return nodes_[location_.node].edges[location_.first_edge_letter.value()];
    }

    char GetNextImplicitLetter() {
        return text_[GetImplicitEdge().text_interval_begin + location_.delta];
    }

    bool IsFirstImplicitNodeOnTheEdgeALeaf(Edge edge) {
        return edge.text_interval_begin + 1 == static_cast<int32_t>(text_.size());
    }

    bool LocationHasLastLetter() {
        char last_letter = text_.back();
        if (location_.IsExplicit()) {
            return nodes_[location_.node].edges.count(last_letter);
        } else {
            return GetNextImplicitLetter() == last_letter;
        }
    }

    void DescendByLastLetter() {
        if (location_.IsExplicit()) {

            location_.first_edge_letter = text_.back();
            Edge edge_to_descend = GetImplicitEdge();

            if (IsFirstImplicitNodeOnTheEdgeALeaf(edge_to_descend)) {
                return;
            } else if (not edge_to_descend.IsLeaf() and edge_to_descend.Length() == 1) {
                location_.node = edge_to_descend.child_node.value();
            } else {
                location_.delta = 1;
            }
        } else {

            Edge edge_to_descend = GetImplicitEdge();

            if (not edge_to_descend.IsLeaf() and location_.delta + 1 == edge_to_descend.Length()) {
                location_ = Location{edge_to_descend.child_node.value()};
            } else {
                ++location_.delta;
            }
        }
    }

    void AddLastLetterAtLocation() {
        if (location_.IsExplicit()) {
            nodes_[location_.node].edges.emplace(text_.back(), text_.size() - 1);
            return;
        }

        auto new_node = static_cast<int32_t>(nodes_.size());
        auto implicit_edge = GetImplicitEdge();

        nodes_.emplace_back(text_.back(), text_.size() - 1);
        Edge edge_lower_half = implicit_edge.SplitChildEdge(location_.delta, new_node);
        nodes_[location_.node].edges[location_.first_edge_letter.value()] = implicit_edge;
        nodes_[new_node].edges[GetNextImplicitLetter()] = edge_lower_half;

        location_ = Location{new_node};
    }

    void TraverseSuffixLink(Location previous_location) {
        if (location_.node == previous_location.node) {
            location_.node = nodes_[location_.node].suffix_link.value();
            return;
        }

        Edge previous_edge =
            nodes_[previous_location.node].edges[previous_location.first_edge_letter.value()];
        location_ = previous_location;

        if (location_.IsRoot()) {
            ++previous_edge.text_interval_begin;
            location_.first_edge_letter = text_[previous_edge.text_interval_begin];
            --location_.delta;
        } else {
            location_.node = nodes_[location_.node].suffix_link.value();
        }

        Edge implicit_edge = GetImplicitEdge();

        while (not implicit_edge.IsLeaf() and implicit_edge.Length() <= location_.delta) {

            previous_edge.text_interval_begin += implicit_edge.Length();
            location_.delta -= implicit_edge.Length();
            location_.node = implicit_edge.child_node.value();
            location_.first_edge_letter = text_[previous_edge.text_interval_begin];
            implicit_edge = GetImplicitEdge();
        }
    }
};

}  // namespace suffix_tree

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

void TestPrefixTree(int32_t test_case_id) {

    int32_t n_strings = 1 + test_case_id;
    int32_t string_size = 1 + test_case_id;

    std::vector<std::string> strings;
    for (int32_t i = 0; i < n_strings; ++i) {
        strings.emplace_back(GenerateRandomString(string_size, 'a', 'c'));
    }

    std::unordered_set<std::string> set{strings.begin(), strings.end()};
    strings = std::vector<std::string>{set.begin(), set.end()};

    prefix_tree::PrefixTree prefix_tree{strings};

    std::string text;
    for (auto &s : strings) {
        text += s;
    }
    assert(prefix_tree.GetOccurrencesAtCurrentPosition().empty());

    for (auto letter : text) {

        prefix_tree.SendNextTextLetter(letter);
        auto string_occurrences = prefix_tree.GetOccurrencesAtCurrentPosition();
        std::unordered_set<int32_t> set_occurrences{string_occurrences.begin(),
                                                    string_occurrences.end()};
        auto text_prefix = prefix_tree.GetCurrentSizeOfProcessedText();
        std::unordered_set<int32_t> expected_set_occurrences;
        for (int32_t i = 0; i < static_cast<int32_t>(strings.size()); ++i) {
            auto string = strings[i];
            bool isin = true;
            for (size_t k = 0; k < string.size(); ++k) {
                if (text[text_prefix - string.size() + k] != string[k]) {
                    isin = false;
                    break;
                }
            }
            if (isin) {
                expected_set_occurrences.emplace(i);
            }
        }

        if (set_occurrences != expected_set_occurrences) {
            assert(false);
        }
    }
}

}  // namespace test

#endif  // STRING_H
