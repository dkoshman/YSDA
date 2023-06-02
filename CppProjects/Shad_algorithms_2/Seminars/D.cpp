#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace io {

class Input {
public:
    std::vector<std::string> strings;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t n_strings = 0;
        in >> n_strings;
        strings.resize(n_strings);
        for (auto &string : strings) {
            in >> string;
        }
    }
};

class Output {
public:
    std::string smallest_super_string;

    Output() = default;

    explicit Output(std::string string) : smallest_super_string{std::move(string)} {
    }

    std::ostream &Write(std::ostream &out) const {
        out << smallest_super_string;
        return out;
    }

    bool operator!=(const Output &other) const {
        return smallest_super_string != other.smallest_super_string;
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

template <typename Node, typename NodeIterable = std::vector<Node>>
class GraphTraversal : public VirtualBaseClass {
public:
    virtual NodeIterable NodesAdjacentTo(Node node) = 0;

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

template <typename Iterator, typename Comparator>
std::vector<size_t> ArgSort(Iterator begin, Iterator end) {
    Comparator comparator;
    std::vector<size_t> indices(end - begin);
    std::iota(indices.begin(), indices.end(), 0);
    auto arg_comparator = [&begin, &comparator](size_t left, size_t right) -> bool {
        return comparator(*(begin + left), *(begin + right));
    };
    std::sort(indices.begin(), indices.end(), arg_comparator);
    return indices;
}

struct ComparatorLess {
    template <typename T>
    bool operator()(const T &left, const T &right) const {
        return left < right;
    }
};

template <typename Iterator>
std::vector<size_t> ArgSort(Iterator begin, Iterator end) {
    return ArgSort<Iterator, ComparatorLess>(begin, end);
}

}  // namespace utils

namespace string {

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
    auto z_function = string::PatternInTextZFunction(pattern, text);

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

bool CompareStringSize(const std::string &left, const std::string &right) {
    return left.size() < right.size();
}

}  // namespace string

std::string BuildSmallestSuperStringInOrder(const std::vector<std::string> &strings) {
    std::string super_string;

    for (auto &string : strings) {

        if (string::FindPatternBeginInText(string, super_string)) {
            continue;
        }

        auto intersection = string::BiggestSuffixOfFirstThatIsPrefixOfSecond(super_string, string);
        super_string += {string.begin() + intersection, string.end()};
    }
    return super_string;
}

class SmallestSuperStringComputer {
public:
    static const int32_t kMaxNumberOfStrings = 18;

    std::optional<std::string> super_string;

    explicit SmallestSuperStringComputer(const std::vector<std::string> &strings)
        : strings_{DropStringsThatAreSubstringsOfOthers(strings)},
          set_of_seen_super_string_nodes_at_ends_of_strings_{strings_.size()} {

        if (strings_.size() > kMaxNumberOfStrings) {
            throw std::invalid_argument{"Too many strings."};
        }
    }

    static std::vector<std::string> DropStringsThatAreSubstringsOfOthers(
        const std::vector<std::string> &strings) {

        auto strings_by_decreasing_size = strings;
        std::sort(strings_by_decreasing_size.begin(), strings_by_decreasing_size.end(),
                  string::CompareStringSize);
        std::reverse(strings_by_decreasing_size.begin(), strings_by_decreasing_size.end());

        std::vector<std::string> filtered_strings;
        for (auto &string : strings_by_decreasing_size) {
            auto is_substring_of_filtered_string =
                [&string](const std::string &not_shorter_filtered_string) {
                    return string::FindPatternBeginInText(string, not_shorter_filtered_string);
                };
            if (not std::any_of(filtered_strings.begin(), filtered_strings.end(),
                                is_substring_of_filtered_string)) {
                filtered_strings.emplace_back(string);
            }
        }

        return filtered_strings;
    }

    std::string FindSmallestSuperString() {

        ComputeEndOfStringsTransitionsInLexicographicalOrder();

        BfsTraversalThroughSuperStringsInLexicographicalOrder traversal{this};

        std::deque<SuperStringNode> starting_bfs_queue;
        for (auto string_id : utils::ArgSort(strings_.begin(), strings_.end())) {
            starting_bfs_queue.emplace_back(static_cast<int32_t>(string_id));
        }

        try {
            implementation::BreadthFirstSearch(&traversal, starting_bfs_queue);
        } catch (const StopTraverseError &e) {
        }

        return super_string.value();
    }

private:
    class StopTraverseError : public std::exception {};

    struct StringLocation {
        int8_t string_id = 0;
        int8_t position_in_string = 0;

        explicit StringLocation(int32_t string_id) : string_id{static_cast<int8_t>(string_id)} {
        }

        StringLocation(int32_t string_id, int32_t position_in_string)
            : string_id{static_cast<int8_t>(string_id)},
              position_in_string{static_cast<int8_t>(position_in_string)} {
        }

        [[nodiscard]] std::string GetStringSuffixToTraverse(
            const std::vector<std::string> &strings) const {
            auto &string = strings[string_id];
            return {string.begin() + position_in_string, string.end()};
        }

        [[nodiscard]] bool HasReachedEndOfString(size_t string_size) const {
            return position_in_string == static_cast<int8_t>(string_size);
        }
    };

    struct SuperStringNode {
        StringLocation location;
        std::array<int8_t, kMaxNumberOfStrings> seen_strings_order{};
        std::bitset<kMaxNumberOfStrings> seen_strings_flags{};
        int8_t n_strings_seen = 0;

        explicit SuperStringNode(int32_t starting_string_id) : location{starting_string_id} {
        }

        [[nodiscard]] bool HasVisitedString(int32_t string_id) const {
            return seen_strings_flags[string_id];
        }

        void OnStringEnd() {
            seen_strings_order[n_strings_seen] = static_cast<int8_t>(location.string_id);
            seen_strings_flags[location.string_id] = true;
            ++n_strings_seen;
        }
    };

    struct SetOfStringIdAndSetOfStringIds {
        std::vector<std::bitset<1 << kMaxNumberOfStrings>> sets_of_strings_ids_by_string_id;

        explicit SetOfStringIdAndSetOfStringIds(size_t n_strings)
            : sets_of_strings_ids_by_string_id(n_strings) {
        }

        [[nodiscard]] bool IsIn(const SuperStringNode &node) const {
            return sets_of_strings_ids_by_string_id[GetNodeStringId(node)]
                                                   [GetNodeSeenStringsSetId(node)];
        }

        void Insert(const SuperStringNode &node) {
            sets_of_strings_ids_by_string_id[GetNodeStringId(node)][GetNodeSeenStringsSetId(node)] =
                true;
        }

        [[nodiscard]] static int32_t GetNodeStringId(const SuperStringNode &node) {
            return node.location.string_id;
        }

        [[nodiscard]] static size_t GetNodeSeenStringsSetId(const SuperStringNode &node) {
            return node.seen_strings_flags.to_ulong();
        }
    };

    std::vector<std::string> strings_;
    SetOfStringIdAndSetOfStringIds set_of_seen_super_string_nodes_at_ends_of_strings_;
    std::vector<std::vector<StringLocation>> end_of_strings_transitions_in_lexicographical_order_;
    std::vector<SuperStringNode> adjacent_nodes_buffer_;

    void ComputeEndOfStringsTransitionsInLexicographicalOrder() {

        end_of_strings_transitions_in_lexicographical_order_.resize(strings_.size());
        auto n_strings = static_cast<int32_t>(strings_.size());

        for (int32_t from = 0; from < n_strings; ++from) {
            auto &transitions = end_of_strings_transitions_in_lexicographical_order_[from];

            for (int32_t to = 0; to < n_strings; ++to) {
                transitions.emplace_back(to, string::BiggestSuffixOfFirstThatIsPrefixOfSecond(
                                                 strings_[from], strings_[to]));
            }

            auto comparator_lexicographical_order_of_string_location_suffixes =
                [this](const StringLocation &left, const StringLocation &right) {
                    return left.GetStringSuffixToTraverse(this->strings_) <
                           right.GetStringSuffixToTraverse(this->strings_);
                };
            std::sort(transitions.begin(), transitions.end(),
                      comparator_lexicographical_order_of_string_location_suffixes);
        }
    }

    const std::vector<SuperStringNode> &NodesAdjacentTo(SuperStringNode node) {

        adjacent_nodes_buffer_.clear();
        ++node.location.position_in_string;

        if (not node.location.HasReachedEndOfString(strings_[node.location.string_id].size())) {
            adjacent_nodes_buffer_.emplace_back(node);
            return adjacent_nodes_buffer_;
        }

        node.OnStringEnd();
        if (set_of_seen_super_string_nodes_at_ends_of_strings_.IsIn(node)) {
            return adjacent_nodes_buffer_;
        }
        set_of_seen_super_string_nodes_at_ends_of_strings_.Insert(node);

        for (auto &transition :
             end_of_strings_transitions_in_lexicographical_order_[node.location.string_id]) {
            if (not node.HasVisitedString(transition.string_id)) {
                adjacent_nodes_buffer_.emplace_back(node);
                adjacent_nodes_buffer_.back().location = transition;
            }
        }

        if (adjacent_nodes_buffer_.empty()) {
            super_string = BuildSuperStringInOrder(node.seen_strings_order);
            throw StopTraverseError{};
        }

        return adjacent_nodes_buffer_;
    }

    [[nodiscard]] std::string BuildSuperStringInOrder(
        std::array<int8_t, kMaxNumberOfStrings> strings_order) const {
        std::vector<std::string> strings_in_order;
        for (size_t i = 0; i < strings_.size(); ++i) {
            strings_in_order.emplace_back(strings_[strings_order[i]]);
        }
        return BuildSmallestSuperStringInOrder(strings_in_order);
    }

    class BfsTraversalThroughSuperStringsInLexicographicalOrder
        : public interface::GraphTraversal<SuperStringNode, const std::vector<SuperStringNode> &> {
    public:
        using Node = SuperStringNode;
        using NodeIterable = const std::vector<Node> &;

        SmallestSuperStringComputer *computer = nullptr;

        explicit BfsTraversalThroughSuperStringsInLexicographicalOrder(
            SmallestSuperStringComputer *tree)
            : computer{tree} {
        }

        NodeIterable NodesAdjacentTo(Node node) override {
            return computer->NodesAdjacentTo(node);
        }
    };
};

io::Output Solve(const io::Input &input) {
    SmallestSuperStringComputer tree{input.strings};
    return io::Output{tree.FindSmallestSuperString()};
}

namespace test {

class NotImplementedError : public std::logic_error {
public:
    NotImplementedError() : std::logic_error("Function not yet implemented."){};
};

bool CompareStringSizeAndLexicographically(const std::string &left, const std::string &right) {
    if (left.size() != right.size()) {
        return string::CompareStringSize(left, right);
    } else {
        return left < right;
    }
}

struct ComparatorStringSizeAndLexicographically {
    bool operator()(const std::string &left, const std::string &right) const {
        return CompareStringSizeAndLexicographically(left, right);
    }
};

io::Output BruteForceSolve(const io::Input &input) {
    if (input.strings.size() >= 8) {
        throw NotImplementedError{};
    }

    std::unordered_set<std::string> set{input.strings.begin(), input.strings.end()};
    std::vector<std::string> strings_permutation{set.begin(), set.end()};

    std::sort(strings_permutation.begin(), strings_permutation.end());

    std::string smallest_super_string;
    do {
        auto super_string = BuildSmallestSuperStringInOrder(strings_permutation);

        smallest_super_string = smallest_super_string.empty()
                                    ? super_string
                                    : std::min(smallest_super_string, super_string,
                                               ComparatorStringSizeAndLexicographically{});

    } while (std::next_permutation(strings_permutation.begin(), strings_permutation.end()));

    for (const auto &string : input.strings) {
        if (smallest_super_string.find(string) != std::string::npos) {
            assert(string::FindPatternBeginInText(string, smallest_super_string).value() ==
                   smallest_super_string.find(string));
        } else {
            assert(false);
        }
    }

    return io::Output{smallest_super_string};
}

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

struct TestIo {
    io::Input input;
    std::optional<io::Output> optional_expected_output;

    explicit TestIo(io::Input input) : input{std::move(input)} {
    }

    TestIo(io::Input input, io::Output output)
        : input{std::move(input)}, optional_expected_output{std::move(output)} {
    }
};

std::string GenerateRandomString(int32_t size, char letter_from = 'a', char letter_to = 'z') {
    std::uniform_int_distribution<char> letters_dist{letter_from, letter_to};
    std::string string;
    for (int32_t i = 0; i < size; ++i) {
        string += letters_dist(*rng::GetEngine());
    }
    return string;
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t n_strings = std::min(18, 3 + test_case_id / 20);
    auto mean_max_letter = M_E * std::log(1 + std::log(1 + test_case_id));

    std::poisson_distribution<char> max_letter{mean_max_letter};
    io::Input input;
    for (int32_t i = 0; i < n_strings; ++i) {
        //        input.strings.emplace_back(GenerateRandomString(
        //            3, 'a', static_cast<char>('a' + std::min('z',
        //            max_letter(*rng::GetEngine())))));
        input.strings.emplace_back(GenerateRandomString(
            1 + i * 100 / n_strings / n_strings, 'a',
            static_cast<char>('a' + std::min('z', max_letter(*rng::GetEngine())))));
    }

    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    return GenerateRandomTestIo(20 * 18 - 1);
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
                BruteForceSolve(test_io.input);
                std::stringstream ss;
                ss << "\n================================Expected=============================="
                      "==\n"
                   << expected_output
                   << "\n================================Received=============================="
                      "==\n"
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
        "3\n"
        "abc\n"
        "bcedf\n"
        "edfh",
        "abcedfh");

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
