#include <algorithm>
#include <array>
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
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rng {

uint32_t GetSeed() {
    auto random_device = std::random_device{};
    static auto seed = random_device();
    return 2506609939;
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

template <class Dividend, class Divisor>
Divisor NonNegativeMod(Dividend value, Divisor divisor) {
    if (divisor == 0) {
        throw std::invalid_argument("Zero divisor.");
    }
    if (divisor < 0) {
        throw std::invalid_argument("Negative divisor.");
    }
    auto mod = value % divisor;
    if (mod < 0) {
        mod += divisor;
    }
    return static_cast<Divisor>(mod);
}

template <typename Integral>
static Integral LargestPowerOfTwoNotGreaterThan(Integral value) {
    if (value <= 0) {
        throw std::invalid_argument{"Non positive logarithm argument."};
    }
    Integral log_two = 0;
    while (value >>= 1) {
        ++log_two;
    }
    return log_two;
}

template <typename Number>
Number PowerOfTwo(Number value) {
    return value * value;
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

template <typename T, typename I>
std::vector<T> Take(const std::vector<T> &values, const std::vector<I> &indices) {
    std::vector<T> slice;
    slice.reserve(values.size());
    for (auto i : indices) {
        slice.emplace_back(values[i]);
    }
    return slice;
}

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

int32_t inline Max(int32_t first, int32_t second) {
    return first > second ? first : second;
}

}  // namespace utils

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

namespace sort {

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

}  // namespace sort

namespace io {

class Input {
public:
    std::vector<std::string> dictionary;
    std::vector<std::string> approximate_strings_to_look_up;
    const int32_t max_word_length = 20;
    const int32_t max_edit_distance_to_search = 2;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t dictionary_size = 0;
        in >> dictionary_size;
        dictionary.resize(dictionary_size);
        for (auto &string : dictionary) {
            in >> string;
        }
        int32_t n_strings_to_look_up = 0;
        in >> n_strings_to_look_up;
        approximate_strings_to_look_up.resize(n_strings_to_look_up);
        for (auto &string : approximate_strings_to_look_up) {
            in >> string;
        }
    }
};

class Output {
public:
    std::vector<std::optional<std::string>> closest_strings_in_dictionary;

    Output() = default;

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        std::string word;
        while (ss >> word) {
            if (word == "-1") {
                closest_strings_in_dictionary.emplace_back();
            } else {
                closest_strings_in_dictionary.emplace_back(word);
            }
        }
    }

    std::ostream &Write(std::ostream &out) const {
        for (auto string : closest_strings_in_dictionary) {
            out << (string ? string.value() : "-1") << '\n';
        }
        return out;
    }

    bool operator!=(const Output &other) const {
        return closest_strings_in_dictionary != other.closest_strings_in_dictionary;
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

    std::vector<int32_t> FindOccurrencesAtEndOfText(const std::string &text) {
        std::vector<int32_t> occurrences;
        ResetTextIterator();
        for (auto c : text) {
            SendNextTextLetter(c);
        }
        return GetOccurrencesAtCurrentPosition();
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

namespace bk_tree {

template <typename T>
struct BkNode {
public:
    T value;
    std::unordered_map<int32_t, int32_t> edges;

    explicit BkNode(const T &value) : value{value} {
    }
};

struct SearchCandidate {
    int32_t node = 0;
    int32_t distance = 0;

    SearchCandidate(int32_t closest_node, int32_t distance)
        : node{closest_node}, distance{distance} {
    }
};

template <typename T>
struct SearchResponse {
    T closest_value;
    int32_t distance = 0;
};

struct TryEmplaceResponse {
    int32_t node = 0;
    bool did_emplace = false;
};

template <typename T, typename M = std::function<int32_t(const T &, const T &)>>
class BkTree {
public:
    M discrete_metric;

    explicit BkTree(M discrete_metric) : discrete_metric{std::move(discrete_metric)} {
    }

    void Insert(const T &value) {
        if (nodes_.empty()) {
            nodes_.emplace_back(value);
        } else {
            for (TryEmplaceResponse response; not response.did_emplace;
                 response = TryEmplace(response.node, value)) {
            }
        }
    }

    SearchResponse<T> SearchClosest(const T &value) {
        if (nodes_.empty()) {
            throw std::runtime_error("Tree empty.");
        }
        auto closest_candidate = BuildSearchCandidate(0, value);
        search_candidates_.clear();
        search_candidates_.emplace_back(closest_candidate);

        while (not search_candidates_.empty()) {
            auto candidate = search_candidates_.back();
            search_candidates_.pop_back();

            if (candidate.distance < closest_candidate.distance) {
                closest_candidate = candidate;
            }

            for (auto &[distance, node] : nodes_[candidate.node].edges) {
                if (abs(candidate.distance - distance) < closest_candidate.distance) {
                    search_candidates_.emplace_back(BuildSearchCandidate(node, value));
                }
            }
        }
        return {nodes_[closest_candidate.node].value, closest_candidate.distance};
    }

    std::optional<T> SearchClosestValueAtDistanceInterval(const T &value, int32_t not_closer_than,
                                                          int32_t not_further_than) {
        std::optional<SearchCandidate> closest_candidate;
        search_candidates_.clear();
        search_candidates_.emplace_back(BuildSearchCandidate(0, value));
        if (not_closer_than <= search_candidates_.back().distance and
            search_candidates_.back().distance <= not_further_than) {
            closest_candidate = search_candidates_.back();
            not_further_than = search_candidates_.back().distance - 1;
            if (not_closer_than > not_further_than) {
                return nodes_[closest_candidate->node].value;
            }
        }

        while (not search_candidates_.empty()) {
            auto candidate = search_candidates_.back();
            search_candidates_.pop_back();

            for (auto &[distance, node] : nodes_[candidate.node].edges) {
                if (abs(candidate.distance - distance) <= not_further_than) {
                    auto new_candidate = BuildSearchCandidate(node, value);
                    search_candidates_.emplace_back(new_candidate);
                    if (not_closer_than <= new_candidate.distance and
                        new_candidate.distance <= not_further_than) {
                        closest_candidate = new_candidate;
                        not_further_than = new_candidate.distance - 1;
                        if (not_closer_than > not_further_than) {
                            return nodes_[new_candidate.node].value;
                        }
                    }
                }
            }
        }

        if (closest_candidate) {
            return nodes_[closest_candidate->node].value;
        } else {
            return std::nullopt;
        }
    }

private:
    std::vector<BkNode<T>> nodes_;
    std::vector<SearchCandidate> search_candidates_;
    std::vector<int32_t> distance_cache_;

    TryEmplaceResponse TryEmplace(int32_t root, const T &value) {
        auto distance = discrete_metric(nodes_[root].value, value);
        auto [map_iterator, did_emplace] = nodes_[root].edges.try_emplace(distance, nodes_.size());
        if (did_emplace) {
            nodes_.emplace_back(value);
        }
        return {map_iterator->second, did_emplace};
    }

    [[nodiscard]] inline SearchCandidate BuildSearchCandidate(int32_t node, const T &value) {
        auto &distance = distance_cache_[node];
        if (distance == -1) {
            distance = discrete_metric(nodes_[node].value, value);
        }
        return {node, distance};
    }
};

}  // namespace bk_tree

struct LevenshteinMatrixDiagonalNode {
    int32_t id = 0;
    int32_t position = 0;

    LevenshteinMatrixDiagonalNode(int32_t id, int32_t position) : id{id}, position{position} {
    }
};

class EditDistanceComputer {
public:
    std::optional<int32_t> Compute(const std::string &source_string,
                                   const std::string &target_string, int32_t max_edits) {
        edits_made_ = 0;
        max_edits_ = max_edits;
        source_size_ = static_cast<int32_t>(source_string.size());
        target_size_ = static_cast<int32_t>(target_string.size());
        source_string_ = &source_string;
        target_string_ = &target_string;

        diagonal_candidates_.clear();
        diagonal_candidates_.emplace_back(0, 0);

        if (TraverseEqualCharsUntilEnd(diagonal_candidates_.front())) {
            return 0;
        }

        for (; edits_made_ < max_edits; ++edits_made_) {
            if (UpdateCandidates()) {
                return edits_made_ + 1;
            }
        }

        return std::nullopt;
    }

private:
    int32_t edits_made_ = 0;
    int32_t max_edits_ = 0;
    int32_t source_size_ = 0;
    int32_t target_size_ = 0;
    const std::string *source_string_ = nullptr;
    const std::string *target_string_ = nullptr;
    std::deque<LevenshteinMatrixDiagonalNode> diagonal_candidates_;

    [[nodiscard]] inline bool IsDiagonalCandidateValid(int32_t id) const {
        return abs(source_size_ - target_size_ - id) <= max_edits_ - edits_made_;
    }

    void AddAndRemoveCandidates() {
        if (not diagonal_candidates_.empty()) {
            auto front_id = diagonal_candidates_.front().id;
            if (not IsDiagonalCandidateValid(front_id)) {
                diagonal_candidates_.pop_front();
            } else if (IsDiagonalCandidateValid(front_id - 1)) {
                diagonal_candidates_.emplace_front(front_id - 1, -1);
            }
        }

        if (not diagonal_candidates_.empty()) {
            auto back_id = diagonal_candidates_.back().id;
            if (not IsDiagonalCandidateValid(back_id)) {
                diagonal_candidates_.pop_back();
            } else if (IsDiagonalCandidateValid(back_id + 1)) {
                diagonal_candidates_.emplace_back(back_id + 1, -1);
            }
        }
    }

    [[nodiscard]] inline static int32_t GetInsertPositionOffset(
        const LevenshteinMatrixDiagonalNode &diagonal) {
        return diagonal.id > 0;
    }

    [[nodiscard]] inline static int32_t GetDeletePositionOffset(
        const LevenshteinMatrixDiagonalNode &diagonal) {
        return diagonal.id < 0;
    }

    bool UpdateCandidates() {
        AddAndRemoveCandidates();

        if (diagonal_candidates_.empty()) {
            return false;
        }

        auto original_position = diagonal_candidates_.front().position;
        ++diagonal_candidates_.front().position;
        for (auto it = diagonal_candidates_.begin(), next_it = it + 1;
             next_it != diagonal_candidates_.end(); ++it, ++next_it) {
            auto &diagonal = *it;
            auto &next_diagonal = *next_it;

            diagonal.position = utils::Max(
                diagonal.position, next_diagonal.position + GetInsertPositionOffset(next_diagonal));
            if (TraverseEqualCharsUntilEnd(diagonal)) {
                return true;
            }

            auto next_diagonal_delete_position =
                original_position + GetDeletePositionOffset(diagonal);
            original_position = next_diagonal.position;
            next_diagonal.position =
                utils::Max(next_diagonal.position + 1, next_diagonal_delete_position);
        }

        return TraverseEqualCharsUntilEnd(diagonal_candidates_.back());
    }

    [[nodiscard]] static inline int32_t GetSourceStringPosition(
        const LevenshteinMatrixDiagonalNode &diagonal) {
        return diagonal.position + utils::Max(diagonal.id, 0);
    }

    [[nodiscard]] static inline int32_t GetTargetStringPosition(
        const LevenshteinMatrixDiagonalNode &diagonal) {
        return diagonal.position + utils::Max(-diagonal.id, 0);
    }

    bool TraverseEqualCharsUntilEnd(LevenshteinMatrixDiagonalNode &diagonal) {
        auto source_position = GetSourceStringPosition(diagonal);
        auto target_position = GetTargetStringPosition(diagonal);
        auto &source_string = *source_string_;
        auto &target_string = *target_string_;
        while (source_position < source_size_ and target_position < target_size_ and
               source_string[source_position] == target_string[target_position]) {
            ++source_position;
            ++target_position;
            ++diagonal.position;
        }
        return source_position == source_size_ and target_position == target_size_;
    }
};

std::optional<int32_t> FindEditDistanceUpTo(const std::string &source_string,
                                            const std::string &target_string, int32_t max_edits) {
    return EditDistanceComputer{}.Compute(source_string, target_string, max_edits);
}

io::Output Solve(io::Input input) {
    std::unordered_set<std::string> unique_dictionary{input.dictionary.begin(),
                                                      input.dictionary.end()};
    input.dictionary = {unique_dictionary.begin(), unique_dictionary.end()};
    prefix_tree::PrefixTree prefix_tree{input.dictionary};
    EditDistanceComputer computer;
    io::Output output;
    output.closest_strings_in_dictionary.reserve(input.approximate_strings_to_look_up.size());
    for (auto &lookup : input.approximate_strings_to_look_up) {
        output.closest_strings_in_dictionary.emplace_back();

        for (auto exact_occurrence : prefix_tree.FindOccurrencesAtEndOfText(lookup)) {
            auto &occurrence = input.dictionary[exact_occurrence];
            if (lookup.size() == occurrence.size()) {
                output.closest_strings_in_dictionary.back() = occurrence;
                break;
            }
        }

        if (output.closest_strings_in_dictionary.back()) {
            continue;
        }

        std::optional<int32_t> min_distance;
        for (auto &word : input.dictionary) {
            auto distance = computer.Compute(word, lookup, input.max_edit_distance_to_search);
            if (distance and (not min_distance or distance < min_distance)) {
                output.closest_strings_in_dictionary.back() = word;
                if (distance == 1) {
                    break;
                }
                min_distance = distance;
            }
        }
    }
    return output;
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
    if (input.approximate_strings_to_look_up.size() > 100 or input.dictionary.size() > 100) {
        throw NotImplementedError{};
    }
    io::Output output;
    output.closest_strings_in_dictionary.reserve(input.approximate_strings_to_look_up.size());
    for (auto &lookup : input.approximate_strings_to_look_up) {
        output.closest_strings_in_dictionary.emplace_back();
        std::optional<int32_t> min_distance;
        for (auto &word : input.dictionary) {
            auto distance = FindEditDistanceUpTo(word, lookup, input.max_edit_distance_to_search);
            if (distance) {
                if (not min_distance or distance < min_distance) {
                    output.closest_strings_in_dictionary.back() = word;
                    min_distance = distance;
                }
            }
        }
    }
    return output;
}

std::string GenerateRandomString(int32_t size, char letter_from = 'a', char letter_to = 'z') {
    std::uniform_int_distribution<char> letters_dist{letter_from, letter_to};
    std::string string;
    for (int32_t i = 0; i < size; ++i) {
        string += letters_dist(*rng::GetEngine());
    }
    return string;
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t dictionary_size = std::min(5'000, 1 + test_case_id / 10);
    int32_t n_words = std::min(5'000, 1 + test_case_id / 10);
    int32_t dictionary_word_max_size = std::min(15, 1 + test_case_id / 10);
    int32_t lookup_word_max_size = std::min(20, 1 + test_case_id / 10);
    auto max_char = static_cast<char>('c' + ('z' - 'c') * std::min(1, test_case_id / 100));

    std::uniform_int_distribution<int32_t> dictionary_word_size{1, dictionary_word_max_size};
    std::uniform_int_distribution<int32_t> lookup_word_size{1, lookup_word_max_size};
    io::Input input;
    input.dictionary.resize(dictionary_size);
    for (auto &s : input.dictionary) {
        s = GenerateRandomString(dictionary_word_size(*rng::GetEngine()), 'a', max_char);
    }
    input.approximate_strings_to_look_up.resize(n_words);
    for (auto &s : input.approximate_strings_to_look_up) {
        s = GenerateRandomString(lookup_word_size(*rng::GetEngine()), 'a', max_char);
    }
    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    //    return GenerateRandomTestIo(5000);
    int32_t dictionary_size = 1000;
    int32_t n_words = 1000;
    int32_t dictionary_word_max_size = 15;
    auto max_char = 'z';

    auto root_word = GenerateRandomString(dictionary_word_max_size, 'a', max_char);
    std::uniform_int_distribution<int32_t> root_word_index{0, dictionary_word_max_size - 1};
    io::Input input;
    input.dictionary.resize(dictionary_size);
    for (auto &s : input.dictionary) {
        s = root_word;
        s[root_word_index(*rng::GetEngine())] = GenerateRandomString(1, 'a', max_char)[0];
        s[root_word_index(*rng::GetEngine())] = GenerateRandomString(1, 'a', max_char)[0];
        s[root_word_index(*rng::GetEngine())] = GenerateRandomString(1, 'a', max_char)[0];
        s[root_word_index(*rng::GetEngine())] = GenerateRandomString(1, 'a', max_char)[0];
        s[root_word_index(*rng::GetEngine())] = GenerateRandomString(1, 'a', max_char)[0];
    }
    input.approximate_strings_to_look_up.resize(n_words);
    for (auto &s : input.approximate_strings_to_look_up) {
        s = root_word;
        s[root_word_index(*rng::GetEngine())] = GenerateRandomString(1, 'a', max_char)[0];
        s[root_word_index(*rng::GetEngine())] = GenerateRandomString(1, 'a', max_char)[0];
        s[root_word_index(*rng::GetEngine())] = GenerateRandomString(1, 'a', max_char)[0];
        s[root_word_index(*rng::GetEngine())] = GenerateRandomString(1, 'a', max_char)[0];
        s[root_word_index(*rng::GetEngine())] = GenerateRandomString(1, 'a', max_char)[0];
    }
    return TestIo{input};
}

void TestEditDistance() {
    assert(not FindEditDistanceUpTo("abc", "def", 1));
    assert(not FindEditDistanceUpTo("a", "b", 0));
    assert(FindEditDistanceUpTo("a", "b", 2).value() == 1);
    assert(FindEditDistanceUpTo("abc", "abb", 2).value() == 1);
    assert(FindEditDistanceUpTo("abcde", "bcdea", 2).value() == 2);
    assert(not FindEditDistanceUpTo("abcde", "bcdea", 1));
    assert(not FindEditDistanceUpTo("abcabc", "bcdea", 1));
    assert(not FindEditDistanceUpTo("abcdef", "xyz", 2));
    assert(FindEditDistanceUpTo("abcd", "bbb", 5).value() == 3);
    assert(FindEditDistanceUpTo("aabcdef", "abcdefg", 2).value() == 2);
    assert(not FindEditDistanceUpTo("abccdef", "abcdefg", 1));
    assert(FindEditDistanceUpTo("abcdefg", "abcdefg", 0).value() == 0);
    assert(FindEditDistanceUpTo("book", "back", 20).value() == 2);
    assert(FindEditDistanceUpTo("levenshtein", "distance", 20).value() == 10);
    assert(FindEditDistanceUpTo("distance", "levenshtein", 20).value() == 10);
    assert(FindEditDistanceUpTo("applications", "algorithm", 20).value() == 9);
    assert(FindEditDistanceUpTo("algorithm", "applications", 20).value() == 9);
    assert(FindEditDistanceUpTo("insertions", "deletions", 20).value() == 4);
    assert(FindEditDistanceUpTo(
               "aaaaaaaaaabbbbbbcasdcasdcasdccazxcvasdfasdasdvasdcasdacdbbbbbbbbaaaaaaaaaa",
               "bbbbbbcasdcasdcasdccazxcvasdfasdasdvasdcasdacdbbbbbbbbaaaaaaaaaaaaaaaaaaaa", 20)
               .value() <= 20);
    assert(not FindEditDistanceUpTo("abcdasdasdfasdffasdfasdfef", "z", 20));
    assert(not FindEditDistanceUpTo("a", "zasdfasdfasdfasdfasdfasdfasdf", 20));
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

            for (size_t index = 0; index < test_io.input.approximate_strings_to_look_up.size();
                 ++index) {
                auto &lookup = test_io.input.approximate_strings_to_look_up[index];
                auto &answer = output.closest_strings_in_dictionary[index];
                auto &expected_answer = expected_output.closest_strings_in_dictionary[index];
                if (answer and expected_answer) {
                    auto dist_a = FindEditDistanceUpTo(lookup, answer.value(), 2);
                    auto dist_b = FindEditDistanceUpTo(lookup, expected_answer.value(), 2);
                    if (dist_a.value() != dist_b.value()) {
                        Solve(test_io.input);
                        throw WrongAnswerException{};
                    }
                } else {
                    if (answer != expected_answer) {
                        Solve(test_io.input);
                        throw WrongAnswerException{};
                    }
                }
            }

            //            if (output != expected_output) {
            //                Solve(test_io.input);
            //
            //                std::stringstream ss;
            //                ss <<
            //                "\n================================Expected================================\n"
            //                   << expected_output
            //                   <<
            //                   "\n================================Received================================\n"
            //                   << output << "\n";
            //
            //                throw WrongAnswerException{ss.str()};
            //            }
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

    TestEditDistance();

    TimedChecker timed_checker;

    timed_checker.Check(
        "3\n"
        "hello\n"
        "world\n"
        "test\n"
        "5\n"
        "helo\n"
        "wodrl\n"
        "testt\n"
        "world\n"
        "tsef",
        "hello\n"
        "world\n"
        "test\n"
        "world\n"
        "-1");

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
