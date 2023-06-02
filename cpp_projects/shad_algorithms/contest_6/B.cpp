#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <memory>
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

struct RightCyclicShiftOfMessageInterval {

    int32_t interval_begin;
    int32_t interval_end;
    int32_t shift_by;

    RightCyclicShiftOfMessageInterval(int32_t begin, int32_t end, int32_t by)
        : interval_begin{begin}, interval_end{end}, shift_by{by} {
    }
};

class Input {
public:
    std::string message;
    std::vector<RightCyclicShiftOfMessageInterval> shifts;

    Input() = default;

    explicit Input(std::istream &in) {
        in >> message;
        int32_t n_shifts = 0;
        in >> n_shifts;
        shifts.reserve(n_shifts);

        for (int32_t index = 0; index < n_shifts; ++index) {
            int32_t from = 0;
            int32_t to = 0;
            int32_t by = 0;
            in >> from >> to >> by;
            --from;
            shifts.emplace_back(from, to, by);
        }
    }
};

class Output {
public:
    std::string decoded_message;

    Output() = default;

    explicit Output(std::string string) : decoded_message{std::move(string)} {
    }

    std::ostream &Write(std::ostream &out) const {
        out << decoded_message << '\n';
        return out;
    }

    bool operator!=(const Output &other) const {
        return decoded_message != other.decoded_message;
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

template <class NodeId = int32_t>
class ConnectByNodesInterface : public VirtualBaseClass {
public:
    virtual void Connect(NodeId from, NodeId to) = 0;
};

template <class Edge>
class ConnectByEdgesInterface : public VirtualBaseClass {
public:
    virtual void AddEdge(Edge edge) = 0;
};

template <class Node, class NodeId = int32_t>
class NodeByIdInterface : public VirtualBaseClass {
public:
    virtual Node &operator[](NodeId node_id) = 0;

    virtual Node operator[](NodeId node_id) const = 0;
};

template <class NodeId = int32_t>
class AdjacentInterface : public VirtualBaseClass {
public:
    //    [[nodiscard]] virtual NodeIdIterable NodesAdjacentTo(NodeId node_id) const = 0;
};

template <class NodeId = int32_t>
class VisitAdjacentInterface : public VirtualBaseClass {
public:
    virtual void VisitNodesAdjacentTo(NodeId node_id,
                                        const std::function<void(NodeId node_id)> &function) = 0;
};

template <class NodeState>
class StateInterface : public VirtualBaseClass {
public:
    [[nodiscard]] virtual NodeState GetState() const = 0;

    virtual void SetState(NodeState state) = 0;
};

template <class NodeState, class NodeId = int32_t>
class NodeStateInterface : public VirtualBaseClass {
public:
    [[nodiscard]] virtual NodeState GetNodeState(NodeId node_id) const = 0;

    virtual void SetNodeState(NodeId node_id, NodeState state) = 0;

    virtual ~NodeStateInterface() = default;
};

class GraphTraversal {
public:
    template <class Node>
    void OnNodeEnter(Node &node) {
    }

    template <class Node>
    void OnEdgeDiscovery(Node &from, Node &to) {
    }

    template <class Edge>
    void OnEdgeDiscovery(Edge edge) {
    }

    template <class Node>
    void OnEdgeTraverse(Node &from, Node &to) {
    }

    template <class Edge>
    void OnEdgeTraverse(Edge &edge) {
    }

    template <class Node>
    void OnNodeExit(Node &node) {
    }

    template <class Node>
    void SetNodeStateEntered(Node &node) {
    }

    template <class Node>
    void SetNodeStateExited(Node &node) {
    }

    template <class Node>
    bool IsNodeUnvisited(Node &node) {
        throw NotImplementedError{};
    }

    template <class Node, class NodeIterable = std::vector<Node> &>
    NodeIterable NodesAdjacentTo(Node &node) {
        throw NotImplementedError{};
    }
};

template <class Value>
class GeneratorInterface : public interface::VirtualBaseClass {
    virtual std::optional<Value> Next() = 0;
};

class Treap {
public:
    template <class T>
    void SetLeftChild(T new_left_child) {
        throw NotImplementedError{};
    }

    template <class Treap>
    void SetRightChild(Treap new_right_child) {
        throw NotImplementedError{};
    }

    template <class Treap>
    Treap DropLeftChild() {
        throw NotImplementedError{};
    }

    template <class Treap>
    Treap DropRightChild() {
        throw NotImplementedError{};
    }

    template <class Treap>
    [[nodiscard]] bool CompareKey(const Treap &other) const {
        throw NotImplementedError{};
    }

    template <class Treap>
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

template <class Edge, class = std::enable_if_t<std::is_base_of_v<DirectedEdge, Edge>>>
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

        if (std::is_base_of_v<UndirectedEdge, Edge> and edge.from != edge.to) {
            std::swap(edge.from, edge.to);
            AddDirectedEdge(edge);
        }
    }

    void AddDirectedEdge(Edge edge) {
        edges_table[edge.from].emplace_back(std::move(edge));
    }
};

template <class Node>
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

class AdjacentByIdImplementation : public interface::AdjacentInterface<int32_t>,
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

class DepthFirstSearch {
public:
    template <class Node>
    void Traverse(Node &starting_node) {
        TraverseRecursive(starting_node);
    }

private:
    template <class Node>
    void TraverseRecursive(Node &node) {

        OnNodeEnter(node);
        SetNodeStateEntered(node);

        for (auto &adjacent_node : NodeIdsAdjacentTo(node)) {

            OnEdgeDiscovery(node, adjacent_node);

            if (IsNodeUnvisited(adjacent_node)) {

                OnEdgeTraverse(node, adjacent_node);
                TraverseRecursive(adjacent_node);
            }
        }

        OnNodeExit(node);
        SetNodeStateExited(node);
    }
};

template <class GraphTraversal, class Node,
          class = std::enable_if_t<std::is_base_of_v<interface::GraphTraversal, GraphTraversal>>>
void BreadthFirstSearch(GraphTraversal *graph_traversal, std::deque<Node> starting_nodes_queue) {

    auto &queue = starting_nodes_queue;

    while (not queue.empty()) {
        auto node = queue.front();
        queue.pop_front();

        graph_traversal->OnNodeEnter(node);
        graph_traversal->SetNodeStateEntered(node);

        for (auto &adjacent_node : graph_traversal->NodesAdjacentTo(node)) {

            graph_traversal->OnEdgeDiscovery(node, adjacent_node);

            if (graph_traversal->ShouldVisitNode(adjacent_node)) {

                graph_traversal->OnEdgeTraverse(node, adjacent_node);
                queue.emplace_back(adjacent_node);
            }
        }

        graph_traversal->OnNodeEnter(node);
        graph_traversal->SetNodeStateExited(node);
    }
}

template <class Generator,
          class = std::enable_if_t  // @formatter:off
          <std::is_base_of_v<       // @formatter:on
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

template <class Treap>
bool ComparatorTreapKey(const Treap &left, const Treap &right) {
    return left.CompareKey(right);
}

template <class TreapPtr>
bool ComparatorTreapPtrKey(const TreapPtr &left, const TreapPtr &right) {
    return left->CompareKey(*right);
}

template <class Treap>
bool ComparatorTreapPriority(const Treap &left, const Treap &right) {
    return left.ComparePriority(right);
}

template <class TreapPtr>
bool ComparatorTreapPtrPriority(const TreapPtr &left, const TreapPtr &right) {
    return left->ComparePriority(*right);
}

template <class NodePtrIterator, class NodeComparator>
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

template <class TreapPtrIterator>
typename TreapPtrIterator::value_type ConstructTreap(TreapPtrIterator begin, TreapPtrIterator end) {

    std::sort(begin, end, ComparatorTreapPtrKey<typename TreapPtrIterator::value_type>);

    return ConstructCartesianTree(
        begin, end, ComparatorTreapPtrPriority<typename TreapPtrIterator::value_type>);
}

template <class TreapPtr>
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

template <class Value>
struct Pair {
    Value left;
    Value right;
};

template <class TreapPtr>
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

}  // namespace implementation

class TreapWithSubtreeSize;

using TreapPtr = std::shared_ptr<TreapWithSubtreeSize>;

class TreapWithSubtreeSize : public interface::Treap {
public:
    int32_t original_index = 0;
    int32_t priority = 0;
    int32_t subtree_size = 1;
    TreapPtr left;
    TreapPtr right;
    TreapWithSubtreeSize *parent = nullptr;

    TreapWithSubtreeSize(int32_t original_index, int32_t priority)
        : original_index{original_index}, priority{priority} {
    }

    [[nodiscard]] bool CompareKey(const TreapWithSubtreeSize &other) const {
        return original_index < other.original_index;
    }

    [[nodiscard]] bool ComparePriority(const TreapWithSubtreeSize &other) const {
        return priority < other.priority;
    }

    [[nodiscard]] int32_t GetLeftSubtreeSize() const {
        return left ? left->subtree_size : 0;
    }

    [[nodiscard]] int32_t GetRightSubtreeSize() const {
        return right ? right->subtree_size : 0;
    }

    TreapPtr DropLeftChild() {
        if (parent) {
            throw NodeAlreadyHasParentError{};
        }
        subtree_size -= GetLeftSubtreeSize();
        auto left_child = left;
        if (left_child) {
            left_child->parent = nullptr;
        }
        left.reset();
        return left_child;
    }

    TreapPtr DropRightChild() {
        if (parent) {
            throw NodeAlreadyHasParentError{};
        }
        subtree_size -= GetRightSubtreeSize();
        auto right_child = right;
        if (right_child) {
            right_child->parent = nullptr;
        }
        right.reset();
        return right_child;
    }

    void SetLeftChild(TreapPtr new_left) {
        if (new_left and new_left->parent) {
            throw NodeAlreadyHasParentError{};
        }
        DropLeftChild();
        left = std::move(new_left);
        if (left) {
            left->parent = this;
        }
        subtree_size += GetLeftSubtreeSize();
    }

    void SetRightChild(TreapPtr new_right) {
        if (new_right and new_right->parent) {
            throw NodeAlreadyHasParentError{};
        }
        DropRightChild();
        right = std::move(new_right);
        if (right) {
            right->parent = this;
        }
        subtree_size += GetRightSubtreeSize();
    }

private:
    class NodeAlreadyHasParentError : public std::logic_error {
    public:
        NodeAlreadyHasParentError()
            : std::logic_error(
                  "To ensure correct subtree_size values, compressor_nodes_ should not have parents upon "
                  "dropping children and upon being assigned a new parent."){};
    };
};

template <class Treap>
Treap ConstructTreapWithRandomPrioritiesAndIndexEqualToRangeOfSize(size_t size) {

    auto &engine = *rng::GetEngine();
    std::uniform_int_distribution<int32_t> distribution(INT32_MIN, INT32_MAX);
    std::vector<Treap> treaps_by_index;
    treaps_by_index.reserve(size);

    for (size_t index = 0; index < size; ++index) {
        auto priority = distribution(engine);
        treaps_by_index.emplace_back(
            std::make_shared<typename Treap::element_type>(index, priority));
    }

    return implementation::ConstructTreap(treaps_by_index.begin(), treaps_by_index.end());
}

template <class Treap>
implementation::Pair<Treap> SplitOffLeftSubtreapOfSize(Treap treap,
                                                       int32_t target_left_subtree_size) {
    if (not treap) {
        return {};
    }

    if (treap->GetLeftSubtreeSize() < target_left_subtree_size) {

        auto right_child = treap->DropRightChild();
        auto right_child_split = SplitOffLeftSubtreapOfSize(
            right_child, target_left_subtree_size - treap->GetLeftSubtreeSize() - 1);
        treap->SetRightChild(right_child_split.left);
        return {treap, right_child_split.right};
    } else {

        auto left_child = treap->DropLeftChild();
        auto left_child_split_by_index =
            SplitOffLeftSubtreapOfSize(left_child, target_left_subtree_size);
        treap->SetLeftChild(left_child_split_by_index.right);
        return {left_child_split_by_index.left, treap};
    }
}

void ApplyShiftsInReverse(TreapPtr treap,
                          const std::vector<io::RightCyclicShiftOfMessageInterval> &shifts) {

    for (auto iterator = shifts.rbegin(); iterator != shifts.rend(); ++iterator) {
        auto &shift = *iterator;

        auto [start_and_middle, end] = SplitOffLeftSubtreapOfSize(treap, shift.interval_end);

        auto [start, middle] = SplitOffLeftSubtreapOfSize(start_and_middle, shift.interval_begin);

        auto [middle_left, middle_right] = SplitOffLeftSubtreapOfSize(middle, shift.shift_by);

        using implementation::Merge;
        treap = Merge(Merge(start, middle_right), Merge(middle_left, end));
    }
}

void DecodeTreapRecursive(TreapPtr treap, const std::string &original_message,
                          std::string &decoded_message) {
    while (treap) {
        DecodeTreapRecursive(treap->left, original_message, decoded_message);
        auto letter = original_message[treap->original_index];
        decoded_message.push_back(letter);
        treap = treap->right;
    }
}

std::string DecodeTreapMessage(const TreapPtr &treap, const std::string &original_message) {
    std::string decoded_message;
    decoded_message.reserve(original_message.size());
    DecodeTreapRecursive(treap, original_message, decoded_message);
    return decoded_message;
}

io::Output Solve(const io::Input &input) {

    auto treap = ConstructTreapWithRandomPrioritiesAndIndexEqualToRangeOfSize<TreapPtr>(
        input.message.size());

    ApplyShiftsInReverse(treap, input.shifts);

    auto decoded_message = DecodeTreapMessage(treap, input.message);

    return io::Output{decoded_message};
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
    throw NotImplementedError{};
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    throw NotImplementedError{};
    io::Input input;
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
        "logoduck\n"
        "3\n"
        "1 3 1\n"
        "4 5 1\n"
        "1 4 1\n",
        "goodluck");

    for (int32_t make_sure = 0; make_sure < 5; ++make_sure) {

        int32_t size = 50'000;
        std::mt19937 mersenne_twister{42};
        std::uniform_int_distribution<char> char_distribution('a', 'z');

        io::Input input;
        input.message.reserve(size);

        for (int32_t i = 0; i < size; ++i) {
            input.message.push_back(char_distribution(mersenne_twister));
        }

        std::uniform_int_distribution<int32_t> int_distribution(0, size - 1);
        input.shifts.reserve(size);

        for (int32_t i = 0; i < size; ++i) {
            auto from = int_distribution(mersenne_twister);
            auto to = int_distribution(mersenne_twister);
            while (from == to) {
                to = int_distribution(mersenne_twister);
            }
            if (to < from) {
                std::swap(from, to);
            }
            auto by = 1 + (int_distribution(mersenne_twister) % (to - from));
            input.shifts.emplace_back(from, to, by);
        }

        auto output = timed_check.TimedSolve(input);

        io::Input reversed_input;
        reversed_input.message = output.decoded_message;
        reversed_input.shifts = input.shifts;

        std::reverse(reversed_input.shifts.begin(), reversed_input.shifts.end());
        for (auto &shift : reversed_input.shifts) {
            shift.shift_by = (shift.interval_end - shift.interval_begin) - shift.shift_by;
        }

        auto reversed_output = timed_check.TimedSolve(reversed_input);

        assert(input.message == reversed_output.decoded_message);
    }

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
